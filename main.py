import math
import os
import re
import sys
import time
import warnings
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

import unicodedata

from run_finalize import finalize_run
from src.code.filter import apply_filters_pipeline, solo_match_uo_then_links_check, solo_shortname_guard, \
     solo_validation
from src.code.get_cases import get_cases, get_cases_from_txt
from src.code.get_matched_ids import get_matched_ids
from src.code.name_normalisation import exact_normalization_series
from src.code.delaware_output import (
    select_delaware_single_matches,
    select_multiple_directid,
    build_ownership_map,
    split_direct_linked,
    stamp_ownership,
)
from src.code.directid_grouping import build_directid_groups
from src.db_connexion.data_extraction import load_query, connect_db
from utils.config import PROJECT_NAME

OUTPUT_DIR = Path("output")

# Registered-agent addresses (Delaware). Editable Excel file: one column named
# "Address", one address per row. Used to exempt matches from the
# StreetSimilarity rule (see the validation step in run()).
AGENT_ADDRESS_FILE = Path("data/delaware_agent_addresses.xlsx")


def _load_agent_addresses(path: Path = AGENT_ADDRESS_FILE) -> list:
    """
    Load the registered-agent address list from Excel.

    Returns a list of normalized address strings. If the file is missing or
    unreadable the pipeline continues with an empty list (the agent-address
    exemption simply never fires) rather than failing the run.
    """
    if not path.exists():
        print(f"[AGENT] {path} not found — agent-address exemption disabled.")
        return []
    try:
        df = pd.read_excel(path)
    except Exception as e:
        print(f"[AGENT] could not read {path} ({e}) — agent-address exemption disabled.")
        return []

    if "Address" in df.columns:
        col = df["Address"]
    elif len(df.columns) >= 1:
        col = df.iloc[:, 0]  # fall back to the first column
    else:
        print(f"[AGENT] {path} has no usable column — agent-address exemption disabled.")
        return []

    addresses = []
    for raw in col.dropna().astype(str):
        norm = normalize_text(raw)
        if norm:
            addresses.append(norm)

    addresses = sorted(set(addresses))
    print(f"[AGENT] loaded {len(addresses)} registered-agent addresses from {path}")
    return addresses


try:
    import usaddress
    _HAS_USADDRESS = True
except ImportError:
    _HAS_USADDRESS = False


def _usaddress_city_state(text):
    """Pull (city, state) out of a street string. Returns (None, None) on failure."""
    if not _HAS_USADDRESS or not text or not str(text).strip():
        return None, None
    try:
        tokens = usaddress.parse(str(text).strip())
    except Exception:
        return None, None

    city = " ".join(t.strip(" ,") for t, lab in tokens if lab == "PlaceName").strip()
    state = " ".join(t.strip(" ,") for t, lab in tokens if lab == "StateName").strip()
    return (city or None), (state or None)


### Helpers ###

def _insert_blank_rows_between_groups(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Insert a blank row between groups (no trailing blank row).
    """
    if df.empty:
        return df
    parts = []
    for _, g in df.groupby(group_col, sort=False):
        parts.append(g)
        parts.append(pd.DataFrame({c: [None] for c in df.columns}))
    parts.pop()
    return pd.concat(parts, ignore_index=True)



def chunked_insert(cur, ids: list):
    chunk_size = 1000
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i:i+chunk_size]
        values_str = ",".join(f"('{x}')" for x in chunk)
        sql_insert = f"INSERT INTO #TempIds (Id) VALUES {values_str};"
        cur.execute(sql_insert)


@connect_db
def _fetch_ownership_links(conn, ids):
    """Fetch parent/child links and UO links for a set of Ids.
    Returns (link_rows, uo_rows) as lists of tuples."""
    clean_ids = sorted({str(x).rstrip("!") for x in ids if pd.notna(x)})
    if not clean_ids:
        return [], []

    link_rows, uo_rows = [], []
    with conn.cursor() as cur:
        cur.execute("CREATE TABLE #TempIds (Id varchar(255) NOT NULL PRIMARY KEY);")
        chunk_size = 1000
        for i in range(0, len(clean_ids), chunk_size):
            chunk = clean_ids[i:i + chunk_size]
            values_str = ",".join(f"('{x}')" for x in chunk)
            cur.execute(f"INSERT INTO #TempIds (Id) VALUES {values_str};")

        try:
            sql = load_query("queries/get_ownership_links.sql").format()
            cur.execute(sql)
            link_rows = [(r[0], r[1]) for r in cur.fetchall()]
        except Exception as e:
            print(f"[OWNERSHIP][WARN] links fetch failed: {e}")

        try:
            sql_uo = load_query("queries/get_uo_links_by_ids.sql").format()
            cur.execute(sql_uo)
            uo_rows = [(r[0], r[1]) for r in cur.fetchall()]
        except Exception as e:
            print(f"[OWNERSHIP][WARN] UO links fetch failed: {e}")

        cur.execute("DROP TABLE #TempIds;")

    print(f"[OWNERSHIP] {len(link_rows):,} links, {len(uo_rows):,} UO links fetched")
    return link_rows, uo_rows


@connect_db
def _fetch_cities(conn, ids):
    mapping_data = []
    clean_ids_set = set()

    for x in ids:
        if pd.isna(x):
            continue
        s_orig = str(x)
        s_clean = s_orig.rstrip('!')
        mapping_data.append((s_orig, s_clean))
        clean_ids_set.add(s_clean)

    df_map = pd.DataFrame(mapping_data, columns=["IdOriginal", "IdLookup"])

    unique_clean_ids = list(clean_ids_set)

    if not unique_clean_ids:
        # Si aucun ID valide, on renvoie un DF vide avec les colonnes attendues
        return pd.DataFrame(columns=["Id", "PCCity", "Postcode", "Address", "NrLinks"])

    with conn.cursor() as cur:
        cur.execute("CREATE TABLE #TempIds (Id varchar(255) NOT NULL PRIMARY KEY);")

        chunk_size = 1000
        for i in range(0, len(unique_clean_ids), chunk_size):
            chunk = unique_clean_ids[i:i + chunk_size]
            # Attention : on insère les IDs NETTOYÉS
            values_str = ",".join(f"('{x}')" for x in chunk)
            cur.execute(f"INSERT INTO #TempIds (Id) VALUES {values_str};")

        sql_tpl = load_query("queries/get_pccity_by_ids.sql")
        sql = sql_tpl.format()
        cur.execute(sql)

        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()

        # Remplace None par ""
        rows_cleaned = []
        for row in rows:
            if row is None:
                continue
            cleaned_row = tuple("" if v is None else v for v in row)
            rows_cleaned.append(cleaned_row)

        cur.execute("DROP TABLE #TempIds;")

    df_sql = pd.DataFrame(rows_cleaned, columns=cols)


    df_final = df_map.merge(df_sql, left_on="IdLookup", right_on="Id", how="inner")

    cols_to_drop = ["IdLookup", "Id"]

    df_final = df_final.drop(columns=cols_to_drop)

    df_final = df_final.rename(columns={"IdOriginal": "Id"})

    return df_final



_ADDRESS_ABBREV = [
    (r"\bgrn\b", "green"),
    (r"\bst\b", "street"),
    (r"\brd\b", "road"),
    (r"\bave\b", "avenue"),
    (r"\bblvd\b", "boulevard"),
    (r"\bdr\b", "drive"),
    (r"\bln\b", "lane"),
    (r"\bhwy\b", "highway"),
    (r"\bste\b", "suite"),
    (r"\bapt\b", "apartment"),
]

def expand_address_abbreviations(text: str) -> str:
    if not text:
        return text
    for pattern, repl in _ADDRESS_ABBREV:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text

def normalize_text(s: str) -> str:
    if not s:
        return ""

    s = s.lower()

    # Supprime accents, mais garde chiffres/lettres
    s = ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

    # Sépare 221b -> 221 b (utile pour unités)
    s = re.sub(r"(\d)([a-z])\b", r"\1 \2", s)

    # Remplace ponctuation par espaces, garde [a-z0-9]
    s = re.sub(r"[^a-z0-9]+", " ", s)

    # Expand abréviations (grn -> green, st -> street, etc.)
    s = expand_address_abbreviations(s)

    # Normalise espaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def calculate_similarity(city1, city2):
    city1_norm = normalize_text(city1) if pd.notna(city1) else ""
    city2_norm = normalize_text(city2) if pd.notna(city2) else ""
    if city1_norm == "" and city2_norm == "":
        return 0
    return SequenceMatcher(None, city1_norm, city2_norm).ratio() * 100



def add_city_similarity_columns(unique_df):
    # Merges below fail on an empty frame; return the expected schema instead.
    if unique_df is None or unique_df.empty:
        out = unique_df.copy() if unique_df is not None else pd.DataFrame()
        for col in ("City 1", "City 2", "Postcode 1", "Postcode 2",
                    "Address 1", "Address 2", "NrLinks 1", "NrLinks 2",
                    "LastTransferDate", "Similarity"):
            if col not in out.columns:
                out[col] = pd.Series(dtype=object)
        return out

    unique_ids = pd.concat([unique_df["OriginalId"], unique_df["MatchedId"]]).astype(str).unique().tolist()
    city_df = _fetch_cities(unique_ids)  # Décorée, connexion gérée automatiquement

    # Fusion pour OriginalId (ville 1 + autres colonnes)
    unique_df = unique_df.merge(
        city_df.rename(columns={
            "Id": "OriginalId",
            "PCCity": "City 1",
            "Postcode": "Postcode 1",
            "Address": "Address 1",
            "NrLinks": "NrLinks 1",
            "LastTransferDate": "_LastTransferDate_orig"
        }),
        on="OriginalId",
        how="left",
    )

    # LastTransferDate n'est conservé que côté MatchedId (retire le côté OriginalId).
    if "_LastTransferDate_orig" in unique_df.columns:
        unique_df = unique_df.drop(columns=["_LastTransferDate_orig"])

    # Fusion pour MatchedId (ville 2 + autres colonnes)
    unique_df = unique_df.merge(
        city_df.rename(columns={
            "Id": "MatchedId",
            "PCCity": "City 2",
            "Postcode": "Postcode 2",
            "Address": "Address 2",
            "NrLinks": "NrLinks 2"
        }),
        on="MatchedId",
        how="left",
    )

    def vectorized_calculate_similarity(city1_series, city2_series):
        return [
            calculate_similarity(c1, c2) for c1, c2 in zip(city1_series, city2_series)
        ]

    unique_df["Similarity"] = vectorized_calculate_similarity(unique_df["City 1"], unique_df["City 2"])

    cols = list(unique_df.columns)
    # Colonnes à placer après Similarity
    new_cols_order = []
    for prefix in ["1", "2"]:
        new_cols_order += [f"Postcode {prefix}", f"Address {prefix}", f"NrLinks {prefix}"]
    # LastTransferDate (MatchedId) placé juste après NrLinks 2
    if "LastTransferDate" in cols:
        new_cols_order.append("LastTransferDate")
    # Colonnes avant Similarity
    cols_before = [c for c in cols if c not in new_cols_order + ["Similarity"]]
    # Construire nouvelle liste de colonnes
    final_cols = cols_before + ["Similarity"] + new_cols_order
    unique_df = unique_df[final_cols]

    return unique_df




STATE_ABBR_TO_FULL = {
    'AL': 'ALABAMA', 'AK': 'ALASKA', 'AZ': 'ARIZONA', 'AR': 'ARKANSAS',
    'CA': 'CALIFORNIA', 'CO': 'COLORADO', 'CT': 'CONNECTICUT',
    'DE': 'DELAWARE', 'FL': 'FLORIDA', 'GA': 'GEORGIA', 'HI': 'HAWAII',
    'ID': 'IDAHO', 'IL': 'ILLINOIS', 'IN': 'INDIANA', 'IA': 'IOWA',
    'KS': 'KANSAS', 'KY': 'KENTUCKY', 'LA': 'LOUISIANA',
    'ME': 'MAINE', 'MD': 'MARYLAND', 'MA': 'MASSACHUSETTS',
    'MI': 'MICHIGAN', 'MN': 'MINNESOTA', 'MS': 'MISSISSIPPI',
    'MO': 'MISSOURI', 'MT': 'MONTANA', 'NE': 'NEBRASKA',
    'NV': 'NEVADA', 'NH': 'NEW HAMPSHIRE', 'NJ': 'NEW JERSEY',
    'NM': 'NEW MEXICO', 'NY': 'NEW YORK', 'NC': 'NORTH CAROLINA',
    'ND': 'NORTH DAKOTA', 'OH': 'OHIO', 'OK': 'OKLAHOMA',
    'OR': 'OREGON', 'PA': 'PENNSYLVANIA', 'RI': 'RHODE ISLAND',
    'SC': 'SOUTH CAROLINA', 'SD': 'SOUTH DAKOTA', 'TN': 'TENNESSEE',
    'TX': 'TEXAS', 'UT': 'UTAH', 'VT': 'VERMONT', 'VA': 'VIRGINIA',
    'WA': 'WASHINGTON', 'WV': 'WEST VIRGINIA', 'WI': 'WISCONSIN',
    'WY': 'WYOMING'
}

def clean_text(text):
    return text.strip()

def detect_state_short_or_full(tokens):
    found_state = None
    for token in tokens[:]:
        token_up = token.upper()
        if token_up in STATE_ABBR_TO_FULL:
            found_state = STATE_ABBR_TO_FULL[token_up]
            tokens.remove(token)
            break
        for full in STATE_ABBR_TO_FULL.values():
            if token_up == full.upper():
                found_state = full
                tokens.remove(token)
                break
    return found_state

def detect_state_full(text):
    text_up = text.upper()
    for full in STATE_ABBR_TO_FULL.values():
        if full.upper() in text_up:
            return full
    return None


def process_separated_address(address, postcode, city):
    IGNORED_ABBR_IN_ADDRESS = {"FL", "MT", "CT", "CO"}
    COUNTRY_VARIANTS = {
        "USA", "UNITED STATES", "UNITED STATES OF AMERICA", "U.S.A", "U.S.",
        "U S A", "U S",
    }

    def remove_punctuation(text):
        if not text:
            return ''
        # Separators, not noise: "CHICAGO,IL" and "ST.PAUL" have no space.
        text = re.sub(r'[,.]', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def remove_country_variants(text):
        if not text:
            return text
        for variant in COUNTRY_VARIANTS:
            pattern = r'\b' + re.escape(variant) + r'\b'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
        return text

    # Nettoyer ponctuation
    address = remove_punctuation(address)
    postcode = remove_punctuation(postcode)
    city = remove_punctuation(city)

    # Supprimer variantes pays dans chaque champ dès le départ
    address = remove_country_variants(address)
    postcode = remove_country_variants(postcode)
    city = remove_country_variants(city)

    def detect_state(text, is_address=False):
        if not text:
            return None, text
        tokens = text.split()
        state_found = None
        text_upper = text.upper()
        # Read the tail first: an anywhere-scan misfires on cities named after
        # a state ("NEW YORK CITY NY", "LA JOLLA CA"). Longest tail wins.
        for n in (3, 2, 1):
            if len(tokens) > n:  # never consume the whole string
                tail = ' '.join(tokens[-n:]).upper()
                if tail in {v.upper() for v in STATE_ABBR_TO_FULL.values()}:
                    state_found = next(
                        v for v in STATE_ABBR_TO_FULL.values() if v.upper() == tail
                    )
                    return state_found, ' '.join(tokens[:-n]).strip()
        if len(tokens) > 1:
            last = tokens[-1].upper()
            if last in STATE_ABBR_TO_FULL and not (
                is_address and last in IGNORED_ABBR_IN_ADDRESS
            ):
                return STATE_ABBR_TO_FULL[last], ' '.join(tokens[:-1]).strip()

        for full_state in STATE_ABBR_TO_FULL.values():
            if full_state.upper() in text_upper:
                pattern = r"\b" + re.escape(full_state) + r"\b"
                text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
                state_found = full_state
                break
        if not state_found:
            for i, t in enumerate(tokens[:]):
                if is_address and t.upper() in IGNORED_ABBR_IN_ADDRESS:
                    continue
                if t.upper() in STATE_ABBR_TO_FULL:
                    state_found = STATE_ABBR_TO_FULL[t.upper()]
                    tokens.pop(i)
                    text = ' '.join(tokens).strip()
                    break
        return state_found, text

    state, city = detect_state(city, is_address=False)
    # detect_state already stripped it from the city; a second pass would eat
    # part of the name ("NEW YORK CITY" -> "CITY").
    state_came_from_city = state is not None
    if not state:
        state, address = detect_state(address, is_address=True)

    city_part = []
    if city:
        city_tokens = city.split()
        city_words = [t for t in city_tokens if not t.isdigit()]
        # Extraction du postcode dans city améliorée (chercher un code postal et le retirer)
        match_postcode_in_city = re.search(r'\b\d{5}\b', city)
        if match_postcode_in_city:
            postcode_found = match_postcode_in_city.group(0)
            city = re.sub(r'\b' + re.escape(postcode_found) + r'\b', '', city).strip()
            if not postcode:
                postcode = postcode_found
            city_tokens = city.split()
            city_words = [t for t in city_tokens if not t.isdigit()]

        if len(city_words) <= 3:
            city = ' '.join(city_words)
        else:
            city_part = city_tokens
            city = None
    else:
        city = None

    def contains_five_digits(text):
        return bool(text and re.search(r'\b\d{5}\b', text))

    def extract_postcode(text):
        if not text:
            return None, text
        match = re.search(r'\b\d{5}\b', text)
        if match:
            pc = match.group(0)
            text = text.replace(pc, "").strip(", ")
            return pc, text
        return None, text

    # Recherche poste dans postcode, puis dans address si pas déjà trouvé
    if not (postcode and contains_five_digits(postcode)):
        pc, new_address = extract_postcode(address)
        if pc:
            postcode = pc
            address = new_address if new_address else None

    city_tokens = city.split() if city else []
    postcode_tokens = postcode.split() if postcode else []
    state_tokens = state.split() if state else []
    address_tokens = address.split() if address else []

    street_tokens = address_tokens + city_part
    # The state word is only noise in the street if it was read from there.
    # When it came from the City field, an occurrence in the street is a street
    # name ("580 CALIFORNIA ST"), not the state.
    to_remove = set(t.upper() for t in postcode_tokens)
    if not state_came_from_city:
        to_remove |= set(t.upper() for t in state_tokens)
    street_tokens_filtered = [t for t in street_tokens if t.upper() not in to_remove]
    street = ' '.join(street_tokens_filtered).strip()

    # --- Suppression des doublons entre colonnes (state, postcode, city dans street et city) ---

    def remove_word_from_text(word, text):
        if not word or not text:
            return text
        pattern = r'\b' + re.escape(word) + r'\b'
        return re.sub(pattern, '', text, flags=re.IGNORECASE).strip()

    # Supprimer country variants dans city et street aussi au cas où il restait
    for variant in COUNTRY_VARIANTS:
        city = remove_word_from_text(variant, city)
        street = remove_word_from_text(variant, street)

    # Suppression de state dans city et street
    if state:
        if not state_came_from_city:
            city = remove_word_from_text(state, city)
            street = remove_word_from_text(state, street)

    # Suppression de postcode dans city et street
    if postcode:
        city = remove_word_from_text(postcode, city)
        street = remove_word_from_text(postcode, street)

    # Suppression de city dans street (après nettoyage éventuel plus haut)
    if city:
        street = remove_word_from_text(city, street)

    # Fallback: where the City field yielded nothing, the city may be sitting
    # inside the street ("2880 ZELL DRIVE LAGUNA BEACH"). usaddress can tell a
    # city apart from a street name, which a token scan cannot.
    if street and (not city or not state):
        u_city, u_state = _usaddress_city_state(street)
        if not city and u_city:
            city = u_city
            street = remove_word_from_text(u_city, street)
        if not state and u_state:
            state = STATE_ABBR_TO_FULL.get(u_state.upper(), u_state.upper())
            street = remove_word_from_text(u_state, street)

    return {
        "street": street if street else None,
        "city": city if city else None,
        "postcode": postcode if postcode else None,
        "state": state if state else None,
    }



def parse_separated(row, prefix):
    addr = row.get(f"Address {prefix}", "")
    post = row.get(f"Postcode {prefix}", "")
    city = row.get(f"City {prefix}", "")
    parsed = process_separated_address(addr, post, city)
    return pd.Series({
        f"Street{prefix}": parsed["street"],
        f"City{prefix}": parsed["city"],
        f"State{prefix}": parsed["state"],
        f"PostalCode{prefix}": parsed["postcode"]
    })

def calculate_address_similarity(addr1, addr2):
    val1norm = normalize_text(addr1) if pd.notna(addr1) else ""
    val2norm = normalize_text(addr2) if pd.notna(addr2) else ""

    if not val1norm or not val2norm:
        return 0.0

    # Inclusion “forte” (maintenant que GRN->GREEN et que le numéro est conservé)
    if len(val1norm) >= 5 and len(val2norm) >= 5:
        if val1norm in val2norm or val2norm in val1norm:
            return 100.0

    return SequenceMatcher(None, val1norm, val2norm).ratio() * 100



def add_parsed_addresses(unique_df):
    PARSED_COLS = [
        "Street1", "City1", "State1", "PostalCode1",
        "Street2", "City2", "State2", "PostalCode2",
        "IsSamePostalCode", "StreetSimilarity", "CitySimilarity", "IsSameState",
    ]

    # parallel_parse yields a column-less frame on empty input; return the
    # expected schema so callers still get a usable empty frame.
    if unique_df is None or unique_df.empty:
        out = unique_df.copy() if unique_df is not None else pd.DataFrame()
        for col in PARSED_COLS:
            if col not in out.columns:
                out[col] = pd.Series(dtype=object)
        return out

    def parallel_parse(df, prefix):
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda r: parse_separated(r, prefix), [row for _, row in df.iterrows()]))
        return pd.DataFrame(results)

    parsed_orig = parallel_parse(unique_df, "1")
    parsed_match = parallel_parse(unique_df, "2")

    unique_df = pd.concat([unique_df, parsed_orig, parsed_match], axis=1)

    unique_df["IsSamePostalCode"] = (
        (unique_df["PostalCode1"] == unique_df["PostalCode2"]) &
        unique_df["PostalCode1"].notnull() & unique_df["PostalCode2"].notnull()
    ).astype(int)

    # --- MODIFICATION ICI ---
    # Utilisation de la nouvelle fonction pour les adresses
    unique_df["StreetSimilarity"] = unique_df.apply(
        lambda row: calculate_address_similarity(row["Street1"], row["Street2"]), axis=1
    )

    # Pour la ville, on garde la méthode classique (souvent plus courte que 5 chars donc l'inclusion est risquée)
    unique_df["CitySimilarity"] = unique_df.apply(
        lambda row: calculate_similarity(row["City1"], row["City2"]), axis=1
    )

    # def same_state(row):
    #     s1 = row.get("State1")
    #     s2 = row.get("State2")
    #     if not s1 or not s2:
    #         return 0
    #     return 1 if s1.upper() == s2.upper() else 0

    def same_state(row):
        s1, s2 = row["State1"], row["State2"]  # actual col names
        if pd.isna(s1) or pd.isna(s2):
            return 0
        return 1 if str(s1).upper() == str(s2).upper() else 0

    unique_df["IsSameState"] = unique_df.apply(same_state, axis=1)

    raw_addr1 = unique_df["Address 1"].fillna("").str.lower()
    raw_addr2 = unique_df["Address 2"].fillna("").str.lower()

    # --- RÈGLE 1 : CT CORPORATION ---
    # "corporation trust center" VS "1209 orange street"
    pat_ctc = "corporation trust center"
    pat_orange = "1209 orange street"

    # Cas A : Gauche = CTC, Droite = Orange
    mask_r1_a = raw_addr1.str.contains(pat_ctc, regex=False) & raw_addr2.str.contains(pat_orange, regex=False)
    # Cas B : Gauche = Orange, Droite = CTC
    mask_r1_b = raw_addr1.str.contains(pat_orange, regex=False) & raw_addr2.str.contains(pat_ctc, regex=False)

    mask_rule_1 = mask_r1_a | mask_r1_b

    # --- RÈGLE 2 : CSC (Corporation Service Company) ---
    # "corporation service company" VS ("251 little falls" OU "2711 centerville road")
    pat_csc = "corporation service company"
    # Note : On cherche la partie fixe de l'adresse. "2711 centerville road" matchera automatiquement "... road, suite 400"
    pat_locs_regex = r"251 little falls|2711 centerville road"

    mask_csc_1 = raw_addr1.str.contains(pat_csc, regex=False)
    mask_csc_2 = raw_addr2.str.contains(pat_csc, regex=False)

    mask_locs_1 = raw_addr1.str.contains(pat_locs_regex, regex=True)
    mask_locs_2 = raw_addr2.str.contains(pat_locs_regex, regex=True)

    # Cas A : Gauche = CSC, Droite = Locs
    mask_r2_a = mask_csc_1 & mask_locs_2
    # Cas B : Gauche = Locs, Droite = CSC
    mask_r2_b = mask_locs_1 & mask_csc_2

    mask_rule_2 = mask_r2_a | mask_r2_b

    # --- APPLICATION DES OVERRIDES ---
    final_override_mask = mask_rule_1 | mask_rule_2

    if final_override_mask.any():
        # On force à 100% si l'une des règles est remplie
        unique_df.loc[final_override_mask, "StreetSimilarity"] = 100.0
        unique_df.loc[final_override_mask, "CitySimilarity"] = 100.0


    return unique_df







@connect_db
def _fetch_oldest_link(conn, orig_ids: List[str]) -> pd.DataFrame:
    ids = [str(x) for x in orig_ids if pd.notna(x)]
    if not ids:
        return pd.DataFrame(columns=["OriginalId", "OldestSourceDate"])

    with conn.cursor() as cur:
        # Création table temporaire
        cur.execute("CREATE TABLE #TempIds (Id varchar(255) NOT NULL PRIMARY KEY);")

        # Insertion par lots
        chunk_size = 1000
        for i in range(0, len(ids), chunk_size):
            chunk = ids[i:i + chunk_size]
            values_str = ",".join(f"('{x}')" for x in chunk)
            cur.execute(f"INSERT INTO #TempIds (Id) VALUES {values_str};")

        # Charger la requête principale depuis fichier
        sql_tpl = load_query("queries/get_oldest_link_date.sql")
        sql = sql_tpl.format(orig_values="")  # Orig_values n'est plus utilisé ici

        # Exécution de la requête principale
        cur.execute(sql)
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()

        # Drop table temporaire
        cur.execute("DROP TABLE #TempIds;")

    if not rows:
        return pd.DataFrame(columns=["OriginalId", "OldestSourceDate"])
    return pd.DataFrame.from_records(rows, columns=cols).astype({"OriginalId": "string", "OldestSourceDate": "string"})


def run() -> tuple[int, dict]:
    """
    Orchestrates the full matching pipeline and produces:
      - matches_delaware.xlsx (Entries, Matching_Propositions, Delaware_Single_Match,
        DirectLinked, directid_Same_City_State, directid_Same_city,
        multiple_directid_delaware, NoMatch)
      - output.txt   (only UniqueMatch as OldID/NewID/UserID)

    The tier tables (UniqueMatch, the ExactName sheets and so on) are still
    calculated — they feed the Delaware phases — but are no longer written to a
    workbook of their own.

    High-level steps:
      1) Load cases (SQL) and prepare base columns.
      2) Parallel per-country exact matching (normalized keys).
      3) Deduplicate (prefer current ID over 'ID!').
      4) Apply isolation filters + solo fallback rules.
      5) Normalize names for export.
      6) Export Excel and TXT.
    """

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    import pandas as pd

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 0)
    pd.set_option('display.max_colwidth', None)
    """
    s = pd.Series("E-CAPITAL I")
    print(exact_normalization_series(s))
    return
    """
    warnings.filterwarnings("ignore")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1) Load + base prep -------------------------------------------------
    print("[1] Fetching cases...")
    df_cases = get_cases()
    #df_cases = get_cases_from_txt("Entries US.txt")
    print(f"→ {len(df_cases)} cases retrieved.")

    df_cases = (
        df_cases
        .rename(columns={"Id": "OriginalId"})
        .assign(CleanedName=lambda df: df["Name"])
    )

    df_cases["CountryCode"] = df_cases["OriginalId"].astype(str).str[:2]
    df_cases = df_cases.sort_values(by=["CleanedName", "CountryCode"])

    print(f"[2] {len(df_cases)} cases to process.")
    print("Countries to process:", df_cases["CountryCode"].unique())

    # Country scheduling (priorities first, then the rest)
    priority_countries = ["US", "CN", "FR"]
    all_countries = list(df_cases["CountryCode"].unique())
    sorted_countries = priority_countries + [c for c in all_countries if c not in priority_countries]
    """
    # Test subset
    test_countries = {"BR"}
    sorted_countries = [c for c in sorted_countries if c in test_countries]
    print(f"[INFO] Selected countries for test run: {', '.join(sorted_countries) if sorted_countries else '(none)'}")
    """

    # --- 2) Matching per-country -----------------------------
    print("[MATCH] Launching country matching...")
    unique_countries = list(dict.fromkeys(sorted_countries))

    matched_df = pd.DataFrame()

    tCOuntries = time.perf_counter()

    if unique_countries:
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_country = {
                executor.submit(
                    get_matched_ids,
                    df_cases[df_cases["CountryCode"] == country][["OriginalId", "Name", "CleanedName", "CountryCode"]],
                    country
                ): country
                for country in unique_countries
                if not df_cases[df_cases["CountryCode"] == country].empty
            }
            for fut in as_completed(future_to_country):
                country = future_to_country[fut]
                try:
                    df_matched, elapsed = fut.result()
                    print(f"[MATCH] {country} done: {len(df_matched)} matches in {elapsed:.2f}s")
                    if not df_matched.empty:
                        matched_df = pd.concat([matched_df, df_matched], ignore_index=True)
                except Exception as e:
                    print(f"[ERROR] Matching failed for {country}: {e}")
    else:
        print("[MATCH] No countries selected, skipping matching.")

    elapsed_countries = time.perf_counter() - tCOuntries
    hours, rem = divmod(elapsed_countries, 3600)
    minutes, seconds = divmod(rem, 60)

    matched_df_raw = matched_df.copy()

    # --- 2bis) Isolated ids: self-match against the full Entries pool --------
    # "Isolated" = OriginalId got ZERO proposals from the normal matching pass.
    # We re-run the SAME matcher for these ids only, then keep proposals whose
    # MatchedId is itself an Entries OriginalId (Entries-vs-Entries only).
    # Resulting rows are merged straight into matched_df_raw so they flow
    # through the exact same de-dup / tiering logic as every other match —
    # no separate sheet, no separate logic.
    try:
        _proposed_ids = set(matched_df_raw["OriginalId"].astype(str)) if not matched_df_raw.empty else set()
        _entries_id_set = set(df_cases["OriginalId"].astype(str))
        _isolated_ids = _entries_id_set - _proposed_ids

        if _isolated_ids:
            print(f"[ISOLATED] {len(_isolated_ids)} isolated ids (zero proposals) — self-matching against Entries...")
            iso_cases = df_cases[df_cases["OriginalId"].astype(str).isin(_isolated_ids)][
                ["OriginalId", "Name", "CleanedName", "CountryCode"]
            ]
            iso_matched_df = pd.DataFrame()
            iso_countries = list(iso_cases["CountryCode"].unique())

            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_country = {
                    executor.submit(
                        get_matched_ids,
                        iso_cases[iso_cases["CountryCode"] == c],
                        c
                    ): c
                    for c in iso_countries
                    if not iso_cases[iso_cases["CountryCode"] == c].empty
                }
                for fut in as_completed(future_to_country):
                    c = future_to_country[fut]
                    try:
                        df_m, _el = fut.result()
                        if df_m is not None and not df_m.empty:
                            iso_matched_df = pd.concat([iso_matched_df, df_m], ignore_index=True)
                    except Exception as e:
                        print(f"[ISOLATED][ERROR] self-match failed for {c}: {e}")

            if not iso_matched_df.empty:
                iso_matched_df["OriginalId"] = iso_matched_df["OriginalId"].astype(str)
                iso_matched_df["MatchedId"] = iso_matched_df["MatchedId"].astype(str)
                # Entries-vs-Entries only: keep matches whose MatchedId is a real Entries id.
                iso_matched_df = iso_matched_df[
                    iso_matched_df["MatchedId"].str.rstrip("!").isin(_entries_id_set)
                ]
                print(f"[ISOLATED] {len(iso_matched_df)} self-matched rows found within Entries pool.")
                matched_df_raw = pd.concat([matched_df_raw, iso_matched_df], ignore_index=True)
    except Exception as e:
        print(f"[ISOLATED][ERROR] isolated-id self-matching step failed, skipping: {e}")

    matched_df_raw = matched_df_raw[matched_df_raw["OriginalId"] != matched_df_raw["MatchedId"]]

    matched_df_raw["_min_id"] = np.where(
        matched_df_raw["OriginalId"].astype(str) < matched_df_raw["MatchedId"].astype(str),
        matched_df_raw["OriginalId"], matched_df_raw["MatchedId"]
    )
    matched_df_raw["_max_id"] = np.where(
        matched_df_raw["OriginalId"].astype(str) >= matched_df_raw["MatchedId"].astype(str),
        matched_df_raw["OriginalId"], matched_df_raw["MatchedId"]
    )
    mask_keep = ~matched_df_raw.duplicated(subset=["_min_id", "_max_id"], keep="first")
    removed_df = matched_df_raw.loc[~mask_keep]  # Pour log éventuellement
    matched_df_raw = matched_df_raw.loc[mask_keep].copy()
    matched_df_raw = matched_df_raw.drop(columns=["_min_id", "_max_id"])

    print(f"[COUNTRIES AND TIMER] Total: {len(matched_df)} matches in {hours:.0f}h {minutes:.0f}m {seconds:.1f}s "
          f"({elapsed_countries / 60:.2f} min, {elapsed_countries:.1f} s)")


    # --- 3) Deduplicate and filtering pipeline -------------------------------

    if matched_df_raw.empty:
        export_df = pd.DataFrame(columns=[
            "OriginalId", "Name", "NameNormalized", "MatchedId", "MatchedName", "IsolationFilter"
        ])
    else:
        export_df = matched_df_raw.copy()

        # De-dupe rule: prefer current ID over ex-ID (ID!) for the same OriginalId/MatchedId
        export_df["OriginalId"] = export_df["OriginalId"].astype(str)
        export_df["MatchedId"] = export_df["MatchedId"].astype(str)

        export_df = export_df[export_df["OriginalId"] != export_df["MatchedId"]]

        s = export_df["MatchedId"]
        export_df["_base"] = s.str.rstrip("!")
        export_df["_is_current"] = ~s.str.endswith("!")

        dup = export_df.duplicated(subset=["OriginalId", "_base"], keep=False)
        if dup.any():
            idx_pref = export_df.loc[dup].groupby(["OriginalId", "_base"])["_is_current"].idxmax()
            keep_idx = pd.Index(idx_pref).union(export_df.index[~dup])
            export_df = export_df.loc[keep_idx]

        export_df = export_df.drop(columns=["_base", "_is_current"])

        # Sets IsolationFilter when a step makes the group unique
        #export_df = apply_filters_pipeline(export_df)

        # Solo fallback 1: UO check, then links check (subsidiaries/shareholders)
        t_solo = time.perf_counter()
        #export_df = solo_match_uo_then_links_check(export_df)
        print(f"[FILTER] SOLO_UO {time.perf_counter() - t_solo:.3f}s")

        # Ensure CountryCode exists for normalization
        if "CountryCode" not in export_df.columns:
            export_df["CountryCode"] = export_df["OriginalId"].astype(str).str[:2]


        export_df["NameNormalized"] = pd.Series(index=export_df.index, dtype=object)
        for cc, group in export_df.groupby("CountryCode"):
            export_df.loc[group.index, "NameNormalized"] = exact_normalization_series(
                group["Name"], country_code=cc
            )["compact"].values


        # Solo fallback 2: guard for very short normalized names (<=3)
        #export_df = solo_shortname_guard(export_df)



        # Solo fallback 3: Solo Validation
        t_solo = time.perf_counter()
        #export_df = solo_validation(export_df)
        print(f"[FILTER] SOLO_VALIDATION {time.perf_counter() - t_solo:.3f}s")




        cols = ["OriginalId", "Name", "NameNormalized", "MatchedId", "MatchedName"]

        # Ajout plus direct de IsolationFilter sans mutation répétée
        if "IsolationFilter" not in export_df.columns:
            export_df = export_df.assign(IsolationFilter=pd.NA)

        cols.append("IsolationFilter")

        export_df = export_df.loc[:, cols]

        # Casting des clés en catégorie pour accélérer le tri et opérations ultérieures
        export_df["OriginalId"] = export_df["OriginalId"].astype("category")
        export_df["MatchedId"] = export_df["MatchedId"].astype("category")

        export_df = export_df.sort_values(["OriginalId", "MatchedId"], kind="stable", ignore_index=True)



    # --- 4) UniqueMatch view --------------------------------------------------
    if export_df.empty:
        unique_df = pd.DataFrame(columns=[
            "OriginalId", "Name", "NameNormalized", "MatchedId", "MatchedName", "IsolationFilter"
        ])
    else:
        counts = export_df["OriginalId"].value_counts()
        single_ids = counts[counts == 1].index

        # Préparer un masque booléen (plus rapide sur gros DataFrames)
        mask_single = export_df["OriginalId"].isin(single_ids)
        unique_df = export_df.loc[mask_single].copy()

        unique_df = unique_df.sort_values(
            ["OriginalId", "MatchedId"], kind="stable", ignore_index=True
        )

    unique_df = add_city_similarity_columns(unique_df)

    unique_df = add_parsed_addresses(unique_df)

    # --- Add the oldest OriginalId link date in UniqueMatch -----------------

    if not unique_df.empty:
        # Query once for all unique OriginalIds
        unique_ids = unique_df["OriginalId"].astype(str).unique().tolist()
        oldest = _fetch_oldest_link(unique_ids)

        # Left-merge to keep all rows in UniqueMatch
        unique_df = (
            unique_df.merge(oldest, on="OriginalId", how="left", validate="one_to_one")
            .assign(OldestSourceDate=lambda d: d["OldestSourceDate"].astype("string"))
        )

        desired_cols = [
            "OriginalId", "Name", "NameNormalized", "MatchedId", "MatchedName",
            "IsolationFilter", "OldestSourceDate",
        ]

        existing_cols = [c for c in desired_cols if c in unique_df.columns]

        # Ajout explicite des colonnes City à la fin
        right_cols = ["City 1", "City 2", "Similarity", "Postcode 1", "Postcode 2", "Address 1", "Address 2", "NrLinks 1", "NrLinks 2"]

        unique_df = unique_df[
            existing_cols +
            [c for c in unique_df.columns if c not in existing_cols + right_cols] +
            right_cols
            ]

    # Snapshot of the fully enriched table (State1/State2/Street1/Street2 and
    # similarity columns all present) taken before the tier extraction starts
    # consuming unique_df. This is the input for matches_delaware.xlsx.
    enriched_snapshot_df = unique_df.copy()

    import pandas as pd
    import re

    # Fonction utilitaire : vérifier si chaîne vide ou sans lettre
    def is_empty_or_no_letter(s):
        if pd.isna(s) or s == "":
            return True
        return not any(c.isalpha() for c in str(s))

    def extract_and_remove(df_source, condition, key_cols=None):
        """
        Sépare df_source en (extracted, remaining) à partir d'un masque booléen.
        - condition doit être un Series booléen aligné sur df_source.index.
        """
        # Alignement explicite par index pour être sûr
        condition = condition.reindex(df_source.index).fillna(False)

        extracted_df = df_source[condition].copy()
        remaining_df = df_source[~condition].copy()

        # Pour l'export, on peut remettre les indices à plat pour le df extrait
        extracted_df = extracted_df.reset_index(drop=True)

        # On NE reset PAS remaining_df pour conserver la cohérence avec les masques déjà calculés
        return extracted_df, remaining_df

    # Normalisation des noms pour comparaisons exactes
    def normalize_name(name):
        name_lower = name.lower()
        name_no_parens = re.sub(r'[()]', '', name_lower)
        name_no_and = re.sub(r'\band\b', '', name_no_parens)
        # Hyphens separate: "1997-XXX" must not glue into one token, or the
        # sequence marker is lost and two different funds compare as one name.
        name_no_and = re.sub(r'-', ' ', name_no_and)
        name_clean = re.sub(r'[^\w\s]', '', name_no_and)
        name_clean = re.sub(r'\s+', ' ', name_clean).strip()
        return name_clean

    # Conversion plus robuste NrLinks 2 en float
    def to_float_nullable(series):
        return pd.to_numeric(series, errors='coerce')

    # Conditions sur unique_df
    condition_address_empty = ~unique_df["Address 1"].fillna("").str.contains(r'[A-Za-z]', regex=True)
    condition_city_empty = ~unique_df["City 1"].fillna("").str.contains(r'[A-Za-z]', regex=True)
    condition_postcode_empty = ~unique_df["Postcode 1"].fillna("").str.contains(r'\d', regex=True)
    condition_address_city = condition_address_empty & condition_city_empty & condition_postcode_empty


    condition_matchedid = unique_df["MatchedId"].str.endswith(("L", "L!"), na=False)

    nr_links_2_num = to_float_nullable(unique_df["NrLinks 2"])
    condition_nrlinks2_zero_or_empty = nr_links_2_num.isna() | (nr_links_2_num == 0)
    condition_nrlinks2_greater_than_zero = (nr_links_2_num > 0).fillna(False)

    # 1. Extraire cas NoAddresses0Links (NrLinks null, 0, ou vide)
    special_cases_no_links_df, unique_df = extract_and_remove(
        unique_df, condition_address_city & condition_matchedid & condition_nrlinks2_zero_or_empty
    )

    # Normalisation des noms dans special_cases_no_links_df
    special_cases_no_links_df["NormName"] = special_cases_no_links_df["Name"].astype(str).apply(normalize_name)
    special_cases_no_links_df["NormMatchedName"] = special_cases_no_links_df["MatchedName"].astype(str).apply(
        normalize_name)

    mask_exact_name = special_cases_no_links_df["NormName"] == special_cases_no_links_df["NormMatchedName"]

    special_cases_no_links_exact_name_df = special_cases_no_links_df[mask_exact_name].copy()

    # Double sécurité en filtrant special_cases_no_links_exact_name_df sur NrLinks 2 nul ou vide, similaire à zero_or_empty
    mask_zero_or_empty_exact = special_cases_no_links_exact_name_df["NrLinks 2"].fillna("").astype(str).isin(["0", ""])
    special_cases_no_links_exact_name_df = special_cases_no_links_exact_name_df[mask_zero_or_empty_exact]

    special_cases_no_links_df = special_cases_no_links_df[~mask_exact_name].copy()

    # Assurer suppression stricte dans unique_df pour special_cases_no_links_df
    common_cols = special_cases_no_links_df.columns.intersection(unique_df.columns).tolist()
    if len(common_cols) > 0:
        mask = unique_df[common_cols].apply(tuple, axis=1).isin(
            special_cases_no_links_df[common_cols].apply(tuple, axis=1))
        unique_df = unique_df.loc[~mask].copy()


    special_cases_no_links_df.drop(columns=["NormName", "NormMatchedName"], inplace=True)
    special_cases_no_links_exact_name_df.drop(columns=["NormName", "NormMatchedName"], inplace=True)

    # 2. Extraire cas NoAddresses avec NrLinks > 0 (après conversion robuste)
    special_cases_df, unique_df = extract_and_remove(
        unique_df, condition_address_city & condition_matchedid & condition_nrlinks2_greater_than_zero
    )

    special_cases_df["NormName"] = special_cases_df["Name"].astype(str).apply(normalize_name)
    special_cases_df["NormMatchedName"] = special_cases_df["MatchedName"].astype(str).apply(normalize_name)

    mask_exact_name = special_cases_df["NormName"] == special_cases_df["NormMatchedName"]

    special_cases_exact_name_df = special_cases_df[mask_exact_name].copy()

    # Plus de robustesse ici aussi sur NrLinks 2 > 0 avec conversion float + safety
    nr_links_2_exact_num = to_float_nullable(special_cases_exact_name_df["NrLinks 2"])
    mask_gt_zero_exact = (nr_links_2_exact_num > 0).fillna(False)
    special_cases_exact_name_df = special_cases_exact_name_df[mask_gt_zero_exact]

    special_cases_df = special_cases_df[~mask_exact_name].copy()

    # Suppression stricte dans unique_df pour special_cases_df
    common_cols = special_cases_df.columns.intersection(unique_df.columns).tolist()
    if len(common_cols) > 0:
        mask = unique_df[common_cols].apply(tuple, axis=1).isin(special_cases_df[common_cols].apply(tuple, axis=1))
        unique_df = unique_df.loc[~mask].copy()

    special_cases_df.drop(columns=["NormName", "NormMatchedName"], inplace=True)
    special_cases_exact_name_df.drop(columns=["NormName", "NormMatchedName"], inplace=True)

    # ---------------------------------------------------------
    # 3. Extraire cas Same State + Postcode
    # ---------------------------------------------------------
    # Condition : IsSamePostalCode == 1 ET IsSameState == 1
    condition_state_postcode = (unique_df["IsSamePostalCode"] == 1) & (unique_df["IsSameState"] == 1)

    special_cases_state_postcode_df, unique_df = extract_and_remove(
        unique_df, condition_state_postcode
    )

    # Gestion Exact Name pour State+Postcode
    special_cases_state_postcode_df["NormName"] = special_cases_state_postcode_df["Name"].astype(str).apply(
        normalize_name)
    special_cases_state_postcode_df["NormMatchedName"] = special_cases_state_postcode_df["MatchedName"].astype(
        str).apply(normalize_name)

    mask_exact_name_sp = special_cases_state_postcode_df["NormName"] == special_cases_state_postcode_df[
        "NormMatchedName"]

    special_cases_state_postcode_exact_df = special_cases_state_postcode_df[mask_exact_name_sp].copy()
    special_cases_state_postcode_df = special_cases_state_postcode_df[~mask_exact_name_sp].copy()  # Reste (non exact)

    # Suppression stricte dans unique_df (déjà fait par extract_and_remove, mais double check si besoin, ici extract_and_remove suffit généralement si les index sont alignés, mais gardons la logique safe si tu préfères)
    # Note: extract_and_remove renvoie déjà le "remaining" unique_df, donc pas besoin de refaire le filtrage manuel "isin".

    special_cases_state_postcode_df.drop(columns=["NormName", "NormMatchedName"], inplace=True)
    special_cases_state_postcode_exact_df.drop(columns=["NormName", "NormMatchedName"], inplace=True)

    # ---------------------------------------------------------
    # 4. Extraire cas Same State + City (Similarity 100)
    # ---------------------------------------------------------
    # Condition : IsSameState == 1 ET CitySimilarity == 100
    # Attention: CitySimilarity est float, parfois prudent de faire >= 99.99 ou conversion int, mais == 100 marche souvent si calculé proprement.
    condition_state_city = (unique_df["IsSameState"] == 1) & (unique_df["CitySimilarity"] >= 100)

    special_cases_state_city_df, unique_df = extract_and_remove(
        unique_df, condition_state_city
    )

    # Gestion Exact Name pour State+City
    special_cases_state_city_df["NormName"] = special_cases_state_city_df["Name"].astype(str).apply(normalize_name)
    special_cases_state_city_df["NormMatchedName"] = special_cases_state_city_df["MatchedName"].astype(str).apply(
        normalize_name)

    mask_exact_name_sc = special_cases_state_city_df["NormName"] == special_cases_state_city_df["NormMatchedName"]

    special_cases_state_city_exact_df = special_cases_state_city_df[mask_exact_name_sc].copy()
    special_cases_state_city_df = special_cases_state_city_df[~mask_exact_name_sc].copy()

    special_cases_state_city_df.drop(columns=["NormName", "NormMatchedName"], inplace=True)
    special_cases_state_city_exact_df.drop(columns=["NormName", "NormMatchedName"], inplace=True)

    # ---------------------------------------------------------
    # 5. Extraire cas Exact Name (restant)
    # ---------------------------------------------------------
    # On isole les cas restants dans unique_df qui ont exactement le même nom normalisé.
    # Comme on n'a pas les colonnes normalisées dans unique_df, on crée le masque à la volée.

    # Calcul des séries temporaires pour la condition
    temp_norm_name = unique_df["Name"].astype(str).apply(normalize_name)
    temp_norm_match = unique_df["MatchedName"].astype(str).apply(normalize_name)

    condition_exact_name = (temp_norm_name == temp_norm_match)

    exact_name_df, unique_df = extract_and_remove(
        unique_df, condition_exact_name
    )

    # ---------------------------------------------------------

    columns_to_remove = [
        "City 1", "City 2", "Similarity", "Postcode 1", "Postcode 2", "Address 1", "Address 2"
    ]

    cols_to_drop = [col for col in columns_to_remove if col in unique_df.columns]

    # Nettoyage final des colonnes pour TOUS les dataframes
    dfs_to_clean = [
        unique_df,
        special_cases_no_links_df, special_cases_df, special_cases_no_links_exact_name_df, special_cases_exact_name_df,
        special_cases_state_postcode_df, special_cases_state_postcode_exact_df,
        special_cases_state_city_df, special_cases_state_city_exact_df,
        exact_name_df
    ]

    for df in dfs_to_clean:
        # On utilise intersection pour éviter erreur si colonne déjà absente
        cols = df.columns.intersection(columns_to_remove)
        if len(cols) > 0:
            df.drop(columns=cols, inplace=True)

    # --- 4bis) Client-approved validation rules (rows failing a rule are ------
    #           dropped from that sheet; they remain visible only in
    #           Matching_Propositions, nothing else changes for them). --------
    LAST_TRANSFER_CUTOFF = 20260401  # yyyymmdd, "recent" per client email

    def _filter_recent_last_transfer(df):
        if df.empty or "LastTransferDate" not in df.columns:
            return df
        val = pd.to_numeric(df["LastTransferDate"], errors="coerce")
        return df[val >= LAST_TRANSFER_CUTOFF].copy()

    def _is_delaware(series):
        """True where the parsed state is Delaware."""
        return series.fillna("").astype(str).str.strip().str.upper().eq("DELAWARE")

    def _delaware_no_street_exemption(df):
        """
        Delaware carve-out: many DE entities are registered with an agent and
        carry no street address, so StreetSimilarity collapses to 0 and good
        exact-name matches get dropped.

        Exempt from the StreetSimilarity rule ONLY when BOTH:
          - Street1 is empty (OriginalId side has no street), AND
          - both sides are Delaware (State1 == State2 == DELAWARE).

        All other rules (e.g. LastTransferDate) still apply to these rows.
        """
        if df.empty:
            return pd.Series(False, index=df.index)
        for col in ("Street1", "State1", "State2"):
            if col not in df.columns:
                return pd.Series(False, index=df.index)

        street1_empty = ~df["Street1"].fillna("").astype(str).str.contains(
            r"[A-Za-z0-9]", regex=True
        )
        both_delaware = _is_delaware(df["State1"]) & _is_delaware(df["State2"])
        return street1_empty & both_delaware

    def _filter_street_similarity(df, threshold):
        if df.empty or "StreetSimilarity" not in df.columns:
            return df
        val = pd.to_numeric(df["StreetSimilarity"], errors="coerce")
        keep = (val >= threshold) | _delaware_no_street_exemption(df)
        return df[keep].copy()

    # --- Registered-agent address exemption ---------------------------------
    AGENT_ADDRESSES = _load_agent_addresses()

    def _agent_address_exemption(df):
        """
        Registered-agent carve-out: a company's Delaware record often carries the
        address of its registered agent (CSC, CT Corporation, ...) rather than its
        own. Comparing that against the operating company's real street produces a
        near-zero StreetSimilarity and drops an otherwise sound exact-name match.

        Exempt from the StreetSimilarity rule when Street2 (the MatchedId side) is
        a known registered-agent address. Matching is done on normalized text with
        containment in either direction, so suite/unit suffixes still match.

        All other rules (e.g. LastTransferDate) still apply to these rows.
        """
        if df.empty or not AGENT_ADDRESSES or "Street2" not in df.columns:
            return pd.Series(False, index=df.index)

        def _is_agent(value):
            norm = normalize_text(value) if pd.notna(value) else ""
            if not norm:
                return False
            return any(a in norm or norm in a for a in AGENT_ADDRESSES)

        return df["Street2"].apply(_is_agent)

    def _filter_street_similarity_with_agent(df, threshold):
        """StreetSimilarity rule with both carve-outs applied."""
        if df.empty or "StreetSimilarity" not in df.columns:
            return df
        val = pd.to_numeric(df["StreetSimilarity"], errors="coerce")
        keep = (
            (val >= threshold)
            | _delaware_no_street_exemption(df)
            | _agent_address_exemption(df)
        )
        return df[keep].copy()

    def _add_validation_note(df):
        """
        Flag rows that were kept via an exemption rather than on street
        similarity, so a reviewer can spot them at a glance:
            "DE agent address" - Street2 is a registered-agent address
            "DE no street"     - Street1 empty and both sides Delaware
            ""                 - passed the StreetSimilarity rule normally
        The column is appended at the end of the row.
        """
        if df.empty:
            df["ValidationNote"] = pd.Series(dtype=object)
            return df

        note = pd.Series("", index=df.index, dtype=object)
        note = note.mask(_delaware_no_street_exemption(df), "DE no street")
        # Agent-address takes precedence when both would apply.
        note = note.mask(_agent_address_exemption(df), "DE agent address")

        df = df.copy()
        df["ValidationNote"] = note
        cols = [c for c in df.columns if c != "ValidationNote"] + ["ValidationNote"]
        return df[cols]

    # NoAddress_ExactName_0Links: LastTransferDate recent (>= 20260401)
    special_cases_no_links_exact_name_df = _filter_recent_last_transfer(
        special_cases_no_links_exact_name_df
    )

    # NoAddress_ExactName: same rule as the 0Links variant
    special_cases_exact_name_df = _filter_recent_last_transfer(special_cases_exact_name_df)

    # Same_state_postcode_ExactName: StreetSimilarity >= 60
    special_cases_state_postcode_exact_df = _filter_street_similarity_with_agent(
        special_cases_state_postcode_exact_df, 60
    )
    special_cases_state_postcode_exact_df = _add_validation_note(
        special_cases_state_postcode_exact_df
    )

    # Same_state_city_ExactName: StreetSimilarity >= 60
    special_cases_state_city_exact_df = _filter_street_similarity_with_agent(
        special_cases_state_city_exact_df, 60
    )
    special_cases_state_city_exact_df = _add_validation_note(
        special_cases_state_city_exact_df
    )

    # Exact_name: LastTransferDate recent (>= 20260401) AND StreetSimilarity >= 78
    exact_name_df = _filter_recent_last_transfer(exact_name_df)
    exact_name_df = _filter_street_similarity_with_agent(exact_name_df, 78)
    exact_name_df = _add_validation_note(exact_name_df)

    print(
        "[VALIDATION] NoAddress_ExactName_0Links="
        f"{len(special_cases_no_links_exact_name_df)} | "
        f"NoAddress_ExactName={len(special_cases_exact_name_df)} | "
        f"Same_state_postcode_ExactName={len(special_cases_state_postcode_exact_df)} | "
        f"Same_state_city_ExactName={len(special_cases_state_city_exact_df)} | "
        f"Exact_name={len(exact_name_df)}"
    )




    # --- 5) NoMatch view ------------------------------
    base_cases = (
        df_cases[["OriginalId", "Name", "CountryCode"]]
        .assign(OriginalId=lambda d: d["OriginalId"].astype(str))
        .drop_duplicates(subset=["OriginalId"], keep="first")
    )
    matched_ids = set(export_df["OriginalId"].unique()) if not export_df.empty else set()
    df_nomatch = base_cases[~base_cases["OriginalId"].isin(matched_ids)].copy()

    if not df_nomatch.empty:
        # Normalize for display consistency (uses country code per row)
        df_nomatch["NameNormalized"] = pd.Series(index=df_nomatch.index, dtype=object)
        for cc, group in df_nomatch.groupby("CountryCode"):
            df_nomatch.loc[group.index, "NameNormalized"] = exact_normalization_series(
                group["Name"], country_code=cc
            )["compact"].values

        df_nomatch = df_nomatch[["OriginalId", "Name", "NameNormalized"]].sort_values(
            ["OriginalId"], kind="stable", ignore_index=True
        )
    else:
        df_nomatch = pd.DataFrame(columns=["OriginalId", "Name", "NameNormalized"])

    # --- 6) Display/Export artifacts -----------------------------------------

    max_excel_rows = 1_000_000

    def write_df_split(df, writer, base_sheet_name):
        num_sheets = math.ceil(len(df) / max_excel_rows)
        for i in range(num_sheets):
            start = i * max_excel_rows
            end = start + max_excel_rows
            sheet_df = df.iloc[start:end]
            sheet_name = f"{base_sheet_name}_{i + 1}" if num_sheets > 1 else base_sheet_name
            sheet_df.to_excel(writer, index=False, sheet_name=sheet_name)

    # Matching_Propositions is built here rather than at write time because the
    # Delaware workbook below is the only consumer of it.
    if matched_df_raw.empty:
        matching_props = pd.DataFrame(columns=[
            "OriginalId", "Name", "NameNormalized", "MatchedId", "MatchedName"
        ])
    else:
        matched_df_raw["OriginalId"] = matched_df_raw["OriginalId"].astype(str)
        df_cases["OriginalId"] = df_cases["OriginalId"].astype(str)
        cases_base = df_cases[["OriginalId", "Name"]].copy()
        df_merge = matched_df_raw.merge(cases_base, on="OriginalId", how="left")
        df_merge = df_merge.rename(columns={"Name_y": "Name"})
        if "Name_x" in df_merge.columns:
            df_merge = df_merge.drop(columns=["Name_x"])
        if "CountryCode" not in df_merge.columns:
            df_merge["CountryCode"] = df_merge["OriginalId"].str[:2]

        def safe_normalize(name, cc):
            try:
                return exact_normalization_series(pd.Series([name]), country_code=cc)["compact"].iloc[0]
            except Exception as e:
                print(f"[WARN] Normalization failed for '{name}' ({cc}): {e}")
                return pd.NA

        df_merge["NameNormalized"] = [
            safe_normalize(n, c) for n, c in zip(df_merge["Name"], df_merge["CountryCode"])
        ]

        matching_props = df_merge[
            ["OriginalId", "Name", "NameNormalized", "MatchedId", "MatchedName"]
        ]
        matching_props = _insert_blank_rows_between_groups(matching_props, "OriginalId")

    print(
        f"[TIERS] Matching_Propositions={len(matching_props)} | UniqueMatch={len(unique_df)} | "
        f"Exact_name={len(exact_name_df)} | NoMatch={len(df_nomatch)}"
    )

    # --- 7) Second output file: matches_delaware.xlsx ------------------------
    # Same Entries / Matching_Propositions / NoMatch as above, but the tier
    # sheets are replaced by a single Delaware-focused sheet plus DirectLinked.
    try:
        de_selection = select_delaware_single_matches(
            enriched_snapshot_df,
            agent_addresses=AGENT_ADDRESSES,
            normalize_text=normalize_text,
            normalize_name=normalize_name,
        )
        print(f"[DELAWARE] {len(de_selection)} single-proposition Delaware rows selected")

        # Phases 2 and 3 both need to know which OriginalIds phase 1 has already
        # taken. de_selection is used rather than the post-split frames because
        # it holds the same ids and is available before the ownership pass —
        # which now has to wait until phase 3 has chosen its rows too.
        _de_ids_both_sides = set()
        for _col in ("OriginalId", "MatchedId"):
            if _col in de_selection.columns:
                _de_ids_both_sides |= set(de_selection[_col].dropna().astype(str))
        _de_original_ids = (
            set(de_selection["OriginalId"].dropna().astype(str))
            if "OriginalId" in de_selection.columns else set()
        )

        # --- Step 2: same-name OriginalId grouping, address sheets ----------
        # Needs the parsed address columns, so Entries ids are enriched here with
        # the same city fetch + address parser the match table uses. The enriched
        # frame is reused by step 4, which groups the address-less ids later in
        # the cascade.
        try:
            # Only the OriginalId side of Delaware_Single_Match is out of scope.
            # Its MatchedIds stay available: they are Entries ids too and may
            # legitimately anchor or join a group here.
            de_handled = set(_de_original_ids)

            all_ids = [
                i for i in df_cases["OriginalId"].astype(str).unique().tolist()
                if i not in de_handled
            ]
            print(
                f"[STEP2] {len(de_handled):,} ids excluded (Delaware_Single_Match "
                f"OriginalId) — {len(all_ids):,} ids to consider"
            )
            cities_all = _fetch_cities(all_ids)

            base = df_cases[["OriginalId", "Name"]].copy()
            base["OriginalId"] = base["OriginalId"].astype(str)
            base = base[base["OriginalId"].isin(all_ids)]
            base = base.merge(
                cities_all.rename(columns={
                    "Id": "OriginalId",
                    "PCCity": "City 1",
                    "Postcode": "Postcode 1",
                    "Address": "Address 1",
                    "NrLinks": "NrLinks 1",
                }),
                on="OriginalId",
                how="left",
            )
            if "LastTransferDate" in base.columns:
                base = base.drop(columns=["LastTransferDate"])

            parsed = pd.DataFrame(
                [parse_separated(r, "1") for _, r in base.iterrows()]
            )
            base = pd.concat([base.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)

            base["NameNormalized"] = pd.Series(index=base.index, dtype=object)
            _cc = df_cases.set_index(df_cases["OriginalId"].astype(str))["CountryCode"]
            base["CountryCode"] = base["OriginalId"].map(_cc)
            for cc, grp in base.groupby("CountryCode"):
                base.loc[grp.index, "NameNormalized"] = exact_normalization_series(
                    grp["Name"], country_code=cc
                )["compact"].values

            same_city_state_df, same_city_df, _ = build_directid_groups(
                base,
                normalize_name=normalize_name,
                normalize_text=normalize_text,
                excluded_ids=de_handled,
                buckets=("address",),
            )
            print(
                f"[STEP2] directid_Same_City_State={len(same_city_state_df)} | "
                f"directid_Same_city={len(same_city_df)}"
            )
        except Exception as e:
            print(f"[STEP2][ERROR] same-name grouping failed: {e}")
            base = pd.DataFrame()
            same_city_state_df = pd.DataFrame()
            same_city_df = pd.DataFrame()

        def _ids_of(frames, columns=("OriginalId",)):
            """Collect the ids a set of frames has already claimed."""
            out = set()
            for _df in frames:
                if _df is None or _df.empty:
                    continue
                for _col in columns:
                    if _col in _df.columns:
                        out |= set(_df[_col].dropna().astype(str))
            return out

        _BOTH = ("OriginalId", "MatchedId")
        _step2_frames = (same_city_state_df, same_city_df)

        # --- Step 3: several OriginalIds sharing one Delaware MatchedId ------
        # Only the OriginalId side of steps 1 and 2 is out of scope. The
        # MatchedIds step 2 produced are Entries ids and may appear here on the
        # OriginalId side.
        try:
            _step3_excluded = _de_original_ids | _ids_of(_step2_frames)
            p3_selection = select_multiple_directid(
                enriched_snapshot_df,
                excluded_ids=_step3_excluded,
                agent_addresses=AGENT_ADDRESSES,
                normalize_text=normalize_text,
                normalize_name=normalize_name,
                relations=("same_legal",),
                delaware="require",
                last_transfer_cutoff=LAST_TRANSFER_CUTOFF,
                label="STEP3",
            )
        except Exception as e:
            print(f"[STEP3][ERROR] selection failed: {e}")
            p3_selection = pd.DataFrame()

        # --- Step 4: the address-less grouping bucket ------------------------
        # Runs after step 3 so an id whose proposition belongs to a shared
        # Delaware cluster is not claimed here on the strength of its name
        # alone. Both sides of steps 1-3 are out of scope.
        try:
            _step4_excluded = (
                _de_ids_both_sides
                | _ids_of(_step2_frames, _BOTH)
                | _ids_of((p3_selection,), _BOTH)
            )
            _, _, no_address_df = build_directid_groups(
                base,
                normalize_name=normalize_name,
                normalize_text=normalize_text,
                excluded_ids=_step4_excluded,
                buckets=("no_address",),
            )
            print(f"[STEP4] directid_no_address={len(no_address_df)}")
        except Exception as e:
            print(f"[STEP4][ERROR] address-less grouping failed: {e}")
            no_address_df = pd.DataFrame()

        # --- Step 5: the remaining shared-target sheets ----------------------
        # Same shape as step 3, narrowing by legal form and dropping the
        # Delaware requirement. Each pass skips what the previous ones took, so
        # an id lands in one sheet only.
        try:
            _handled = (
                _de_original_ids
                | _ids_of(_step2_frames)
                | _ids_of((p3_selection,), _BOTH)
                | _ids_of((no_address_df,))
            )

            _passes = [
                ("p3_same_legal", ("same_legal",), "exclude", "STEP5-SAMELEGAL"),
                ("p3_diff_legal", ("diff_legal",), "ignore", "STEP5-DIFFLEGAL"),
                ("p3_no_legal", ("a_no_legal",), "ignore", "STEP5-NOLEGAL"),
            ]
            _p3_frames = {}
            for key, rels, de_mode, tag in _passes:
                sel = select_multiple_directid(
                    enriched_snapshot_df,
                    excluded_ids=_handled,
                    agent_addresses=AGENT_ADDRESSES,
                    normalize_text=normalize_text,
                    normalize_name=normalize_name,
                    relations=rels,
                    delaware=de_mode,
                    last_transfer_cutoff=LAST_TRANSFER_CUTOFF,
                    label=tag,
                )
                _p3_frames[key] = sel
                if not sel.empty and "OriginalId" in sel.columns:
                    _handled |= set(sel["OriginalId"].astype(str))

            p3_same_legal_sel = _p3_frames["p3_same_legal"]
            p3_diff_legal_sel = _p3_frames["p3_diff_legal"]
            p3_no_legal_sel = _p3_frames["p3_no_legal"]
        except Exception as e:
            print(f"[STEP5][ERROR] selection failed: {e}")
            p3_same_legal_sel = pd.DataFrame()
            p3_diff_legal_sel = pd.DataFrame()
            p3_no_legal_sel = pd.DataFrame()

        # --- Ownership pass: one fetch covering phase 1 and phase 3 ----------
        # Phase 3's ids are not known until its selection has run, so the fetch
        # waits for both and covers them in a single round trip.
        try:
            _own_ids = set()
            for _df in (de_selection, p3_selection, p3_same_legal_sel,
                        p3_diff_legal_sel, p3_no_legal_sel,
                        same_city_state_df, same_city_df, no_address_df):
                if _df is None or _df.empty:
                    continue
                for _col in ("OriginalId", "MatchedId"):
                    if _col in _df.columns:
                        _own_ids |= set(_df[_col].dropna().astype(str))

            if _own_ids:
                _link_rows, _uo_rows = _fetch_ownership_links(_own_ids)
                _ownership = build_ownership_map(_link_rows, _uo_rows)
            else:
                _ownership = {}

            de_main_df, _dl_phase1 = split_direct_linked(de_selection, _ownership)
            p3_main_df, _dl_p3 = split_direct_linked(p3_selection, _ownership)
            p3_same_legal_df, _dl_sl = split_direct_linked(p3_same_legal_sel, _ownership)
            p3_diff_legal_df, _dl_dl = split_direct_linked(p3_diff_legal_sel, _ownership)
            p3_no_legal_df, _dl_nl = split_direct_linked(p3_no_legal_sel, _ownership)

            # Every phase contributes to the one DirectLinked sheet.
            _dl_parts = [d for d in (_dl_phase1, _dl_p3, _dl_sl, _dl_dl, _dl_nl)
                         if d is not None and not d.empty]
            direct_linked_df = (
                pd.concat(_dl_parts, ignore_index=True) if _dl_parts else _dl_phase1
            )

            same_city_state_df = stamp_ownership(same_city_state_df, _ownership)
            same_city_df = stamp_ownership(same_city_df, _ownership)
            no_address_df = stamp_ownership(no_address_df, _ownership)
            direct_linked_df = stamp_ownership(direct_linked_df, _ownership)
        except Exception as e:
            print(f"[OWNERSHIP][ERROR] ownership pass failed: {e}")
            de_main_df = de_selection.copy()
            p3_main_df = p3_selection.copy()
            p3_same_legal_df = p3_same_legal_sel.copy()
            p3_diff_legal_df = p3_diff_legal_sel.copy()
            p3_no_legal_df = p3_no_legal_sel.copy()
            direct_linked_df = pd.DataFrame()

        print(
            f"[DELAWARE] Delaware_Single_Match={len(de_main_df)} | "
            f"multiple_directid_delaware={len(p3_main_df)} | "
            f"DirectLinked={len(direct_linked_df)}"
        )

        de_path = os.path.join(OUTPUT_DIR, "matches_delaware.xlsx")
        with pd.ExcelWriter(de_path) as de_writer:
            wrote_any = False

            df_entries_de = df_cases[["OriginalId", "Name"]].copy()
            if not df_entries_de.empty:
                write_df_split(df_entries_de, de_writer, "Entries")
                wrote_any = True

            if not matching_props.empty:
                write_df_split(matching_props, de_writer, "Matching_Propositions")
                wrote_any = True

            if not de_main_df.empty:
                write_df_split(de_main_df, de_writer, "Delaware_Single_Match")
                wrote_any = True

            if not direct_linked_df.empty:
                write_df_split(direct_linked_df, de_writer, "DirectLinked")
                wrote_any = True

            if not same_city_state_df.empty:
                write_df_split(same_city_state_df, de_writer, "directid_Same_City_State")
                wrote_any = True

            if not same_city_df.empty:
                write_df_split(same_city_df, de_writer, "directid_Same_city")
                wrote_any = True

            if not no_address_df.empty:
                write_df_split(no_address_df, de_writer, "directid_no_address")
                wrote_any = True

            if not p3_main_df.empty:
                write_df_split(p3_main_df, de_writer, "multiple_directid_delaware")
                wrote_any = True

            for _frame, _sheet in (
                (p3_same_legal_df, "multiple_directid_same_legal"),
                (p3_diff_legal_df, "multiple_directid_diff_legal"),
                (p3_no_legal_df, "multiple_directid_no_legal"),
            ):
                if _frame is not None and not _frame.empty:
                    write_df_split(_frame, de_writer, _sheet)
                    wrote_any = True

            if not df_nomatch.empty:
                write_df_split(df_nomatch, de_writer, "NoMatch")
                wrote_any = True

            if not wrote_any:
                pd.DataFrame(["No data to export"]).to_excel(
                    de_writer, index=False, sheet_name="Empty"
                )

        print(f"[EXPORT] Excel: {de_path}")
    except Exception as e:
        print(f"[EXPORT][ERROR] Impossible d'écrire matches_delaware.xlsx: {e}")




if __name__ == "__main__":
    t0 = time.perf_counter()

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    exit_code = 0
    metrics = {}
    extra=[]
    try:
        run()

    except Exception as e:
        exit_code = 1
        metrics = {"error": 1}
        traceback.print_exc()


    finally:
        out_file = os.path.join(OUTPUT_DIR, "matches_delaware.xlsx")
        # Construis la liste d'artefacts additionnels
        extra = []
        p = Path(out_file)
        if p.exists():
            extra.append((str(p), "file", "xlsx", False))


        finalize_run(
            project_name=PROJECT_NAME,
            started_at=started_at,
            exit_code=exit_code,
            metrics=metrics,
            require_primary_txt=True,  # TXT ici, donc True pour lever une erreur si on ne le trouve pas
            extra_artifacts=extra,
        )

        elapsed_s = time.perf_counter() - t0
        print(f"[TIMER] Total: {elapsed_s / 60:.2f} min ({elapsed_s:.1f} s)")

        sys.exit(exit_code)



