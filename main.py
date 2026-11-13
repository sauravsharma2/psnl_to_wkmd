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
from src.code.directid_grouping import build_direct_vs_good, build_directid_groups
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
        if not city and u_city
