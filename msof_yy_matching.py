# Databricks notebook source
# MAGIC %md
# MAGIC # MSOF Direct Entries — YY matching (Orbis + SQL Server dates / native names)
# MAGIC All Orbis data fetching lives in `queries/*.sql` — change filters there, not here.
# MAGIC
# MAGIC **Part A (Databricks)** · Step 1 data → Step 2 name features → Step 3 candidates
# MAGIC (blocks → score ≥ threshold → numbers → legal form same on both sides) → YY list CSV for SQL Server
# MAGIC
# MAGIC **PC** · `yy_sql_info.py` reads the YY list, gets OldestSourceDate + NativeName from bvdaffils, writes a CSV.
# MAGIC Upload that CSV to the Volume.
# MAGIC
# MAGIC **Part B (Databricks)** · Step 4 date check → CN native-name check → ownership tie-break → best score
# MAGIC → unique matches with a `ResultSheet` label → Step 5 export (one CSV + one Excel)
# MAGIC
# MAGIC **Session expired?** Set `REUSE_SAVED = True` and run the cells again — saved steps are read back.

# COMMAND ----------

# MAGIC %pip install -q rapidfuzz Metaphone xlsxwriter

# COMMAND ----------

# ═══════════════════════════ CONFIG ═══════════════════════════
import os

DIRECT_COUNTRIES = ["YY"]               # direct IDs to match (this notebook is built for YY)
POOL_FILTER = "1=1"                     # good pool = all countries (YY can be any country)
PERSON_TYPES = ["I", "D"]               # entities.Entity_type meaning a person
ACTIVE_PREFIX = "Active"                # legal_info.Status starting with this = active (info only)

THRESHOLD = 96                          # MatchScore cut-off
BLOCK_MAX_POOL = 10_000                 # skip a name block with more pool companies than this (None = no cap)
NATIVE_CHECK_PREFIX = "CN"              # native-name check only for MatchedIds starting with this

USER_ID = "SHARMS73"
REUSE_SAVED = False                     # True = reuse tables already saved (resume after session expiry)

SRC = "prd_bronze_orbis.orbis"
OUT_SCHEMA = "training_sandbox_dev.msof_direct"
OUT_DIR = "/Volumes/training_sandbox_dev/msof_direct/files"
LEGAL_FORM_CSV = f"{OUT_DIR}/legal_form_mapping.csv"
FEATURE_PARTITIONS = 2000               # split names into this many parts so all machines work at once
TABLE_PREFIX = "yy_"                    # every output table starts with this

# Part B input: the CSV made on the PC by yy_sql_info.py (uploaded to OUT_DIR).
# Leave None to pick the newest yy_sql_info_*.csv in OUT_DIR.
SQL_INFO_CSV = None

# queries/ folder next to this notebook (change if you keep it elsewhere)
QUERY_DIR = os.path.join(os.getcwd(), "queries")

# COMMAND ----------

# ═══════════════════════════ HELPERS ═══════════════════════════
import re
import shutil
import unicodedata
import uuid
from datetime import datetime, timedelta, timezone
from functools import reduce

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {OUT_SCHEMA}")

IST = timezone(timedelta(hours=5, minutes=30))
RUN_TAG = f"YY_{datetime.now(IST):%Y%m%d}"     # file names end with today's India date, e.g. _20261015


def _sql_list(values):
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


SQL_PARAMS = {
    "src": SRC,
    "direct_countries": _sql_list(DIRECT_COUNTRIES),
    "person_types": _sql_list(PERSON_TYPES),
    "pool_filter": POOL_FILTER,
    "active_prefix": ACTIVE_PREFIX,
}


def run_sql(file_name, **extra):
    """Read queries/<file_name>, fill the {placeholders}, run it."""
    with open(os.path.join(QUERY_DIR, file_name), encoding="utf-8") as f:
        text = f.read()
    return spark.sql(text.format(**{**SQL_PARAMS, **extra}))


def with_ids_view(ids_df, file_name):
    """Register a list of IDs as a temp view and run a query that reads it."""
    view = f"v_ids_{uuid.uuid4().hex[:8]}"
    ids_df.select("Id").distinct().createOrReplaceTempView(view)
    return run_sql(file_name, ids_view=view)


def tname(name):
    return f"{OUT_SCHEMA}.{TABLE_PREFIX}{name}"


def save(df, name):
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(tname(name))
    return spark.table(tname(name))


def get_or_build(name, build):
    """Reuse a saved table when REUSE_SAVED is on, otherwise build and save it."""
    if REUSE_SAVED and spark.catalog.tableExists(tname(name)):
        print(f"[reuse] {tname(name)}")
        return spark.table(tname(name))
    return save(build(), name)


def same(a, b):
    """True/False when both sides have a value, NULL when either is missing."""
    return F.when(F.col(a).isNotNull() & F.col(b).isNotNull(), F.col(a) == F.col(b))


def write_one_csv(df, path):
    """Write a DataFrame as ONE CSV file at `path` (Spark writes a folder; we keep only the part file)."""
    tmp_dir = path + "_tmp"
    (df.coalesce(1).write.mode("overwrite")
       .option("header", True).option("escape", '"').option("quoteAll", True).csv(tmp_dir))
    part = [f.path for f in dbutils.fs.ls(tmp_dir) if f.name.startswith("part-")][0]
    dbutils.fs.rm(path)
    dbutils.fs.mv(part, path)
    dbutils.fs.rm(tmp_dir, recurse=True)
    print(f"CSV written: {path}")

# COMMAND ----------

# ═══════════════════════ NAME FEATURES (definitions only) ═══════════════════════
WORDS_TO_DELETE = [
    "the", "as", "of", "for", "an", "and", "&", "et", "y", "this", "or", "le", "la",
    "de", "du", "co", "company", "compnay", "in", "en", "liquidation", "liquidazione",
    "liquiazione", "di", "del", "?", "unipessoal", "-", "anonyme",
    "societe", "unipersonal", "sociedad", "de'", "par action", "simplifiee", "societa",
    "responsabilita", "limitata", "societa'", "responsabilita'", "consortile",
    "semplificata", "responsabilit", "soccoop", "cooperativa",
    "v likvidaci", "extinguida", "(extinguida)", "liquidacao",
    "associe unique", "responsabilite", "limitee", "limitiee", "civile", "simplifie", "cokg", "gmbh",
    "haftungsbeschrankt", "ldta",
]

# Numbers 1–20 in EN/DE/FR/IT/ES/PT/NL (accents stripped). Words with a second common
# meaning ('un', 'uno', 'um', 'een', 'elf', 'once') are left out on purpose.
_NUM_WORDS = {
    "en": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
          "sixteen seventeen eighteen nineteen twenty",
    "de": "eins zwei drei vier funf sechs sieben acht neun zehn - zwolf dreizehn vierzehn funfzehn "
          "sechzehn siebzehn achtzehn neunzehn zwanzig",
    "fr": "- deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze seize "
          "dixsept dixhuit dixneuf vingt",
    "it": "- due tre quattro cinque sei sette otto nove dieci undici dodici tredici quattordici "
          "quindici sedici diciassette diciotto diciannove venti",
    "es": "- dos tres cuatro cinco seis siete ocho nueve diez - doce trece catorce quince dieciseis "
          "diecisiete dieciocho diecinueve veinte",
    "pt": "- dois tres quatro cinco seis sete oito nove dez onze doze treze catorze quinze dezesseis "
          "dezessete dezoito dezenove vinte",
    "nl": "- twee drie vier vijf zes zeven acht negen tien - twaalf dertien veertien vijftien zestien "
          "zeventien achttien negentien twintig",
}
NUM_WORDS = {w: str(i + 1) for words in _NUM_WORDS.values()
             for i, w in enumerate(words.split()) if w != "-"}
ROMAN = {r: str(i + 1) for i, r in enumerate(
    "i ii iii iv v vi vii viii ix x xi xii xiii xiv xv xvi xvii xviii xix xx".split())}

# ── legal forms: turn every regex in legal_form_mapping.csv into word sequences once ──
# A name is then cleaned by looking its words up in a dictionary (fast), instead of
# running ~1,500 regex alternatives over every character (slow).
import re._parser as _sp
import re._constants as _sc

SHORT_FORM_MAX = 3            # forms this short keep their start/end position (AB, AD, KK, MB, SA…)
ALWAYS_ANCHORED = {"PRIVATE"} # ordinary words that are only a legal form at the start/end
TYPO_MIN_LEN = 6              # legal-form typo fix ('limted' -> LIMITED) only for words this long
TOKEN_RE = re.compile(r"[^\W_]+")


def split_top(rx):
    """Split a regex on '|' that is not inside (...) or [...]."""
    parts, depth, cls, cur, esc = [], 0, False, "", False
    for ch in rx:
        if esc:
            cur += ch; esc = False; continue
        if ch == "\\":
            cur += ch; esc = True; continue
        if cls:
            cls = ch != "]"; cur += ch; continue
        if ch == "[":
            cls = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "|" and depth == 0:
            parts.append(cur); cur = ""; continue
        cur += ch
    parts.append(cur)
    return [p for p in (x.strip() for x in parts) if p]


def _expand(nodes, cap=256):
    """Every text a (finite) regex can match; \b and ^/$ are ignored here."""
    outs = [""]
    for op, av in nodes:
        if op == _sc.LITERAL:
            alts = [chr(av)]
        elif op in (_sc.NOT_LITERAL, _sc.ANY, _sc.CATEGORY):
            alts = [" "]
        elif op == _sc.AT:
            continue
        elif op == _sc.IN:
            alts = list(dict.fromkeys(
                chr(v) if o == _sc.LITERAL else chr(v[0]) if o == _sc.RANGE else " " for o, v in av)) or [" "]
        elif op == _sc.BRANCH:
            alts = [x for br in av[1] for x in _expand(br, cap)]
        elif op == _sc.SUBPATTERN:
            alts = _expand(av[-1], cap)
        elif op in (_sc.MAX_REPEAT, _sc.MIN_REPEAT):
            lo, hi, sub = av
            inner, alts = _expand(sub, cap), []
            for n in range(lo, min(hi, max(lo, 1)) + 1):
                acc = [""]
                for _ in range(n):
                    acc = [x + y for x in acc for y in inner]
                alts += acc
        else:
            raise ValueError(f"unsupported regex op {op}")
        outs = [o + x for o in outs for x in alts][:cap]
    return outs


def _toks(text):
    return tuple(TOKEN_RE.findall(text.lower().replace(".", "")))


def build_legal_lookup(df):
    """-> anywhere / start / end dicts {word tuple: legal-form group}, max length, single-word forms for typos."""
    anywhere, start, end = {}, {}, {}
    for group, rx in zip(df["LegalForm"], df["Regex"]):
        if not rx.strip():
            continue
        for alt in split_top(rx):
            core = re.sub(r"[^A-Za-z0-9]", "", re.sub(r"\\[bs]", "", alt)).upper()
            short = len(core) <= SHORT_FORM_MAX or core in ALWAYS_ANCHORED
            s_anchor = alt.lstrip().startswith("^")
            e_anchor = bool(re.search(r"(?<!\\)\$", alt))
            for txt in _expand(_sp.parse(alt, re.IGNORECASE)):
                t = _toks(txt)
                if not t:
                    continue
                if not short or not (s_anchor or e_anchor):
                    anywhere.setdefault(t, group)
                else:
                    if s_anchor:
                        start.setdefault(t, group)
                    if e_anchor:
                        end.setdefault(t, group)
    maxlen = max(len(t) for d in (anywhere, start, end) for t in d)
    typo = {t[0]: g for d in (anywhere, end) for t, g in d.items() if len(t) == 1 and len(t[0]) >= TYPO_MIN_LEN}
    return anywhere, start, end, maxlen, typo


_lf = pd.read_csv(LEGAL_FORM_CSV, dtype=str, keep_default_na=False)
LEGAL_ANY, LEGAL_START, LEGAL_END, LEGAL_MAXLEN, LEGAL_TYPO = build_legal_lookup(_lf)
WORDS_SET = {t for t in (_toks(w) for w in WORDS_TO_DELETE) if t}
WORDS_MAXLEN = max(map(len, WORDS_SET))
print(f"legal forms: {len(LEGAL_ANY) + len(LEGAL_START) + len(LEGAL_END):,} word patterns, "
      f"{len(set(_lf['LegalForm'])):,} groups, {len(LEGAL_TYPO):,} typo targets")


def _remove(tokens, anywhere, start=None, end=None, maxlen=8, found=None):
    """One left-to-right pass; the longest form at each position wins."""
    start, end = start or {}, end or {}
    out, i, n = [], 0, len(tokens)
    while i < n:
        hit = 0
        for L in range(min(maxlen, n - i), 0, -1):
            t = tuple(tokens[i:i + L])
            g = anywhere.get(t) or (start.get(t) if i == 0 else None) or (end.get(t) if i + L == n else None)
            if g is not None:
                hit = L
                if found is not None and g is not True:
                    found.append(g)
                break
        if hit:
            i += hit
        else:
            out.append(tokens[i]); i += 1
    return out


TYPO_FORMS = sorted(LEGAL_TYPO)
WORDS_DICT = {t: True for t in WORDS_SET}


# common words that look like a legal-form typo but are not ('parts' ~ PartG, 'operation' ~ corporation)
TYPO_PROTECT = {"parts", "partner", "partners", "operation", "operations", "corporate", "limits", "limitless",
                "cooperation", "incorporation", "company", "companies", "private", "public"}


def _typo_group(w, cache):
    """Legal-form group that w is a one-letter typo of, else None. Cached per batch (words repeat a lot).
    Strict on purpose: same first letter, at least 6 letters, exactly 1 edit (swap counts as 1)."""
    if w in cache:
        return cache[w]
    g = None
    if len(w) >= TYPO_MIN_LEN and (w,) not in WORDS_SET and w not in TYPO_PROTECT:
        from rapidfuzz import process
        from rapidfuzz.distance import DamerauLevenshtein
        hit = process.extractOne(w, [f for f in TYPO_FORMS if f[0] == w[0]],
                                 scorer=DamerauLevenshtein.distance, score_cutoff=1)
        if hit is not None:
            g = LEGAL_TYPO[hit[0]]
    cache[w] = g
    return g


def _typo_end(tokens, found, cache):
    """'Sifi Global Limted' / 'Corpora Tio': last word (or last two joined) one typo away from a legal form."""
    for k in (2, 1):
        if len(tokens) > k:
            g = _typo_group("".join(tokens[-k:]), cache)
            if g is not None:
                found.append(g)
                return tokens[:-k]
    return tokens


def clean_name(text, legal=None, cache=None):
    """-> (no_legal text, legal-form groups found). Same rules as before, done with word lookups."""
    anywhere, start, end, maxlen = legal or (LEGAL_ANY, LEGAL_START, LEGAL_END, LEGAL_MAXLEN)
    low = "".join(c for c in unicodedata.normalize("NFD", str(text or "").lower())
                  if unicodedata.category(c) != "Mn")
    toks, found = TOKEN_RE.findall(low.replace(".", "")), []
    for _ in range(2):  # legal -> stopwords -> legal again (catches 'AG in liquidation')
        toks = _remove(toks, anywhere, start, end, maxlen, found)
        toks = _remove(toks, WORDS_DICT, maxlen=WORDS_MAXLEN)
    toks = _typo_end(toks, found, {} if cache is None else cache)   # 'Limted', 'Corpration' (last word only)
    return " ".join(toks), list(dict.fromkeys(found))


# self-check: fails fast if the CSV or the cleaning logic breaks
for _name, _want, _forms in [
        ("Thor United Corporation", "thor united", ["CORP"]),
        ("Zwicky AB", "zwicky", ["AB"]),
        ("Bau Gesellschaft mit beschränkter Haftung", "bau", None),
        ("AD Services", "ad services", []),
        ("Sifi Global Limted", "sifi global", ["LIMITED"]),
        ("Zwicky Holding AG in Liquidation", "zwicky holding", ["AG"])]:
    _got, _f = clean_name(_name)
    assert _got == _want and (_forms is None or _f == _forms), \
        f"Name-cleaning check failed: '{_name}' -> '{_got}' {_f} (expected '{_want}' {_forms})"

FEAT_SCHEMA = ("NoLegal string, Compact string, Numbers string, Phonetic string, SortKey string, "
               "LegalForms array<string>")
FEAT_COLS = ["NoLegal", "Compact", "Numbers", "Phonetic", "SortKey", "LegalForms"]


@F.pandas_udf(FEAT_SCHEMA)
def feat_udf(s: pd.Series) -> pd.DataFrame:
    from metaphone import doublemetaphone
    typo_cache = {}   # plain dict built on the worker, one per batch

    def one(t):
        t = "" if t is None else str(t)
        low = "".join(c for c in unicodedata.normalize("NFD", t.lower())
                      if unicodedata.category(c) != "Mn")

        no_legal, forms = clean_name(t, cache=typo_cache)

        # numbers: digits, roman (standalone I–XX), number words -> one sorted set.
        # Read from the name AFTER legal forms are removed, so 'B.V.' / 'B V' is not read as roman 5.
        nums = set()
        for tok in no_legal.split():
            if tok.isdigit():
                nums.add(str(int(tok)))
            elif tok in ROMAN:
                nums.add(ROMAN[tok])
            elif tok in NUM_WORDS:
                nums.add(NUM_WORDS[tok])
        numbers = "|".join(sorted(nums, key=int))
        compact = no_legal.replace(" ", "")
        if not compact:  # name was only a legal form
            compact = "".join(TOKEN_RE.findall(low))
            no_legal = compact

        tokens = no_legal.split()
        phonetic = "".join(doublemetaphone(tok)[0] for tok in tokens)
        sort_key = "".join(sorted(tokens))
        return no_legal, compact, numbers, phonetic, sort_key, forms

    return pd.DataFrame([one(t) for t in s], columns=FEAT_COLS)


SCORE_SCHEMA = "Ratio double, TokenSort double, TokenSet double, JaroWinkler double"


@F.pandas_udf(SCORE_SCHEMA)
def score_udf(c1: pd.Series, c2: pd.Series, n1: pd.Series, n2: pd.Series) -> pd.DataFrame:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
    rows = []
    for a, b, x, y in zip(c1, c2, n1, n2):
        a, b, x, y = a or "", b or "", x or "", y or ""
        rows.append((
            round(fuzz.ratio(a, b), 2),
            round(fuzz.token_sort_ratio(x, y), 2),
            round(fuzz.token_set_ratio(x, y), 2),          # review only, not in MatchScore
            round(JaroWinkler.similarity(a, b) * 100, 2),
        ))
    return pd.DataFrame(rows, columns=["Ratio", "TokenSort", "TokenSet", "JaroWinkler"])



_BRACKETS = "()（）[]［］【】{}｛｝〔〕〈〉《》「」『』"


@F.pandas_udf("string")
def native_norm_udf(s: pd.Series) -> pd.Series:
    """Native-name key: full-width -> normal (NFKC), lower case, brackets removed (text inside kept),
    spaces and punctuation removed. '中保投义德（苏州）…（有限合伙）' == '中保投义德 苏州 … 有限合伙'."""
    table = str.maketrans("", "", _BRACKETS)

    def one(t):
        if t is None:
            return None
        x = unicodedata.normalize("NFKC", str(t)).lower().translate(table)
        x = "".join(c for c in x if c.isalnum())
        return x or None

    return s.map(one)


# COMMAND ----------

# ═══════════════════════ STEP 1: DATA (SQL) ═══════════════════════
direct_ids = get_or_build("direct_ids", lambda: run_sql("01_direct_ids.sql"))
good_ids = get_or_build("good_ids", lambda: run_sql("02_good_ids.sql"))
info = get_or_build("company_info", lambda: run_sql("03_company_info.sql"))   # DOI, legal form, status

print(f"direct IDs: {direct_ids.count():,}   good IDs: {good_ids.count():,}")

# COMMAND ----------

# ═══════════════════════ STEP 2: NAME FEATURES ═══════════════════════
feats = get_or_build(
    "name_features",
    lambda: (direct_ids.unionByName(good_ids)
             .repartition(FEATURE_PARTITIONS)
             .withColumn("f", feat_udf("Name"))
             .select("Id", "Name", F.substring("Id", 1, 2).alias("cc"), "f.*")),
)
# check: B.V. / B V / BV must all give the same Compact, Numbers empty, LegalForms ["BV"]
display(feats.filter(F.col("Id").isin("NL33166750", "NL24143896", "YY*1047781105"))
             .select("Id", "Name", "Compact", "Numbers", "LegalForms"))

# COMMAND ----------

# ═══════════════════════ BUILDING BLOCKS ═══════════════════════
LINK_PRIORITY = {"shared_shareholder": 1, "shared_subsidiary": 2, "shared_BO": 3,
                 "shared_BO_subsidiary": 4, "shared_GUO": 5, "shared_DUO": 6}


def blocks(df):
    """YY blocking: same sound (Metaphone) OR same words in any order. No prefix rule (too big globally)."""
    return (df.select("Id", "cc", F.lit("phonetic").alias("Rule"), F.col("Phonetic").alias("Key"))
              .unionByName(df.select("Id", "cc", F.lit("sorted_words").alias("Rule"), F.col("SortKey").alias("Key")))
              .filter(F.col("Key").isNotNull() & (F.col("Key") != "")))


def candidate_pairs(cases, pool, rnd):
    a = blocks(feats.join(cases.select("Id"), "Id", "left_semi"))
    b = blocks(feats.join(pool.select("Id"), "Id", "left_semi"))
    if BLOCK_MAX_POOL:
        big = save(b.groupBy("Rule", "Key").agg(F.count("*").alias("PoolSize"))
                    .filter(F.col("PoolSize") > BLOCK_MAX_POOL), f"{rnd}_skipped_name_blocks")
        a = a.join(big.select("Rule", "Key"), ["Rule", "Key"], "left_anti")
        b = b.join(big.select("Rule", "Key"), ["Rule", "Key"], "left_anti")
    a, b = a.alias("a"), b.alias("b")
    return (a.join(b, (F.col("a.Rule") == F.col("b.Rule")) & (F.col("a.Key") == F.col("b.Key")))
             .filter(F.col("a.Id") != F.col("b.Id"))
             .groupBy(F.col("a.Id").alias("OriginalId"), F.col("b.Id").alias("MatchedId"))
             .agg(F.concat_ws("|", F.array_sort(F.collect_set("a.Rule"))).alias("BlockRules")))


def add_scores(pairs):
    f1 = feats.select(F.col("Id").alias("OriginalId"), "Name", F.col("Compact").alias("NameNormalized"),
                      F.col("NoLegal").alias("NoLegal_1"), F.col("Numbers").alias("Numbers_1"),
                      F.col("LegalForms").alias("LegalForms_1"))
    f2 = feats.select(F.col("Id").alias("MatchedId"), F.col("Name").alias("MatchedName"),
                      F.col("Compact").alias("MatchedNameNormalized"),
                      F.col("NoLegal").alias("NoLegal_2"), F.col("Numbers").alias("Numbers_2"),
                      F.col("LegalForms").alias("LegalForms_2"))
    s = (pairs.join(f1, "OriginalId").join(f2, "MatchedId")
              .withColumn("s", score_udf("NameNormalized", "MatchedNameNormalized", "NoLegal_1", "NoLegal_2"))
              .select("*", "s.*").drop("s", "NoLegal_1", "NoLegal_2"))
    return add_legal_check(s.withColumn("MatchScore", F.greatest("Ratio", "TokenSort", "JaroWinkler")))


def add_legal_check(df):
    """LegalFormCheck: same only when BOTH names carry a legal form from the SAME row of
    legal_form_mapping.csv (Ltd = Limited is fine; PTY LTD, CO LTD, PVT LTD are their own rows)."""
    n1 = F.size(F.coalesce(F.col("LegalForms_1"), F.array().cast("array<string>")))
    n2 = F.size(F.coalesce(F.col("LegalForms_2"), F.array().cast("array<string>")))
    return (df.withColumn("NameLegalForm_1", F.concat_ws("|", "LegalForms_1"))
              .withColumn("NameLegalForm_2", F.concat_ws("|", "LegalForms_2"))
              .withColumn("LegalFormCheck",
                          F.when((n1 == 0) & (n2 == 0), "both_missing")
                           .when((n1 == 0) | (n2 == 0), "one_side_missing")
                           .when(F.arrays_overlap("LegalForms_1", "LegalForms_2"), "same")
                           .otherwise("different"))
              .drop("LegalForms_1", "LegalForms_2"))


def add_ownership(pairs):
    """ownership_link, HasLink, StrongestLink and its % on both sides (edges come from 04_ownership_edges.sql)."""
    ids = (pairs.select(F.col("OriginalId").alias("Id"))
                .unionByName(pairs.select(F.col("MatchedId").alias("Id"))))
    e = with_ids_view(ids, "04_ownership_edges.sql")
    e1 = e.select(F.col("CompanyId").alias("OriginalId"), "HubId", "LinkType",
                  F.col("DirectPct").alias("DirectPct_1"), F.col("TotalPct").alias("TotalPct_1"))
    e2 = e.select(F.col("CompanyId").alias("MatchedId"), "HubId", "LinkType",
                  F.col("DirectPct").alias("DirectPct_2"), F.col("TotalPct").alias("TotalPct_2"))
    j = (pairs.select("OriginalId", "MatchedId").join(e1, "OriginalId")
              .join(e2, ["MatchedId", "HubId", "LinkType"]))

    prio = F.coalesce(*[F.when(F.col("LinkType") == k, F.lit(v)) for k, v in LINK_PRIORITY.items()])
    strength = F.greatest(F.coalesce("TotalPct_1", "DirectPct_1", F.lit(-1.0)),
                          F.coalesce("TotalPct_2", "DirectPct_2", F.lit(-1.0)))
    w = Window.partitionBy("OriginalId", "MatchedId").orderBy(strength.desc(), prio.asc())
    strongest = (j.withColumn("rn", F.row_number().over(w)).filter("rn = 1")
                  .select("OriginalId", "MatchedId", F.col("LinkType").alias("StrongestLink"),
                          "DirectPct_1", "DirectPct_2", "TotalPct_1", "TotalPct_2"))
    types = j.groupBy("OriginalId", "MatchedId").agg(
        F.concat_ws("|", F.array_sort(F.collect_set("LinkType"))).alias("ownership_link"))

    return (pairs.join(types.join(strongest, ["OriginalId", "MatchedId"]), ["OriginalId", "MatchedId"], "left")
                 .withColumn("HasLink", F.col("ownership_link").isNotNull().cast("int"))
                 .withColumn("ownership_link", F.coalesce("ownership_link", F.lit("no ownership link")))
                 .withColumn("IsDirectSame", same("DirectPct_1", "DirectPct_2"))
                 .withColumn("IsTotalSame", same("TotalPct_1", "TotalPct_2")))

# COMMAND ----------

# ═══════════════════════ ROUND ENGINE ═══════════════════════
DEBUG_COLS = ["Round", "OriginalId", "Name", "MatchedId", "MatchScore", "Numbers_1", "Numbers_2",
              "NameLegalForm_1", "NameLegalForm_2", "DateCheck", "NativeCheck",
              "ownership_link", "MatchCount", "FilterDecision"]


def as_debug(df, rnd, decision):
    out = df
    for c in DEBUG_COLS:
        if c not in out.columns:
            out = out.withColumn(c, F.lit(None).cast("string"))
    dec = F.lit(decision) if isinstance(decision, str) else decision
    return (out.withColumn("Round", F.lit(rnd)).withColumn("FilterDecision", dec)
               .select(*[F.col(c).cast("string") for c in DEBUG_COLS]))


def find_candidates(rnd, cases, pool):
    """PART A: blocks -> score >= THRESHOLD -> numbers identical -> legal form same on both sides.
    Saves <rnd>_candidates (what Part B starts from) and <rnd>_debug_a."""
    if REUSE_SAVED and spark.catalog.tableExists(tname(f"{rnd}_candidates")):
        print(f"[reuse] {rnd}_candidates")
        return spark.table(tname(f"{rnd}_candidates"))

    debug = []
    pairs = save(candidate_pairs(cases, pool, rnd), f"{rnd}_pairs")
    debug.append(as_debug(
        cases.join(pairs.select(F.col("OriginalId").alias("Id")).distinct(), "Id", "left_anti")
             .withColumnRenamed("Id", "OriginalId"),
        rnd, "DROPPED (no candidate in any name block)"))

    scored = save(add_scores(pairs), f"{rnd}_scored")
    passed = scored.filter(F.col("MatchScore") >= THRESHOLD)
    debug.append(as_debug(
        scored.groupBy("OriginalId", "Name").agg(F.max("MatchScore").alias("MatchScore"))
              .join(passed.select("OriginalId").distinct(), "OriginalId", "left_anti"),
        rnd, F.concat(F.lit("DROPPED (best score "), F.col("MatchScore").cast("string"),
                      F.lit(f" < {THRESHOLD})"))))

    num_ok = passed.filter(F.col("Numbers_1") == F.col("Numbers_2"))
    debug.append(as_debug(passed.filter(F.col("Numbers_1") != F.col("Numbers_2")), rnd,
                          "REMOVE (numbers in name differ)"))

    # legal form must be present on BOTH sides and the same
    legal_ok = num_ok.filter(F.col("LegalFormCheck") == "same")
    debug.append(as_debug(num_ok.filter(F.col("LegalFormCheck") != "same"), rnd,
                          F.concat(F.lit("REMOVE (legal form "), F.col("LegalFormCheck"), F.lit(": "),
                                   F.coalesce(F.col("NameLegalForm_1"), F.lit("")), F.lit(" vs "),
                                   F.coalesce(F.col("NameLegalForm_2"), F.lit("")), F.lit(")"))))

    cands = save(legal_ok, f"{rnd}_candidates")
    save(reduce(lambda x, y: x.unionByName(y), debug), f"{rnd}_debug_a")
    return cands


def _num(c):
    """100.0 -> '100', 98.5 -> '98.5' (for readable WonBy text)."""
    return F.regexp_replace(F.col(c).cast("string"), r"\.0$", "")


def finish_round(rnd, cands, sql_info):
    """PART B: date check -> CN native-name check -> pick one (ownership link, then score).
    Candidates with a DOI before the source date and candidates with no DOI compete equally.
    Returns (all remaining rows: unique winners + tied groups, debug)."""
    debug = []
    w = Window.partitionBy("OriginalId")
    is_cn = F.col("MatchedId").startswith(NATIVE_CHECK_PREFIX)

    # ── STEP 1 date check: YY oldest source date (SQL Server) vs candidate's incorporation date (Orbis) ──
    # only 'reject' (DOI after the source date) is removed; 'pass' and 'no_doi' both stay
    osd = sql_info.select("OriginalId", "OldestSourceDate_SQL", "NativeName_1")
    p = (cands.join(osd, "OriginalId", "left")
              .join(info.select(F.col("Id").alias("MatchedId"), F.col("DOI").alias("DOI_Matched")),
                    "MatchedId", "left")
              .withColumn("_osd", F.expr("try_to_date(OldestSourceDate_SQL, 'yyyyMMdd')"))
              .withColumn("DateCheck",
                          F.when(F.col("_osd").isNull(), "no_source_date")
                           .when(F.col("DOI_Matched").isNull(), "no_doi")
                           .when(F.col("_osd") < F.col("DOI_Matched"), "reject")
                           .otherwise("pass"))
              .drop("_osd")
              .withColumn("_n0", F.count("*").over(w))
              .withColumn("_rej_date", F.sum((F.col("DateCheck") == "reject").cast("int")).over(w)))
    debug.append(as_debug(p.filter(F.col("DateCheck") == "reject"), rnd,
                          "REMOVE (incorporation date after the YY oldest source date)"))
    p = p.filter(F.col("DateCheck") != "reject").withColumn("_n1", F.count("*").over(w))

    # ── STEP 2 native name check, only when MatchedId starts with CN ──
    native_cn = (spark.table(f"{SRC}.contact_info")
                 .filter(F.col("BvD_ID_number").startswith(NATIVE_CHECK_PREFIX))
                 .groupBy(F.col("BvD_ID_number").alias("MatchedId"))
                 .agg(F.first("NAME_NATIVE", ignorenulls=True).alias("NativeName_2")))
    p = (p.join(native_cn, "MatchedId", "left")
          .withColumn("_k1", native_norm_udf("NativeName_1"))
          .withColumn("_k2", native_norm_udf("NativeName_2"))
          .withColumn("NativeCheck",
                      F.when(~is_cn, "n/a")
                       .when(F.col("_k1").isNull() | F.col("_k2").isNull(), "missing")
                       .when(F.col("_k1") == F.col("_k2"), "same")
                       .otherwise("different"))
          .drop("_k1", "_k2")
          .withColumn("_any_same", F.max((F.col("NativeCheck") == "same").cast("int")).over(w)))
    keep_native = (F.col("NativeCheck") == "n/a") | \
        F.when(F.col("_any_same") == 1, F.col("NativeCheck") == "same") \
         .otherwise(F.col("NativeCheck") == "missing")
    debug.append(as_debug(p.filter(~keep_native), rnd,
                          F.when(F.col("NativeCheck") == "different", "REMOVE (native name differs)")
                           .otherwise("REMOVE (native name missing, another CN candidate has the same native name)")))
    p = (p.filter(keep_native).drop("_any_same")
          .withColumn("_n2", F.count("*").over(w))
          .withColumn("_rej_native", F.col("_n1") - F.col("_n2"))
          .withColumn("_npass", F.sum((F.col("DateCheck") == "pass").cast("int")).over(w)))

    # ── STEP 3a pick one: the only candidate with a shared ownership link wins ──
    p = (add_ownership(p)
         .withColumn("LinkCount", F.sum("HasLink").over(w)))
    p = p.withColumn("_owner_win", (F.col("_n2") > 1) & (F.col("LinkCount") == 1) & (F.col("HasLink") == 1))
    keep_link = (F.col("_n2") == 1) | (F.col("LinkCount") != 1) | (F.col("HasLink") == 1)
    debug.append(as_debug(p.filter(~keep_link), rnd, "REMOVE (no ownership link, another candidate has one)"))
    p = p.filter(keep_link).withColumn("_n3", F.count("*").over(w))

    # ── STEP 3b pick one: highest name score wins ──
    p = (p.withColumn("Best", F.max("MatchScore").over(w))
          .withColumn("_next", F.max(F.when(F.col("MatchScore") < F.col("Best"), F.col("MatchScore"))).over(w)))
    debug.append(as_debug(
        p.filter(F.col("MatchScore") < F.col("Best")), rnd,
        F.concat(F.lit("REMOVE (score "), F.col("MatchScore").cast("string"),
                 F.lit(" lost to "), F.col("Best").cast("string"), F.lit(")"))))
    kept = (p.filter(F.col("MatchScore") == F.col("Best"))
             .withColumn("MatchCount", F.count("*").over(w))
             .withColumn("_any_cn", F.max(is_cn.cast("int")).over(w)))

    # ── sheet: 1 left -> INC / No_INC by the winner's DOI; 2+ left -> Multi_INC_review ──
    inc = F.when(F.col("DateCheck") == "pass", "INC").otherwise("No_INC")
    native_tag = F.when(F.col("NativeCheck") == "same", "Same_Native").otherwise("No_Native")
    kept = kept.withColumn(
        "ResultSheet",
        F.when(F.col("MatchCount") > 1,
               F.when(F.col("_any_cn") == 1, "CN_Multi_INC_review").otherwise("Multi_INC_review"))
         .when(~is_cn, inc)
         .otherwise(F.concat(F.lit("CN_"), inc, F.lit("_"), native_tag)))

    # ── WonBy: why this candidate was chosen (last column) ──
    others = F.col("_n2") - 1
    others_pass = F.col("_npass") - (F.col("DateCheck") == "pass").cast("int")
    others_mix = F.concat(F.lit(" ("), others_pass.cast("string"), F.lit(" pass, "),
                          (others - others_pass).cast("string"), F.lit(" no_doi)"))
    doi_txt = F.when(F.col("DateCheck") == "pass",
                     F.concat(F.lit("DOI "), F.col("DOI_Matched").cast("string"), F.lit(" <= source date "),
                              F.date_format(F.expr("try_to_date(OldestSourceDate_SQL, 'yyyyMMdd')"), "yyyy-MM-dd"))) \
                .otherwise(F.lit("no DOI"))
    removed = F.concat_ws(", ",
                          F.when(F.col("_rej_date") > 0, F.concat(F.col("_rej_date").cast("string"),
                                                                  F.lit(" rejected (DOI after source date)"))),
                          F.when(F.col("_rej_native") > 0, F.concat(F.col("_rej_native").cast("string"),
                                                                    F.lit(" removed (native name)"))))
    kept = kept.withColumn(
        "WonBy",
        F.when(F.col("MatchCount") > 1,
               F.concat(F.lit("tie: "), F.col("MatchCount").cast("string"), F.lit(" candidates left, same score "),
                        _num("Best"),
                        F.when(F.col("LinkCount") == 0, F.lit(", none has an ownership link"))
                         .otherwise(F.lit(", more than one has an ownership link"))))
         .when(F.col("_n0") == 1,
               F.concat(F.lit("only_candidate: 1 candidate found, "), doi_txt))
         .when(F.col("_n2") == 1,
               F.concat(F.when((F.col("_rej_native") > 0) & (F.col("_rej_date") == 0), F.lit("native: "))
                         .otherwise(F.lit("date: ")),
                        removed, F.lit(", 1 left, "), doi_txt))
         .when(F.col("_owner_win"),
               F.concat(F.lit("ownership: only candidate with a shared link ("), F.col("StrongestLink"),
                        F.lit("), beat "), others.cast("string"), F.lit(" other(s)"), others_mix))
         .otherwise(
               F.concat(F.lit("score: "), _num("Best"), F.lit(" vs next best "), _num("_next"),
                        F.lit(", beat "), (F.col("_n3") - 1).cast("string"), F.lit(" other(s)"))))

    kept = kept.drop(*[c for c in kept.columns if c.startswith("_")], "Best", "LinkCount")
    kept = save(kept, f"{rnd}_proposals")
    debug.append(as_debug(kept, rnd, F.when(F.col("MatchCount") == 1, "KEEP (unique)")
                                       .otherwise("KEEP (tie -> Multi_INC_review)")))
    dbg = save(reduce(lambda x, y: x.unionByName(y), debug), f"{rnd}_debug_b")
    return kept, dbg


EXPORT_COLS = [
    "OriginalId", "Name", "NameNormalized", "MatchedId", "MatchedName", "MatchedNameNormalized",
    "MatchScore", "Ratio", "TokenSort", "JaroWinkler", "TokenSet", "BlockRules", "Numbers_1", "Numbers_2",
    "NameLegalForm_1", "NameLegalForm_2",
    "ownership_link", "StrongestLink", "DirectPct_1", "DirectPct_2", "IsDirectSame",
    "TotalPct_1", "TotalPct_2", "IsTotalSame",
    "OldestSourceDate_SQL", "DOI_Matched", "DateCheck",
    "NativeName_1", "NativeName_2", "NativeCheck",
    "MatchCount", "ResultSheet", "WonBy",
]

# COMMAND ----------

# ═══════════════════════ STEP 3 (PART A): candidates + YY list for SQL Server ═══════════════════════
cands = find_candidates("r1", direct_ids, good_ids)

yy_for_sql = cands.select("OriginalId").distinct().orderBy("OriginalId")
yy_csv = f"{OUT_DIR}/yy_ids_for_sql_{RUN_TAG}.csv"
write_one_csv(yy_for_sql, yy_csv)
print(f"candidate rows: {cands.count():,}   YY IDs for SQL Server: {yy_for_sql.count():,}")
print(f"NEXT: download {yy_csv.split('/')[-1]}, run yy_sql_info.py on the PC, upload its CSV to {OUT_DIR}")

# COMMAND ----------

# ═══════════════════════ STEP 4 (PART B): SQL Server info -> date + native checks -> unique ═══════════════════════
if SQL_INFO_CSV:
    sql_csv = SQL_INFO_CSV
else:
    found = sorted(f.path for f in dbutils.fs.ls(OUT_DIR) if f.name.startswith("yy_sql_info_") and f.name.endswith(".csv"))
    assert found, f"No yy_sql_info_*.csv in {OUT_DIR} — run yy_sql_info.py on the PC and upload its output first."
    sql_csv = found[-1]
print(f"SQL Server info: {sql_csv}")

# SQL Server sends NativeName as UTF-8 bytes read as cp1252 ('æ±Ÿè‹…'); repair it back to Chinese ('江苏…')
def _fix_mojibake(s):
    """UTF-8 text that was read as cp1252/latin-1 ('æ±Ÿè‹…') -> real text ('江苏…'). Real text is returned unchanged."""
    if s is None:
        return None
    try:
        b = bytearray()
        for ch in s:
            o = ord(ch)
            b.extend(bytes([o]) if o < 256 else ch.encode("cp1252"))
        return b.decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


fix_mojibake_udf = F.udf(_fix_mojibake, "string")

sql_info = save(
    spark.read.option("header", True).option("multiLine", True).option("escape", '"')
         .option("encoding", "UTF-8").csv(sql_csv)
         .select("OriginalId", "OldestSourceDate_SQL",
                 fix_mojibake_udf(F.col("NativeName_SQL")).alias("NativeName_1"))
         .dropDuplicates(["OriginalId"]),
    "sql_info",
)
cands = spark.table(tname("r1_candidates"))
proposals, dbg_b = finish_round("r1", cands, sql_info)
unique = proposals   # unique winners (INC/No_INC/CN_*) + tied rows (Multi_INC_review / CN_Multi_INC_review)

print(f"output rows: {unique.count():,}   unique winners: {unique.filter(F.col('MatchCount') == 1).count():,}")
display(unique.groupBy("ResultSheet").count().orderBy("ResultSheet"))

# COMMAND ----------

# ═══════════════════════ STEP 5: EXPORT — one CSV + one Excel (UniqueMatch, with ResultSheet) ═══════════════════════
import xlsxwriter

out_cols = [c for c in EXPORT_COLS if c in unique.columns]
unique_out = unique.select(*[F.col(c).cast("string") if c in ("DOI_Matched",) else F.col(c) for c in out_cols]) \
                   .orderBy("ResultSheet", "OriginalId")

write_one_csv(unique_out, f"{OUT_DIR}/matches_{RUN_TAG}.csv")

MAX_EXCEL_ROWS = 1_048_000
n_unique = unique.count()
if n_unique > MAX_EXCEL_ROWS:
    print(f"[WARN] {n_unique:,} rows, first {MAX_EXCEL_ROWS:,} in Excel (all in CSV)")
unique_pd = unique_out.limit(MAX_EXCEL_ROWS).toPandas()

summary = [("Run date (IST)", datetime.now(IST).strftime("%Y-%m-%d %H:%M")),
           ("MatchScore threshold", THRESHOLD),
           ("Direct IDs", direct_ids.count()),
           ("SQL Server info file", sql_csv.split("/")[-1]),
           ("Unique matches", n_unique)]
summary += [(f"  {r['ResultSheet']}", r["count"])
            for r in unique.groupBy("ResultSheet").count().orderBy("ResultSheet").collect()]

tmp = "/tmp/matches_YY.xlsx"
wb = xlsxwriter.Workbook(tmp, {"constant_memory": True, "strings_to_urls": False,
                               "strings_to_formulas": False, "nan_inf_to_errors": True})
hdr = wb.add_format({"bold": True, "bg_color": "#DDEBF7", "border": 1})


def write_cell(ws, r, c, v):
    """constant_memory needs row-by-row, left-to-right writes with explicit types."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NA:
        return
    if isinstance(v, str):
        ws.write_string(r, c, v)
    elif isinstance(v, bool):
        ws.write_boolean(r, c, v)
    else:
        try:
            ws.write_number(r, c, float(v))
        except (TypeError, ValueError):
            ws.write_string(r, c, str(v))


ws0 = wb.add_worksheet("Summary")
ws0.write_row(0, 0, ["Item", "Value"], hdr)
for i, (k, v) in enumerate(summary, start=1):
    write_cell(ws0, i, 0, k)
    write_cell(ws0, i, 1, v)
ws0.set_column(0, 0, 30)
ws0.set_column(1, 1, 45)

ws1 = wb.add_worksheet("UniqueMatch")
ws1.write_row(0, 0, list(unique_pd.columns), hdr)
for i, row in enumerate(unique_pd.itertuples(index=False, name=None), start=1):
    for j, v in enumerate(row):
        write_cell(ws1, i, j, v)
ws1.set_column(0, len(unique_pd.columns) - 1, 20)
ws1.freeze_panes(1, 0)

wb.close()
dest = f"{OUT_DIR}/matches_{RUN_TAG}.xlsx"
shutil.copy(tmp, dest)
print(f"Excel written: {dest}")
