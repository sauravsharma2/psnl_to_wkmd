import re
import time
from pathlib import Path

try:
    import jellyfish
except ImportError:
    jellyfish = None

try:
    from unidecode import unidecode
except ImportError:
    unidecode = None


_LIMITED_TOKENS = {
    "limited", "limite", "limit", "limi", "lim", "ltd", "lt", "li", "l",
    "lted", "limted", "limtied", "ldt", "litd", "limimted", "linited", "limied", "eld",
    "ltp", "ktd", "tld", "td", "lrd", "ltf", "lts", "lta", "lyd", "lmt", "mtd", "lc", "ltfd", "lmted", "ltds",
    "limimited", "limt", "ld",
    "ldt", "lmited", "lmtd",
    "llc", "gmbh", "sarl", "nv", "kg",
}

_PRIVATE_TOKENS = {
    "pty", "pte", "p",
    "pyt", "ptu", "ptv", "ptd", "pti", "py", "pt", "ply", "ptyx",
    "proprietari", "proprieta",
    "proprietary", "propriety", "proprietry", "proprietory",
    "private", "privat", "priv", "pri", "pvt", "fvt", "pie", "pv",
    "ptc", "ptl", "ptf", "pta", "ptle",
    "bvba", "bv",
}

_INC_TOKENS = {
    "inc", "incorporated", "incoporated", "incorpoated", "incorpated",
    "incroporated", "incorpoarted", "incorproated", "incorp",
}

_CORP_TOKENS = {"corp", "corporation", "corportion", "corpn"}
_LLP_TOKENS  = {"llp", "lp"}
_NL_TOKENS   = {"nl"}
_PL_TOKENS   = {"p/l", "pl"}
_OTHER_TOKENS = {"ag", "pti", "rte", "privated", "litmited", "litmied", "ptel"}

_ALL_LEGAL_TOKENS = (
    _LIMITED_TOKENS | _PRIVATE_TOKENS |
    _INC_TOKENS | _CORP_TOKENS | _LLP_TOKENS |
    _NL_TOKENS | _PL_TOKENS | _OTHER_TOKENS
)

_FUSED_SUFFIXES = sorted(
    [
        "proprietarylimited", "proprietarylimitd", "proprietaryltd", "proprietorylimited",
        "ptylimited", "ptylimite", "ptylimit", "ptylimi", "ptylimted", "ptylinited", "ptylimimited",
        "ptyltd", "ptuimited", "ptvltd", "pytltd", "ptdltd", "p/lltd", "plltd",
        "ptelimited", "pteltd",
        "privatelimited", "privatelimite", "privatelimit", "privatelimi",
        "privatelim", "privateli", "privateltd", "privateld",
        "privatelimted", "privatelinited", "privatelimimited",
        "pvtlimited", "pvtlimite", "pvtlimit", "pvtlimi",
        "pvtltd", "pvtlimted", "pvtlmited", "pvtlinited",
        "fvtlimited", "fvtltd", "fvtlimit", "copvtltd", "copvtlimited",
        "plimited", "plimite", "plimit", "plimi", "pltd",
        "limited", "limite", "limit", "limi", "lim",
        "limted", "limied", "liited", "lmited", "linited", "limimited", "limt",
        "ltmtred", "lmtred", "ltmted",
        "private", "privat", "priv", "pri", "pvt", "fvt",
        "ltd", "lt", "li", "ld", "pr",
    ],
    key=len, reverse=True,
)
_FUSED_MIN_LEN = 4
_FOREIGN_SA_MIN_TOKENS = 2

_LEGAL_DETECT_RE = re.compile(
    r'\b('
    r'proprietary\s+limited|propriety\s+limited|proprietry\s+limited|proprietory\s+limited|'
    r'proprietary\s+ltd\.?|'
    r'private\s+limited\s+company|private\s+limited|private\s+ltd\.?|'
    r'p\s+limited|p\s+ltd\.?|pvt\.?\s*ltd\.?|pvt\.?|pte\.?|pty\.?|'
    r'p\.?/l\.?|'
    r'l\.?\s*t\.?\s*d\.?|lmt\.?|limited|ltd\.?|'
    r'limite\b|limit\b|limi\b|lim\b|limt\b|'
    r'limted\b|lmited\b|linited\b|limimited\b|limied\b|'
    r'lted\b|limtied\b|ldt\b|litd\b|limimted\b|'
    r'privat\b|private\b|\bld\b|\bli\b|pvtltd|privatelimited|'
    r'aboriginal\s+corporation'
    r')',
    re.I,
)

_AU_GEO_BRACKET_RE = re.compile(
    r'\(\s*(?:'
    r'australia|aust\.?|aus\.?|'
    r'nsw|vic|qld|sa|wa|nt|act|tas|'
    r'asia(?:\s*[-–]?\s*pacific)?|south\s*east\s*asia|southeast\s*asia|'
    r's\.?e\.?a\.?|s\.?g\.?|ap|'
    r'singapore|singapour|'
    r'foreign\s+co[^)]*'
    r')\s*\)',
    re.I,
)

_BRACKETED_LEGAL_RE = re.compile(
    r'\(\s*(?:proprietary|propriety|proprietry|proprietory|private|privat|priv|pvt|pte|pty|p)\s*\)',
    re.I,
)

_S_TOKEN_RE = re.compile(r'\(\s*[Ss]\s*\)')


def load_synonyms(path=None) -> dict:
    import os
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synonyms.txt")
    synonyms = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                src, tgt = line.split("=", 1)
                src = src.strip().lower()
                tgt = tgt.strip().lower()
                if src and tgt:
                    synonyms[src] = tgt
    except FileNotFoundError:
        pass
    return synonyms


def _build_synonym_re(synonyms: dict):
    if not synonyms:
        return None
    words = sorted(synonyms.keys(), key=len, reverse=True)
    escaped = [re.escape(w) for w in words]
    return re.compile(r'\b(?:' + '|'.join(escaped) + r')\b', re.I)


_SYNONYMS: dict = {}
_SYNONYM_RE = None


def _init_synonyms():
    global _SYNONYMS, _SYNONYM_RE
    if not _SYNONYMS:
        _SYNONYMS = load_synonyms()
        _SYNONYM_RE = _build_synonym_re(_SYNONYMS)


def apply_synonyms(text: str) -> str:
    _init_synonyms()
    if not text or not _SYNONYM_RE:
        return text
    return _SYNONYM_RE.sub(lambda m: _SYNONYMS[m.group(0).lower()], text)


def _prenormalise_name(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r'\([^)]*\)', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'((?:LTD|LIMITED))[^A-Za-z0-9]+$', r'\1', s, flags=re.I)
    s = re.sub(r'\bPTY(LIMITED|LTD|LIM|LT)\b', r'PTY \1', s, flags=re.I)
    s = re.sub(r'\bP\s+TY\b', 'PTY', s, flags=re.I)
    s = re.sub(r'\bPT\s+Y\b',  'PTY', s, flags=re.I)
    s = re.sub(r'(?<=[A-Za-z])0(?=[A-Za-z])', 'O', s)
    return s


def strip_geo_token(name: str) -> str:
    s = re.sub(r'\s*' + _S_TOKEN_RE.pattern + r'\s*', ' ', name).strip()
    s = re.sub(_AU_GEO_BRACKET_RE.pattern, ' ', s, flags=re.I).strip()
    s = re.sub(_BRACKETED_LEGAL_RE.pattern, ' ', s, flags=re.I).strip()
    s = re.sub(r"\(\s*s'pore\s*\)", ' ', s, flags=re.I).strip()
    s = re.sub(r"\bs'pore\b", ' ', s, flags=re.I).strip()
    s = re.sub(r'\s+', ' ', s).strip()
    return s

strip_s_token = strip_geo_token


def load_words_to_remove(path: str = "words_to_remove.txt") -> set:
    p = Path(path)
    if not p.exists():
        return set()
    with open(p, encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip() and not line.strip().startswith('#')}


def build_words_regex(words: set):
    if not words:
        return None
    pattern = r'\b(' + '|'.join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + r')\b'
    return re.compile(pattern, re.I)


_LEGAL_CLUSTER_RE = re.compile(
    r'(?<!\w)'
    r'('
    r'proprietary\s+limited|propriety\s+limited|proprietry\s+limited|proprietory\s+limited|'
    r'proprietary\s+ltd\.?|'
    r'private\s+limited|private\s+ltd\.?|pvt\.?\s*ltd\.?|pvt\.?|pte\.?|pty\.?|'
    r'p\.?/l\.?|'
    r'p\.?\s*ltd\.?|p\.?\s*limited|'
    r'limited|limite|limit|limi|lim|limt|limted|lmited|linited|limimited|limied|'
    r'lted\b|limtied\b|ldt\b|litd\b|limimted\b|'
    r'ltd\.?|lt\b|ld\b|li\b|'
    r'nl\b|n\.l\.?\b|'
    r'privat\b|private\b|priv\b|pri\b|'
    r'pvtltd|privatelimited|copvtltd|ptyltd|ptylimited'
    r')'
    r'(?!\w)',
    re.I,
)

_JUNK_AFTER_LEGAL_RE = re.compile(
    r'\s+(?:'
    r'represented\s+through|authoris[e]?d\s+representative|authorized\s+representative|'
    r'through\s+its|company\s+secretary|through\s+its\s+director|'
    r'\(first\s+holder\)|\(second\s+holder\)|\(third\s+holder\)|'
    r'&\s+sh(?:ri|\.?\s)|&\s+smt|&\s+mr(?:\s|s)|&\s+dr|'
    r'atf\b|as\s+trustee|as\s+ttee'
    r')',
    re.I,
)


def truncate_at_legal_suffix(name: str) -> str:
    if not name:
        return name

    junk_match = _JUNK_AFTER_LEGAL_RE.search(name)
    if junk_match:
        legal_before = list(_LEGAL_CLUSTER_RE.finditer(name[:junk_match.start()]))
        if legal_before:
            return name[:legal_before[-1].end()].strip()

    matches = list(_LEGAL_CLUSTER_RE.finditer(name))
    if not matches:
        return name

    if len(matches) == 1:
        m = matches[0]
        text_before = name[:m.start()].strip()
        after       = name[m.end():].strip()
        if after and after[0] in ',;&([<':
            return name[:m.end()].strip()
        if len(text_before.split()) < 2 and len(after) > 3:
            return name
        if len(after) > 3:
            return name[:m.end()].strip()
        return name

    for i, m in enumerate(matches):
        text_before = name[:m.start()].strip()
        after       = name[m.end():].strip()
        if after and after[0] in ',;&([<':
            return name[:m.end()].strip()
        if len(text_before.split()) >= 1 and len(after) > 3:
            return name[:m.end()].strip()

    return name


def _strip_fused(token: str) -> str:
    t = token.lower()
    if len(t) < _FUSED_MIN_LEN:
        return token
    for suffix in _FUSED_SUFFIXES:
        if t.endswith(suffix):
            prefix = token[:len(token) - len(suffix)]
            if len(prefix) <= 3:
                continue
            return prefix
    return token


def strip_legal_suffix(name: str) -> str:
    tokens = name.split()
    changed = True

    while changed and tokens:
        changed = False
        last = tokens[-1].lower().rstrip(".,;:()/")

        if last in _PL_TOKENS and len(tokens) >= 2:
            tokens.pop()
            changed = True
            continue

        last_alpha = re.sub(r'[^a-z]', '', last)
        if last_alpha in _NL_TOKENS and len(tokens) >= 2:
            tokens.pop()
            changed = True
            continue

        if last == "corporation" and len(tokens) >= 2:
            prev = tokens[-2].lower().rstrip(".,;:()")
            if prev == "aboriginal":
                tokens.pop()
                tokens.pop()
                changed = True
                continue

        if last in _ALL_LEGAL_TOKENS and last not in {"l", "p"}:
            was_limited = last in _LIMITED_TOKENS
            tokens.pop()
            changed = True
            if was_limited and tokens:
                second = tokens[-1].lower().rstrip(".,;:()/")
                if second in _PRIVATE_TOKENS and second not in {"p"}:
                    tokens.pop()
                elif second in {"p"} and len(tokens) >= 2:
                    tokens.pop()
            continue

        if last in {"l", "p"} and len(tokens) >= 3:
            was_limited = last == "l"
            tokens.pop()
            changed = True
            if was_limited and tokens:
                second = tokens[-1].lower().rstrip(".,;:()/")
                if second in _PRIVATE_TOKENS and second not in {"p"}:
                    tokens.pop()
                elif second in {"p"} and len(tokens) >= 2:
                    tokens.pop()
            continue

        if len(tokens) >= 3 and tokens[-1].lower().rstrip(".,;:()") == "co":
            if tokens[-2].rstrip(".,;:()") == "&":
                tokens.pop()
                tokens.pop()
                changed = True
                continue

        if last == "ag" and len(tokens) >= 3:
            tokens.pop()
            changed = True
            continue

        _last_nodot = tokens[-1].lower().replace(".", "").rstrip(",;:()")
        if _last_nodot == "lp" and len(tokens) >= 3:
            tokens.pop()
            changed = True
            continue

        raw_last = tokens[-1]
        stripped_prefix = _strip_fused(raw_last)
        if stripped_prefix != raw_last:
            tokens.pop()
            if stripped_prefix:
                tokens.append(stripped_prefix)
            changed = True
            continue

    if tokens and tokens[-1].lower().rstrip(".,;:()") == "sa" and len(tokens) >= _FOREIGN_SA_MIN_TOKENS:
        tokens.pop()

    return " ".join(tokens)


# ─── VECTORIZED REGEX (for _series functions) ────────────────────────────────

_VEC_GEO_BRACKET_STR  = _AU_GEO_BRACKET_RE.pattern
_VEC_LEGAL_BRACKET_STR = _BRACKETED_LEGAL_RE.pattern

_VSTRIPPA = re.compile(
    r'\s+(?:proprietary|proprietory|proprietry|propriety|private|privat|priv|pvt|pte|pty|pyt|ptu|ptv|ptd|ply|ptyx|py)\.?'
    r'\s+(?:limited|limite|limit|limi|lim|limt|limted|lmited|linited|ltd|lts|lta|ltp|ktd|tld|lrd|ltf|lyd|lmt|mtd|lc|eld|lt)\.?$',
    re.I
)
_VSTRIPPB = re.compile(
    r'\s+(?:limited|limite|limit|limi|lim|limt|limted|lmited|linited|ltd|lts|lta|ltp|ktd|tld|lrd|ltf|lyd|lmt|mtd|lc|eld|lt)\.?$',
    re.I
)
_VSTRIPPC = re.compile(
    r'\s+(?:proprietary|proprietory|proprietry|propriety|private|privat|priv|pvt|pte|pty|pyt|ptu|ptv|ply|ptyx|py)\.?$',
    re.I
)
_VSTRIPPD = re.compile(
    r'\s+(?:inc|incorporated|incoporated|incorpoated|incorpated|incroporated|incorpoarted|'
    r'incorproated|incorp|corp|corporation|corportion|llc|llp|gmbh|sarl|nl|n\.l\.?|l\.?p\.?|p\.?/l\.?|\bpl\b)\.?$',
    re.I
)

_VEC_TRUNC_RE = re.compile(
    r'((?:pty|pte|pvt|ptu|ptv|limited|ltd|lts|ltp|ktd|tld|lrd|ltf|lta|lyd|lmt|mtd|lc|eld|lmted|ltds)\.?)'
    r'(\s+(?:(?!(?:pty|pte|pvt|ptu|limited|ltd|lts|ltp|ktd|tld|lrd|ltf|lta|lyd|lmt|mtd|lc|eld)\b)'
    r'[A-Za-z0-9]\S*\s*){1,10})$',
    re.I
)


def make_condensed_series(series, words_re=None):
    """Vectorised make_condensed for bulk preprocessing. ~30-60s for 23M rows."""
    import pandas as pd

    s = series.fillna("").astype(str).str.strip()

    # Step 0 — bracket removal
    s = s.str.replace(r'\([^)]*\)', ' ', regex=True)
    s = s.str.replace(r'((?:LTD|LIMITED))[^A-Za-z0-9]+$', r'\1', regex=True, flags=re.I)
    s = s.str.replace(r'\bPTY(LIMITED|LTD|LIM|LT)\b', r'PTY \1', regex=True, flags=re.I)
    s = s.str.replace(r'\bP\s+TY\b', 'PTY', regex=True, flags=re.I)
    s = s.str.replace(r'\bPT\s+Y\b', 'PTY', regex=True, flags=re.I)
    s = s.str.replace(r'(?<=[A-Za-z])0(?=[A-Za-z])', 'O', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()

    # Step 0a — synonyms (only on rows with synonym words)
    _init_synonyms()
    if _SYNONYMS:
        _syn_detect = re.compile(
            r'\b(?:' + '|'.join(re.escape(k) for k in sorted(_SYNONYMS, key=len, reverse=True)) + r')\b',
            re.I
        )
        _syn_replace = lambda m: _SYNONYMS[m.group(0).lower()]
        syn_mask = s.str.contains(_syn_detect, na=False)
        if syn_mask.sum() > 0:
            s[syn_mask] = s[syn_mask].apply(lambda x: _syn_detect.sub(_syn_replace, x))

    # Step 0b — geo bracket stripping
    s = s.str.replace(_VEC_GEO_BRACKET_STR, ' ', regex=True, flags=re.I)
    s = s.str.replace(_VEC_LEGAL_BRACKET_STR, ' ', regex=True, flags=re.I)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()

    # Step 1 — truncation
    s = s.str.replace(_VEC_TRUNC_RE, r'\1', regex=True).str.strip()

    # Step 2 — legal suffix strip
    for _pat in [_VSTRIPPA, _VSTRIPPB, _VSTRIPPC, _VSTRIPPD]:
        s = s.str.replace(_pat.pattern, '', case=False, regex=True).str.strip()
    for _pat in [_VSTRIPPA, _VSTRIPPB]:
        s = s.str.replace(_pat.pattern, '', case=False, regex=True).str.strip()

    # Step 3 — lowercase
    s = s.str.lower()

    # Step 4 — noise words
    if words_re:
        s = s.str.replace(words_re, ' ', regex=True)

    # Step 5 — remove non-alphanum
    s = s.str.replace(r'[^a-z0-9]', '', regex=True)

    return s


def make_match_name_series(series, words_re=None):
    """Vectorised make_match_name for bulk preprocessing."""
    s = series.fillna("").astype(str).str.strip()

    s = s.str.replace(r'\([^)]*\)', ' ', regex=True)
    s = s.str.replace(r'((?:LTD|LIMITED))[^A-Za-z0-9]+$', r'\1', regex=True, flags=re.I)
    s = s.str.replace(r'\bPTY(LIMITED|LTD|LIM|LT)\b', r'PTY \1', regex=True, flags=re.I)
    s = s.str.replace(r'\bP\s+TY\b', 'PTY', regex=True, flags=re.I)
    s = s.str.replace(r'\bPT\s+Y\b', 'PTY', regex=True, flags=re.I)
    s = s.str.replace(r'(?<=[A-Za-z])0(?=[A-Za-z])', 'O', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()

    _init_synonyms()
    if _SYNONYM_RE:
        s = s.str.replace(_SYNONYM_RE, lambda m: _SYNONYMS[m.group(0).lower()], regex=True)

    s = s.str.replace(_VEC_GEO_BRACKET_STR, ' ', regex=True, flags=re.I)
    s = s.str.replace(_VEC_LEGAL_BRACKET_STR, ' ', regex=True, flags=re.I)

    s = s.str.replace(_VEC_TRUNC_RE, r'\1', regex=True).str.strip()
    for _pat in [_VSTRIPPA, _VSTRIPPB, _VSTRIPPC, _VSTRIPPD]:
        s = s.str.replace(_pat.pattern, '', case=False, regex=True).str.strip()
    for _pat in [_VSTRIPPA, _VSTRIPPB]:
        s = s.str.replace(_pat.pattern, '', case=False, regex=True).str.strip()

    s = s.str.lower()

    if words_re:
        s = s.str.replace(words_re, ' ', regex=True)

    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()

    return s


def make_legal_form_series(series):
    """Vectorised legal form detection."""
    import pandas as pd
    lf  = pd.Series([None] * len(series), dtype=object, index=series.index)
    su  = series.str.upper().fillna("")
    nil = lf.isna()

    m = su.str.contains(r'\bLLC\b', regex=True, na=False)
    lf[m & nil] = "FOREIGN_CLOSE"; nil = lf.isna()
    m = su.str.contains(r'\s+L\.?P\.?$', regex=True, na=False)
    lf[m & nil] = "FOREIGN_CLOSE"; nil = lf.isna()

    m = su.str.contains(r'\b(?:LLP|GMBH|SARL|BVBA|\bNV\b|\bBV\b|\bKG\b|\bAG\b)\b', regex=True, na=False)
    lf[m & nil] = "FOREIGN_OTHER"; nil = lf.isna()

    m = su.str.contains(r'\bN\.?L\.?\b', regex=True, na=False)
    lf[m & nil] = "AU_NL"; nil = lf.isna()

    m = su.str.contains(r'\b(?:INC|INCORPORATED|INCOPORATED|INCORP|CORP(?:ORATION)?)\b', regex=True, na=False)
    lf[m & nil] = "AU_CORP"; nil = lf.isna()
    m = su.str.contains(r'ABORIGINAL\s+CORPORATION', regex=True, na=False)
    lf[m & nil] = "AU_CORP"; nil = lf.isna()

    m = su.str.contains(
        r'\b(?:PTY|PTE|PVT|PTU|PTV|PTD|PTI|PYT|PLY|PTYX|PROPRIETAR[Y\w]*|PROPRIET[A-Z]+|PRIVATE|PRIVAT|P/L)\b',
        regex=True, na=False
    )
    lf[m & nil] = "AU_PTY_LTD"; nil = lf.isna()

    m = su.str.contains(
        r'\b(?:LTD|LIMITED|LIMTED|LMITED|LINITED|LIMIMITED|LIMIED|LDT|LTED|LIMTIED|LITD|LIMIMTED|LMTD|LTP|KTD|TLD|LRD|LTF|LTS|LTA|LYD|LMT|MTD|LC|ELD|LTFD|LMTED|LTDS)\b',
        regex=True, na=False
    )
    lf[m & nil] = "AU_PTY_LTD"; nil = lf.isna()

    unclass = nil & series.str.contains(
        r'\b(?:pty|pte|pvt|ltd|limited|inc|corp|llc|llp|nl|lp|gmbh|p/l)\b',
        case=False, regex=True, na=False
    )
    if unclass.sum() > 0:
        lf[unclass] = series[unclass].apply(extract_legal_form)

    return lf


def make_legal_form_orig_series(series):
    import pandas as pd
    has_legal_mask = series.str.contains(_LEGAL_DETECT_RE, na=False)
    result = pd.Series([None] * len(series), dtype=object, index=series.index)
    if has_legal_mask.sum() > 0:
        result[has_legal_mask] = series[has_legal_mask].apply(extract_legal_form_orig)
    return result


def make_condensed(name: str, words_re=None) -> str:
    if not name or not str(name).strip():
        return ""

    s = _prenormalise_name(str(name))
    s = apply_synonyms(s)
    s = strip_geo_token(s)
    s = truncate_at_legal_suffix(s)
    s = strip_legal_suffix(s)
    s = s.lower()

    if words_re:
        s = words_re.sub(" ", s)

    s = re.sub(r'[^a-z0-9]', '', s)
    return s.strip()


def build_phonetic_block_key(condensed: str, n_tokens: int = 2, size: int = 4) -> str:
    if not condensed:
        return ""

    key_src = condensed
    for prefix in ("australian", "australia", "indian", "india"):
        if key_src.startswith(prefix) and len(key_src) > len(prefix) + 3:
            key_src = key_src[len(prefix):]
            break

    prefix_len = n_tokens * size
    return key_src[:prefix_len].upper()


def make_match_name(name: str, words_re=None) -> str:
    if not name or not str(name).strip():
        return ""

    s = _prenormalise_name(str(name))
    s = apply_synonyms(s)
    s = strip_geo_token(s)
    s = truncate_at_legal_suffix(s)
    s = strip_legal_suffix(s)
    s = s.lower()

    if words_re:
        s = words_re.sub(" ", s)

    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def has_legal(name: str) -> bool:
    if not name:
        return False
    if _LEGAL_DETECT_RE.search(str(name)):
        return True
    tokens = str(name).split()
    if tokens:
        last = tokens[-1].lower().rstrip(".,;:()/")
        if last in _ALL_LEGAL_TOKENS:
            return True
        if len(tokens) >= 2:
            second = tokens[-2].lower().rstrip(".,;:()/")
            if second in _ALL_LEGAL_TOKENS:
                return True
    return False


def extract_legal_form(name: str) -> str | None:
    if not name:
        return None
    s = str(name)

    _tokens = s.strip().split()
    if re.search(r'\bllc\b', s, re.I):
        return "FOREIGN_CLOSE"
    if len(_tokens) >= 3 and re.match(r'^l\.?p\.?$', _tokens[-1], re.I):
        return "FOREIGN_CLOSE"
    if re.search(r'\bllp\b', s, re.I):
        return "FOREIGN_OTHER"
    if re.search(r'\bgmbh\b', s, re.I):
        return "FOREIGN_OTHER"
    if re.search(r'\bsarl\b', s, re.I):
        return "FOREIGN_OTHER"
    if re.search(r'\bbvba\b', s, re.I):
        return "FOREIGN_OTHER"
    if re.search(r'\bbv\b', s, re.I):
        return "FOREIGN_OTHER"
    if re.search(r'\bnv\b', s, re.I):
        return "FOREIGN_OTHER"
    if re.search(r'\bkg\b', s, re.I):
        return "FOREIGN_OTHER"
    if re.search(r'\bag\b', s, re.I):
        return "FOREIGN_OTHER"
    if len(_tokens) >= 2 and re.match(r'^sa[.,]?$', _tokens[-1], re.I):
        return "FOREIGN_OTHER"

    if re.search(r'\bn\.?l\.?\b', s, re.I):
        _last = re.sub(r'[^a-z.]', '', _tokens[-1].lower()) if _tokens else ''
        if _last in ('nl', 'n.l', 'n.l.'):
            return "AU_NL"

    if re.search(r'\baboriginal\s+corporation\b', s, re.I):
        return "AU_CORP"
    if re.search(r'\b(inc|incorporated|incoporated|incorpoated|incorpated|incroporated|incorpoarted|incorproated|incorp)\b', s, re.I):
        return "AU_CORP"
    if re.search(r'\b(corp|corporation|corportion|corpn)\b', s, re.I):
        return "AU_CORP"

    if re.search(r'\bp\.?/l\.?\b', s, re.I):
        return "AU_PTY_LTD"
    if len(_tokens) >= 1 and re.match(r'^pl[.,]?$', _tokens[-1], re.I) and len(_tokens) >= 2:
        return "AU_PTY_LTD"
    if re.search(r'\b(proprietary|propriety|proprietry|proprietory)\b', s, re.I):
        return "AU_PTY_LTD"
    if re.search(
        r'\b(pvt|pte|pty|pie|private|priv|ptd|ptc|ptl|ptf|pta|ptle|pti|ptel|privated|'
        r'pyt|ptu|ptv|py)\b',
        s, re.I
    ):
        return "AU_PTY_LTD"

    if re.search(
        r'\b(ltd|limited|limi|lim|limt|limted|lmited|linited|limimited|limied|'
        r'ldt|lted|limtied|litd|limimted|lmtd|litmited|litmied)\b',
        s, re.I
    ):
        return "AU_PTY_LTD"

    if re.search(r'\b(plc|p\.l\.c)\b', s, re.I):
        return "AU_PTY_LTD"

    return None


def extract_legal_form_orig(name: str) -> str | None:
    if not name:
        return None

    matches = list(_LEGAL_DETECT_RE.finditer(str(name)))
    if matches:
        matches.sort(key=lambda m: m.end())
        best = matches[-1]
        matched_text = best.group(0).strip()

        if re.match(r'^(private|privat|priv|pri|pvt\.?)$', matched_text, re.I):
            after = name[best.end():].strip()
            if after:
                next_tok = after.split()[0].rstrip(".,;:").lower()
                if next_tok in _LIMITED_TOKENS:
                    matched_text = f"{matched_text} {after.split()[0]}"

        if len(matches) >= 2:
            prev = matches[-2]
            gap = name[prev.end():best.start()].strip()
            if not gap:
                combined = name[prev.start():best.end()].strip()
                if re.search(r'(pvt|pte|pty|priv|private)', prev.group(0), re.I):
                    matched_text = combined

        return matched_text

    tokens = str(name).split()
    if tokens:
        last = tokens[-1].lower().rstrip(".,;:()")
        if last in _ALL_LEGAL_TOKENS and last not in {"l", "p"}:
            second = tokens[-2].lower() if len(tokens) >= 2 else ""
            if second in _ALL_LEGAL_TOKENS:
                return f"{tokens[-2]} {tokens[-1]}"
            return tokens[-1]
        if last in {"l", "p"} and len(tokens) >= 2:
            second = tokens[-2].lower()
            if second in _ALL_LEGAL_TOKENS:
                return f"{tokens[-2]} {tokens[-1]}"

    if tokens:
        raw_last = tokens[-1].rstrip(".,;:()")
        stripped = _strip_fused(raw_last)
        if stripped != raw_last:
            return raw_last[len(stripped):]

    return None


_SLASH_RE        = re.compile(r'/')
_MULTI_AND_RE    = re.compile(r'(?:\band\b|&).*(?:\band\b|&)', re.I)
_LTD_AND_SUFFIX  = re.compile(r'\b(ltd|limited)\b.*\b(and|&|with)\b', re.I)
_ONLY_NUMBERS_RE = re.compile(r'^[\d\s\W]+$')
_SHAREHOLDER_RE  = re.compile(
    r'(?:'
    r'represented\s+through|authoris[e]?d\s+representative|authorized\s+representative|'
    r'through\s+its|company\s+secretary|through\s+its\s+director|'
    r'\(first\s+holder\)|\(second\s+holder\)|\(third\s+holder\)|'
    r'&\s+sh(?:ri|\.?\s)|&\s+smt\b|&\s+mr(?:\s|s\b)|&\s+dr\b'
    r')',
    re.I,
)

_PL_SLASH_RE = re.compile(r'\bP/L\b', re.I)


def remove_invalid_names(df, label: str):
    print(f"[{label}] Hard name filtering started...")
    import time as _time
    import pandas as pd
    t0 = _time.time()

    names = df["Name"].fillna("").astype(str).str.strip()

    m_empty    = (names == "") | (names.str.lower() == "nan")
    names_npl  = names.str.replace(_PL_SLASH_RE, "PL", regex=True)
    m_slash    = names_npl.str.contains(r'/', na=False, regex=False)
    m_multand  = names.str.contains(r'(?:and|&).*(?:and|&)', na=False, regex=True, flags=re.I)
    m_ltdand   = names.str.contains(r'(?:ltd|limited).*(?:and|&|with)', na=False, regex=True, flags=re.I)
    m_nums     = names.str.fullmatch(r'[\d\s\W]+', na=False)
    m_short    = names.str.len() <= 3
    m_sholder  = names.str.contains(
        r'(?:represented\s+through|authoris[e]?d\s+representative|authorized\s+representative|'
        r'through\s+its|company\s+secretary|through\s+its\s+director|'
        r'\(first\s+holder\)|\(second\s+holder\)|\(third\s+holder\)|'
        r'&\s+sh(?:ri|\.?\s)|&\s+smt|&\s+mr(?:\s|s)|&\s+dr)',
        na=False, regex=True, flags=re.I)

    reason = pd.Series("", index=df.index)
    reason[m_sholder] = "ShareholderRecord"
    reason[m_short]   = "TooShort"
    reason[m_nums]    = "OnlyNumbers"
    reason[m_ltdand]  = "LtdBeforeAnd"
    reason[m_multand] = "MultipleAnd"
    reason[m_slash]   = "ContainsSlash"
    reason[m_empty]   = "EmptyName"

    drop_mask  = reason != ""
    clean_df   = df[~drop_mask].copy()
    dropped_df = df[drop_mask].copy()
    dropped_df["DropReason"] = reason[drop_mask]

    print(f"[{label}] Hard filter done in {_time.time()-t0:.2f}s | "
          f"Kept: {len(clean_df):,} | Dropped: {len(dropped_df):,}")
    return clean_df, dropped_df.to_dict("records")
