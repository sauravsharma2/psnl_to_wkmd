# src/code/directid_grouping.py
"""
Same-name OriginalId grouping.

The address sheets (steps 2) and the address-less sheet (step 4) sit at
different points in the cascade, so main.py calls this module twice and selects
the bucket it wants with the `buckets` argument.

Phase 1 deliberately drops groups where several different OriginalIds all point
at one Delaware MatchedId: with the same name spread across several cities there
is no way to tell which OriginalId is the real one. Those OriginalIds are still
plausible duplicates *of each other* though, so this module pairs them up
directly instead of via the Delaware entity.

How the grouping works
----------------------
Every OriginalId in Entries is considered (a Delaware match is not required).
Ids are grouped on exact normalised name + exact city, then each group is split
by whether the state is populated and the two halves are handled independently:

    directid_Same_City_State : every id in the subgroup has a state, and they
                               all agree on it. A city name shared across two
                               different states never produces a pairing.
    directid_Same_city       : every id in the subgroup has NO state at all.
                               Same name, same city, nothing else known.

An id whose state status differs from the rest of its group ends up alone in its
half and is dropped — a stateful id never pairs with a stateless one, since
"Birmingham" alone could be AL or MI.

City names are compared exactly after normalisation, with one exception: a small
alias table folds known variants together (ST PAUL / SAINT PAUL, MAIMI / MIAMI).
No fuzzy matching is used, so unlisted typos simply will not group. That is the
accepted trade-off: a false merge between two genuinely different companies is
worse than a missed one.

Within each subgroup the anchor is the id that everything else points at. A
plain id (no '*' at the 4th character) always wins over a direct-entry '*' id,
so the '*' records fold into the cleaner one; NrLinks only decides between ids
of the same kind. Every other id in the subgroup is emitted as one proposition
pointing at the anchor. A subgroup of 11 therefore produces 10 rows, all naming
the same anchor. Subgroups of 2 are kept (anchor + 1 proposition); a subgroup of
1 produces nothing.

City matching is exact after normalisation, by design — no fuzzy matching, so a
typo like "MAIMI" simply will not group. That is the accepted trade-off: a false
merge between two genuinely different companies is worse than a missed one.
"""

from __future__ import annotations

import re
from typing import Callable

import pandas as pd

from src.code.name_matching import (
    NAME_FUZZY_THRESHOLD, _threshold_for, cluster_by_similarity, cluster_names,
    name_score, split_name,
)

try:
    import jellyfish
    _HAS_JELLYFISH = True
except ImportError:  # pragma: no cover
    _HAS_JELLYFISH = False


# Alternative names, not typos — no similarity measure catches these because
# the strings are genuinely different words. Typos go through the fuzzy pass.
CITY_ALIASES = {
    "st paul": "saint paul",
    "st louis": "saint louis",
    "st petersburg": "saint petersburg",
    "st cloud": "saint cloud",
    "st charles": "saint charles",
    "st joseph": "saint joseph",
    "st augustine": "saint augustine",
    "st george": "saint george",
    "new york city": "new york",
    "nyc": "new york",
}

# Jaro-Winkler cutoff for treating two city names as the same city. 94 sits in
# the gap between observed misspellings (>=94) and genuinely different cities
# (<=93.8). Jaro-Winkler rather than a plain ratio: it survives transposition.
CITY_FUZZY_THRESHOLD = 94.0


def _city_cluster_keys(city_keys: list, counts: dict = None) -> dict:
    """
    Fold near-identical city names onto one canonical key.

    counts maps each key to its frequency; the most common spelling becomes
    canonical, so typos fold into the correct form.

    Returns {city_key: canonical_key}. Without jellyfish every key maps to
    itself, i.e. the old exact-match behaviour.
    """
    if not _HAS_JELLYFISH or len(city_keys) < 2:
        return {k: k for k in city_keys}

    items = [(k, k) for k in city_keys if k]
    return cluster_by_similarity(
        items,
        lambda a, b: _city_score(a, b) >= CITY_FUZZY_THRESHOLD,
        counts or {},
    )


def _city_score(a: str, b: str) -> float:
    if a == b:
        return 100.0
    try:
        return jellyfish.jaro_winkler_similarity(a, b) * 100
    except Exception:
        return 0.0


OUTPUT_COLUMNS = [
    "OriginalId", "Name", "NameNormalized",
    "MatchedId", "MatchedName",
    "NameScore",
    "Street1", "City1", "State1", "PostalCode1",
    "Street2", "City2", "State2", "PostalCode2",
    "NrLinks 1", "NrLinks 2",
    "GroupSize", "OwnershipLink",
]


def _blank(value) -> bool:
    """True when a state/city value carries no usable content."""
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def _simple_key(value: str) -> str:
    """Lower case, strip punctuation, collapse spaces. No abbreviation expansion."""
    s = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    return re.sub(r"\s+", " ", s).strip()


def _norm_key(value, normalize_text: Callable) -> str:
    """
    Normalised key for exact city comparison.

    Aliases are applied before normalize_text, which expands address
    abbreviations and would turn "ST PAUL" into "street paul".
    """
    if pd.isna(value):
        return ""
    aliased = CITY_ALIASES.get(_simple_key(value), value)
    return normalize_text(str(aliased))


def build_directid_groups(
    df: pd.DataFrame,
    *,
    normalize_name: Callable,
    normalize_text: Callable,
    excluded_ids: set = None,
    buckets: tuple = ("address", "no_address"),
) -> tuple:
    """
    Build the direct-id grouping sheets.

    df must be one row per OriginalId, carrying the parsed address columns
    Street1 / City1 / State1 / PostalCode1 plus "NrLinks 1".

    excluded_ids drops those OriginalIds before grouping. buckets selects which
    halves to emit — "address" for the city/state sheets, "no_address" for the
    address-less one. The two are built in separate calls because they sit at
    different points in the cascade.

    Returns (same_city_state_df, same_city_df, no_address_df); a bucket that was
    not requested comes back empty.
    """
    empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
    if df is None or df.empty:
        return empty.copy(), empty.copy(), empty.copy()

    work = df.copy()
    work["OriginalId"] = work["OriginalId"].astype(str)

    if excluded_ids:
        before = len(work)
        work = work[~work["OriginalId"].isin(set(excluded_ids))]
        print(f"[GROUPING] {before - len(work):,} ids excluded (claimed earlier)")
        if work.empty:
            return empty.copy(), empty.copy(), empty.copy()

    # One row per OriginalId — the input may carry several propositions per id.
    work = work.drop_duplicates(subset=["OriginalId"], keep="first")

    work["_name_key"] = work["Name"].fillna("").astype(str).apply(normalize_name)
    work["_city_key"] = work["City1"].apply(lambda v: _norm_key(v, normalize_text))
    work["_state_key"] = work["State1"].apply(lambda v: _norm_key(v, normalize_text))
    work["_state_null"] = work["State1"].apply(_blank)
    work["_no_address"] = (
        work["Street1"].apply(_blank)
        & work["City1"].apply(_blank)
        & work["State1"].apply(_blank)
        & work["PostalCode1"].apply(_blank)
    )

    work = work[work["_name_key"] != ""]
    if work.empty:
        return empty.copy(), empty.copy(), empty.copy()

    # Records with no address at all are grouped on the name alone, in their
    # own bucket; every id in such a group must be equally address-less.
    no_addr = work[work["_no_address"]].copy()
    work = work[~work["_no_address"] & (work["_city_key"] != "")]

    # Name clustering spans both buckets so they agree on canonical spellings.
    _all_names = pd.concat([work["_name_key"], no_addr["_name_key"]])
    _name_freq = _all_names.value_counts().to_dict()
    _name_canon = cluster_names(sorted(_all_names.unique()), counts=_name_freq)
    _name_folded = sum(1 for k, v in _name_canon.items() if k != v)
    if _name_folded:
        print(f"[GROUPING] {_name_folded} name spelling(s) folded onto a canonical form")
    for frame in (work, no_addr):
        frame["_name_orig"] = frame["_name_key"]
        frame["_name_key"] = frame["_name_key"].map(lambda k: _name_canon.get(k, k))

    # Fold misspellings onto one canonical city key before grouping.
    _freq = work["_city_key"].value_counts().to_dict()
    _canon = _city_cluster_keys(sorted(work["_city_key"].unique()), counts=_freq)
    _folded = sum(1 for k, v in _canon.items() if k != v)
    if _folded:
        print(f"[GROUPING] {_folded} city spelling(s) folded onto a canonical form")
    work["_city_key"] = work["_city_key"].map(lambda k: _canon.get(k, k))

    # '*' at the 4th character marks a direct entry. The plain id anchors the
    # group and outranks NrLinks; within one class NrLinks decides.
    for frame in (work, no_addr):
        frame["_links"] = pd.to_numeric(frame.get("NrLinks 1"), errors="coerce").fillna(-1)
        frame["_has_star"] = frame["OriginalId"].str[3:4].eq("*")

    rows_state, rows_city, rows_no_addr = [], [], []

    def _emit(subgroup, target):
        """Pick an anchor for a subgroup and emit one row per other member."""
        if len(subgroup) < 2:
            return
        # Anchor: plain id, then highest NrLinks, then id for a stable result.
        subgroup = subgroup.sort_values(
            ["_has_star", "_links", "OriginalId"],
            ascending=[True, False, True],
        )
        anchor = subgroup.iloc[0]
        anchor_words, _m, _l = split_name(anchor.get("_name_orig", ""))

        # Clustering picked its own canonical name, which need not be the id that
        # anchors the group here. Re-check each member against the anchor it will
        # actually be shown against, so no row is published below the threshold.
        members = []
        for _, row in subgroup.iloc[1:].iterrows():
            row_words, _m2, _l2 = split_name(row.get("_name_orig", ""))
            if not row_words and not anchor_words:
                score = 100.0
            else:
                score = name_score(row_words, anchor_words)
            if score >= _threshold_for(row_words, anchor_words, NAME_FUZZY_THRESHOLD):
                members.append((row, round(score, 1)))
        if not members:
            return

        for row, score in members:
            target.append({
                "OriginalId": row["OriginalId"],
                "Name": row.get("Name"),
                "NameNormalized": row.get("NameNormalized"),
                "MatchedId": anchor["OriginalId"],
                "MatchedName": anchor.get("Name"),
                "NameScore": score,
                "Street1": row.get("Street1"),
                "City1": row.get("City1"),
                "State1": row.get("State1"),
                "PostalCode1": row.get("PostalCode1"),
                "Street2": anchor.get("Street1"),
                "City2": anchor.get("City1"),
                "State2": anchor.get("State1"),
                "PostalCode2": anchor.get("PostalCode1"),
                "NrLinks 1": row.get("NrLinks 1"),
                "NrLinks 2": anchor.get("NrLinks 1"),
                "GroupSize": len(members) + 1,
                "OwnershipLink": "",
            })

    for (_name_key, _city_key), group in (
        work.groupby(["_name_key", "_city_key"], sort=False)
        if "address" in buckets else []
    ):
        if len(group) < 2:
            continue

        # Split on whether the state is populated; the halves never mix, and an
        # id left alone in its half is dropped.
        with_state = group[~group["_state_null"]]
        without_state = group[group["_state_null"]]

        # Each distinct state forms its own subgroup, so a city name shared
        # across two states never pairs.
        for _state_key, state_group in with_state.groupby("_state_key", sort=False):
            _emit(state_group, rows_state)

        # No state on either side: same name and city is all that is known.
        _emit(without_state, rows_city)

    # No address at all on any side: the name is the only evidence.
    if "no_address" in buckets:
        for _name_key, group in no_addr.groupby("_name_key", sort=False):
            _emit(group, rows_no_addr)

    def _finish(rows):
        out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS) if rows else empty.copy()
        if not out.empty:
            out.sort_values(["MatchedId", "OriginalId"], inplace=True, kind="stable")
            out.reset_index(drop=True, inplace=True)
        return out

    return _finish(rows_state), _finish(rows_city), _finish(rows_no_addr)


def build_direct_vs_good(
    df: pd.DataFrame,
    *,
    normalize_text: Callable,
    excluded_ids: set = None,
    last_transfer_cutoff: int = 20260401,
) -> tuple:
    """
    Direct-vs-Good: a proposition kept when both sides sit in the same place.

    df is the enriched proposition table — OriginalId from Entries, MatchedId
    from the company pool, with both sides' parsed address columns.

    The address test is the one the directid sheets use: cities are compared on
    the normalised, alias-folded, fuzzy-clustered key, and the state decides the
    bucket. No state is privileged — the two sides simply have to agree.

        same_city_state : both sides carry a state and it is the same
        same_city       : neither side carries a state
        (dropped)       : states differ, or only one side has one

    An OriginalId with more than one qualifying MatchedId is ambiguous and is
    dropped outright. Several OriginalIds may share one MatchedId.

    Returns (same_city_state_df, same_city_df).
    """
    empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
    if df is None or df.empty:
        return empty.copy(), empty.copy()

    work = df.copy()
    work["OriginalId"] = work["OriginalId"].astype(str)
    work["MatchedId"] = work["MatchedId"].astype(str)

    if excluded_ids:
        before = len(work)
        work = work[~work["OriginalId"].isin(set(excluded_ids))]
        print(f"[DVG] {before - len(work):,} rows excluded (claimed earlier)")

    if "LastTransferDate" in work.columns:
        val = pd.to_numeric(work["LastTransferDate"], errors="coerce")
        work = work[val >= last_transfer_cutoff]
        print(f"[DVG] after LastTransferDate >= {last_transfer_cutoff}: {len(work):,} rows")

    for col in ("City1", "City2", "State1", "State2"):
        if col not in work.columns:
            work[col] = ""
    if work.empty:
        return empty.copy(), empty.copy()

    work["_city1"] = work["City1"].apply(lambda v: _norm_key(v, normalize_text))
    work["_city2"] = work["City2"].apply(lambda v: _norm_key(v, normalize_text))
    work["_state1"] = work["State1"].apply(lambda v: _norm_key(v, normalize_text))
    work["_state2"] = work["State2"].apply(lambda v: _norm_key(v, normalize_text))

    # Fold misspellings across both sides at once so "DEVNER"/"DENVER" meet.
    _keys = pd.concat([work["_city1"], work["_city2"]])
    _freq = _keys.value_counts().to_dict()
    _canon = _city_cluster_keys(sorted(k for k in _keys.unique() if k), counts=_freq)
    _folded = sum(1 for k, v in _canon.items() if k != v)
    if _folded:
        print(f"[DVG] {_folded} city spelling(s) folded onto a canonical form")
    work["_city1"] = work["_city1"].map(lambda k: _canon.get(k, k))
    work["_city2"] = work["_city2"].map(lambda k: _canon.get(k, k))

    work = work[(work["_city1"] != "") & (work["_city1"] == work["_city2"])]
    if work.empty:
        return empty.copy(), empty.copy()

    both_state = (work["_state1"] != "") & (work["_state2"] != "")
    no_state = (work["_state1"] == "") & (work["_state2"] == "")
    state_rows = work[both_state & (work["_state1"] == work["_state2"])]
    city_rows = work[no_state]

    def _finish(sub):
        if sub.empty:
            return empty.copy()
        # One MatchedId per OriginalId; an id proposing several is ambiguous.
        counts = sub.groupby("OriginalId")["MatchedId"].nunique()
        keep = set(counts[counts == 1].index)
        dropped = len(counts) - len(keep)
        if dropped:
            print(f"[DVG] {dropped:,} OriginalId(s) dropped — several qualifying MatchedIds")
        sub = sub[sub["OriginalId"].isin(keep)]
        if sub.empty:
            return empty.copy()

        sizes = sub.groupby("MatchedId")["OriginalId"].nunique()
        out = pd.DataFrame({
            "OriginalId": sub["OriginalId"],
            "Name": sub.get("Name"),
            "NameNormalized": sub.get("NameNormalized"),
            "MatchedId": sub["MatchedId"],
            "MatchedName": sub.get("MatchedName"),
            "NameScore": sub.get("NameScore"),
            "Street1": sub.get("Street1"), "City1": sub.get("City1"),
            "State1": sub.get("State1"), "PostalCode1": sub.get("PostalCode1"),
            "Street2": sub.get("Street2"), "City2": sub.get("City2"),
            "State2": sub.get("State2"), "PostalCode2": sub.get("PostalCode2"),
            "NrLinks 1": sub.get("NrLinks 1"), "NrLinks 2": sub.get("NrLinks 2"),
            "GroupSize": sub["MatchedId"].map(sizes),
            "OwnershipLink": "",
        })
        return out.sort_values(
            ["MatchedId", "OriginalId"], kind="stable"
        ).reset_index(drop=True)

    return _finish(state_rows), _finish(city_rows)
