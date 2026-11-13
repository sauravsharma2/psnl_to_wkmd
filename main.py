# src/code/delaware_output.py
"""
Second output file: matches_delaware.xlsx

Contents
--------
    Entries                 (unchanged, same as matches.xlsx)
    Matching_Propositions   (unchanged, same as matches.xlsx)
    NoMatch                 (unchanged, same as matches.xlsx)
    Delaware_Single_Match   OriginalIds with exactly ONE proposition whose
                            MatchedId is a Delaware entity, enriched with an
                            OwnershipLink column
    DirectLinked            rows pulled out of Delaware_Single_Match because the
                            OriginalId and MatchedId are literally parent/child
                            of one another, with a LinkType column

Selection rules for Delaware_Single_Match, applied in order:
    0. Exact name only — the OriginalId and MatchedId names must match under the
       same normalisation used by the ExactName sheets in matches.xlsx.
    1. OriginalId must have exactly one proposition.
    2. The MatchedId must be Delaware — either State2 == DELAWARE or Street2
       is a known registered-agent address.
    3. Drop every row for a MatchedId that is claimed by two or more different
       OriginalIds (ambiguous: same name, different cities, one DE target).
    4. Fetch ownership links, split out the DirectLinked rows, then stamp
       OwnershipLink on whatever remains.

Rows are kept whether or not the OriginalId has an address.

This module owns its own logic and does not modify anything in the existing
pipeline; main.py calls build_delaware_output() and passes in the helpers it
needs, so there is no circular import.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable

import pandas as pd

from src.code.name_matching import name_relation, names_match


# The four literal parent/child relationships between an OriginalId ("Direct")
# and its MatchedId ("Good"). Wording kept identical to the sibling project so
# the column reads the same way for reviewers.
def _direct_link_type(orig_id: str, matched_id: str, ownership: dict) -> str | None:
    o_own = ownership.get(orig_id, {})
    m_own = ownership.get(matched_id, {})
    if matched_id in o_own.get("children", set()):
        return "Direct→Good (Parent→Child)"
    if matched_id in o_own.get("parents", set()):
        return "Direct→Good (Child→Parent)"
    if orig_id in m_own.get("children", set()):
        return "Good→Direct (Parent→Child)"
    if orig_id in m_own.get("parents", set()):
        return "Good→Direct (Child→Parent)"
    return None


def _ownership_label(o_own: dict, m_own: dict) -> str:
    labels = []
    if o_own.get("parents", set()) & m_own.get("parents", set()):
        labels.append("Same Parent")
    if o_own.get("children", set()) & m_own.get("children", set()):
        labels.append("Same Child")
    if o_own.get("uos", set()) & m_own.get("uos", set()):
        labels.append("Same UO")
    return " | ".join(labels) if labels else "No shared ownership"


def build_ownership_map(link_rows: Iterable, uo_rows: Iterable) -> dict:
    """
    Build {id: {"parents": set, "children": set, "uos": set}}.

    link_rows : iterable of (ParentId, ChildId) from Links
    uo_rows   : iterable of (ChildId, ParentId) from UO_links
    """
    ownership = defaultdict(lambda: {"parents": set(), "children": set(), "uos": set()})

    for parent, child in link_rows:
        if parent is None or child is None:
            continue
        parent, child = str(parent), str(child)
        ownership[parent]["children"].add(child)
        ownership[child]["parents"].add(parent)

    for child, uo in uo_rows:
        if child is None or uo is None:
            continue
        ownership[str(child)]["uos"].add(str(uo))

    return ownership


def _is_delaware_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper().eq("DELAWARE")


def select_delaware_single_matches(
    df: pd.DataFrame,
    *,
    agent_addresses: list,
    normalize_text: Callable,
    normalize_name: Callable = None,
) -> pd.DataFrame:
    """
    Apply the selection rules above and return the surviving rows.

    df is expected to be the full enriched match table (one row per proposition,
    carrying State2 / Street2 / MatchedId / OriginalId).

    normalize_name is the same helper the ExactName tier sheets use, so
    "exact name" means the same thing in both output files.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    out["OriginalId"] = out["OriginalId"].astype(str)
    out["MatchedId"] = out["MatchedId"].astype(str)

    # Rule 0 — name match, via the shared structured comparison.
    if normalize_name is not None and "MatchedName" in out.columns:
        norm_orig = out["Name"].fillna("").astype(str).apply(normalize_name)
        norm_match = out["MatchedName"].fillna("").astype(str).apply(normalize_name)
        results = [names_match(a, b) for a, b in zip(norm_orig, norm_match)]
        out["NameScore"] = [r[1] for r in results]
        before = len(out)
        out = out[[r[0] for r in results]]
        print(f"[DELAWARE] name filter: {before} -> {len(out)} rows")
        if out.empty:
            return out

    # Rule 1 — exactly one proposition per OriginalId.
    counts = out["OriginalId"].value_counts()
    single_ids = counts[counts == 1].index
    out = out[out["OriginalId"].isin(single_ids)]
    if out.empty:
        return out

    # Rule 2 — MatchedId is Delaware: by state, or by registered-agent street.
    missing = [c for c in ("State2", "Street2") if c not in out.columns]
    if missing:
        print(
            f"[DELAWARE][WARN] missing column(s) {missing} — the Delaware test cannot "
            f"run and no rows will be selected. Pass an enriched table "
            f"(after add_parsed_addresses)."
        )

    is_de_state = (
        _is_delaware_series(out["State2"])
        if "State2" in out.columns
        else pd.Series(False, index=out.index)
    )

    if agent_addresses and "Street2" in out.columns:
        def _is_agent(value):
            norm = normalize_text(value) if pd.notna(value) else ""
            if not norm:
                return False
            return any(a in norm for a in agent_addresses)

        is_de_agent = out["Street2"].apply(_is_agent)
    else:
        is_de_agent = pd.Series(False, index=out.index)

    out = out[is_de_state | is_de_agent]
    if out.empty:
        return out

    # Rule 3 — a MatchedId claimed by several OriginalIds is ambiguous; drop
    # the whole cluster rather than guess.
    m_counts = out.groupby("MatchedId")["OriginalId"].nunique()
    ambiguous = set(m_counts[m_counts >= 2].index)
    if ambiguous:
        print(f"[DELAWARE] dropping {len(ambiguous)} MatchedId(s) claimed by 2+ OriginalIds")
        out = out[~out["MatchedId"].isin(ambiguous)]

    return out.copy()


def select_multiple_directid(
    df: pd.DataFrame,
    *,
    excluded_ids: set,
    agent_addresses: list,
    normalize_text: Callable,
    normalize_name: Callable,
    relations: tuple = ("same_legal",),
    delaware: str = "require",
    last_transfer_cutoff=None,
    label: str = "PHASE3",
) -> pd.DataFrame:
    """
    Several OriginalIds proposing one shared MatchedId.

    Phase 1 drops these clusters because a single entity cannot be the
    counterpart of thirty operating records. This surfaces them grouped around
    the target they all claim, with no constraint on the OriginalId's address.

    relations selects which legal-form relationships qualify (see
    name_relation); delaware is "require", "exclude" or "ignore".

    Rules, in order:
        1. OriginalIds handled by an earlier phase are excluded.
        2. The name relation must be one of `relations`.
        3. The Delaware condition, per `delaware`.
        4. LastTransferDate on the matched record must be recent.
        5. The MatchedId must be claimed by two or more distinct OriginalIds.
    """
    empty = pd.DataFrame(columns=list(df.columns) if df is not None else [])
    if df is None or df.empty:
        return empty

    out = df.copy()
    out["OriginalId"] = out["OriginalId"].astype(str)
    out["MatchedId"] = out["MatchedId"].astype(str)

    # Rule 1 — skip anything already resolved by an earlier phase.
    if excluded_ids:
        before = len(out)
        out = out[~out["OriginalId"].isin(excluded_ids)]
        print(f"[{label}] excluded {before - len(out):,} rows already handled earlier")
    if out.empty:
        return out

    # Rule 2 — the name must relate in one of the requested ways. "same_legal"
    # is the strict case; the others cover a differing or missing legal form.
    if "MatchedName" in out.columns:
        norm_orig = out["Name"].fillna("").astype(str).apply(normalize_name)
        norm_match = out["MatchedName"].fillna("").astype(str).apply(normalize_name)
        results = [name_relation(a, b) for a, b in zip(norm_orig, norm_match)]
        out["NameScore"] = [r[1] for r in results]
        out["NameRelation"] = [r[0] for r in results]
        out = out[[r[0] in relations for r in results]]
        print(f"[{label}] after name filter {relations}: {len(out):,} rows")
    if out.empty:
        return out

    # Rule 3 — MatchedId is Delaware (same test as Delaware_Single_Match).
    is_de_state = (
        _is_delaware_series(out["State2"])
        if "State2" in out.columns
        else pd.Series(False, index=out.index)
    )
    if agent_addresses and "Street2" in out.columns:
        def _is_agent(value):
            norm = normalize_text(value) if pd.notna(value) else ""
            if not norm:
                return False
            return any(a in norm for a in agent_addresses)
        is_de_agent = out["Street2"].apply(_is_agent)
    else:
        is_de_agent = pd.Series(False, index=out.index)

    if delaware == "require":
        out = out[is_de_state | is_de_agent]
    elif delaware == "exclude":
        out = out[~(is_de_state | is_de_agent)]
    if delaware != "ignore":
        print(f"[{label}] after Delaware test ({delaware}): {len(out):,} rows")
    if out.empty:
        return out

    # Rule 4 — recency on the matched record. None switches the rule off.
    if last_transfer_cutoff is not None and "LastTransferDate" in out.columns:
        val = pd.to_numeric(out["LastTransferDate"], errors="coerce")
        out = out[val >= last_transfer_cutoff]
        print(f"[{label}] after LastTransferDate >= {last_transfer_cutoff}: {len(out):,} rows")
    if out.empty:
        return out

    # Rule 5 — the shared-target condition. A MatchedId claimed by only one
    # OriginalId is a phase 1 case, not a phase 3 one.
    counts = out.groupby("MatchedId")["OriginalId"].nunique()
    shared = counts[counts >= 2].index
    out = out[out["MatchedId"].isin(shared)]
    print(
        f"[{label}] after shared-target test: {len(out):,} rows "
        f"across {out['MatchedId'].nunique():,} matched entities"
    )

    return out.sort_values(["MatchedId", "OriginalId"], kind="stable").reset_index(drop=True)


def stamp_ownership(df: pd.DataFrame, ownership: dict) -> pd.DataFrame:
    """Set OwnershipLink on a frame of OriginalId/MatchedId pairs."""
    if df is None or df.empty:
        if df is not None and "OwnershipLink" not in df.columns:
            df["OwnershipLink"] = pd.Series(dtype=object)
        return df

    labels = []
    for a, b in zip(df["OriginalId"].astype(str), df["MatchedId"].astype(str)):
        o_own = ownership.get(a.rstrip("!"))
        m_own = ownership.get(b.rstrip("!"))
        labels.append(
            _ownership_label(o_own, m_own) if (o_own and m_own) else "No ownership data"
        )

    out = df.copy()
    out["OwnershipLink"] = labels
    cols = [c for c in out.columns if c != "OwnershipLink"] + ["OwnershipLink"]
    return out[cols]


def split_direct_linked(df: pd.DataFrame, ownership: dict) -> tuple:
    """
    Rule 4 — split the selection into (remaining, direct_linked).

    A row whose OriginalId and MatchedId are literally parent/child of one
    another moves to DirectLinked with a LinkType column and is removed from the
    main sheet. Everything else stays and gets an OwnershipLink column.
    """
    if df.empty:
        empty = df.copy()
        empty["OwnershipLink"] = pd.Series(dtype=object)
        dl = df.copy()
        dl["LinkType"] = pd.Series(dtype=object)
        return empty, dl

    link_types = []
    ownership_labels = []

    for orig_id, matched_id in zip(df["OriginalId"].astype(str), df["MatchedId"].astype(str)):
        # IDs may carry a trailing '!' for ex-IDs; links are stored unsuffixed.
        o_key, m_key = orig_id.rstrip("!"), matched_id.rstrip("!")
        lt = _direct_link_type(o_key, m_key, ownership)
        link_types.append(lt)

        if lt is None:
            o_own = ownership.get(o_key)
            m_own = ownership.get(m_key)
            if o_own and m_own:
                ownership_labels.append(_ownership_label(o_own, m_own))
            else:
                ownership_labels.append("No ownership data")
        else:
            ownership_labels.append("")

    work = df.copy()
    work["_LinkType"] = link_types
    work["_OwnershipLink"] = ownership_labels

    is_direct = work["_LinkType"].notna()

    direct_linked = work[is_direct].copy()
    direct_linked["LinkType"] = direct_linked["_LinkType"]
    direct_linked = direct_linked.drop(columns=["_LinkType", "_OwnershipLink"])

    remaining = work[~is_direct].copy()
    remaining["OwnershipLink"] = remaining["_OwnershipLink"]
    remaining = remaining.drop(columns=["_LinkType", "_OwnershipLink"])

    return remaining, direct_linked


# Kept for callers that predate the generalised signature.
def select_multiple_directid_delaware(df, **kwargs):
    kwargs.setdefault("relations", ("same_legal",))
    kwargs.setdefault("delaware", "require")
    return select_multiple_directid(df, **kwargs)
