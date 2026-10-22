"""
Databrick_YY_SQL_Check — checks Databricks YY UniqueMatch results against SQL Server (bvdaffils).

Input : matches_YY_<date>.xlsx (sheet UniqueMatch)      unique winners, from Databricks
        matches_multi_YY_<date>.csv                          tie rows (Multi_INC_review / CN_Multi_INC_review)
        Pass the xlsx path, or leave it out to use the newest matches_YY_*.xlsx next to this script;
        the multi CSV with the same date is picked up automatically (skipped with a warning if missing).
Output: output/sql_check_YY_<input date>.xlsx        unique winners: UniqueMatch (all) + INC / No_INC / CN_* sheets
        output/sql_check_multi_YY_<input date>.xlsx  ties: Multi_INC_review / CN_Multi_INC_review
                                                     (a sheet over 1,048,000 rows is split: _2, _3 ...)
        Every UniqueMatch column + these new columns:
          OLD_bvdlien, NEW_bvdlien  BvD ID -> BvDLiens ID (bvdaffils.dbo.GetBvDLiensID)
          InSQL                     yes / no / no_liens_id  (is the MatchedId in Companies?)
          Foreign                   foreign / not_foreign   (Companies.[Foreign] in 1, 2)
          SQL_shared_link           rows with 'no ownership link' only: shared links found in SQL Server,
                                    e.g. shared_subsidiary_obo|shared_UO_obo

SQL lives in queries/ (01 … 05). Run:  python main.py [matches_YY_20261006.xlsx]
"""
import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyodbc

from db_connexion import connect_db
from run_finalize import finalize_run, _now_utc_iso

ROOT = Path(__file__).resolve().parent
QUERY_DIR = ROOT / "queries"
OUTDIR = ROOT / "output"
PROJECT_NAME = "Databrick_YY_SQL_Check"

SHEET = "UniqueMatch"
NO_LINK = "no ownership link"
BATCH = 50_000                       # rows per insert into the temp tables
PAIRS_INDEX = "create clustered index ix_pairs on #pairs (old_l, new_l)"


def q(name: str) -> str:
    """Read one SQL file from queries/."""
    return "set nocount on;\n" + (QUERY_DIR / name).read_text(encoding="utf-8")


SHEET_ORDER = ["INC", "No_INC", "CN_INC_Same_Native", "CN_No_INC_Same_Native",
               "CN_INC_No_Native", "CN_No_INC_No_Native"]
MULTI_SHEETS = ["Multi_INC_review", "CN_Multi_INC_review"]


MAX_SHEET_ROWS = 1_048_000          # Excel limit is 1,048,576 rows per sheet


def _to_sheets(xw, df: pd.DataFrame, name: str) -> None:
    """Write df to one sheet, or to name, name_2, name_3 ... when it is over the Excel row limit."""
    if df.empty:
        df.to_excel(xw, sheet_name=name[:31], index=False)
        return
    for k, i in enumerate(range(0, len(df), MAX_SHEET_ROWS), start=1):
        sheet = name if k == 1 else f"{name[:28]}_{k}"
        df.iloc[i:i + MAX_SHEET_ROWS].to_excel(xw, sheet_name=sheet[:31], index=False)


def write_sheets(df: pd.DataFrame, out: Path, order: list[str], all_name: str | None = None) -> None:
    """Optional first sheet with every row; then one sheet per name in `order` (empty sheets still written)."""
    t = time.time()
    try:
        import xlsxwriter  # noqa: F401  (much faster than openpyxl for big sheets)
        engine = "xlsxwriter"
    except ImportError:
        engine = "openpyxl"
    with pd.ExcelWriter(out, engine=engine) as xw:
        if all_name:
            _to_sheets(xw, df, all_name)
        for name in order:
            _to_sheets(xw, df[df["ResultSheet"] == name], name)
    print(f"[xlsx] {out.name}: {len(df):,} rows written in {time.time() - t:.0f}s")


def find_input(arg: str | None) -> Path:
    if arg:
        for p in (Path(arg), ROOT / arg):
            if p.exists():
                return p
        sys.exit(f"Input file not found: {arg}")
    files = sorted(ROOT.glob("matches_YY_*.xlsx"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit("No matches_YY_*.xlsx next to main.py — pass the file path.")
    return files[-1]


def _check_cols(df: pd.DataFrame, src: str) -> pd.DataFrame:
    for col in ("OriginalId", "MatchedId", "ownership_link", "ResultSheet"):
        if col not in df.columns:
            sys.exit(f"Column '{col}' not in {src}. Found: {list(df.columns)}")
    return df


def load(path: Path) -> pd.DataFrame:
    t = time.time()
    try:  # calamine reads big xlsx ~10x faster; falls back to openpyxl if not installed
        df = pd.read_excel(path, sheet_name=SHEET, dtype=str, engine="calamine")
    except Exception:
        df = pd.read_excel(path, sheet_name=SHEET, dtype=str)
    print(f"[read] {len(df):,} rows from {path.name} / {SHEET} in {time.time() - t:.0f}s")
    return _check_cols(df, f"{path.name} / {SHEET}")


def load_multi(xlsx: Path) -> pd.DataFrame | None:
    """matches_multi_YY_<date>.csv next to the unique xlsx (same date)."""
    path = xlsx.with_name(xlsx.stem.replace("matches_", "matches_multi_", 1) + ".csv")
    if not path.exists():
        print(f"[warn] {path.name} not found — multi file skipped")
        return None
    t = time.time()
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8")
    print(f"[read] {len(df):,} rows from {path.name} in {time.time() - t:.0f}s")
    return _check_cols(df, path.name)


def insert(cur, table: str, cols: str, rows: list[tuple]) -> None:
    if not rows:
        return
    marks = ", ".join("?" * len(rows[0]))
    for i in range(0, len(rows), BATCH):
        cur.executemany(f"insert into {table} ({cols}) values ({marks})", rows[i:i + BATCH])


@connect_db
def run_checks(conn: pyodbc.Connection, ids: list[str], pairs: list[tuple[str, str, bool]]):
    """ids: distinct BvD IDs. pairs: (old_bvdid, new_bvdid, no_link)."""
    cur = conn.cursor()
    cur.fast_executemany = True

    t = time.time()
    cur.execute(q("01_create_ids.sql"))
    insert(cur, "#ids", "bvdid", [(i,) for i in ids])
    cur.execute(q("02_liens_map.sql"))
    rows = cur.execute(q("03_companies_lookup.sql")).fetchall()
    ids_info = {r.bvdid: (r.liens, r.in_sql, r.fgn) for r in rows}
    print(f"[sql] 01-03 {len(ids_info):,} IDs converted + looked up ({time.time() - t:.0f}s)")

    t = time.time()
    liens = lambda v: ids_info.get(v, (None,))[0]
    pair_rows = {}
    for o, n, nl in pairs:
        lo, ln = liens(o), liens(n)
        if lo and ln:
            pair_rows[(lo, ln)] = max(pair_rows.get((lo, ln), 0), int(nl))
    cur.execute(q("04_create_pairs.sql"))
    insert(cur, "#pairs", "old_l, new_l, no_link", [(a, b, f) for (a, b), f in pair_rows.items()])
    cur.execute(PAIRS_INDEX)
    print(f"[sql] 04 {len(pair_rows):,} pairs loaded ({time.time() - t:.0f}s)")

    t = time.time()
    shared = {}
    for r in cur.execute(q("05_shared_links.sql")).fetchall():
        shared.setdefault((r.old_l, r.new_l), set()).add(r.link_type)
    print(f"[sql] 05 shared links: {len(shared):,} pairs ({time.time() - t:.0f}s)")

    cur.close()
    return ids_info, shared


def main() -> int:
    started = _now_utc_iso()
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", nargs="?", help="matches_YY_<date>.xlsx (default: newest next to main.py)")
    args = ap.parse_args()

    path = find_input(args.xlsx)
    uniq = load(path)
    multi = load_multi(path)
    df = pd.concat([uniq, multi], ignore_index=True) if multi is not None else uniq
    is_multi = df["ResultSheet"].isin(MULTI_SHEETS)

    clean = lambda s: s.fillna("").str.strip().str.replace("!", "", regex=False)  # same as MatchingOK
    old_ids, new_ids = clean(df["OriginalId"]), clean(df["MatchedId"])
    no_link = df["ownership_link"].fillna("").str.strip().eq(NO_LINK)

    ids = sorted({v for v in pd.concat([old_ids, new_ids]) if v})
    pairs = [(o, n, nl) for o, n, nl in zip(old_ids, new_ids, no_link) if o and n]
    print(f"[ids] {len(ids):,} distinct IDs, {len(pairs):,} pairs ({int(no_link.sum()):,} without Orbis link)")

    ids_info, shared = run_checks(ids, pairs)

    info = lambda v, k: ids_info.get(v, (None, None, None))[k] if v else None
    df["OLD_bvdlien"] = [info(v, 0) for v in old_ids]
    df["NEW_bvdlien"] = [info(v, 0) for v in new_ids]
    df["InSQL"] = ["no_liens_id" if not info(v, 0) else "yes" if info(v, 1) else "no" for v in new_ids]
    df["Foreign"] = ["" if not info(v, 1) else "foreign" if str(info(v, 2)) in ("1", "2") else "not_foreign"
                     for v in new_ids]
    keys = list(zip(df["OLD_bvdlien"], df["NEW_bvdlien"]))
    df["SQL_shared_link"] = ["|".join(sorted(shared.get(k, ()))) if nl else "" for k, nl in zip(keys, no_link)]

    OUTDIR.mkdir(exist_ok=True)
    tag = re.search(r"(YY_\d{8})", path.name)
    out = OUTDIR / f"sql_check_{tag.group(1) if tag else 'YY'}.xlsx"
    out_multi = OUTDIR / f"sql_check_multi_{tag.group(1) if tag else 'YY'}.xlsx"
    write_sheets(df[~is_multi], out, order=SHEET_ORDER, all_name="UniqueMatch")
    artifacts = [(str(out), "table", "xlsx", True)]
    if multi is not None:
        write_sheets(df[is_multi], out_multi, order=MULTI_SHEETS)
        artifacts.append((str(out_multi), "table", "xlsx", True))

    metrics = {
        "rows": len(df),
        "unique_rows": int((~is_multi).sum()),
        "multi_rows": int(is_multi.sum()),
        "in_sql": df["InSQL"].value_counts().to_dict(),
        "foreign": df["Foreign"].value_counts().to_dict(),
        "no_link_rows": int(no_link.sum()),
        "no_link_rows_with_sql_link": int((df["SQL_shared_link"] != "").sum()),
        "multi_yy_ids": int(df.loc[is_multi, "OriginalId"].nunique()),
        "sheets": df["ResultSheet"].value_counts().to_dict(),
    }
    print(f"\n[done] {out}" + (f"\n       {out_multi}" if multi is not None else ""))
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return finalize_run(project_name=PROJECT_NAME, started_at=started,
                        extra_artifacts=artifacts, metrics=metrics)


if __name__ == "__main__":
    sys.exit(main())
