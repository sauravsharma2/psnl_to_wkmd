# ═══════════════════════ STEP 5: EXPORT — unique: CSV + Excel | multi (ties): CSV ═══════════════════════
# Excel max is 1,048,576 rows/sheet, so the tie rows (Multi_INC_review, CN_Multi_INC_review) go to their own CSV.
import xlsxwriter

MULTI_SHEETS = ["Multi_INC_review", "CN_Multi_INC_review"]
MAX_EXCEL_ROWS = 1_048_000

out_cols = [c for c in EXPORT_COLS if c in unique.columns]
all_out = unique.select(*[F.col(c).cast("string") if c in ("DOI_Matched",) else F.col(c) for c in out_cols])
uniq_out = all_out.filter(~F.col("ResultSheet").isin(MULTI_SHEETS)).orderBy("ResultSheet", "OriginalId")
multi_out = all_out.filter(F.col("ResultSheet").isin(MULTI_SHEETS)).orderBy("ResultSheet", "OriginalId", "MatchedId")

write_one_csv(uniq_out, f"{OUT_DIR}/matches_{RUN_TAG}.csv")
write_one_csv(multi_out, f"{OUT_DIR}/matches_multi_{RUN_TAG}.csv")

counts = {r["ResultSheet"]: r["count"] for r in unique.groupBy("ResultSheet").count().collect()}
n_uniq = sum(v for k, v in counts.items() if k not in MULTI_SHEETS)
n_multi = sum(v for k, v in counts.items() if k in MULTI_SHEETS)
n_multi_ids = multi_out.select("OriginalId").distinct().count()
if n_uniq > MAX_EXCEL_ROWS:
    print(f"[WARN] {n_uniq:,} unique rows, first {MAX_EXCEL_ROWS:,} in Excel (all in CSV)")
uniq_pd = uniq_out.limit(MAX_EXCEL_ROWS).toPandas()

summary = [("Run date (IST)", datetime.now(IST).strftime("%Y-%m-%d %H:%M")),
           ("MatchScore threshold", THRESHOLD),
           ("Direct IDs", direct_ids.count()),
           ("SQL Server info file", sql_csv.split("/")[-1]),
           ("Unique matches (this file)", n_uniq)]
summary += [(f"  {k}", counts[k]) for k in sorted(counts) if k not in MULTI_SHEETS]
summary += [("Multi rows (matches_multi CSV)", n_multi),
            ("  YY IDs with a tie", n_multi_ids)]
summary += [(f"  {k}", counts[k]) for k in MULTI_SHEETS if k in counts]

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
ws0.set_column(0, 0, 32)
ws0.set_column(1, 1, 45)

ws1 = wb.add_worksheet("UniqueMatch")
ws1.write_row(0, 0, list(uniq_pd.columns), hdr)
for i, row in enumerate(uniq_pd.itertuples(index=False, name=None), start=1):
    for j, v in enumerate(row):
        write_cell(ws1, i, j, v)
ws1.set_column(0, len(uniq_pd.columns) - 1, 20)
ws1.freeze_panes(1, 0)

wb.close()
dest = f"{OUT_DIR}/matches_{RUN_TAG}.xlsx"
shutil.copy(tmp, dest)
print(f"Excel written: {dest}  ({n_uniq:,} unique rows)")
print(f"Multi CSV    : {OUT_DIR}/matches_multi_{RUN_TAG}.csv  ({n_multi:,} rows, {n_multi_ids:,} YY IDs)")
