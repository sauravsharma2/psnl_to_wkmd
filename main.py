# Databricks notebook source
# MAGIC %pip install xlsxwriter
# MAGIC %restart_python

# COMMAND ----------

import xlsxwriter, shutil, os

# ---------------- CONFIG ----------------
OUT_DIR  = "/Volumes/training_sandbox_dev/msof_direct/files"
MAX_ROWS = 1_000_000          # rows per Excel file (Excel limit 1,048,576)
TMP_DIR  = "/tmp"
# ----------------------------------------

QUERIES = {
    # BvD ID contains '*'  -> direct_id
    "direct_id": """
        SELECT BvD_ID_number
        FROM prd_bronze_orbis.orbis.legal_info AS legal_info
        WHERE legal_info.Information_provider = 'BvD ownership database'
          AND BvD_ID_number LIKE '%*%'
    """,
    # BvD ID without '*'   -> non_direct
    "non_direct": """
        SELECT BvD_ID_number
        FROM prd_bronze_orbis.orbis.legal_info AS legal_info
        WHERE legal_info.Information_provider = 'BvD ownership database'
          AND BvD_ID_number NOT LIKE '%*%'
    """,
}

# COMMAND ----------

def _clean(v):
    # xlsxwriter handles str/int/float/None; cast anything else (Decimal, date...) to str
    return v if v is None or isinstance(v, (str, int, float)) else str(v)


def export_to_excel(df, base_name):
    """Stream a Spark DF into Excel files of MAX_ROWS each.
    File 1 = <base_name>.xlsx, then <base_name>_part2.xlsx, _part3.xlsx ...
    Returns (total_rows, list_of_files)."""
    cols = df.columns
    files = []
    state = {}

    def open_part(p):
        name = f"{base_name}.xlsx" if p == 1 else f"{base_name}_part{p}.xlsx"
        tmp_path = os.path.join(TMP_DIR, name)
        wb = xlsxwriter.Workbook(tmp_path, {"constant_memory": True})
        ws = wb.add_worksheet("data")
        ws.write_row(0, 0, cols)
        state.update(wb=wb, ws=ws, tmp=tmp_path, name=name)

    def close_part():
        state["wb"].close()
        dest = os.path.join(OUT_DIR, state["name"])
        shutil.copyfile(state["tmp"], dest)   # /tmp -> Volume
        os.remove(state["tmp"])
        files.append(dest)

    part, row_in_part, total = 1, 0, 0
    open_part(part)

    for r in df.toLocalIterator(prefetchPartitions=True):
        if row_in_part == MAX_ROWS:          # current file full -> start next part
            close_part()
            part += 1
            row_in_part = 0
            open_part(part)
        state["ws"].write_row(row_in_part + 1, 0, [_clean(v) for v in r])
        row_in_part += 1
        total += 1

    close_part()
    return total, files

# COMMAND ----------

os.makedirs(OUT_DIR, exist_ok=True)
results = {}

for base_name, sql in QUERIES.items():
    print(f"Running '{base_name}' ...")
    df = spark.sql(sql)
    total, files = export_to_excel(df, base_name)
    results[base_name] = (total, files)
    print(f"  done -> {total:,} rows in {len(files)} file(s)")

# COMMAND ----------

print("=" * 55)
print("SUMMARY")
print("=" * 55)
for base_name, (total, files) in results.items():
    print(f"\n{base_name.upper()}  |  Total count: {total:,}")
    for f in files:
        print(f"   {f}")
print("\nGrand total:", f"{sum(t for t, _ in results.values()):,}")
