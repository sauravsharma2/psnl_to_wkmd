%pip install xlsxwriter -q


from pyspark.sql import functions as F
import xlsxwriter, shutil, os, math

# ---------------- CONFIG ----------------
OUT_DIR   = "/Volumes/training_sandbox_dev/msof_direct/files"
OBO_PATH  = f"{OUT_DIR}/obo.parquet"
BASE_NAME = "orbis_vs_obo"
MAX_ROWS  = 1_000_000          # per sheet (Excel limit 1,048,576)
TMP_DIR   = "/local_disk0/tmp"
# ----------------------------------------

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
norm = lambda c: F.upper(F.trim(F.col(c).cast("string")))

# 1. OBO parquet (first column = BvD ID)
obo_raw = spark.read.parquet(OBO_PATH)
obo = (obo_raw.select(norm(obo_raw.columns[0]).alias("bvd_id"))
       .where("bvd_id IS NOT NULL AND bvd_id <> ''")
       .distinct())

# 2. ORBIS query
orbis = (spark.sql("""
            SELECT BvD_ID_number
            FROM prd_bronze_orbis.orbis.legal_info
            WHERE Information_provider = 'BvD ownership database'
         """)
         .select(norm("BvD_ID_number").alias("bvd_id"))
         .where("bvd_id IS NOT NULL AND bvd_id <> ''")
         .distinct())

orbis.cache(); obo.cache()
print(f"ORBIS distinct: {orbis.count():,} | OBO distinct: {obo.count():,}")

# 3. Compare
orbis_only = orbis.join(obo, "bvd_id", "left_anti")   # in ORBIS, not in OBO
obo_only   = obo.join(orbis, "bvd_id", "left_anti")   # in OBO, not in ORBIS

o_ids = sorted(orbis_only.toPandas()["bvd_id"].tolist())
b_ids = sorted(obo_only.toPandas()["bvd_id"].tolist())
print(f"orbis_only: {len(o_ids):,} | obo_only: {len(b_ids):,}")

# 4. Write Excel (split into part files if > MAX_ROWS)
os.makedirs(TMP_DIR, exist_ok=True)
n_parts = max(math.ceil(len(o_ids) / MAX_ROWS),
              math.ceil(len(b_ids) / MAX_ROWS), 1)

for p in range(n_parts):
    fname = f"{BASE_NAME}.xlsx" if n_parts == 1 else f"{BASE_NAME}_part{p+1}.xlsx"
    tmp = f"{TMP_DIR}/{fname}"
    s, e = p * MAX_ROWS, (p + 1) * MAX_ROWS

    wb = xlsxwriter.Workbook(tmp, {"constant_memory": True})
    for sheet, ids in [("orbis_only", o_ids), ("obo_only", b_ids)]:
        ws = wb.add_worksheet(sheet)
        ws.write(0, 0, "bvd_id")
        ws.write_column(1, 0, ids[s:e])
    wb.close()

    shutil.copy(tmp, f"{OUT_DIR}/{fname}")
    print(f"Saved {fname}: orbis_only={len(o_ids[s:e]):,} | obo_only={len(b_ids[s:e]):,}")

orbis.unpersist(); obo.unpersist()
