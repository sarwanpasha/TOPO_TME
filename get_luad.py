"""Fetch only LUAD_IMC_CellType/*.mat, LUAD_IMC_MaskTif/*.tif and the clinical xlsx from LungData.zip by range reads."""
import os, sys
from remotezip import RemoteZip
URL = "https://zenodo.org/records/7760826/files/LungData.zip"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw", "luad")
want = sys.argv[1:] or ["LUAD_IMC_CellType/", "LUAD_IMC_MaskTif/", "LUAD Clinical Data.xlsx"]
with RemoteZip(URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}, support_suffix_range=False) as z:
    n = 0
    for i in z.infolist():
        if i.filename.endswith("/") or "__MACOSX" in i.filename or not any(w in i.filename for w in want):
            continue
        dst = os.path.join(OUT, i.filename.split("LungData/", 1)[1].replace("/", os.sep))
        if os.path.exists(dst) and os.path.getsize(dst) == i.file_size:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with z.open(i) as s, open(dst, "wb") as f:
            f.write(s.read())
        n += 1
        if n % 50 == 0:
            print("got", n, flush=True)
print("done", n, flush=True)
