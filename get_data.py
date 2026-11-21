"""Download only the files we need from the public archives (HTTP range requests into the big zips).
python get_data.py [danenberg|cbioportal|jackson|codex ...]"""
import os, sys, json, shutil, urllib.request, collections, time
import common as C
RAW = C.RAW
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fetch(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return log("have", dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r, open(dst + ".part", "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(dst + ".part", dst); log("got", dst, os.path.getsize(dst))


def from_zip(url, want, outdir):
    from remotezip import RemoteZip
    os.makedirs(outdir, exist_ok=True)
    with RemoteZip(url, headers=UA) as z:
        for info in z.infolist():
            if any(info.filename.endswith(w) for w in want):
                dst = os.path.join(outdir, info.filename.replace("/", os.sep))
                if os.path.exists(dst) and os.path.getsize(dst) == info.file_size:
                    log("have", dst); continue
                log("extracting", info.filename, f"{info.file_size/1e9:.2f} GB (compressed {info.compress_size/1e9:.2f})")
                z.extract(info, outdir)


def danenberg():
    from_zip("https://zenodo.org/records/6036188/files/MBTMEStrIMCPublic.zip?download=1", ["SingleCells.csv"],
             os.path.join(RAW, "danenberg"))


def cbioportal():
    d = os.path.join(RAW, "cbioportal"); os.makedirs(d, exist_ok=True)
    base = "https://www.cbioportal.org/api/studies/brca_metabric/clinical-data?clinicalDataType=%s&projection=SUMMARY&pageSize=10000000"
    for kind, key, fn in [("PATIENT", "patientId", "data_clinical_patient.txt"), ("SAMPLE", "sampleId", "data_clinical_sample.txt")]:
        dst = os.path.join(d, fn)
        if os.path.exists(dst):
            continue
        rows = json.load(urllib.request.urlopen(urllib.request.Request(base % kind, headers={**UA, "Accept": "application/json"}), timeout=600))
        tab = collections.defaultdict(dict); cols = set()
        for r in rows:
            tab[r[key]][r["clinicalAttributeId"]] = r["value"]; cols.add(r["clinicalAttributeId"])
            if kind == "SAMPLE":
                tab[r[key]]["PATIENT_ID"] = r["patientId"]
        cols = sorted(cols - {"PATIENT_ID"})
        with open(dst, "w", encoding="utf-8") as f:
            f.write(("PATIENT_ID" if kind == "PATIENT" else "SAMPLE_ID\tPATIENT_ID") + "\t" + "\t".join(cols) + "\n")
            for k, v in tab.items():
                lead = k if kind == "PATIENT" else k + "\t" + v.get("PATIENT_ID", "")
                f.write(lead + "\t" + "\t".join(str(v.get(c, "")) for c in cols) + "\n")
        log(fn, len(tab), "rows", len(cols), "cols")


def jackson():
    d = os.path.join(RAW, "jackson"); z = "https://zenodo.org/records/4607374/files/%s?download=1"
    from_zip(z % "singlecell_locations.zip", ["_SC_locations.csv"], d)
    from_zip(z % "singlecell_cluster_labels.zip", ["Basel_metaclusters.csv", "Zurich_matched_metaclusters.csv", "Metacluster_annotations.csv"], d)
    from_zip(z % "SingleCell_and_Metadata.zip", ["BaselTMA/SC_dat.csv", "BaselTMA/Basel_PatientMetadata.csv",
                                                  "ZurichTMA/SC_dat.csv", "ZurichTMA/Zuri_PatientMetadata.csv"], d)


def codex():
    fetch("https://data.mendeley.com/public-files/datasets/mpjzbtfgfr/files/c24351b3-76d7-444f-9edf-0246356b0c78/file_downloaded",
          os.path.join(RAW, "codex", "CRC_clusters_neighborhoods_markers.csv"))


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["cbioportal", "danenberg", "codex", "jackson"]):
        try:
            log("==", w); globals()[w](); log("DONE", w)
        except Exception as e:
            log("FAILED", w, repr(e)[:400])
