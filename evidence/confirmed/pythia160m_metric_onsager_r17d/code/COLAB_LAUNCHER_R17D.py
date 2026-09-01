#!/usr/bin/env python3
"""Colab launcher for frozen five-seed R17d confirmation."""
from pathlib import Path
import shutil, subprocess, sys, zipfile

def main():
    from google.colab import files
    uploaded=files.upload()
    bundles=[Path(n) for n in uploaded if n.lower().endswith(".zip") and "r17d" in n.lower()]
    if len(bundles)!=1: raise SystemExit("Upload exactly one pythia160m_metric_confirm_r17d.zip")
    root,out=Path("/content/pythia_r17d_run"),Path("/content/pythia_r17d_results")
    for path in (root,out):
        if path.exists(): shutil.rmtree(path)
        path.mkdir(parents=True)
    with zipfile.ZipFile(bundles[0]) as archive:
        bad=archive.testzip()
        if bad: raise SystemExit(f"Corrupt member: {bad}")
        archive.extractall(root)
    scripts=list(root.rglob("pythia160m_metric_confirm_r17d.py"))
    requirements=list(root.rglob("requirements.txt"))
    if len(scripts)!=1 or len(requirements)!=1: raise SystemExit("Invalid R17d bundle layout")
    subprocess.run([sys.executable,"-m","pip","install","-q","-r",str(requirements[0])],check=True)
    cmd=[sys.executable,str(scripts[0]),"--device","cuda","--outdir",str(out)]
    print("Running:"," ".join(cmd),flush=True)
    code=subprocess.run(cmd).returncode
    summary=out/"run_summary.json"
    if summary.exists(): print(summary.read_text())
    else: print(f"No run_summary.json; child exit code={code}",flush=True)
    result_zip=shutil.make_archive(str(out),"zip",out); files.download(result_zip)
    if code not in (0,2): raise SystemExit(code)

if __name__=="__main__": main()
