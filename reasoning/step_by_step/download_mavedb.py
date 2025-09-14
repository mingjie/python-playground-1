# save as download_mavedb.py
import requests, zipfile, io, json, os
import pandas as pd

# Example: list experiments on MaveDB manually or use web UI to pick an experiment accession.
# Here we download a MaveDB bulk dump (if desired) or fetch an experiment JSON.
# Replace 'urn:mavedb:00000055' with the experiment URN you choose.

experiment_urn = "urn:mavedb:00000055"  # example; pick one from mavedb.org
api_url = f"https://api.mavedb.org/experiments/{experiment_urn}"

r = requests.get(api_url)
r.raise_for_status()
exp = r.json()
print("Experiment title:", exp.get("title"))

# Many MaveDB experiments have associated files - find file URLs
files = exp.get("files", [])
for f in files:
    print(f["filename"], f.get("download_url"))

# If you have a direct CSV url, download it:
csv_url = "https://zenodo.org/records/15653325/files/mavedb-dump.20250612164404.zip"  # example bulk dump
# or use a single experiment file url returned above
