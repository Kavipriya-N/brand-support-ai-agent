"""
Data Downloader: Downloads a configurable range-sliced subsample of the Kaggle TWCS dataset.
Uses HTTP Range requests to fetch the needed sample size quickly without downloading the entire 516MB file.
"""

import os
import sys
import yaml
import httpx
from pathlib import Path

def download_subsample(config_path: str = "configs/default.yaml") -> Path:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_url = cfg["data"]["raw_dataset_url"]
    dest_path = Path(cfg["data"]["raw_sample_path"])
    sample_bytes = cfg["data"].get("raw_sample_bytes", 45000000)

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and dest_path.stat().st_size >= sample_bytes:
        print(f"[OK] Raw sample already exists at {dest_path} ({dest_path.stat().st_size / 1e6:.1f} MB)")
        return dest_path

    print(f"[*] Downloading {sample_bytes / 1e6:.1f} MB slice from {raw_url} to {dest_path}...")
    headers = {"Range": f"bytes=0-{sample_bytes}"}
    
    # Follow redirects (HF redirects to AWS CDN)
    with httpx.Client(follow_redirects=True, verify=False, timeout=120.0) as client:
        with client.stream("GET", raw_url, headers=headers) as response:
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                downloaded = 0
                for chunk in response.iter_bytes(chunk_size=1024 * 512):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (5 * 1024 * 1024) < 1024 * 512:
                        print(f"    Downloaded {downloaded / 1e6:.1f} MB...")

    print(f"[OK] Download complete: {dest_path} ({dest_path.stat().st_size / 1e6:.1f} MB)")
    return dest_path

if __name__ == "__main__":
    download_subsample()
