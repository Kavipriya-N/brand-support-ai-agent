"""
Build script for static frontend deployment (GitHub Pages / Vercel / Netlify).

Prepares the distribution directory (`dist/`) by:
1. Copying frontend assets from `web/`
2. Bundling pre-computed evaluation results and golden samples as static JSON endpoints
3. Ensuring relative API routing for GitHub Pages subpath compatibility
4. Creating `.nojekyll` to disable Jekyll processing
"""

import os
import json
import shutil
import re
from pathlib import Path

def build_dist():
    dist_dir = Path("dist")
    web_dir = Path("web")
    
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy web assets
    for file_name in ["index.html", "style.css", "app.js"]:
        src_file = web_dir / file_name
        if src_file.exists():
            shutil.copy(src_file, dist_dir / file_name)

    # 2. Make fetch URLs relative in dist/app.js to support subpath hosting (e.g. /brand-support-ai-agent/)
    app_js_path = dist_dir / "app.js"
    if app_js_path.exists():
        content = app_js_path.read_text(encoding="utf-8")
        # Replace leading slash API calls with relative API calls
        content = content.replace('fetch("/api/', 'fetch("api/')
        app_js_path.write_text(content, encoding="utf-8")

    # 3. Create static api directory in dist
    api_dir = dist_dir / "api"
    api_dir.mkdir(parents=True, exist_ok=True)

    # 4. Copy eval-results
    eval_src = Path("data/processed/eval_results.json")
    if eval_src.exists():
        eval_data = json.loads(eval_src.read_text(encoding="utf-8"))
        # Save both extension-less and .json for diverse static servers
        (api_dir / "eval-results").write_text(json.dumps(eval_data), encoding="utf-8")
        (api_dir / "eval-results.json").write_text(json.dumps(eval_data), encoding="utf-8")

    # 5. Build samples endpoint from golden_set.jsonl
    golden_src = Path("data/golden_set.jsonl")
    samples = []
    if golden_src.exists():
        with open(golden_src, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        samples_payload = {"total": len(samples), "samples": samples}
        (api_dir / "samples").write_text(json.dumps(samples_payload), encoding="utf-8")
        (api_dir / "samples.json").write_text(json.dumps(samples_payload), encoding="utf-8")

    # 6. Add .nojekyll for GitHub Pages
    (dist_dir / ".nojekyll").write_text("", encoding="utf-8")

    print(f"[OK] Static distribution built successfully in {dist_dir} ({len(list(dist_dir.rglob('*')))} files)")

if __name__ == "__main__":
    build_dist()
