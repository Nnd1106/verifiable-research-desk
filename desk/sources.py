"""sources.py: load the published data files this desk validates.

The desk only ever sees what the public sees: the JSON each project serves
from GitHub Pages. A local sibling checkout is used as a fallback so the desk
can be developed before a project's first deployment; the result records
which one was read, with a SHA-256 of the exact bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from typing import Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCES = {
    "portfolio": {
        "name": "Portfolio Construction Lab",
        "url": "https://nnd1106.github.io/portfolio-construction-lab/data/portfolio-data.json",
        "site": "https://nnd1106.github.io/portfolio-construction-lab/",
        "repo": "https://github.com/Nnd1106/portfolio-construction-lab",
        "local": os.path.join(ROOT, "..", "portfolio-construction-lab", "docs", "data", "portfolio-data.json"),
    },
    "valuation": {
        "name": "Valuation & Deal Lab",
        "url": "https://nnd1106.github.io/valuation-deal-lab/data/deal-lab-data.json",
        "site": "https://nnd1106.github.io/valuation-deal-lab/",
        "repo": "https://github.com/Nnd1106/valuation-deal-lab",
        "local": os.path.join(ROOT, "..", "valuation-deal-lab", "docs", "data", "deal-lab-data.json"),
    },
}


def _fetch(url: str, timeout: int = 60) -> Optional[bytes]:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "research-desk/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read() if response.status == 200 else None
    except Exception as exc:
        print(f"  ~ could not fetch {url}: {exc}")
        return None


def load(key: str, prefer_local: bool = False) -> Dict:
    """Return {"data", "meta"}; meta has name, origin, sha256 and size.

    Raises RuntimeError when neither the published file nor a local copy exists.
    """
    src = SOURCES[key]
    raw, origin = None, None
    if not prefer_local:
        raw = _fetch(src["url"])
        origin = "published"
    if raw is None and os.path.exists(src["local"]):
        with open(src["local"], "rb") as fh:
            raw = fh.read()
        origin = "local checkout"
    if raw is None:
        raise RuntimeError(f"{src['name']}: no published file and no local copy")
    data = json.loads(raw)
    meta = {
        "key": key, "name": src["name"], "url": src["url"], "site": src["site"], "repo": src["repo"],
        "origin": origin, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
        "generated_utc": data.get("meta", {}).get("generated_utc"),
    }
    return {"data": data, "meta": meta}
