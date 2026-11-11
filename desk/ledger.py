"""ledger.py: an append-only, hash-chained forecast ledger.

Each line of the ledger file is a JSON object

    {"hash": ..., "prev": ..., "payload": "<the record as a JSON string>"}

with hash = SHA-256(prev + "\\n" + payload). The payload is stored as text so
that anyone can verify the chain byte for byte in any language (the site does
it in the browser) without having to reproduce a JSON serialiser.

Changing, deleting or reordering any earlier record changes every later
hash. On its own that only proves internal consistency: the evidence that a
forecast existed *before* its outcome comes from git, because each run is
committed by GitHub Actions with a server-side timestamp and the chain head
of one commit is the "prev" of the next.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

GENESIS = "0" * 64


def canonical(record: Dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def link_hash(prev: str, payload: str) -> str:
    return hashlib.sha256((prev + "\n" + payload).encode("utf-8")).hexdigest()


def record_id(key: str) -> str:
    """Stable identifier of a forecast from its natural key."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def read(path: str) -> List[Dict]:
    """All entries as {"hash", "prev", "payload", "record"}; [] if the file does not exist."""
    if not os.path.exists(path):
        return []
    entries = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entry["record"] = json.loads(entry["payload"])
            entries.append(entry)
    return entries


def verify(entries: Iterable[Dict]) -> Tuple[bool, Optional[int]]:
    """Check every link. Returns (ok, index of the first broken entry or None)."""
    prev = GENESIS
    for i, e in enumerate(entries):
        if e["prev"] != prev or e["hash"] != link_hash(prev, e["payload"]):
            return False, i
        if e.get("record", json.loads(e["payload"])).get("seq") != i:
            return False, i
        prev = e["hash"]
    return True, None


def head(entries: List[Dict]) -> str:
    return entries[-1]["hash"] if entries else GENESIS


def append(path: str, records: List[Dict], key_field: str = "id") -> List[Dict]:
    """Append records whose key is not already present. Returns the entries added.

    Refuses to write to a ledger whose chain does not verify.
    """
    entries = read(path)
    ok, bad = verify(entries)
    if not ok:
        raise RuntimeError(f"{path}: chain broken at entry {bad}; refusing to append")
    seen = {e["record"][key_field] for e in entries}
    prev = head(entries)
    added = []
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            if rec[key_field] in seen:
                continue
            body = dict(rec, seq=len(entries) + len(added))
            payload = canonical(body)
            digest = link_hash(prev, payload)
            fh.write(json.dumps({"hash": digest, "prev": prev, "payload": payload}, separators=(",", ":")) + "\n")
            added.append({"hash": digest, "prev": prev, "payload": payload, "record": body})
            seen.add(rec[key_field])
            prev = digest
    return added
