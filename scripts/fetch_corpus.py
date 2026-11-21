#!/usr/bin/env python3
"""Download a Portuguese instruction/chat corpus (Alpaca pt-BR) and format it
as `### Usuário: ... ### Assistente: ...` examples for byte-level training.

Uses only the Python standard library (datasets-server rows API), so it runs
both on the GitHub runner and locally without extra dependencies.
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

DATASET = "dominguesm/alpaca-data-pt-br"
CONFIG = "default"
SPLIT = "train"
BASE = "https://datasets-server.huggingface.co/rows"


def fetch(offset, length, retries=5):
    url = (f"{BASE}?dataset={urllib.parse.quote(DATASET, safe='')}"
           f"&config={CONFIG}&split={SPLIT}&offset={offset}&length={length}")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "slm-pt"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/{retries}: {exc} (sleep {wait}s)",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus.txt")
    ap.add_argument("--limit", type=int, default=0, help="max examples (0 = all)")
    ap.add_argument("--length", type=int, default=100, help="rows per request (max 100)")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    per_req = max(1, min(args.length, 100))
    first = fetch(0, per_req)
    total = int(first.get("num_rows_total", 0))
    if args.limit > 0:
        total = min(total, args.limit)
    print(f"dataset={DATASET} available={first.get('num_rows_total')} using={total}",
          file=sys.stderr)

    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        offset = 0
        while offset < total:
            data = first if offset == 0 else fetch(offset, per_req)
            rows = data.get("rows", [])
            if not rows:
                break
            for item in rows:
                r = item.get("row", {})
                instr = (r.get("instruction") or "").strip()
                inp = (r.get("input") or "").strip()
                out = (r.get("output") or "").strip()
                if not instr or not out:
                    continue
                f.write("### Usuário:\n")
                f.write(instr + "\n")
                if inp:
                    f.write(inp + "\n")
                f.write("\n### Assistente:\n")
                f.write(out + "\n\n")
                written += 1
                if written >= total:
                    break
            offset += len(rows)
            if offset % 5000 == 0:
                print(f"  {written} examples written", file=sys.stderr)

    size = os.path.getsize(args.out)
    print(f"wrote {written} examples ({size / 1048576:.2f} MB) to {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
