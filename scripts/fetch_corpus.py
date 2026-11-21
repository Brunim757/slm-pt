#!/usr/bin/env python3
"""Download the Portuguese instruction corpus (Alpaca pt-BR) in a single request
and format it as `### Usuário: ... ### Assistente: ...` examples.

The dataset ships as one parquet file, so we download it once (no paginated API,
which gets rate-limited) and read it with pyarrow.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.request

DATASET = "dominguesm/alpaca-data-pt-br"
FALLBACK_PARQUET = "data/train-00000-of-00001-63604ec441480f0e.parquet"


def open_url(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "slm-pt"})
    return urllib.request.urlopen(req, timeout=timeout)


def find_parquet():
    try:
        with open_url(f"https://huggingface.co/api/datasets/{DATASET}", timeout=60) as r:
            info = json.load(r)
        files = [s["rfilename"] for s in info.get("siblings", [])]
        parqs = [f for f in files if f.endswith(".parquet")]
        if parqs:
            return parqs[0]
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not query file list: {exc})", file=sys.stderr)
    return FALLBACK_PARQUET


def download(url, path):
    with open_url(url) as r, open(path, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)


def fmt(instr, inp, out):
    parts = ["### Usuário:\n", instr, "\n"]
    if inp:
        parts.append(inp + "\n")
    parts += ["\n### Assistente:\n", out, "\n\n"]
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus.txt")
    ap.add_argument("--limit", type=int, default=0, help="max examples (0 = all)")
    args = ap.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow is required: python3 -m pip install pyarrow")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    parquet = find_parquet()
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{parquet}"
    tmp = tempfile.mktemp(suffix=".parquet")
    print(f"downloading {parquet} ...", file=sys.stderr)
    download(url, tmp)
    print(f"downloaded {os.path.getsize(tmp) / 1048576:.2f} MB", file=sys.stderr)

    table = pq.read_table(tmp, columns=["instruction", "input", "output"])
    data = table.to_pydict()
    n = len(data["instruction"])
    limit = args.limit if args.limit > 0 else n
    limit = min(limit, n)

    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(limit):
            instr = (data["instruction"][i] or "").strip()
            inp = (data["input"][i] or "").strip()
            out = (data["output"][i] or "").strip()
            if not instr or not out:
                continue
            f.write(fmt(instr, inp, out))
            written += 1

    os.remove(tmp)
    size = os.path.getsize(args.out)
    print(f"wrote {written} examples ({size / 1048576:.2f} MB) to {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()