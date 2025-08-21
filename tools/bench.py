#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단 벤치: 스니펫 기반 호출을 N회 반복하고 CSV에 기록
"""
import argparse
import csv
import json
import time
from pathlib import Path

from codeflow import invoke_code_edit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8765")
    ap.add_argument("--path", default="examples/sample_py/app.py")
    ap.add_argument("--function", default="add")
    ap.add_argument("--intent", default="add()에 음수 방지")
    ap.add_argument("-n", "--runs", type=int, default=5)
    ap.add_argument("--csv", default="bench_results.csv")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts", "ok", "edits", "notes"])  # 최소 열

        for i in range(args.runs):
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            ok = False
            edits = 0
            notes = ""
            try:
                out = invoke_code_edit(args.base_url, args.intent, args.path, args.function)
                ok = "error" not in out
                edits = out.get("edits", 0)
                notes = out.get("test", {}).get("status", "")
            except Exception as e:
                notes = str(e)
            w.writerow([ts, ok, edits, notes])
            print(f"[{i+1}/{args.runs}] ok={ok} edits={edits} notes={notes}")


if __name__ == "__main__":
    main()


