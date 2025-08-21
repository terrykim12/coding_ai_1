#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLAN을 건너뛰고 최소 PLAN 객체로 /patch만 호출해 success_logs를 빠르게 쌓는다.
"""
import json
import argparse
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--path", default="examples/sample_py/app.py")
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    plan = {
        "files": [
            {
                "path": args.path.replace("\\", "/"),
                "reason": "log-only",
                "strategy": "regex",
                "tests": [],
            }
        ],
        "notes": "",
    }
    for i in range(max(1, args.runs)):
        try:
            r = requests.post(f"{args.base}/patch", json={"plan": plan}, timeout=180)
            r.raise_for_status()
            print(f"[OK] run {i+1}")
        except Exception as e:
            print(f"[SKIP] run {i+1}: {e}")


if __name__ == "__main__":
    main()


