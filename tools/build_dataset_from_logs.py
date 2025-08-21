#!/usr/bin/env python3
import os, json, glob, random
random.seed(42)

SUCCESS = sorted(glob.glob("training/success_logs/*.jsonl"))
FAIL    = sorted(glob.glob("training/fail_logs/*.jsonl"))

rows = []

# 성공: PATCH 생성 페어 → 바로 학습에 사용
for p in SUCCESS:
    with open(p, encoding="utf-8") as f:
        for ln in f:
            try:
                rec = json.loads(ln)
                if rec.get("role") != "patch_success":
                    continue
                plan = rec.get("plan")
                patch = rec.get("patch")
                if not plan or not patch:
                    continue
                sample = {
                    "instruction": "You are a PATCH generator. Return ONLY the items of the edits array. No markdown or fences.",
                    "input": json.dumps({"plan": plan}, ensure_ascii=False),
                    "output": json.dumps(patch, ensure_ascii=False),
                }
                # 타입 혼합 방지: input/output을 항상 문자열로 강제
                if isinstance(sample.get("output"), (dict, list)):
                    sample["output"] = json.dumps(sample["output"], ensure_ascii=False)
                if isinstance(sample.get("input"), (dict, list)):
                    sample["input"] = json.dumps(sample["input"], ensure_ascii=False)
                rows.append(sample)
            except Exception:
                pass

# 실패 로그는 선택 사용(정답 없음) → 이번 단계에서는 제외

random.shuffle(rows)
k = max(1, int(len(rows) * 0.1)) if rows else 1
val = rows[:k]; train = rows[k:]

os.makedirs("training/data", exist_ok=True)
with open("training/data/train_from_logs.jsonl","w",encoding="utf-8") as f:
    for r in train: f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open("training/data/val_from_logs.jsonl","w",encoding="utf-8") as f:
    for r in val: f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("built:", len(train), "train,", len(val), "val")

