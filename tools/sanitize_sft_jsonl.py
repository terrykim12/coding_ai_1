# -*- coding: utf-8 -*-
import json, sys, io, os

def fix_file(path: str):
    fixed = []
    changed = skipped = 0
    with io.open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                skipped += 1
                continue
            for key in ("output", "input", "response", "prompt"):
                if key in obj:
                    val = obj[key]
                    if isinstance(val, (dict, list)):
                        obj[key] = json.dumps(val, ensure_ascii=False)
                        changed += 1
                    elif val is None:
                        obj[key] = ""
            fixed.append(json.dumps(obj, ensure_ascii=False))
    tmp = path + ".fixed"
    with io.open(tmp, "w", encoding="utf-8", newline="\n") as w:
        w.write("\n".join(fixed) + "\n")
    bak = path + ".bak"
    if os.path.exists(bak):
        os.remove(bak)
    os.replace(path, bak)
    os.replace(tmp, path)
    print(f"[ok] {path} normalized: lines={len(fixed)} changed={changed} skipped={skipped} backup={bak}")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        fix_file(p)


