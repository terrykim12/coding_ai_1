# server/path_resolver.py
from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def norm_slash(p: str) -> str:
    return p.replace("\\", "/")


def is_abs_like(p: str) -> bool:
    p = norm_slash(p)
    return p.startswith("/") or re.match(r"^[A-Za-z]:/", p) is not None


class PathResolver:
    def __init__(
        self,
        root: str | Path,
        exts: Tuple[str, ...] = (".py", ".ts", ".js", ".tsx", ".jsx"),
    ):
        self.root = str(Path(root).resolve())
        self.exts = exts
        self._all: List[str] = []
        self._by_basename: Dict[str, List[str]] = {}
        self._index()

    def _index(self) -> None:
        patterns = [f"**/*{e}" for e in self.exts]
        for pat in patterns:
            for p in glob.glob(os.path.join(self.root, pat), recursive=True):
                rp = norm_slash(os.path.relpath(p, self.root))
                ap = norm_slash(str(Path(self.root, rp)))
                self._all.append(ap)
                self._by_basename.setdefault(os.path.basename(rp), []).append(ap)

    def resolve(self, predicted: str) -> Optional[str]:
        if not predicted:
            return None
        p = norm_slash(predicted).strip()

        # 1) root 기준 상대경로 시도
        candidate = norm_slash(str(Path(self.root, p)))
        if os.path.isfile(candidate):
            return candidate

        # 2) 절대경로 흉내면 접미사 매칭
        if is_abs_like(p):
            parts = norm_slash(p).split("/")
            for k in range(min(6, len(parts)), 1, -1):
                suf = "/".join(parts[-k:])
                for ap in self._all:
                    if ap.endswith(suf) and os.path.isfile(ap):
                        return ap

        # 3) basename만 같은 후보 중 접미사 점수 최대 선택
        base = os.path.basename(p)
        cands = self._by_basename.get(base, [])
        if cands:
            def score(ap: str) -> int:
                a = norm_slash(ap).split("/")
                b = norm_slash(p).split("/")
                s = 0
                while s < min(len(a), len(b)) and a[-1 - s] == b[-1 - s]:
                    s += 1
                return s

            cands = sorted(cands, key=score, reverse=True)
            return cands[0]

        return None

    def fix_patch_paths(self, patch: dict) -> Tuple[dict, Dict[str, str], List[str]]:
        """patch.edits[*].path를 워크스페이스 실제 경로로 보정"""
        edits = patch.get("edits", [])
        remap: Dict[str, str] = {}
        failed: List[str] = []
        for e in edits:
            if not isinstance(e, dict):
                failed.append(f"<bad-edit-type:{type(e).__name__}>")
                continue
            p = e.get("path")
            if not p:
                failed.append("<missing path>")
                continue
            real = self.resolve(p)
            if real:
                new_rel = norm_slash(os.path.relpath(real, self.root))
                if new_rel != p:
                    remap[p] = new_rel
                    e["path"] = new_rel
            else:
                failed.append(p)
        return patch, remap, failed

