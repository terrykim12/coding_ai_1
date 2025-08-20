from __future__ import annotations
import os, re, json, difflib, hashlib, shutil, tempfile, ast
from dataclasses import dataclass
from typing import Tuple, Optional, List

LEDGER_PATH = os.path.join(".llm_patch", "ledger.json")

# ===== 유틸리티 함수들 =====
def _ensure_dir(path: str):
    """디렉토리 생성"""
    os.makedirs(path, exist_ok=True)

def _read(path: str) -> str:
    """파일 읽기"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def _write(path: str, content: str):
    """파일 쓰기"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def _norm(s: str) -> str:
    """정규화 (공백, 줄바꿈 정리)"""
    return re.sub(r'\s+', ' ', s.strip())

def _sha(s: str) -> str:
    """SHA1 해시"""
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def _file_sha(path: str) -> str:
    """파일 SHA1 해시"""
    return _sha(_read(path))

# ===== 데이터 구조 =====
@dataclass
class Range:
    a: int
    b: int
    
    def __len__(self):
        return self.b - self.a

# ===== LEDGER 관리 =====
def _load_ledger() -> dict:
    """LEDGER 로드"""
    _ensure_dir(os.path.dirname(LEDGER_PATH))
    if os.path.exists(LEDGER_PATH):
        try:
            with open(LEDGER_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {"applied": {}}

def _save_ledger(ledger: dict):
    """LEDGER 저장"""
    with open(LEDGER_PATH, 'w', encoding='utf-8') as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)

def _edit_id(edit: dict) -> str:
    """편집 ID 생성 (path + loc + action + code 기반 안정적 해시)"""
    key = json.dumps({
        "path": edit.get("path"),
        "loc": edit.get("loc"),
        "action": edit.get("action"),
        "code": edit.get("code", "")
    }, sort_keys=True, ensure_ascii=False)
    return _sha(key)[:16]

# ===== 위치 찾기 함수들 =====
def _range_from_linecol(src: str, line: int, col: int) -> Range:
    """라인:컬럼 → 문자 위치 변환"""
    lines = src.splitlines()
    if line > len(lines):
        return Range(len(src), len(src))
    
    pos = sum(len(l) + 1 for l in lines[:line-1]) + (col - 1)
    return Range(pos, pos)

def _fuzz_window(src: str, center: int, window: int = 200) -> Range:
    """퓨즈 매칭용 윈도우 추출"""
    start = max(0, center - window)
    end = min(len(src), center + window)
    return Range(start, end)

def _py_ast_func_range(src: str, func: str, part: str = "body") -> Optional[Range]:
    """Python AST 기반 함수 범위 찾기"""
    try:
        tree = ast.parse(src)
    except Exception:
        return None
    
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            target = node
            break
    
    if not target or not hasattr(target, "lineno") or not hasattr(target, "end_lineno"):
        return None
    
    lines = src.splitlines(True)
    def to_ofs(line, col): 
        return sum(len(l) for l in lines[:line-1]) + col
    
    if part == "def":   # def 시그니처 전체
        a = to_ofs(target.lineno, 0)
        b = to_ofs(target.body[0].lineno, 0) if target.body else to_ofs(target.end_lineno, 0)
    else:               # body (기본)
        if not target.body: 
            return None
        first = target.body[0]
        last = target.body[-1]
        a = to_ofs(getattr(first, "lineno", target.lineno), 0)
        b = to_ofs(getattr(last, "end_lineno", last.lineno), len(lines[getattr(last, "end_lineno", last.lineno)-1]))
    
    return Range(a, b)

def _pick_best_match(src: str, matches: List[re.Match], code: str) -> re.Match:
    """다중 매치 시 최적 후보 선택"""
    # (1) 주변 창 추출(매치 전후 200자)
    def ctx(m): 
        a = max(0, m.start() - 200)
        b = min(len(src), m.end() + 200)
        return src[a:b]
    
    # (2) code의 헤더(첫 200자)와 유사도 점수
    header = _norm(code)[:200]
    def score(m):
        return difflib.SequenceMatcher(a=_norm(ctx(m)), b=header).ratio() - 0.01 * len(m.group(0))
    
    return max(matches, key=score)

def _find_anchor_range(src: str, loc: dict, code: str) -> Tuple[Range, str]:
    """위치 찾기 (앵커, regex, 퓨즈 순서)"""
    flags = re.DOTALL | re.MULTILINE
    
    # 1단계: regex 타입
    if loc["type"] == "regex":
        pattern = loc.get("pattern", "")
        if pattern:
            matches = list(re.finditer(pattern, src, flags))
            if matches:
                if len(matches) == 1:
                    m = matches[0]
                else:
                    m = _pick_best_match(src, matches, code)
                return Range(m.start(), m.end()), "regex"
    
    # 2단계: anchor 타입
    if loc["type"] == "anchor":
        before = loc.get("before", "")
        after = loc.get("after", "")
        
        if before and after:
            # before와 after 모두 있는 경우
            before_matches = list(re.finditer(re.escape(before), src, flags))
            after_matches = list(re.finditer(re.escape(after), src, flags))
            
            if before_matches and after_matches:
                # 최적 조합 찾기
                best_range = None
                best_score = -1
                
                for bm in before_matches:
                    for am in after_matches:
                        if bm.end() <= am.start():  # before가 after보다 앞에 있어야 함
                            range_len = am.start() - bm.end()
                            if range_len > 0 and range_len < 1000:  # 합리적인 길이
                                # 주변 컨텍스트와 code 유사도 계산
                                context = src[max(0, bm.start()-100):min(len(src), am.end()+100)]
                                score = difflib.SequenceMatcher(a=_norm(context), b=_norm(code)).ratio()
                                
                                if score > best_score:
                                    best_score = score
                                    best_range = Range(bm.end(), am.start())
                
                if best_range:
                    return best_range, "anchor"
        
        elif before:
            # before만 있는 경우
            matches = list(re.finditer(re.escape(before), src, flags))
            if matches:
                m = matches[0] if len(matches) == 1 else _pick_best_match(src, matches, code)
                return Range(m.end(), m.end()), "anchor_before"
        
        elif after:
            # after만 있는 경우
            matches = list(re.finditer(re.escape(after), src, flags))
            if matches:
                m = matches[0] if len(matches) == 1 else _pick_best_match(src, matches, code)
                return Range(m.start(), m.start()), "anchor_after"
    
    # 3단계: AST 타입 (Python 전용)
    if loc["type"] == "ast":
        if path.endswith(".py") and "func" in loc:
            r = _py_ast_func_range(src, loc["func"], loc.get("part", "body"))
            if r: 
                return r, "ast"
        # AST 실패 시 fuzz로 폴백
    
    # 4단계: range 타입
    if loc["type"] == "range":
        start = loc.get("start", 0)
        end = loc.get("end", len(src))
        return Range(start, end), "range"
    
    # 5단계: fuzz 매칭 (마지막 수단)
    if code:
        # code의 첫 100자로 퓨즈 매칭
        code_start = _norm(code[:100])
        best_pos = -1
        best_ratio = 0.6  # 최소 임계값
        
        for i in range(0, len(src) - 100, 50):  # 50자씩 점프
            window = _norm(src[i:i+200])
            ratio = difflib.SequenceMatcher(a=code_start, b=window).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = i
        
        if best_pos >= 0:
            return Range(best_pos, best_pos), "fuzz"
    
    # 실패 시 전체 범위 반환
    return Range(0, len(src)), "fallback"

# ===== 사전 조건 검사 =====
def _pre_match(text: str, pat: str, use_regex: bool) -> bool:
    """pre 조건용 안전 매칭: 기본은 리터럴. regex 요청 시 컴파일 실패하면 리터럴 fallback."""
    if not use_regex:
        return pat in text
    try:
        return re.search(pat, text, re.DOTALL | re.MULTILINE) is not None
    except re.error:
        # 잘못된 정규식이면 리터럴로 대체
        return pat in text

def _check_pre(src: str, pre: dict, path: str) -> Tuple[bool, Optional[str]]:
    """pre = {'must_contain': [...], 'must_not_contain': [...], 'regex': bool, 'context_sha': '...'}"""
    use_regex = bool(pre.get("regex", False))
    
    # must_contain 검사
    for pat in pre.get("must_contain", []):
        if not _pre_match(src, pat, use_regex):
            return False, f"pre.must_contain miss: {pat}"
    
    # must_not_contain 검사
    for pat in pre.get("must_not_contain", []):
        if _pre_match(src, pat, use_regex):
            return False, f"pre.must_not_contain hit: {pat}"
    
    # context_sha 검사
    if sh := pre.get("context_sha"):
        if sh != _file_sha(path):
            return False, "pre.context_sha mismatch"
    
    return True, None

# ===== 중복/동일성 검사 =====
def _already_contains(src: str, code: str) -> bool:
    """코드가 이미 포함되어 있는지 확인"""
    return _norm(code) in _norm(src)

def _range_equals(src: str, r: Range, code: str) -> bool:
    """범위 내용이 동일한지 확인"""
    return _norm(src[r.a:r.b]) == _norm(code)

# ===== Python 문법 검증 =====
def _python_syntax_ok(path: str, text: str) -> bool:
    """Python 문법 검증"""
    if not path.endswith('.py'):
        return True
    
    try:
        compile(text, path, 'exec')
        return True
    except SyntaxError:
        return False

# ===== 메인 패치 적용 함수 =====
def apply_patch_json(patch: dict, allowed_paths: Optional[List[str]] = None, dry_run: bool = False) -> dict:
    """패치 JSON 적용"""
    report = {
        "applied": [],      # 실제 적용된 파일들
        "skipped": [],      # 건너뛴 파일들 (성공으로 카운트)
        "failed": [],       # 실패한 파일들
        "details": []       # 상세 정보 (status 포함)
    }
    
    ledger = _load_ledger()
    
    for e in patch.get("edits", []):
        path = e["path"]
        action = e.get("action", "replace_range")
        code = e.get("code", "")
        once = e.get("once", True)
        edit_id = _edit_id(e)
        
        # 경로 검증
        if allowed_paths and not any(path.startswith(allowed) for allowed in allowed_paths):
            report["failed"].append({"path": path, "error": "path not allowed"})
            report["details"].append({
                "path": path, "status": "failed", "action": action, 
                "edit_id": edit_id, "delta_chars": 0
            })
            continue
        
        # 파일 읽기
        try:
            src = _read(path)
        except Exception as e:
            report["failed"].append({"path": path, "error": f"file read failed: {e}"})
            report["details"].append({
                "path": path, "status": "failed", "action": action, 
                "edit_id": edit_id, "delta_chars": 0
            })
            continue
        
        # 프리컨디션 검사
        pre = e.get("pre", {})
        if pre:
            ok, why = _check_pre(src, pre, path)
            if not ok:
                report["failed"].append({"path": path, "error": f"precondition failed: {why}"})
                report["details"].append({
                    "path": path, "status": "failed", "action": action, 
                    "edit_id": edit_id, "delta_chars": 0
                })
                continue
        
        # LEDGER 기반 중복 차단 (성공으로 카운트)
        if once and path in ledger["applied"] and edit_id in ledger["applied"][path]:
            report["skipped"].append({"path": path, "reason": f"duplicate(edit_id={edit_id})"})
            report["details"].append({
                "path": path, "status": "duplicate", "action": action, 
                "edit_id": edit_id, "delta_chars": 0
            })
            continue
        
        # 위치 찾기
        try:
            r, anchor_kind = _find_anchor_range(src, e["loc"], code)
        except Exception as e:
            report["failed"].append({"path": path, "error": f"location finding failed: {e}"})
            report["details"].append({
                "path": path, "status": "failed", "action": action, 
                "edit_id": edit_id, "delta_chars": 0
            })
            continue
        
        # 사전 검사 (no-op 케이스들)
        applied = False
        delta = 0
        
        if action == "replace_range":
            if _range_equals(src, r, code):
                # 동일한 내용이 이미 있음 (성공으로 카운트)
                report["skipped"].append({"path": path, "reason": "identical content"})
                report["details"].append({
                    "path": path, "status": "noop_identical", "action": action, 
                    "edit_id": edit_id, "delta_chars": 0
                })
                continue
        elif action.startswith("insert") and _already_contains(src, code):
            # 코드가 이미 존재함 (성공으로 카운트)
            report["skipped"].append({"path": path, "reason": "code already present"})
            report["details"].append({
                "path": path, "status": "noop_present", "action": action, 
                "edit_id": edit_id, "delta_chars": 0
            })
            continue
        
        # 실제 편집 적용
        try:
            if action == "insert_before":
                new_text = src[:r.a] + code + src[r.a:]
                delta = len(code)
            elif action == "insert_after":
                new_text = src[:r.b] + code + src[r.b:]
                delta = len(code)
            elif action == "replace_range":
                new_text = src[:r.a] + code + src[r.b:]
                delta = len(code) - len(r)
            elif action == "delete_range":
                new_text = src[:r.a] + src[r.b:]
                delta = -len(r)
            else:
                raise ValueError(f"Unknown action: {action}")
            
            applied = True
            
        except Exception as e:
            report["failed"].append({"path": path, "error": f"edit application failed: {e}"})
            report["details"].append({
                "path": path, "status": "failed", "action": action, 
                "edit_id": edit_id, "delta_chars": 0
            })
            continue
        
        # 적용 후 Python 문법 검증 (해당 파일일 때만)
        if applied and not dry_run and not _python_syntax_ok(path, new_text):
            report["failed"].append({"path": path, "error": "syntax error after apply — reverted"})
            report["details"].append({
                "path": path, "status": "failed", "action": action, 
                "edit_id": edit_id, "delta_chars": 0
            })
            continue
        
        # 실제 적용 (dry_run이 아닐 때)
        if not dry_run and applied:
            # 백업 & 기록
            backup_dir = os.path.join(".llm_patch", "backups")
            _ensure_dir(backup_dir)
            shutil.copy2(path, os.path.join(backup_dir, os.path.basename(path) + ".bak"))
            
            _write(path, new_text)
            
            # LEDGER 업데이트 (아이템포턴시)
            ledger["applied"].setdefault(path, {})
            ledger["applied"][path][edit_id] = _file_sha(path)
            
            _save_ledger(ledger)
        
        # 결과 기록
        if applied:
            report["applied"].append(path)
            report["details"].append({
                "path": path, "status": "applied", "action": action, 
                "anchor": anchor_kind, "edit_id": edit_id, "delta_chars": delta
            })
        else:
            report["skipped"].append({"path": path, "reason": "no-op"})
            report["details"].append({
                "path": path, "status": "noop", "action": action, 
                "edit_id": edit_id, "delta_chars": 0
            })
    
    return report

# 기존 함수들과의 호환성을 위한 래퍼
def apply_patch_from_file(
    patch_file: str,
    allowed_paths: Optional[list] = None,
    dry_run: bool = False
) -> dict:
    """파일에서 Patch JSON 읽어서 적용"""
    try:
        with open(patch_file, 'r', encoding='utf-8') as f:
            patch_data = json.load(f)
        
        return apply_patch_json(patch_data, allowed_paths, dry_run)
    
    except Exception as e:
        return {"failed": [{"path": patch_file, "error": f"Failed to read patch file: {str(e)}"}]}

def create_backup(file_path: str) -> str:
    """파일 백업 생성"""
    backup_path = f"{file_path}.backup"
    if os.path.exists(file_path):
        import shutil
        shutil.copy2(file_path, backup_path)
    return backup_path

def restore_backup(file_path: str) -> bool:
    """백업에서 파일 복원"""
    backup_path = f"{file_path}.backup"
    if os.path.exists(backup_path):
        import shutil
        shutil.copy2(backup_path, file_path)
        os.remove(backup_path)
        return True
    return False

