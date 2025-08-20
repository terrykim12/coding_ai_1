import os
import re
import math
import glob
from collections import Counter
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def _tokenize(s: str) -> List[str]:
    """간단한 토큰화 - Python 식별자와 특수문자 분리"""
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\S", s)

def _tfidf_rank_simple(chunks: List[Tuple[str, str]], query: str, topk: int = 6) -> List[Tuple[str, str]]:
    """간단한 TF-IDF 랭킹 (순수 파이썬 구현)"""
    docs = [t for _, t in chunks]
    vocab = set()
    dfs = Counter()  # Document frequency
    tokenized = []
    
    # 토큰화 및 어휘 구축
    for d in docs:
        toks = _tokenize(d)
        tokenized.append(toks)
        vocab.update(toks)
    
    # Document frequency 계산
    for t in vocab:
        dfs[t] = sum(1 for toks in tokenized if t in toks)
    
    N = len(docs)
    q = Counter(_tokenize(query))
    scores = []
    
    # 각 문서에 대한 점수 계산
    for i, toks in enumerate(tokenized):
        tf = Counter(toks)
        s = 0.0
        
        for t, qv in q.items():
            if t in tf and dfs[t] > 0:
                idf = math.log((N + 1) / (dfs[t] + 1)) + 1
                s += (tf[t] * idf) * qv
        
        scores.append((s, i))
    
    # 상위 k개 선택
    idx = [i for _, i in sorted(scores, reverse=True)[:topk]]
    return [chunks[i] for i in idx]

def _tfidf_rank_sklearn(chunks: List[Tuple[str, str]], query: str, topk: int = 6) -> List[Tuple[str, str]]:
    """scikit-learn을 사용한 TF-IDF 랭킹 (더 정확함)"""
    try:
        docs = [t for _, t in chunks]
        vectorizer = TfidfVectorizer(
            token_pattern=r'[A-Za-z_][A-Za-z0-9_]*|\S',
            max_features=10000,
            stop_words=None
        )
        
        # 문서 벡터화
        doc_vectors = vectorizer.fit_transform(docs)
        query_vector = vectorizer.transform([query])
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        
        # 상위 k개 인덱스
        top_indices = np.argsort(similarities)[::-1][:topk]
        
        return [chunks[i] for i in top_indices]
    except Exception:
        # sklearn 실패 시 간단한 구현으로 폴백
        return _tfidf_rank_simple(chunks, query, topk)

def _extract_error_context(error: str) -> str:
    """에러 메시지에서 핵심 정보 추출"""
    if not error:
        return ""
    
    # 파일 경로, 라인 번호, 함수명 등 추출
    patterns = [
        r'File "([^"]+)"',  # 파일 경로
        r'line (\d+)',      # 라인 번호
        r'in (\w+)',        # 함수명
        r'(\w+Error)',      # 에러 타입
    ]
    
    context_parts = []
    for pattern in patterns:
        matches = re.findall(pattern, error)
        context_parts.extend(matches)
    
    return " ".join(context_parts)

def _read_file_safe(path: str, max_size: int = 100000) -> Optional[str]:
    """안전하게 파일 읽기 (크기 제한, 인코딩 처리)"""
    try:
        if os.path.getsize(path) > max_size:
            return None
        
        # 텍스트 파일인지 확인
        with open(path, 'rb') as f:
            chunk = f.read(1024)
            if b'\x00' in chunk:  # 바이너리 파일
                return None
        
        # 텍스트로 읽기
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception:
        return None

def build_context(paths=None, code_paste=None, error=None, topk=2, slice_lines=150):
    """컨텍스트 빌드 - 토큰 절약 최적화 (더욱 가벼운 버전)"""
    if not paths:
        paths = ["."]
    
    # 파일 수집
    snippets = []
    for path in paths:
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                snippets.append((path, content))
            except Exception as e:
                logger.warning(f"파일 읽기 실패 {path}: {e}")
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.c', '.h')):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            snippets.append((file_path, content))
                        except Exception as e:
                            logger.warning(f"파일 읽기 실패 {file_path}: {e}")
    
    if not snippets:
        return f"Error: {error or 'No files found'}\nCode: {code_paste or 'No code'}"
    
    # TF-IDF 랭킹으로 Top-k 선택 (더 적게)
    query = f"{error or ''} {code_paste or ''}"
    top_snippets = _tfidf_rank_sklearn(snippets, query, topk=topk)
    
    # 컨텍스트 구성 (토큰 절약)
    ctx_parts = []
    
    # 에러 메시지 (있는 경우)
    if error:
        ctx_parts.append(f"[ERROR]\n{error}")
    
    # 코드 스니펫 (있는 경우)
    if code_paste:
        ctx_parts.append(f"[CODE]\n{code_paste}")
    
    # Top-k 파일 스니펫 (라인 수 제한 - 더 적게)
    for path, content in top_snippets:
        lines = content.splitlines()
        
        # 파일당 slice_lines만큼만 사용 (토큰 절약)
        if len(lines) > slice_lines:
            # 에러나 코드에 나오는 키워드 근처 중심으로 슬라이스
            snippet = "\n".join(lines[:slice_lines])
            snippet += f"\n... (truncated, total {len(lines)} lines)"
        else:
            snippet = content
        
        ctx_parts.append(f"<<FILE {path}>>\n{snippet}\n<<END>>")
    
    return "\n\n".join(ctx_parts)

def build_context_from_error(error: str, project_root: str = ".") -> str:
    """에러 메시지만으로 컨텍스트 구축"""
    return build_context(
        paths=[project_root],
        error=error,
        max_files=10
    )

def build_context_from_intent(intent: str, project_root: str = ".") -> str:
    """의도만으로 컨텍스트 구축"""
    return build_context(
        paths=[project_root],
        code_paste=intent,
        max_files=15
    )

