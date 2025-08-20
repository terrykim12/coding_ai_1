import logging
import os
import re
import math
import glob
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 로거 설정 추가
logger = logging.getLogger(__name__)

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

def build_context(paths: List[str]) -> str:
    """컨텍스트를 빌드하는 함수"""
    context_parts = []
    
    for path_str in paths:
        try:
            file_path = Path(path_str)
            
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        context_parts.append(f"파일: {file_path}\n{content}\n{'='*50}\n")
                except Exception as e:
                    logger.warning(f"파일 읽기 실패 {file_path}: {e}")
                    continue
                    
            elif file_path.is_dir():
                # 디렉토리인 경우 재귀적으로 파일 탐색
                for py_file in file_path.rglob("*.py"):
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            context_parts.append(f"파일: {py_file}\n{content}\n{'='*50}\n")
                    except Exception as e:
                        logger.warning(f"파일 읽기 실패 {py_file}: {e}")
                        continue
            else:
                logger.warning(f"경로를 찾을 수 없음: {file_path}")
                
        except Exception as e:
            logger.error(f"경로 처리 실패 {path_str}: {e}")
            continue
    
    return "\n".join(context_parts) if context_parts else ""

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

