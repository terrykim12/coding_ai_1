#!/usr/bin/env python3
"""
Git 커밋 히스토리에서 (문맥, 패치) 학습 데이터 추출

사용법:
    python build_dataset_from_git.py [repo_path] [output_file]

예시:
    python build_dataset_from_git.py . training/data/train.jsonl
    python build_dataset_from_git.py /path/to/repo training/data/val.jsonl
"""

import os
import sys
import json
import difflib
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from pydriller import Repository
    PYDRILLER_AVAILABLE = True
except ImportError:
    PYDRILLER_AVAILABLE = False
    print("Warning: pydriller not available. Install with: pip install pydriller")

def extract_code_chunks(source_code: str, max_chunk_size: int = 1000) -> List[str]:
    """소스 코드를 청크로 분할"""
    if not source_code:
        return []
    
    lines = source_code.split('\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for line in lines:
        line_size = len(line) + 1  # +1 for newline
        
        if current_size + line_size > max_chunk_size and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_size = line_size
        else:
            current_chunk.append(line)
            current_size += line_size
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

def create_diff_patch(before: str, after: str) -> str:
    """unified diff 형식의 패치 생성"""
    if not before and not after:
        return ""
    
    before_lines = before.splitlines() if before else []
    after_lines = after.splitlines() if after else []
    
    diff = list(difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile='before',
        tofile='after',
        lineterm=''
    ))
    
    return '\n'.join(diff)

def extract_commit_context(commit) -> Dict[str, Any]:
    """커밋에서 컨텍스트 정보 추출"""
    context = {
        "hash": commit.hash,
        "author": commit.author.name,
        "date": commit.author_date.isoformat(),
        "message": commit.msg,
        "files_changed": len(commit.modifications),
        "insertions": commit.insertions,
        "deletions": commit.deletions
    }
    
    # 커밋 메시지에서 태그 추출
    message = commit.msg.lower()
    if any(tag in message for tag in ['fix', 'bug', 'error', 'issue']):
        context["type"] = "bug_fix"
    elif any(tag in message for tag in ['feat', 'add', 'implement', 'new']):
        context["type"] = "feature_add"
    elif any(tag in message for tag in ['refactor', 'clean', 'improve']):
        context["type"] = "refactor"
    else:
        context["type"] = "other"
    
    return context

def create_training_record(
    file_path: str,
    before_code: str,
    after_code: str,
    commit_context: Dict[str, Any],
    chunk_size: int = 1000
) -> List[Dict[str, Any]]:
    """학습 레코드 생성"""
    records = []
    
    # 코드를 청크로 분할
    before_chunks = extract_code_chunks(before_code, chunk_size)
    after_chunks = extract_code_chunks(after_code, chunk_size)
    
    # 각 청크에 대해 레코드 생성
    for i, (before_chunk, after_chunk) in enumerate(zip(before_chunks, after_chunks)):
        if before_chunk == after_chunk:
            continue  # 변경이 없는 청크는 건너뛰기
        
        # 변경 사항 설명 생성
        if commit_context["type"] == "bug_fix":
            instruction = f"Fix the bug in {os.path.basename(file_path)}. The code has issues that need to be corrected."
        elif commit_context["type"] == "feature_add":
            instruction = f"Implement the requested feature in {os.path.basename(file_path)}. Add the new functionality as shown in the diff."
        else:
            instruction = f"Apply the code changes to {os.path.basename(file_path)} as specified in the commit message."
        
        # 입력 컨텍스트 구성
        input_context = f"""Commit: {commit_context['message']}
File: {file_path}
Author: {commit_context['author']}
Date: {commit_context['date']}

Original code:
{before_chunk}

Changes needed:
{create_diff_patch(before_chunk, after_chunk)}"""
        
        # Patch JSON 생성
        patch_json = {
            "version": "1",
            "edits": [
                {
                    "path": file_path,
                    "loc": {
                        "type": "anchor",
                        "before": before_chunk[:100] if before_chunk else "",
                        "after": after_chunk[:100] if after_chunk else ""
                    },
                    "action": "replace_range",
                    "code": after_chunk
                }
            ]
        }
        
        record = {
            "instruction": instruction,
            "input": input_context,
            "output": patch_json,
            "metadata": {
                "commit_hash": commit_context["hash"],
                "file_path": file_path,
                "chunk_index": i,
                "commit_type": commit_context["type"],
                "lines_changed": len(after_chunk.split('\n')) - len(before_chunk.split('\n'))
            }
        }
        
        records.append(record)
    
    return records

def process_repository(
    repo_path: str,
    output_file: str,
    max_commits: Optional[int] = None,
    file_extensions: Optional[List[str]] = None,
    min_file_size: int = 50,
    max_file_size: int = 50000
) -> int:
    """저장소 처리 및 데이터셋 생성"""
    
    if not PYDRILLER_AVAILABLE:
        print("Error: pydriller is required but not available")
        return 0
    
    if file_extensions is None:
        file_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs']
    
    print(f"Processing repository: {repo_path}")
    print(f"Output file: {output_file}")
    print(f"File extensions: {file_extensions}")
    
    # 출력 디렉토리 생성
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    all_records = []
    commit_count = 0
    
    try:
        # 저장소 순회
        for commit in Repository(repo_path).traverse_commits():
            if max_commits and commit_count >= max_commits:
                break
            
            commit_count += 1
            print(f"Processing commit {commit_count}: {commit.hash[:8]} - {commit.msg[:50]}...")
            
            commit_context = extract_commit_context(commit)
            
            # 파일 수정사항 처리
            for modification in commit.modifications:
                if not modification.new_path:
                    continue
                
                # 파일 확장자 확인
                if not any(modification.new_path.endswith(ext) for ext in file_extensions):
                    continue
                
                # 파일 크기 확인
                if modification.source_code:
                    code_size = len(modification.source_code)
                    if code_size < min_file_size or code_size > max_file_size:
                        continue
                
                # 변경 전후 코드 확인
                before_code = modification.source_code_before or ""
                after_code = modification.source_code or ""
                
                if before_code == after_code:
                    continue  # 변경이 없는 경우
                
                # 학습 레코드 생성
                records = create_training_record(
                    modification.new_path,
                    before_code,
                    after_code,
                    commit_context
                )
                
                all_records.extend(records)
                
                if commit_count % 10 == 0:
                    print(f"  Generated {len(all_records)} records so far...")
        
        # JSONL 파일로 저장
        print(f"Saving {len(all_records)} records to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in all_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"Successfully created dataset with {len(all_records)} records")
        print(f"Processed {commit_count} commits")
        
        return len(all_records)
        
    except Exception as e:
        print(f"Error processing repository: {str(e)}")
        return 0

def main():
    parser = argparse.ArgumentParser(
        description="Git 저장소에서 학습 데이터셋 생성"
    )
    parser.add_argument(
        "repo_path",
        help="Git 저장소 경로 (기본값: 현재 디렉토리)",
        nargs='?',
        default='.'
    )
    parser.add_argument(
        "output_file",
        help="출력 JSONL 파일 경로",
        nargs='?',
        default='training/data/train.jsonl'
    )
    parser.add_argument(
        "--max-commits",
        type=int,
        help="처리할 최대 커밋 수"
    )
    parser.add_argument(
        "--extensions",
        nargs='+',
        default=['.py', '.js', '.ts', '.java', '.cpp', '.c'],
        help="처리할 파일 확장자 목록"
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=50,
        help="최소 파일 크기 (바이트)"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=50000,
        help="최대 파일 크기 (바이트)"
    )
    
    args = parser.parse_args()
    
    # 저장소 경로 확인
    repo_path = os.path.abspath(args.repo_path)
    if not os.path.exists(os.path.join(repo_path, '.git')):
        print(f"Error: {repo_path} is not a Git repository")
        sys.exit(1)
    
    # 데이터셋 생성
    record_count = process_repository(
        repo_path=repo_path,
        output_file=args.output_file,
        max_commits=args.max_commits,
        file_extensions=args.extensions,
        min_file_size=args.min_size,
        max_file_size=args.max_size
    )
    
    if record_count > 0:
        print(f"\nDataset creation completed successfully!")
        print(f"Total records: {record_count}")
        print(f"Output file: {args.output_file}")
    else:
        print("\nNo records were generated. Check the repository and parameters.")
        sys.exit(1)

if __name__ == "__main__":
    main()

