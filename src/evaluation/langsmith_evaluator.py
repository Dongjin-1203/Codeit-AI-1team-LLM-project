# ===== langsmith_evaluators.py (새 파일) =====

from typing import Dict, List, Any

# === 1. 유틸리티 함수: 문서 매칭 ===
def normalize_text(text: str) -> str:
    """
    텍스트 정규화 강화
    """
    import re
    
    # 소문자 변환
    normalized = text.lower()
    
    # 특수문자 제거 (\r, \n 등)
    normalized = re.sub(r'[\r\n\t]+', ' ', normalized)
    
    # 연속 공백 하나로
    normalized = ' '.join(normalized.split())
    
    # strip
    normalized = normalized.strip()
    
    return normalized


def is_matching_context(retrieved_text: str, ground_truth_text: str, threshold: float = 0.5) -> bool:
    """
    두 문서가 같은 문서인지 판단 (개선 버전)
    """
    # 정규화 강화
    normalized_retrieved = normalize_text(retrieved_text)
    normalized_truth = normalize_text(ground_truth_text)
    
    # 방법 1: 완전 포함 (빠른 체크)
    if normalized_truth in normalized_retrieved:
        return True
    
    if normalized_retrieved in normalized_truth:
        return True
    
    # 방법 2: 정답 단어 커버리지 (핵심 개선!)
    truth_words = set(normalized_truth.split())
    retrieved_words = set(normalized_retrieved.split())
    
    if len(truth_words) == 0:
        return False
    
    # 정답 단어 중 검색 결과에 포함된 비율
    matched_words = truth_words & retrieved_words
    coverage = len(matched_words) / len(truth_words)  # ✅ 정답 기준!
    
    return coverage >= threshold


def count_matching_contexts(
    retrieved_contexts: List[str],
    ground_truth_contexts: List[str],
    threshold: float = 0.5
) -> int:
    """
    매칭되는 문서 개수 계산
    
    Args:
        retrieved_contexts: 검색된 문서 리스트
        ground_truth_contexts: 정답 문서 리스트
        threshold: 유사도 임계값
        
    Returns:
        매칭 개수
    """
    matched_count = 0
    
    for retrieved in retrieved_contexts:
        for truth in ground_truth_contexts:
            if is_matching_context(retrieved, truth, threshold):
                matched_count += 1
                break  # 하나의 retrieved는 하나의 truth만 매칭
    
    return matched_count


# === 수정된 Evaluator ===

def context_precision_evaluator(run: Any, example: Any) -> Dict[str, float]:
    try:
        # 1. 검색 결과 추출 (수정!)
        # run.outputs가 dict이고 실제 결과는 'output' 키에 있음
        if isinstance(run.outputs, dict):
            retrieved_results = run.outputs.get('output', [])  # ✅ 수정!
        else:
            retrieved_results = run.outputs
        
        # 텍스트만 추출
        retrieved_contexts = []
        for result in retrieved_results:
            if isinstance(result, dict):
                text = result.get('content', '')
                if text:
                    retrieved_contexts.append(text)
        
        # 2. 정답 추출 (수정!)
        # example.outputs에 있음!
        ground_truth_contexts = example.outputs.get('ground_truth_contexts', [])  # ✅ 수정!
        
        # 3. 검증
        if len(retrieved_contexts) == 0:
            return {"key": "context_precision", "score": 0.0, "comment": "검색 결과 없음"}
        
        if len(ground_truth_contexts) == 0:
            return {"key": "context_precision", "score": 0.0, "comment": "정답 없음"}
        
        # 4. 매칭 개수 계산
        matched_count = count_matching_contexts(
            retrieved_contexts,
            ground_truth_contexts,
            threshold=0.5
        )
        
        # 5. Precision 계산
        precision = matched_count / len(retrieved_contexts)
        
        return {
            "key": "context_precision",
            "score": precision,
            "comment": f"매칭: {matched_count}/{len(retrieved_contexts)}"
        }
        
    except Exception as e:
        print(f"Context Precision 계산 오류: {e}")
        import traceback
        traceback.print_exc()
        return {"key": "context_precision", "score": 0.0, "comment": f"오류: {str(e)}"}


def context_recall_evaluator(run: Any, example: Any) -> Dict[str, float]:
    try:
        # 1. 검색 결과 추출 (수정!)
        if isinstance(run.outputs, dict):
            retrieved_results = run.outputs.get('output', [])  # ✅ 수정!
        else:
            retrieved_results = run.outputs
        
        retrieved_contexts = []
        for result in retrieved_results:
            if isinstance(result, dict):
                text = result.get('content', '')
                if text:
                    retrieved_contexts.append(text)
        
        # 2. 정답 추출 (수정!)
        ground_truth_contexts = example.outputs.get('ground_truth_contexts', [])  # ✅ 수정!
        
        # 3. 검증
        if len(ground_truth_contexts) == 0:
            return {"key": "context_recall", "score": 0.0, "comment": "정답 없음"}
        
        if len(retrieved_contexts) == 0:
            return {"key": "context_recall", "score": 0.0, "comment": "검색 결과 없음"}
        
        # 4. 매칭 개수 계산
        matched_count = 0
        for truth in ground_truth_contexts:
            for retrieved in retrieved_contexts:
                if is_matching_context(retrieved, truth, threshold=0.5):
                    matched_count += 1
                    break
        
        # 5. Recall 계산
        recall = matched_count / len(ground_truth_contexts)
        
        return {
            "key": "context_recall",
            "score": recall,
            "comment": f"발견: {matched_count}/{len(ground_truth_contexts)}"
        }
        
    except Exception as e:
        print(f"Context Recall 계산 오류: {e}")
        import traceback
        traceback.print_exc()
        return {"key": "context_recall", "score": 0.0, "comment": f"오류: {str(e)}"}


# === 4. (선택) 검색 시간 Evaluator ===
def retrieval_time_evaluator(run: Any, example: Any) -> Dict[str, float]:
    """
    검색 시간 측정 (성능 지표)
    
    Returns:
        {"key": "retrieval_time", "score": 시간(초)}
    """
    try:
        # LangSmith run에서 실행 시간 추출
        latency = run.execution_time  # 초 단위
        
        return {
            "key": "retrieval_time",
            "score": latency,
            "comment": f"{latency:.3f}초"
        }
        
    except Exception as e:
        return {"key": "retrieval_time", "score": 0.0, "comment": "시간 측정 실패"}