import numpy as np
from typing import List, Dict
from sklearn.metrics import ndcg_score

class RAGMetrics:
    """RAG 검색기 평가 지표 계산"""
    
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], 
                     relevant_ids: List[str], 
                     k: int = 5) -> float:
        """
        Recall@K: 상위 K개 중 관련 문서 재현율
        
        Args:
            retrieved_ids: 검색된 문서 ID 리스트 (순서대로)
            relevant_ids: 정답 문서 ID 리스트
            k: 상위 몇 개를 볼지
        
        Returns:
            0.0 ~ 1.0 사이 값
        """
        if not relevant_ids:
            return 0.0
        
        retrieved_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        hits = retrieved_k.intersection(relevant_set)
        recall = len(hits) / len(relevant_set)
        
        return recall
    
    @staticmethod
    def mrr(retrieved_ids: List[str], 
            relevant_ids: List[str]) -> float:
        """
        Mean Reciprocal Rank: 첫 정답 문서의 역순위
        
        Returns:
            0.0 ~ 1.0 (1위면 1.0, 2위면 0.5, 3위면 0.33...)
        """
        relevant_set = set(relevant_ids)
        
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_set:
                return 1.0 / rank
        
        return 0.0  # 정답을 못 찾음
    
    @staticmethod
    def hit_rate_at_k(retrieved_ids: List[str], 
                       relevant_ids: List[str], 
                       k: int = 5) -> float:
        """
        Hit Rate@K: 상위 K개 중 하나라도 있으면 성공
        
        Returns:
            0.0 (실패) or 1.0 (성공)
        """
        retrieved_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        return 1.0 if retrieved_k.intersection(relevant_set) else 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], 
                   relevant_ids: List[str], 
                   k: int = 5) -> float:
        """
        NDCG@K: 순위를 고려한 검색 품질
        
        Returns:
            0.0 ~ 1.0
        """
        # relevance scores: 관련 문서면 1, 아니면 0
        relevance = [1 if doc_id in relevant_ids else 0 
                     for doc_id in retrieved_ids[:k]]
        
        if sum(relevance) == 0:
            return 0.0
        
        # ideal ranking: 관련 문서를 모두 상위에
        ideal_relevance = sorted(relevance, reverse=True)
        
        # sklearn의 ndcg_score 사용
        # shape: (1, k)
        relevance_array = np.array([relevance])
        ideal_array = np.array([ideal_relevance])
        
        return ndcg_score(ideal_array, relevance_array)


class AggregateMetrics:
    """여러 쿼리에 대한 집계 메트릭"""
    
    # 평균 계산할 숫자 메트릭 키 목록
    NUMERIC_METRICS = {
        'recall@3', 'recall@5', 'recall@10',
        'mrr',
        'ndcg@3', 'ndcg@5', 'ndcg@10',
        'hit_rate@3', 'hit_rate@5', 'hit_rate@10',
        'keyword_match', 'max_relevance', 'avg_relevance',
        'answer_length'
    }
    
    @staticmethod
    def calculate_all(results: List[Dict]) -> Dict:
        """
        모든 쿼리의 평균 메트릭 계산
        
        Args:
            results: 각 쿼리별 메트릭 리스트
                [
                  {"recall@5": 0.8, "mrr": 1.0, ...},
                  {"recall@5": 0.6, "mrr": 0.5, ...},
                ]
        
        Returns:
            평균 메트릭
        """
        metrics = {}
        
        if not results:
            return metrics
        
        # 숫자 메트릭만 평균 계산
        for key in AggregateMetrics.NUMERIC_METRICS:
            # 해당 키가 있는 결과들만 필터링
            values = [r[key] for r in results if key in r and isinstance(r.get(key), (int, float))]
            
            if values:  # 값이 있을 때만 계산
                metrics[f"avg_{key}"] = float(np.mean(values))
                metrics[f"std_{key}"] = float(np.std(values))
        
        # top_scores의 경우 리스트이므로 평균 처리
        if 'top_scores' in results[0]:
            all_scores = []
            for r in results:
                scores = r.get('top_scores', [])
                if scores and isinstance(scores, list):
                    all_scores.extend([s for s in scores if isinstance(s, (int, float))])
            
            if all_scores:
                metrics['avg_all_scores'] = float(np.mean(all_scores))
        