"""
RAG 검색기 평가 모듈 (에러 처리 강화)

test_queries_mapped.json을 사용하여 표준 평가 지표로 성능 측정
Retriever 또는 Pipeline 모두 지원
"""

import json
import wandb
from datetime import datetime
from typing import Dict, List, Optional, Union
from pathlib import Path
from tqdm import tqdm
import traceback

from src.evaluation.metrics import RAGMetrics, AggregateMetrics


class RAGEvaluator:
    """RAG 시스템 평가 (에러 처리 강화 버전)"""

    def __init__(self, 
                 pipeline=None,
                 retriever=None,
                 test_data_path: str = "test_queries_mapped.json",
                 use_wandb: bool = False):
        """
        Args:
            pipeline: RAG 파이프라인 인스턴스 (선택)
            retriever: RAG 검색기 인스턴스 (선택)
            test_data_path: 테스트 쿼리 JSON 파일 경로
            use_wandb: WandB 로깅 사용 여부
        """
        if pipeline is None and retriever is None:
            raise ValueError("pipeline 또는 retriever 중 하나는 반드시 제공해야 합니다.")
        
        self.pipeline = pipeline
        self.retriever = retriever
        self.test_data_path = test_data_path
        self.use_wandb = use_wandb
        
        # 테스트 데이터 로드
        with open(test_data_path, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
        
        print(f"✅ 테스트 데이터 로드: {len(self.test_data['test_cases'])}개 쿼리")
        
        # WandB 초기화 (필요시)
        if self.use_wandb:
            self._init_wandb()
    
    def _init_wandb(self):
        """WandB 초기화"""
        wandb.init(
            project="rag-document-summarization",
            name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "test_queries": len(self.test_data['test_cases']),
                "mode": "pipeline" if self.pipeline else "retriever"
            }
        )
    
    def evaluate_single_query(self, test_case: Dict, k: int = 5) -> Optional[Dict]:
        """
        단일 쿼리 평가 (에러 처리 포함)
        
        Args:
            test_case: 테스트 케이스 (query, ground_truth 포함)
            k: 상위 K개 검색 결과 평가
        
        Returns:
            평가 결과 딕셔너리 또는 None (에러 발생시)
        """
        try:
            query = test_case['query']
            relevant_ids = test_case['ground_truth']['relevant_doc_ids']
            
            # Pipeline 또는 Retriever 실행
            if self.pipeline:
                # Pipeline 사용
                rag_result = self.pipeline.generate_answer(query)
                sources = rag_result.get('sources', [])
                answer = rag_result.get('answer', '')
            else:
                # Retriever 사용
                sources = self.retriever.search(query, top_k=10)
                answer = ""
            
            # Sources에서 문서 ID 추출
            retrieved_ids = []
            scores = []
            
            for source in sources:
                # 다양한 ID 필드명 지원
                doc_id = (source.get('id') or 
                         source.get('chunk_id') or 
                         source.get('doc_id') or
                         source.get('_id'))
                
                if doc_id:
                    retrieved_ids.append(doc_id)
                    
                    # 다양한 score 필드명 지원
                    score = (source.get('relevance_score') or 
                            source.get('score') or
                            source.get('similarity') or
                            (1 - source.get('distance', 1)))
                    
                    scores.append(float(score) if score is not None else 0.0)
            
            # 평가 지표 계산
            metrics = RAGMetrics()
            
            result = {
                "query_id": test_case['id'],
                "query": query,
                "category": test_case.get('category', 'unknown'),
                "difficulty": test_case.get('difficulty', 'unknown'),
                "retrieved_ids": retrieved_ids[:k],
                "relevant_ids": relevant_ids,
                
                # 표준 메트릭
                "recall@3": metrics.recall_at_k(retrieved_ids, relevant_ids, k=3),
                "recall@5": metrics.recall_at_k(retrieved_ids, relevant_ids, k=5),
                "recall@10": metrics.recall_at_k(retrieved_ids, relevant_ids, k=10),
                "mrr": metrics.mrr(retrieved_ids, relevant_ids),
                "ndcg@3": metrics.ndcg_at_k(retrieved_ids, relevant_ids, k=3),
                "ndcg@5": metrics.ndcg_at_k(retrieved_ids, relevant_ids, k=5),
                "ndcg@10": metrics.ndcg_at_k(retrieved_ids, relevant_ids, k=10),
                "hit_rate@3": metrics.hit_rate_at_k(retrieved_ids, relevant_ids, k=3),
                "hit_rate@5": metrics.hit_rate_at_k(retrieved_ids, relevant_ids, k=5),
                "hit_rate@10": metrics.hit_rate_at_k(retrieved_ids, relevant_ids, k=10),
                
                # 추가 정보
                "top_scores": scores[:k],
                "answer_length": len(answer),
                
                # 레거시 메트릭
                "keyword_match": self._calculate_keyword_match(
                    answer,
                    test_case['ground_truth'].get('expected_keywords', [])
                ),
                "max_relevance": scores[0] if scores else 0.0,
                "avg_relevance": sum(scores[:2]) / min(2, len(scores)) if scores else 0.0,
            }
            
            return result
            
        except Exception as e:
            print(f"\n⚠️ 쿼리 평가 실패 (ID: {test_case.get('id', 'unknown')})")
            print(f"   쿼리: {test_case.get('query', 'unknown')[:50]}...")
            print(f"   에러: {str(e)}")
            traceback.print_exc()
            return None
    
    def _calculate_keyword_match(self, answer: str, keywords: List[str]) -> float:
        """키워드 매칭률 계산"""
        if not keywords or not answer:
            return 0.0
        
        matches = sum(1 for kw in keywords if kw in answer)
        return matches / len(keywords)
    
    def evaluate_all(self, save_results: bool = True) -> Dict:
        """
        전체 테스트 쿼리 평가 (에러 처리 포함)
        
        Returns:
            {
                "summary": {...},
                "details": [...],
                "errors": [...]
            }
        """
        all_results = []
        errors = []
        
        print("\n" + "="*80)
        print("RAG 검색기 평가 시작")
        print(f"총 {len(self.test_data['test_cases'])}개 쿼리")
        print(f"모드: {'Pipeline' if self.pipeline else 'Retriever'}")
        print("="*80 + "\n")
        
        # 각 쿼리 평가
        for test_case in tqdm(self.test_data['test_cases'], desc="평가 진행"):
            result = self.evaluate_single_query(test_case)
            
            if result is not None:
                all_results.append(result)
                
                # WandB 로깅
                if self.use_wandb:
                    self._log_to_wandb(result)
            else:
                errors.append({
                    "query_id": test_case.get('id', 'unknown'),
                    "query": test_case.get('query', 'unknown')
                })
        
        # 결과 확인
        if not all_results:
            print("\n❌ 오류: 성공적으로 평가된 쿼리가 없습니다.")
            print("\n문제 해결:")
            print("1. Pipeline/Retriever가 올바르게 초기화되었는지 확인")
            print("2. Sources 구조 확인 (id, relevance_score 필드)")
            print("3. 위 에러 메시지 확인")
            
            return {
                "timestamp": datetime.now().isoformat(),
                "mode": "pipeline" if self.pipeline else "retriever",
                "summary": {},
                "details": [],
                "errors": errors
            }
        
        # 집계 메트릭 계산
        summary = AggregateMetrics.calculate_all(all_results)
        
        # summary가 None이면 빈 dict로 초기화
        if summary is None:
            summary = {}
        
        # 난이도별, 카테고리별 분석
        summary['by_difficulty'] = self._group_by(all_results, 'difficulty')
        summary['by_category'] = self._group_by(all_results, 'category')
        
        # 결과 출력
        self._print_summary(summary, all_results, errors)
        
        # 결과 저장
        eval_result = {
            "timestamp": datetime.now().isoformat(),
            "mode": "pipeline" if self.pipeline else "retriever",
            "summary": summary,
            "details": all_results,
            "errors": errors,
            "success_rate": len(all_results) / len(self.test_data['test_cases'])
        }
        
        if save_results:
            self._save_results(eval_result)
        
        # WandB summary 업데이트
        if self.use_wandb:
            wandb.summary.update(summary)
            wandb.summary.update({"success_rate": eval_result['success_rate']})
            wandb.finish()
        
        return eval_result
    
    def _group_by(self, results: List[Dict], key: str) -> Dict:
        """난이도별/카테고리별 그룹화"""
        groups = {}
        
        for result in results:
            group_name = result.get(key, 'unknown')
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(result)
        
        # 각 그룹별 평균 계산
        group_stats = {}
        for group_name, group_results in groups.items():
            group_stats[group_name] = AggregateMetrics.calculate_all(group_results)
        
        return group_stats
    
    def _log_to_wandb(self, result: Dict):
        """WandB 로깅"""
        try:
            wandb.log({
                "recall@5": result['recall@5'],
                "mrr": result['mrr'],
                "ndcg@5": result['ndcg@5'],
                "hit_rate@5": result['hit_rate@5'],
                "keyword_match": result['keyword_match'],
                "max_relevance": result['max_relevance'],
            })
        except Exception as e:
            print(f"WandB 로깅 실패: {e}")
    
    def _print_summary(self, summary: Dict, all_results: List[Dict], errors: List[Dict]):
        """결과 출력"""
        print("\n" + "="*80)
        print("📊 평가 결과 요약")
        print("="*80 + "\n")
        
        # 성공률
        total = len(all_results) + len(errors)
        success_rate = len(all_results) / total if total > 0 else 0
        print(f"평가 성공률: {success_rate:.1%} ({len(all_results)}/{total})")
        
        if errors:
            print(f"⚠️ 실패한 쿼리: {len(errors)}개")
        
        print("\n전체 평균 메트릭:")
        print(f"  Recall@3:    {summary.get('avg_recall@3', 0):.3f} (±{summary.get('std_recall@3', 0):.3f})")
        print(f"  Recall@5:    {summary.get('avg_recall@5', 0):.3f} (±{summary.get('std_recall@5', 0):.3f})")
        print(f"  Recall@10:   {summary.get('avg_recall@10', 0):.3f} (±{summary.get('std_recall@10', 0):.3f})")
        print(f"  MRR:         {summary.get('avg_mrr', 0):.3f} (±{summary.get('std_mrr', 0):.3f})")
        print(f"  NDCG@5:      {summary.get('avg_ndcg@5', 0):.3f} (±{summary.get('std_ndcg@5', 0):.3f})")
        print(f"  Hit Rate@5:  {summary.get('avg_hit_rate@5', 0):.3f} (±{summary.get('std_hit_rate@5', 0):.3f})")
        
        print("\n레거시 메트릭 (참고):")
        print(f"  키워드 매칭:  {summary.get('avg_keyword_match', 0):.3f}")
        print(f"  최고 관련도:  {summary.get('avg_max_relevance', 0):.3f}")
        
        if summary.get('by_difficulty'):
            print("\n난이도별 성능:")
            for difficulty, stats in summary.get('by_difficulty', {}).items():
                print(f"\n  [{difficulty}]")
                print(f"    Recall@5: {stats.get('avg_recall@5', 0):.3f}")
                print(f"    MRR:      {stats.get('avg_mrr', 0):.3f}")
                print(f"    NDCG@5:   {stats.get('avg_ndcg@5', 0):.3f}")
        
        if summary.get('by_category'):
            print("\n카테고리별 성능:")
            for category, stats in summary.get('by_category', {}).items():
                print(f"\n  [{category}]")
                print(f"    Recall@5: {stats.get('avg_recall@5', 0):.3f}")
                print(f"    MRR:      {stats.get('avg_mrr', 0):.3f}")
        
        # 실패 케이스 분석
        failed_queries = [r for r in all_results if r.get('hit_rate@5', 0) == 0]
        
        if failed_queries:
            print(f"\n⚠️ 검색 실패 쿼리: {len(failed_queries)}개")
            print("\n상위 5개:")
            for i, fail in enumerate(failed_queries[:5], 1):
                print(f"  {i}. [{fail['category']}] {fail['query']}")
        
        # 종합 평가
        print("\n🎯 종합 평가:")
        recall_5 = summary.get('avg_recall@5', 0)
        mrr = summary.get('avg_mrr', 0)
        
        if recall_5 >= 0.80 and mrr >= 0.70:
            print("  ✅ 우수 - 프로덕션 준비 완료")
        elif recall_5 >= 0.60 and mrr >= 0.50:
            print("  ⚠️  양호 - 추가 개선 권장")
        else:
            print("  ❌ 미흡 - 재검토 필요")
            if success_rate < 1.0:
                print(f"  💡 평가 성공률이 {success_rate:.1%}입니다. 에러 확인 필요")
    
    def _save_results(self, results: Dict):
        """결과 JSON 파일로 저장"""
        output_dir = Path("evaluation/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = output_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 결과 저장: {filename}")