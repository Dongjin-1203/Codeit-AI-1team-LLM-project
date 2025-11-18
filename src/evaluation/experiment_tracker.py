# ===== experiment_tracker.py =====
"""
RAG 검색 시스템 실험 추적 및 비교 도구

기능:
1. 실험 결과 자동 저장
2. 이전 실험과 비교
3. 성능 차트 생성
4. 최적 설정 추천
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 서버 환경 대응


class ExperimentTracker:
    """실험 추적 및 비교 클래스"""
    
    def __init__(self, log_dir: str = "src/evaluation/results/experiments"):
        """
        Args:
            log_dir: 실험 로그 저장 디렉토리
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / "experiments_log.json"
        self.summary_file = self.log_dir / "experiments_summary.csv"
        
        # 로그 파일 초기화
        if not self.log_file.exists():
            self._save_log([])
    
    
    # === 1. 실험 결과 저장 ===
    
    def log_experiment(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        langsmith_url: Optional[str] = None,
        notes: str = ""
    ) -> None:
        """
        실험 결과 저장
        
        Args:
            experiment_name: 실험 이름 (예: "baseline", "embedding-small")
            config: 설정 정보 (임베딩 모델, Top-K 등)
            metrics: 평가 지표 (precision, recall 등)
            langsmith_url: LangSmith 결과 URL
            notes: 추가 메모
        """
        # 실험 데이터 구성
        experiment_data = {
            "timestamp": datetime.now().isoformat(),
            "experiment_name": experiment_name,
            "config": config,
            "metrics": metrics,
            "langsmith_url": langsmith_url,
            "notes": notes
        }
        
        # 기존 로그 로드
        logs = self._load_log()
        
        # 새 실험 추가
        logs.append(experiment_data)
        
        # 저장
        self._save_log(logs)
        self._update_summary()
        
        print(f"✅ 실험 '{experiment_name}' 저장 완료")
        print(f"   Precision: {metrics.get('precision', 0):.4f}")
        print(f"   Recall: {metrics.get('recall', 0):.4f}")
    
    
    # === 2. 실험 비교 ===
    
    def compare_experiments(
        self,
        experiment_names: Optional[List[str]] = None,
        top_n: int = 5
    ) -> pd.DataFrame:
        """
        실험 결과 비교
        
        Args:
            experiment_names: 비교할 실험 이름 리스트 (None이면 최근 실험)
            top_n: experiment_names가 None일 때 최근 몇 개 비교할지
            
        Returns:
            비교 결과 DataFrame
        """
        logs = self._load_log()
        
        if not logs:
            print("⚠️ 저장된 실험이 없습니다")
            return pd.DataFrame()
        
        # 비교할 실험 선택
        if experiment_names is None:
            # 최근 N개
            selected_logs = logs[-top_n:]
        else:
            # 지정된 실험들
            selected_logs = [
                log for log in logs 
                if log['experiment_name'] in experiment_names
            ]
        
        if not selected_logs:
            print("⚠️ 비교할 실험을 찾을 수 없습니다")
            return pd.DataFrame()
        
        # DataFrame 생성
        comparison_data = []
        for log in selected_logs:
            row = {
                "실험명": log['experiment_name'],
                "날짜": log['timestamp'][:10],
                "임베딩": log['config'].get('embedding_model', 'N/A'),
                "Top-K": log['config'].get('top_k', 'N/A'),
                "Precision": log['metrics'].get('precision', 0),
                "Recall": log['metrics'].get('recall', 0),
                "F1": self._calculate_f1(
                    log['metrics'].get('precision', 0),
                    log['metrics'].get('recall', 0)
                ),
                "검색시간(초)": log['metrics'].get('avg_time', 0)
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # 출력
        print("\n" + "="*80)
        print("📊 실험 비교 결과")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        return df
    
    
    def show_improvement(self, baseline_name: str, current_name: str) -> None:
        """
        Baseline 대비 개선 효과 출력
        
        Args:
            baseline_name: 기준 실험 이름
            current_name: 비교할 실험 이름
        """
        logs = self._load_log()
        
        # 실험 찾기
        baseline = next((log for log in logs if log['experiment_name'] == baseline_name), None)
        current = next((log for log in logs if log['experiment_name'] == current_name), None)
        
        if not baseline or not current:
            print("⚠️ 실험을 찾을 수 없습니다")
            return
        
        # 개선율 계산
        baseline_precision = baseline['metrics'].get('precision', 0)
        baseline_recall = baseline['metrics'].get('recall', 0)
        
        current_precision = current['metrics'].get('precision', 0)
        current_recall = current['metrics'].get('recall', 0)
        
        precision_improvement = (current_precision - baseline_precision) / baseline_precision * 100 if baseline_precision > 0 else 0
        recall_improvement = (current_recall - baseline_recall) / baseline_recall * 100 if baseline_recall > 0 else 0
        
        # 출력
        print("\n" + "="*80)
        print(f"📈 개선 효과: {baseline_name} → {current_name}")
        print("="*80)
        print(f"\nPrecision:")
        print(f"  {baseline_name}: {baseline_precision:.4f}")
        print(f"  {current_name}: {current_precision:.4f}")
        print(f"  개선율: {precision_improvement:+.2f}% {'✅' if precision_improvement > 0 else '❌'}")
        
        print(f"\nRecall:")
        print(f"  {baseline_name}: {baseline_recall:.4f}")
        print(f"  {current_name}: {current_recall:.4f}")
        print(f"  개선율: {recall_improvement:+.2f}% {'✅' if recall_improvement > 0 else '❌'}")
        
        print("\n" + "="*80)
    
    
    # === 3. 시각화 ===
    
    def plot_metrics(
        self,
        experiment_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        실험 결과 차트 생성
        
        Args:
            experiment_names: 차트에 포함할 실험 (None이면 전체)
            save_path: 차트 저장 경로 (None이면 화면 출력)
        """
        logs = self._load_log()
        
        if not logs:
            print("⚠️ 저장된 실험이 없습니다")
            return
        
        # 실험 선택
        if experiment_names is not None:
            logs = [log for log in logs if log['experiment_name'] in experiment_names]
        
        if not logs:
            print("⚠️ 차트를 그릴 실험이 없습니다")
            return
        
        # 데이터 준비
        names = [log['experiment_name'] for log in logs]
        precisions = [log['metrics'].get('precision', 0) for log in logs]
        recalls = [log['metrics'].get('recall', 0) for log in logs]
        
        # 차트 생성
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(names))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], precisions, width, label='Precision', alpha=0.8)
        ax.bar([i + width/2 for i in x], recalls, width, label='Recall', alpha=0.8)
        
        ax.set_xlabel('실험')
        ax.set_ylabel('점수')
        ax.set_title('실험별 성능 비교')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # 저장 또는 출력
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 차트 저장: {save_path}")
        else:
            default_path = self.log_dir / "comparison_chart.png"
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            print(f"✅ 차트 저장: {default_path}")
        
        plt.close()
    
    
    # === 4. 최적 설정 추천 ===
    
    def recommend_best(self, metric: str = "f1") -> Dict[str, Any]:
        """
        최적 설정 추천
        
        Args:
            metric: 기준 지표 ("precision", "recall", "f1")
            
        Returns:
            최적 실험 정보
        """
        logs = self._load_log()
        
        if not logs:
            print("⚠️ 저장된 실험이 없습니다")
            return {}
        
        # F1 점수 계산
        for log in logs:
            if 'f1' not in log['metrics']:
                p = log['metrics'].get('precision', 0)
                r = log['metrics'].get('recall', 0)
                log['metrics']['f1'] = self._calculate_f1(p, r)
        
        # 최적 실험 찾기
        best = max(logs, key=lambda x: x['metrics'].get(metric, 0))
        
        print("\n" + "="*80)
        print(f"🏆 최적 설정 ({metric.upper()} 기준)")
        print("="*80)
        print(f"실험명: {best['experiment_name']}")
        print(f"날짜: {best['timestamp'][:10]}")
        print(f"\n설정:")
        for key, value in best['config'].items():
            print(f"  {key}: {value}")
        print(f"\n성능:")
        print(f"  Precision: {best['metrics'].get('precision', 0):.4f}")
        print(f"  Recall: {best['metrics'].get('recall', 0):.4f}")
        print(f"  F1: {best['metrics'].get('f1', 0):.4f}")
        print("="*80)
        
        return best
    
    
    # === 5. 유틸리티 ===
    
    def list_experiments(self) -> None:
        """저장된 실험 목록 출력"""
        logs = self._load_log()
        
        if not logs:
            print("⚠️ 저장된 실험이 없습니다")
            return
        
        print("\n" + "="*80)
        print("📋 저장된 실험 목록")
        print("="*80)
        
        for i, log in enumerate(logs, 1):
            print(f"\n{i}. {log['experiment_name']}")
            print(f"   날짜: {log['timestamp'][:10]}")
            print(f"   Precision: {log['metrics'].get('precision', 0):.4f}")
            print(f"   Recall: {log['metrics'].get('recall', 0):.4f}")
        
        print("="*80)
    
    
    def clear_experiments(self) -> None:
        """모든 실험 로그 삭제 (주의!)"""
        confirm = input("⚠️ 모든 실험 로그를 삭제하시겠습니까? (yes/no): ")
        if confirm.lower() == 'yes':
            self._save_log([])
            self._update_summary()
            print("✅ 모든 실험 로그 삭제 완료")
        else:
            print("❌ 취소됨")
    
    
    # === 내부 함수 ===
    
    def _load_log(self) -> List[Dict]:
        """로그 파일 로드"""
        if not self.log_file.exists():
            return []
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    
    def _save_log(self, logs: List[Dict]) -> None:
        """로그 파일 저장"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    
    
    def _update_summary(self) -> None:
        """요약 CSV 업데이트"""
        logs = self._load_log()
        
        if not logs:
            return
        
        summary_data = []
        for log in logs:
            row = {
                "timestamp": log['timestamp'],
                "experiment_name": log['experiment_name'],
                "embedding_model": log['config'].get('embedding_model', 'N/A'),
                "top_k": log['config'].get('top_k', 'N/A'),
                "precision": log['metrics'].get('precision', 0),
                "recall": log['metrics'].get('recall', 0),
                "f1": self._calculate_f1(
                    log['metrics'].get('precision', 0),
                    log['metrics'].get('recall', 0)
                ),
                "avg_time": log['metrics'].get('avg_time', 0)
            }
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df.to_csv(self.summary_file, index=False, encoding='utf-8-sig')
    
    
    @staticmethod
    def _calculate_f1(precision: float, recall: float) -> float:
        """F1 점수 계산"""
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)


# ===== 사용 예시 =====

if __name__ == "__main__":
    # Tracker 초기화
    tracker = ExperimentTracker()
    
    # 예시 1: 실험 결과 저장
    tracker.log_experiment(
        experiment_name="baseline",
        config={
            "embedding_model": "text-embedding-3-small",
            "top_k": 5,
            "chunk_size": 1000
        },
        metrics={
            "precision": 0.30,
            "recall": 0.65,
            "avg_time": 0.41
        },
        notes="초기 baseline 실험"
    )
    
    # 예시 2: 실험 비교
    tracker.compare_experiments()
    
    # 예시 3: 개선 효과 확인
    # tracker.show_improvement("baseline", "embedding-small")
    
    # 예시 4: 차트 생성
    # tracker.plot_metrics()
    
    # 예시 5: 최적 설정 추천
    # tracker.recommend_best(metric="f1")