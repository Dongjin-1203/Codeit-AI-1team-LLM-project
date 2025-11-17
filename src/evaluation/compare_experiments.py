# ===== compare_experiments.py =====
"""
실험 결과 비교 및 분석 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 경로 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from evaluation.experiment_tracker import ExperimentTracker


def main():
    """메인 실행"""
    tracker = ExperimentTracker()
    
    print("\n" + "="*80)
    print("🔍 실험 비교 도구")
    print("="*80)
    
    while True:
        print("\n메뉴:")
        print("  1. 모든 실험 목록 보기")
        print("  2. 최근 실험 비교 (최근 5개)")
        print("  3. 특정 실험 비교")
        print("  4. 개선 효과 확인")
        print("  5. 차트 생성")
        print("  6. 최적 설정 추천")
        print("  0. 종료")
        
        choice = input("\n선택: ").strip()
        
        if choice == "1":
            # 목록
            tracker.list_experiments()
        
        elif choice == "2":
            # 최근 비교
            tracker.compare_experiments(top_n=5)
        
        elif choice == "3":
            # 특정 실험 비교
            names = input("실험 이름들 (쉼표로 구분): ").strip()
            if names:
                experiment_names = [n.strip() for n in names.split(',')]
                tracker.compare_experiments(experiment_names=experiment_names)
        
        elif choice == "4":
            # 개선 효과
            baseline = input("Baseline 실험 이름: ").strip()
            current = input("비교할 실험 이름: ").strip()
            
            if baseline and current:
                tracker.show_improvement(baseline, current)
        
        elif choice == "5":
            # 차트
            names_input = input("실험 이름들 (쉼표로 구분, 엔터: 전체): ").strip()
            
            if names_input:
                experiment_names = [n.strip() for n in names_input.split(',')]
            else:
                experiment_names = None
            
            tracker.plot_metrics(experiment_names=experiment_names)
        
        elif choice == "6":
            # 최적 설정
            metric = input("기준 지표 (precision/recall/f1, 엔터: f1): ").strip()
            if not metric:
                metric = "f1"
            
            tracker.recommend_best(metric=metric)
        
        elif choice == "0":
            print("👋 종료합니다")
            break
        
        else:
            print("❌ 잘못된 선택입니다")


if __name__ == "__main__":
    main()