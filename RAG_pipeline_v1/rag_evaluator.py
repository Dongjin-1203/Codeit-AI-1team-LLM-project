from rag_pipeline import RAGPipeline


class RAGEvaluator:
    """RAG 시스템 평가"""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def create_test_set(self):
        """평가용 테스트 데이터셋"""
        return [
            {
                'query': '한영대학교의 특성화 교육환경 구축 사업은?',
                'expected_keywords': ['한영대학', '트랙운영', '학사정보', '특성화'],
                'expected_organization': '한영대학'
            },
            {
                'query': '재난 안전 관리 시스템은?',
                'expected_keywords': ['재난', '안전', '관리', '시스템'],
            },
            {
                'query': '서울시립대학교의 학업성취도 시스템은?',
                'expected_keywords': ['서울시립대', '학업성취도', '분석'],
                'expected_organization': '서울시립대학교'
            },
            {
                'query': '인천광역시 도시계획위원회 시스템은?',
                'expected_keywords': ['인천', '도시계획', '위원회'],
            },
            {
                'query': '고려대학교 포털 학사 정보시스템은?',
                'expected_keywords': ['고려대', '포털', '학사', '정보시스템'],
            }
        ]

    def evaluate(self):
        """평가 실행"""
        test_set = self.create_test_set()
        results = []

        print("="*60)
        print("RAG 시스템 평가")
        print("="*60)

        for i, test in enumerate(test_set, 1):
            print(f"\n[{i}/{len(test_set)}] {test['query']}")

            # RAG 실행
            result = self.pipeline.generate_answer(test['query'])

            # 평가 지표 계산
            answer = result['answer']
            sources = result['sources']

            # 1. 키워드 매칭률
            keyword_match = sum(
                1 for kw in test['expected_keywords']
                if kw in answer
            ) / len(test['expected_keywords'])

            # 2. 상위 2개 평균 관련도 (수정됨)
            top_sources = sources[:2]
            avg_relevance = sum(s['relevance_score'] for s in top_sources) / len(top_sources)

            # 3. 최고 관련도
            max_relevance = sources[0]['relevance_score']

            # 4. 발주기관 정확도
            org_correct = None
            if 'expected_organization' in test:
                org_correct = any(
                    test['expected_organization'] in s['organization']
                    for s in sources[:3]
                )

            eval_result = {
                'query': test['query'],
                'keyword_match': keyword_match,
                'avg_relevance': avg_relevance,
                'max_relevance': max_relevance,
                'org_correct': org_correct,
                'answer_length': len(answer)
            }

            results.append(eval_result)

            print(f"  키워드 매칭: {keyword_match:.1%}")
            print(f"  최고 관련도: {max_relevance:.3f}")
            print(f"  상위 2개 평균: {avg_relevance:.3f}")
            if org_correct is not None:
                print(f"  발주기관 정확: {'✅' if org_correct else '❌'}")

        # 전체 통계
        self._print_summary(results)

        return results

    def _print_summary(self, results):
        """평가 결과 요약"""
        print("\n" + "="*60)
        print("전체 평가 결과")
        print("="*60)

        avg_keyword = sum(r['keyword_match'] for r in results) / len(results)
        avg_relevance = sum(r['avg_relevance'] for r in results) / len(results)
        avg_max = sum(r['max_relevance'] for r in results) / len(results)

        org_results = [r['org_correct'] for r in results if r['org_correct'] is not None]
        org_accuracy = sum(org_results) / len(org_results) if org_results else 0

        print(f"\n📊 성능 지표:")
        print(f"  평균 키워드 매칭률: {avg_keyword:.1%}")
        print(f"  평균 최고 관련도: {avg_max:.3f}")
        print(f"  평균 상위2개 관련도: {avg_relevance:.3f}")
        print(f"  발주기관 정확도: {org_accuracy:.1%}")
        print(f"  테스트 쿼리 수: {len(results)}")

        # 판정
        print(f"\n🎯 종합 평가:")
        if avg_keyword >= 0.7 and avg_max >= 0.15:
            print("  ✅ 우수 - MVP 품질 충족")
            print("  💡 키워드 매칭과 검색 정확도 모두 우수합니다!")
        elif avg_keyword >= 0.5 and avg_max >= 0.10:
            print("  ⚠️  양호 - 개선 여지 있음")
        else:
            print("  ❌ 미흡 - 재검토 필요")