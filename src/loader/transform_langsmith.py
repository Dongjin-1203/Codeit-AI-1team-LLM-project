# === 1. 환경 설정 ===
import os 
import json
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

# LangSmith 클라이언트 설정
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

# === 2. 데이터 로드 함수 ===
def load_synthetic_testset(input_path):
    """합성 데이터셋 로드"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            record = json.loads(line)
            data.append(record)
    return data

# === 3. 데이터 변환 함수 ===
def convert_to_langsmith_format(testset):
    """langsmith 형식으로 변환"""
    langsmith_data = []
    for idx, record in enumerate(testset):
        langsmith_example = {
            "inputs": {
                "question": record["user_input"]
            },
            "outputs": {
                "ground_truth_contexts": record["retrieved_contexts"]
            },
            "metadata": {
                "reference_answer": record.get("reference", ""),
                "source": "synthetic_testset_100",
                "index": idx
            }
        }
        langsmith_data.append(langsmith_example)
    return langsmith_data

# === 4. LangSmith 업로드 함수 ===
def upload_to_langsmith(langsmith_data, dataset_name):
    try:
        os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
        client = Client()
        
        # 데이터셋 가져오기 시도
        try:
            ls_dataset = client.read_dataset(dataset_name=dataset_name)
            print(f"⚠️ 기존 Dataset 발견: {dataset_name}")
            # 기존 것 사용 또는 삭제 후 재생성
            user_choice = input("기존 데이터 삭제하고 새로 만들까요? (y/n): ")
            if user_choice.lower() == 'y':
                client.delete_dataset(dataset_name=dataset_name)
                ls_dataset = client.create_dataset(
                    dataset_name=dataset_name,
                    description="RAG 검색기 평가용 100개 쿼리"
                )
                print("✅ 기존 Dataset 삭제 후 재생성")
            else:
                print("⚠️ 기존 Dataset에 추가합니다")
                
        except Exception as e:  # ✅ 모든 예외 처리
            # Dataset이 없으면 새로 생성
            print(f"새 Dataset 생성 중... (오류: {type(e).__name__})")
            ls_dataset = client.create_dataset(
                dataset_name=dataset_name,
                description="RAG 검색기 평가용 100개 쿼리"
            )
            print("✅ 새 Dataset 생성 완료")
        
        # 예제 업로드
        print(f"업로드 중: {len(langsmith_data)}개...")
        client.create_examples(
            inputs=[ex["inputs"] for ex in langsmith_data],
            outputs=[ex["outputs"] for ex in langsmith_data],
            metadata=[ex["metadata"] for ex in langsmith_data],
            dataset_id=ls_dataset.id
        )
        
        print(f"✅ 업로드 완료: {len(langsmith_data)}개")
        return ls_dataset
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

# === 5. 메인 실행 ===
def main():
    input_path = "src/evaluation/results/synthetic_testset_100.json" 
    dataset_name = "RAG-Retriever-TestSet-v1"

    # 1단계: 데이터 로드
    print("데이터 로딩 중...")
    testset = load_synthetic_testset(input_path)
    print(f"로드 완료: {len(testset)}개 레코드")

    # 2단계: 데이터 변환
    print("데이터 변환 중...")
    langsmith_data = convert_to_langsmith_format(testset)

    # 샘플 출력
    print("\n=== 변환 예시 (첫 번째 레코드) ===")
    print(json.dumps(langsmith_data[0], ensure_ascii=False, indent=2))

    # 3단계: LangSmith 업로드
    user_confirm = input("LangSmith에 업로드하시겠습니까? (y/n): ")
    if user_confirm.lower() == 'y':
        print("\nLangSmith 업로드 중...")
        dataset = upload_to_langsmith(langsmith_data, dataset_name)

        if dataset:
            print(f"\n✅ 성공! Dataset URL: {dataset.url}")
        else:
            print("\n❌ 업로드 실패")
    else:
        print("업로드 취소됨")

# 실행
if __name__ == "__main__":
    main()