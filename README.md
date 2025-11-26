---
title: RFPilot                # 괄호 제거
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker                   # Docker 사용
app_port: 7860                # Streamlit 포트
pinned: false
license: mit
---

# Codeit-AI-1team-LLM-project
---
## 챗봇 서비스 시연
![VectorDB Dashboard](asset/chatbot.gif)

## 벡터 DB 대시보드 영상
![VectorDB Dashboard](asset/vectorDB.gif)

# 1. 프로젝트 개요
- **B2G 입찰지원 전문 컨설팅 스타트업 – 'RFPilot'**
- RFP 문서를 요약하고, 사용자 질문에 실시간으로 응답하는 챗봇 시스템
> **배경**: 매일 수백 건의 기업 및 정부 제안요청서(RFP)가 게시되는데, 각 요청서 당 수십 페이지가 넘는 문건을 모두 검토하는 것은 불가능합니다. 이러한 과정은 비효율적이며, 중요한 정보를 빠르게 파악하기 어렵습니다.
> 
> **목표**: 사용자의 질문에 실시간으로 응답하고, 관련 제안서를 탐색하여 요약 정보를 제공하는 챗봇을 개발하여 컨설턴트의 업무 효율을 향상시키고자 합니다.
> 
> **기대 효과**: RAG 시스템을 통해 중요한 정보를 신속하게 제공함으로써, 제안서 검토 시간을 단축하고 컨설팅 업무에 보다 집중할 수 있는 환경을 조성합니다.
---
# 2. 설치 및 실행(🪟 Windows)
---
### Prerequisites
- Python 3.12.3 설치됨
- Poetry 설치됨
- 저장소 클론 완료
- 데이터셋 로컬에 저장
- 양자화된 모델 파일(.gguf) 저장
- .env 생성(api키 입력)

**env 파일 설정 방법**
```env
OPENAI_API_KEY = "OpenAI API 키"
WANDB_API_KEY = "WanDB API 키"
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY = "LangSmith API 키"
LANGCHAIN_PROJECT = "LangSmith 프로젝트 이름"
```

**코드 실행**
```powershell
# 1. 프로젝트 폴더로 이동
cd Codeit-AI-1team-LLM-project

# 2. 가상환경 설정 및 의존성 설치
python -m poetry config virtualenvs.in-project true
python -m poetry env use 3.12.3
python -m poetry install

# 3. 가상환경 활성화
python -m poetry env activate

# 4. 실행(전처리~벡터DB 구측)
python -m poetry run python main.py --step all

# 5. 벡터 DB 대시보드 실행
python -m poetry run streamlit run src/visualization/streamlit_app.py

# 6. 챗봇 서비스 실행
python -m poetry run streamlit run src/visualization/chatbot_app.py

# 7. LangSmith 실험 실행(API 및 프로젝트 생성 필요)
python -m poetry run python src/evaluation/run_experiment.py              # 대화형 메뉴
python -m poetry run python src/evaluation/run_experiment.py --run        # 실험 실행
python -m poetry run python src/evaluation/run_experiment.py --compare    # 실험 비교
```

# 3. 프로젝트 구조
---
```
CODEIT-AI-1TEAM-LLM-PROJECT/
│
├── main.py                  # 실행 진입점
├── models/                  # 로컬 모델 로드용 양자화 파일 저장 폴더(비공개)
├── data/                    # 문서 및 벡터DB 저장 폴더(비공개)
│   ├── files/               # hwp, pdf 문서
│   └── data_list.csv        # RFP 문서 정보 csv
├── src/
│   ├── loader/              # 문서 로딩 및 전처리
│   ├── evaluation/          # LangSmith 평가
│   ├── embedding/           # 임베딩, 벡터DB 생성
│   ├── retriever/           # 문서 검색기
│   ├── generator/           # 응답 생성기
│   ├── visualization/       # UI 구성
│   ├── notebooks/           # Hugging Face 모델 학습 코드
│   └── utils/               # 공통 함수 모듈
└── README.md
```
- `main.py`: 전체 RAG 파이프라인 실행의 진입점입니다.
- `data/`: 원문 문서, 생성된 벡터DB 등이 저장됩니다.
- `models/`: 로컬 모델 로드용 양자화 모델 파일을 저장하는 곳입니다.
- `src/loader`: PDF, HWP 문서를 텍스트로 추출하고 의미 단위로 분할합니다.
- `src/evaluation`: LangSmith 평가 환경을 관리하고 실험을 진행합니다.
- `src/embedding`: 텍스트 임베딩 벡터를 생성하고 Chroma DB를 구축합니다.
- `src/retriever`: 사용자 질문에 대한 관련 문서를 벡터DB에서 검색합니다.
- `src/generator`: 검색된 문서 기반으로 LLM이 응답을 생성합니다.
- `src/notebooks`: 로컬 모델을 Fine-Tuning하여 양자화 파일을 생성합니다.
- `src/visualization`: Streamlit 기반 사용자 인터페이스를 구성합니다.
- `src/utils`: 설정 확인, 경로 설정 등 공통 유틸리티 함수들을 포함합니다.

# 4. 팀 소개
> 기본에 충실실하며 실제 사용 가능한 모델을 만들기 위해 끊임없이 노력하는 팀입니다.

## 👨🏼‍💻 멤버 구성
|지동진|김진욱|이유노|박지윤|
|-----|------|------|-------|
|<img width="100" height="100" alt="image" src="https://github.com/user-attachments/assets/b9f1a52f-4304-496d-a19c-2d6b4775a5c3" />|<img width="100" height="100" alt="image" src="https://avatars.githubusercontent.com/u/80089860?v=4.png"/>|<img width="100" height="100" alt="image" src="https://github.com/user-attachments/assets/4e635630-f00c-4026-bb1d-c73ec05f37c8" />|<img width="100" height="100" alt="image" src="https://github.com/user-attachments/assets/088a073c-cf1c-40a1-97fb-1d2c1f1b8794" />|
|![https://github.com/Dongjin-1203](https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white)|![https://github.com/Jinuk93](https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white)|![https://github.com/Leeyuno0419](https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white)|![https://github.com/krapnuyij](https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white)|
|![hamubr1203@gmail.com](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)|![rlawlsdnr430@gmail.com](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)|![yoonolee0419@gmail.com](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)|![jiyun1147@gmail.com](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)|

## 👨🏼‍💻 역할 분담
|지동진|김진욱|이유노|박지윤|
|------|--------------|---------------|---------------|
|PM/AI Enginner(Rettriever, Pre-trained, PEFT)|Data Scientist|AI Engineer(API, Prompt)|AI Engineer(HuggingFace, PEFT)|
|프로젝트 총괄. 팀 회의 진행. 팀 혐업 환경 관리. RAG 개발. 대시보드 개발, PEFT 담당|학습 데이터 구성. 데이터 전처리 파이프라인 작성. 개발간 필요한 인사이트 도출 및 정보 수집, 제공|API 모델 개발. 프롬프트 작성. 모델 개선|HuggingFace 모델 학습, 모델 개선|
---
# 5. 프로젝트 타임라인
<img width="1580" height="807" alt="image" src="https://github.com/user-attachments/assets/57f6346a-663f-4ddd-a4b6-fafc2074ff71" />


---
# 6. 서비스 설명

## 서비스 아키텍쳐
<img width="4208" height="2004" alt="image" src="https://github.com/user-attachments/assets/73a0db09-b858-4b69-b93b-a85f928225a9" />

---
# Further Information

## 개발 스택 및 개발환경
- **언어**: <img width="67" height="18" alt="image" src="https://github.com/user-attachments/assets/e8035e3d-cadb-48f5-a4ac-3693faca01a7" /> <img width="67" height="18" alt="image" src="https://github.com/user-attachments/assets/0658c7ba-8039-4dc3-96a2-7c1308b2fafc" />

- **프레임워크**: <img width="79" height="18" alt="image" src="https://github.com/user-attachments/assets/e8814092-7e1e-4b22-8d77-e04fd2b26ae6" /> <img width="79" height="18" alt="image" src="https://img.shields.io/badge/LangChain-ffffff?logo=langchain&logoColor=green" />

- **라이브러리**: <img width="71" height="18" alt="image" src="https://github.com/user-attachments/assets/a428cd24-c8a5-4296-b6da-22eb322afa49" /> <img width="69" height="18" alt="image" src="https://github.com/user-attachments/assets/4325f1d3-d8ba-4bec-a746-4cad4993e925" /> <img width="103" height="18" alt="image" src="https://github.com/user-attachments/assets/a2009044-329d-4dde-b0dc-701122ff8149" /> <img width="53" height="18" alt="image" src="https://github.com/user-attachments/assets/f6225115-0b60-439e-8388-974a0365f8d6" /> 
- **클라우드 서비스**: <img width="71" height="18" alt="image" src="https://img.shields.io/badge/Google%20Cloud-4285F4?&style=plastic&logo=Google%20Cloud&logoColor=white" />
- **도구**: <img width="65" height="18" alt="image" src="https://github.com/user-attachments/assets/52f296c1-c878-4285-abe6-74842522e793" /> <img width="89" height="18" alt="image" src="https://github.com/user-attachments/assets/4ac10441-0753-4e94-9237-1ea6dc2034a2" /><img width="63" height="18" alt="image" src="https://github.com/user-attachments/assets/fea30130-c47c-4fa7-b3cb-7531481cfb28" /> <img width="89" height="18" alt="image" src="https://img.shields.io/badge/google_drive-white?style=for-the-badge&logo=google%20drive&logoColor=white&color=%23EA4336" />



## 협업 Tools
<img width="69" height="18" alt="image" src="https://github.com/user-attachments/assets/2bc2fa93-b01e-4051-9b31-ab83301594df" />
<img width="63" height="18" alt="image" src="https://github.com/user-attachments/assets/6c44ddad-80a4-4098-9727-6dae9a8fcb1c" />
<img width="65" height="18" alt="image" src="https://github.com/user-attachments/assets/a85b2d0f-8cdc-43e7-8e14-da11708a33a4" />
<img width="89" height="18" alt="image" src="https://github.com/user-attachments/assets/28d7f511-a4fe-4aa5-9184-2d3a94a97f29" />
<img width="89" height="18" alt="image" src="https://img.shields.io/badge/weightsandbiases-%23FFBE00?style=for-the-badge&logo=wandb-%23FFBE00&logoColor=%23FFBE00" />

## 기타 링크
