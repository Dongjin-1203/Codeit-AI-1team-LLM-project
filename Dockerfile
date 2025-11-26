# ===== GPU 지원 Dockerfile for Hugging Face Spaces =====
# CUDA 지원 Python 베이스 이미지
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# ===== Python 3.12.3 설치 =====
# deadsnakes PPA를 통해 Python 3.12 설치
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

# Python 3.12 및 필수 패키지 설치
RUN apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    python3.12-venv \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12를 기본 python으로 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# pip 설치 (Python 3.12용)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12
RUN python -m pip install --upgrade pip setuptools wheel

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사
COPY requirements.txt .

# ===== llama-cpp-python CUDA 빌드 =====
# CUDA 지원으로 llama-cpp-python 설치 (먼저 설치)
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
ENV FORCE_CMAKE=1
RUN pip install --no-cache-dir llama-cpp-python==0.3.16

# ===== 나머지 의존성 설치 =====
# llama-cpp-python 제외하고 설치
RUN pip install --no-cache-dir -r requirements.txt

# ===== 프로젝트 파일 복사 =====
COPY . .

# ===== 환경변수 설정 =====
# CUDA 가시성 (GPU 사용)
ENV CUDA_VISIBLE_DEVICES=0

# ===== Streamlit 설정 =====
# HF Spaces는 포트 7860 사용
EXPOSE 7860

# ===== 헬스체크 (선택) =====
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# ===== 실행 명령 =====
CMD ["streamlit", "run", "src/visualization/chatbot_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]