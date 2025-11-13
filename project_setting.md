## 🛠 사전 준비

### 필수 설치 항목

1. **Python 3.12.3**
2. **pyenv** (Python 버전 관리)
3. **Poetry** (의존성 관리)

### 설치 방법

#### 🪟 Windows

```powershell
# 1. pyenv-win 설치
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"
& "./install-pyenv-win.ps1"

# PowerShell 재시작 후

# 2. Python 3.12.3 설치
pyenv install 3.12.3

# 3. Poetry 설치
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

#### 🍎 Mac/Linux

```bash
# 1. pyenv 설치
curl https://pyenv.run | bash

# 환경 변수 설정 (zsh 기준)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc

# 2. Python 3.12.3 설치
pyenv install 3.12.3

# 3. Poetry 설치
curl -sSL https://install.python-poetry.org | python3 -
```

---

## 🚀 환경 설정

### 1. 저장소 클론

#### 🪟 Windows
```powershell
git clone 
cd Codeit-AI-1team-LLM-project
```

#### 🍎 Mac/Linux
```bash
git clone 
cd Codeit-AI-1team-LLM-project
```

### 2. Python 버전 설정

프로젝트 폴더에 `.python-version` 파일이 있으면 자동으로 Python 3.12.3을 사용합니다.

#### 🪟 Windows
```powershell
# 확인
python --version
# Python 3.12.3이 아니면:
pyenv local 3.12.3
```

#### 🍎 Mac/Linux
```bash
# 확인
python --version
# Python 3.12.3이 아니면:
pyenv local 3.12.3
```

### 3. Poetry 설정

#### 🪟 Windows
```powershell
# 가상환경을 프로젝트 내부에 생성
python -m poetry config virtualenvs.in-project true
```

#### 🍎 Mac/Linux
```bash
poetry config virtualenvs.in-project true
```

---

## 📦 의존성 설치

`poetry.lock` 파일을 기준으로 정확히 동일한 버전의 패키지를 설치합니다.

#### 🪟 Windows
```powershell
# Python 버전 지정
python -m poetry env use 3.12.3

# 의존성 설치
python -m poetry install

# 가상환경 활성화
python -m poetry shell
```

#### 🍎 Mac/Linux
```bash
# Python 버전 지정
poetry env use 3.12.3

# 의존성 설치
poetry install

# 가상환경 활성화
poetry shell
```

**설치 완료 확인:**

프롬프트 앞에 `(.venv)`가 붙으면 성공! ✅

```
(.venv) PS C:\Codeit-AI-1team-LLM-project>  # Windows
(codeit-ai-1team-llm-project-py3.12) user@computer:~/project$  # Mac/Linux
```

---

## 🎯 프로젝트 실행

### 기본 실행

#### 🪟 Windows
```powershell
# 가상환경이 활성화된 상태에서
python main.py
```

#### 🍎 Mac/Linux
```bash
# 가상환경이 활성화된 상태에서
python main.py
```

### 가상환경 나가기

#### 🪟 Windows & Mac/Linux
```bash
exit
```

---

## 👥 개발 가이드

### 일상적인 작업 흐름

#### 🪟 Windows
```powershell
# 1. 프로젝트 폴더로 이동
cd C:\Codeit-AI-1team-LLM-project

# 2. 최신 코드 받기
git pull

# 3. 의존성 업데이트 (팀원이 패키지 추가한 경우)
python -m poetry install

# 4. 가상환경 활성화
python -m poetry shell

# 5. 개발 작업...

# 6. 작업 종료
exit
```

#### 🍎 Mac/Linux
```bash
# 1. 프로젝트 폴더로 이동
cd ~/Codeit-AI-1team-LLM-project

# 2. 최신 코드 받기
git pull

# 3. 의존성 업데이트 (팀원이 패키지 추가한 경우)
poetry install

# 4. 가상환경 활성화
poetry shell

# 5. 개발 작업...

# 6. 작업 종료
exit
```

### 새 패키지 추가

#### 🪟 Windows
```powershell
# 패키지 추가
python -m poetry add 

# 예: requests 추가
python -m poetry add requests

# 개발 도구 추가
python -m poetry add --group dev pytest

# Git 커밋
git add pyproject.toml poetry.lock
git commit -m "Add "
git push
```

#### 🍎 Mac/Linux
```bash
# 패키지 추가
poetry add 

# 예: requests 추가
poetry add requests

# 개발 도구 추가
poetry add --group dev pytest

# Git 커밋
git add pyproject.toml poetry.lock
git commit -m "Add "
git push
```

---

## 🐛 문제 해결

### Python 버전이 3.12.3이 아니에요

#### 🪟 Windows
```powershell
pyenv local 3.12.3
python --version
```

#### 🍎 Mac/Linux
```bash
pyenv local 3.12.3
python --version
```

### Poetry 명령어를 찾을 수 없어요

#### 🪟 Windows
```powershell
# Poetry를 python 모듈로 실행
python -m poetry --version

# PATH 추가 (영구적)
[Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path", "User") + ";$env:APPDATA\Python\Scripts", "User")
```

#### 🍎 Mac/Linux
```bash
# PATH 추가
export PATH="$HOME/.local/bin:$PATH"

# 영구 적용
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Import 에러가 나요

```bash
# 가상환경이 활성화되어 있는지 확인
# 프롬프트에 (.venv)가 있어야 함

# 없다면 다시 활성화
poetry shell  # Mac/Linux
python -m poetry shell  # Windows

# 의존성 재설치
poetry install  # Mac/Linux
python -m poetry install  # Windows
```