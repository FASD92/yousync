# User Processor 🎤

사용자 음성을 분석하여 기준 영상과 비교 평가하는 자동화 시스템

## 📋 목차
- [시스템 개요](#시스템-개요)
- [설치 및 설정](#설치-및-설정)
- [사용법](#사용법)
- [결과 해석](#결과-해석)
- [문제 해결](#문제-해결)

## 🎯 시스템 개요

User Processor는 사용자가 녹음한 음성을 기준 영상과 비교하여 발음, 타이밍, 억양을 종합적으로 평가하는 시스템입니다.

### 📊 처리 과정
1. **🧹 음성 전처리** - 배경 제거 및 16kHz 변환
2. **🎯 STT 처리** - Whisper를 통한 음성 인식
3. **📍 MFA 정렬** - Docker MFA로 음소 단위 정렬
4. **🔍 비교 분석** - TextGrid 및 피치 패턴 비교
5. **📊 점수 산출** - 가중치 기반 종합 평가

### 🏗️ 모듈 구조
```
user_processor/
├── main.py                  # 🚀 메인 파이프라인
├── audio_cleaner.py         # 🧹 음성 전처리 모듈
├── stt_processor.py         # 🎯 STT 처리 모듈
├── mfa_aligner.py           # 📍 MFA 정렬 모듈
├── textgrid_comparator.py   # 🔍 TextGrid 비교 분석 모듈
├── voice_to_pitch.py        # 🎵 피치 분석 모듈
├── scorer.py                # 📊 점수 산출 모듈
├── setup_directories.py    # 📁 디렉토리 구조 생성 스크립트
├── requirements.txt         # 📦 Python 의존성 목록
├── README.md               # 📖 이 문서
├── .gitignore              # 🚫 Git 제외 파일
└── venv/                   # 🐍 Python 가상환경
```

## ⚡ 빠른 시작

```bash
# 1. 저장소 클론
git clone <repository-url>
cd user_processor

# 2. 환경 설정
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Docker 이미지 다운로드
docker pull mmcauliffe/montreal-forced-aligner:latest

# 4. 디렉토리 구조 생성
python3 setup_directories.py

# 5. 음성 파일 준비 및 실행
cp your_audio.mp4 ../shared_data/user/input/
python3 main.py --input ../shared_data/user/input/your_audio.mp4
```

## 🛠️ 설치 및 설정

### 1. 사전 요구사항

#### Docker 설치
```bash
# Docker가 설치되어 있는지 확인
docker --version

# 없다면 Docker 설치 (https://docs.docker.com/get-docker/)
```

#### MFA Docker 이미지 다운로드
```bash
docker pull mmcauliffe/montreal-forced-aligner:latest
```

### 2. Python 환경 설정

```bash
# 프로젝트 디렉토리로 이동
cd user_processor

# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows

# 필요한 라이브러리 설치
pip install -r requirements.txt

# 또는 개별 설치
pip install whisper librosa soundfile fastdtw scipy praat-parselmouth
```

### 3. 디렉토리 구조 설정

```bash
# shared_data 디렉토리 구조 자동 생성
python3 setup_directories.py
```

실행 전 다음 구조가 준비되어 있어야 합니다:

```
dev/
├── youtube_downloader/
│   └── Youtube_Downloader/
│       ├── youtube_processor/
│       │   ├── downloads/           # 기준 음성 파일 (MP3)
│       │   └── token_data/          # 기준 토큰 데이터 (JSON)
│       └── syncdata/mfa/            # MFA 모델
│           ├── english_us_arpa.dict
│           └── english_us_arpa/
├── user_processor/                  # 이 시스템
└── shared_data/
    └── user/
        └── input/                   # 분석할 음성 파일 위치
```

## 🚀 사용법

### 1. 음성 파일 준비

분석하고 싶은 음성 파일을 `shared_data/user/input/` 디렉토리에 복사:

```bash
cp your_audio_file.mp4 ../shared_data/user/input/
```

### 2. 기본 실행

```bash
# 가상환경 활성화
source venv/bin/activate

# 기본 실행 (배경 제거 포함)
python3 main.py --input ../shared_data/user/input/your_audio_file.mp4
```

### 3. 실행 옵션

```bash
# 배경 제거 없이 실행 (더 빠름)
python3 main.py --input your_audio.mp4 --no_background_removal

# 특정 기준 비디오 지정
python3 main.py --input your_audio.mp4 --video_id jZOywn1qArI

# 도움말 보기
python3 main.py --help
```

### 4. 실행 예시

```bash
# 1. 디렉토리 이동
cd user_processor

# 2. 가상환경 활성화
source venv/bin/activate

# 3. 음성 파일 복사
cp ~/Downloads/my_recording.mp4 ../shared_data/user/input/

# 4. 분석 실행
python3 main.py --input ../shared_data/user/input/my_recording.mp4

# 5. 결과 확인
cat ../shared_data/final_result.json
```

## 📊 결과 해석

### 실행 완료 시 출력 예시

```
🎉 User Processor 완료!
📝 인식된 텍스트: Stand, I didn't mean to...
🎬 기준 비디오: jZOywn1qArI_Liam_Neeson_Taken_token

📊 점수 결과:
   🗣️  발음 점수:     2.2/100
   ⏰ 타이밍 점수:   58.1/100
   🎵 피치 점수:    54.7/100
   🏆 최종 점수:    29.3/100

⏱️  처리 시간: 86.0초
📁 결과 파일: ../shared_data/final_result.json
```

### 점수 해석

- **🗣️ 발음 점수**: 음소 단위 정확도 (0-100점)
- **⏰ 타이밍 점수**: 발화 속도 및 리듬 (0-100점)
- **🎵 피치 점수**: 억양 및 음조 유사도 (0-100점)
- **🏆 최종 점수**: 가중치 적용 종합 점수
  - 발음 50% + 타이밍 25% + 피치 25%

### 생성되는 파일들

```
shared_data/
├── final_result.json        # 최종 종합 결과
├── comparison_result.json   # 상세 비교 분석
├── mfa_output/             # TextGrid 정렬 결과
│   ├── reference.TextGrid
│   └── user.TextGrid
├── pitch_data/             # 피치 분석 데이터
│   ├── reference/pitch.json
│   ├── user/pitch.json
│   └── segments.json
└── user/
    ├── processed/          # 전처리된 음성
    └── lab/               # STT 결과
```

## ⏱️ 처리 시간

- **배경 제거 포함**: 2-3분
- **배경 제거 없이**: 1-2분
- **파일 크기에 따라 변동**

## 🔧 문제 해결

### Docker 관련 오류

```bash
# Docker 서비스 확인
docker ps

# Docker Desktop 실행 확인 (Mac/Windows)
# 또는 Docker 서비스 시작 (Linux)
sudo systemctl start docker
```

### 모델 다운로드 오류

```bash
# Whisper 모델 미리 다운로드
python3 -c "import whisper; whisper.load_model('base')"
```

### 권한 오류

```bash
# 디렉토리 권한 설정
chmod -R 755 ../shared_data/
```

### 메모리 부족 오류

```bash
# 더 작은 Whisper 모델 사용
python3 main.py --input your_audio.mp4  # 기본: base 모델

# 또는 코드에서 model_size를 'tiny'로 변경
```

### MFA 정렬 실패

1. **Docker 이미지 확인**:
   ```bash
   docker images | grep montreal-forced-aligner
   ```

2. **MFA 모델 파일 확인**:
   ```bash
   ls ../youtube_downloader/Youtube_Downloader/syncdata/mfa/
   ```

3. **파일 권한 확인**:
   ```bash
   ls -la ../shared_data/mfa_corpus/
   ```

## 📞 지원

문제가 발생하면 다음을 확인해주세요:

1. **Docker가 실행 중인가?**
2. **가상환경이 활성화되었나?**
3. **필요한 디렉토리 구조가 준비되었나?**
4. **기준 데이터 파일들이 존재하나?**

## 🎯 사용 팁

- **첫 실행 시**: 배경 제거 없이 테스트해보세요 (`--no_background_removal`)
- **빠른 테스트**: 짧은 음성 파일(10-30초)로 먼저 시도
- **정확한 비교**: 기준 영상과 유사한 내용으로 녹음
- **음질 중요**: 깨끗한 환경에서 녹음할수록 정확한 분석

---

**User Processor v1.0** - 음성 분석 및 평가 자동화 시스템 🎤
