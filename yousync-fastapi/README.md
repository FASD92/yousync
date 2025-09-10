# YouSync FastAPI 오디오 분석 서비스

영화/드라마 스크립트와 오디오 분석을 위한 FastAPI 기반 백엔드 서비스입니다.

## 🎯 서비스 개요

YouSync는 영화나 드라마의 스크립트 데이터와 사용자가 업로드한 오디오를 분석하여 발음, 억양, 감정 등을 평가하는 서비스입니다.

### 주요 기능
- **스크립트 관리**: 영화/드라마 스크립트 및 단어별 타이밍 데이터 관리
- **오디오 분석**: 사용자 음성 업로드 및 AI 기반 분석 (발음, 억양, 감정 등)
- **사용자 관리**: Google OAuth 기반 인증 및 사용자 프로필 관리
- **북마크 시스템**: 사용자별 스크립트 북마크 기능
- **실시간 진행 상황**: Server-Sent Events를 통한 분석 진행 상황 실시간 추적

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   Analysis      │
│   (Client)      │◄──►│   Server        │◄──►│   Server        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   PostgreSQL    │
                       │   Database      │
                       └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   AWS S3        │
                       │   File Storage  │
                       └─────────────────┘
```

## 🛠️ 기술 스택

### Backend
- **FastAPI**: Python 웹 프레임워크
- **SQLAlchemy**: ORM (Object-Relational Mapping)
- **PostgreSQL**: 메인 데이터베이스
- **Alembic**: 데이터베이스 마이그레이션

### 인증 & 보안
- **Google OAuth 2.0**: 소셜 로그인
- **JWT**: 토큰 기반 인증
- **bcrypt**: 비밀번호 해싱

### 파일 처리 & 클라우드
- **AWS S3**: 오디오 파일 저장소
- **boto3**: AWS SDK
- **httpx**: 비동기 HTTP 클라이언트

### 비동기 처리
- **BackgroundTasks**: FastAPI 백그라운드 작업
- **ThreadPoolExecutor**: S3 업로드 비동기 처리
- **Server-Sent Events**: 실시간 진행 상황 스트리밍

## 📁 프로젝트 구조

```
fast-api/
├── back-end/                          # 백엔드 소스코드
│   ├── main.py                        # FastAPI 애플리케이션 진입점
│   ├── models.py                      # SQLAlchemy 데이터베이스 모델
│   ├── schemas.py                     # Pydantic 스키마 정의
│   ├── database.py                    # 데이터베이스 연결 설정
│   ├── dependencies.py                # 의존성 주입 설정
│   ├── requirements.txt               # Python 패키지 의존성
│   ├── .env                          # 환경 변수 (보안)
│   ├── .env.example                  # 환경 변수 예시
│   ├── alembic.ini                   # Alembic 설정
│   ├── migrations/                   # 데이터베이스 마이그레이션
│   ├── services/                     # 비즈니스 로직 서비스
│   └── router/                       # API 라우터 모듈
│       ├── auth_router.py            # 인증 관련 API
│       ├── user_audio_router.py      # 오디오 분석 API (핵심)
│       ├── token_router.py           # 토큰 관리 API
│       ├── script_router.py          # 스크립트 관리 API
│       ├── actor_router.py           # 배우 정보 API
│       ├── mypage_router.py          # 마이페이지 API
│       ├── script_audio_router.py    # 스크립트 오디오 API
│       └── url_router.py             # URL 관리 API
├── docs/                             # 프로젝트 문서
├── figma-assets/                     # 디자인 에셋
├── tmp/                              # 임시 파일
├── add_dummy_data.py                 # 더미 데이터 생성 스크립트
├── request_test.py                   # API 테스트 스크립트
└── README.md                         # 프로젝트 설명서
```

## 🗄️ 데이터베이스 스키마

### 주요 테이블
- **tokens**: 스크립트 토큰 정보 (배우, 시간, S3 URL 등)
- **scripts**: 문장 단위 스크립트 데이터
- **script_words**: 단어 단위 스크립트 데이터 (타이밍, 확률 등)
- **actors**: 배우 정보 및 별칭
- **users**: 사용자 정보 (Google OAuth)
- **analysis_results**: 오디오 분석 결과 저장
- **bookmarks**: 사용자 북마크 데이터
- **urls**: YouTube URL 관리

## 🚀 배포 환경

### 현재 배포: AWS EC2
- **플랫폼**: Amazon EC2
- **데이터베이스**: PostgreSQL (AWS RDS 또는 EC2 내부)
- **파일 저장소**: AWS S3
- **도메인**: EC2 퍼블릭 IP 또는 도메인

### 환경 설정
```bash
# 필수 환경 변수
DATABASE_URL=postgresql://user:password@host:port/dbname
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=ap-northeast-2
S3_BUCKET_NAME=your_bucket_name
TARGET_SERVER_URL=http://analysis-server-url
WEBHOOK_URL=http://your-domain/tokens/webhook/analysis-complete
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
```

## 🔄 오디오 분석 워크플로우

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant S3
    participant AnalysisServer
    participant Webhook

    Client->>FastAPI: POST /tokens/{id}/upload-audio
    FastAPI->>Client: job_id (즉시 응답)
    
    Note over FastAPI: BackgroundTasks 시작
    FastAPI->>S3: 비동기 파일 업로드
    FastAPI->>AnalysisServer: httpx POST 요청
    
    Client->>FastAPI: GET /tokens/analysis-progress/{job_id} (SSE)
    FastAPI-->>Client: 실시간 진행 상황 스트림
    
    AnalysisServer->>Webhook: 분석 완료 알림
    FastAPI->>Client: 분석 완료 (SSE 종료)
```

## 📡 주요 API 엔드포인트

### 인증
- `POST /auth/google` - Google OAuth 로그인
- `POST /auth/refresh` - 토큰 갱신

### 오디오 분석 (핵심 기능)
- `POST /tokens/{token_id}/upload-audio` - 오디오 업로드 및 분석 요청
- `GET /tokens/analysis-result/{job_id}` - 분석 결과 조회
- `GET /tokens/analysis-progress/{job_id}` - 실시간 진행 상황 (SSE)
- `POST /tokens/webhook/analysis-complete` - 분석 완료 웹훅

### 마이페이지 (핵심 기능)
- `POST /mypage/bookmarks/` - 북마크 추가
- `DELETE /mypage/bookmarks/{token_id}` - 북마크 삭제  
- `GET /mypage/bookmarks/` - 북마크 목록 조회 (토큰 정보 포함)
- `GET /mypage/my-dubbed-tokens` - 내가 더빙한 토큰 목록
- `GET /mypage/tokens/{token_id}/analysis-status` - 토큰 분석 상태 확인
- `DELETE /mypage/tokens/{token_id}/my-results` - 분석 결과 삭제 (재더빙용)
- `GET /mypage/overview` - 마이페이지 통합 정보 (새로 구현)

### 모니터링
- `GET /` - 서비스 상태 확인
- `GET /health` - 헬스체크
- `GET /docs` - Swagger UI API 문서

## 🛠️ 로컬 개발 환경 설정

### 1. 저장소 클론
```bash
git clone <repository-url>
cd fast-api/back-end
```

### 2. 가상환경 설정
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정
```bash
cp .env.example .env
# .env 파일을 편집하여 실제 값 입력
```

### 5. 데이터베이스 마이그레이션
```bash
alembic upgrade head
```

### 6. 서버 실행
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 7. API 문서 확인
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🧪 테스트

### API 테스트
```bash
python request_test.py
```

### 더미 데이터 생성
```bash
python add_dummy_data.py
```

## 📊 모니터링 및 로깅

### 로그 파일
- `server.log`: 애플리케이션 로그
- 콘솔 출력: 실시간 요청/응답 로그

### 주요 모니터링 포인트
- API 응답 시간
- 오디오 분석 성공/실패율
- S3 업로드 성능
- 데이터베이스 연결 상태
- 메모리 사용량

## 🔒 보안 고려사항

- **환경 변수**: 민감한 정보는 .env 파일로 관리
- **CORS**: 프론트엔드 도메인만 허용하도록 설정 필요
- **JWT 토큰**: 적절한 만료 시간 설정
- **파일 업로드**: 파일 크기 및 형식 제한
- **Rate Limiting**: API 호출 빈도 제한 (향후 구현 예정)

## 🚧 향후 개선 계획

1. **작업 큐 시스템**: Redis + SQS 기반 비동기 처리로 전환
2. **캐싱**: Redis 캐싱으로 성능 향상
3. **모니터링**: Prometheus + Grafana 도입
4. **CI/CD**: GitHub Actions 자동 배포
5. **컨테이너화**: Docker 기반 배포
6. **로드 밸런싱**: 다중 인스턴스 운영

## 📞 문의 및 지원

- **개발팀**: [개발팀 연락처]
- **이슈 리포팅**: GitHub Issues
- **API 문서**: `/docs` 엔드포인트 참조

---

*이 문서는 2025년 7월 10일 기준으로 작성되었습니다.*
