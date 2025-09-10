# 마이페이지 및 점수 시스템 구현 문서

**작성일**: 2025년 7월 11일  
**작성자**: 개발팀  
**버전**: 1.0.0

## 📋 개요

YouSync 프로젝트에 사용자별 마이페이지 기능과 점수 관리 시스템을 구현했습니다. 이 문서는 구현된 기능들과 변경사항을 상세히 기록합니다.

## 🎯 구현 목표

### 1. 마이페이지 기능
- 사용자가 더빙한 영상(토큰) 목록 표시
- 더빙 기록이 있는 토큰 진입 시 결과창 바로 표시
- 더빙 기록이 없는 토큰 진입 시 더빙 시작 화면 표시
- "다시 더빙하기" 기능으로 기존 결과 삭제 후 재시작

### 2. 데이터 관리 정책
- 한 사용자당 동일 토큰의 분석 결과는 1세트만 존재
- 재더빙 시 기존 분석 결과 모두 삭제
- 데이터 누적 방지로 스토리지 효율성 확보

## 🗄️ 데이터베이스 구조 변경

### AnalysisResult 테이블 수정

**기존 구조**:
```sql
CREATE TABLE analysis_results (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR UNIQUE,
    token_id INTEGER REFERENCES tokens(id),
    status VARCHAR NOT NULL,
    progress INTEGER NOT NULL,
    result JSON,
    message VARCHAR,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**변경된 구조**:
```sql
CREATE TABLE analysis_results (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR UNIQUE,
    script_id INTEGER REFERENCES scripts(id),  -- token_id → script_id 변경
    user_id INTEGER REFERENCES users(id),      -- 사용자 구분용 추가
    status VARCHAR NOT NULL,
    progress INTEGER NOT NULL,
    result JSON,
    message VARCHAR,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 주요 변경사항
1. **`user_id` 컬럼 추가**: 사용자별 분석 결과 구분
2. **`token_id` → `script_id` 변경**: 스크립트 단위 분석으로 전환
3. **관계 설정**: User ↔ AnalysisResult ↔ Script ↔ Token

## 🔧 구현된 API 엔드포인트

### 마이페이지 API (`/mypage`)

#### 1. 더빙한 토큰 목록 조회
```http
GET /mypage/my-dubbed-tokens
```

**응답 예시**:
```json
[
  {
    "token_id": 1,
    "token_name": "아이언맨 명대사",
    "actor_name": "로버트 다우니 주니어",
    "category": "액션",
    "last_dubbed_at": "2025-07-11T10:30:00Z",
    "total_scripts": 5,
    "completed_scripts": 5
  }
]
```

#### 2. 토큰 분석 상태 확인
```http
GET /mypage/tokens/{token_id}/analysis-status
```

**응답 예시**:
```json
{
  "token_id": 1,
  "has_analysis": true,
  "script_results": [
    {
      "script_id": 1,
      "script_text": "I am Iron Man",
      "has_result": true,
      "job_id": "job_123",
      "status": "completed",
      "result": {"overall_score": 85.5},
      "created_at": "2025-07-11T10:30:00Z"
    }
  ]
}
```

#### 3. 재더빙용 결과 삭제
```http
DELETE /mypage/tokens/{token_id}/my-results
```

### 점수 관리 API (`/score`)

#### 1. 토큰별 평균 점수 조회
```http
GET /score/{token_id}/score
```

**응답 예시**:
```json
{
  "token_id": 1,
  "average_score": 85.5
}
```

#### 2. 전체 평균 점수 조회
```http
GET /score/me/average
```

**응답 예시**:
```json
{
  "user_id": 123,
  "average_score": 82.3
}
```

## 📝 스키마 정의

### 새로 추가된 Pydantic 스키마

```python
class MyDubbedTokenResponse(BaseModel):
    token_id: int
    token_name: str
    actor_name: str
    category: Optional[str] = None
    last_dubbed_at: datetime
    total_scripts: int
    completed_scripts: int

class TokenAnalysisStatusResponse(BaseModel):
    token_id: int
    has_analysis: bool
    script_results: List[dict] = Field(default_factory=list)

class TokenScore(BaseModel):
    token_id: int 
    average_score: float

class UserScore(BaseModel):
    user_id: int 
    average_score: float
```

## 🔄 비즈니스 로직

### 1. 마이페이지 토큰 목록 조회 로직

```sql
-- 사용자가 더빙한 토큰들 조회
SELECT 
    t.id as token_id,
    t.token_name,
    t.actor_name,
    t.category,
    MAX(ar.created_at) as last_dubbed_at,
    COUNT(s.id) as total_scripts,
    COUNT(ar.id) as completed_scripts
FROM tokens t
JOIN scripts s ON s.token_id = t.id
JOIN analysis_results ar ON ar.script_id = s.id
WHERE ar.user_id = {current_user_id}
GROUP BY t.id, t.token_name, t.actor_name, t.category
ORDER BY last_dubbed_at DESC;
```

### 2. 재더빙 시 기존 결과 삭제 로직

```python
def delete_my_token_results(token_id: int, user_id: int):
    # 1. 해당 토큰의 스크립트 ID들 조회
    script_ids = db.query(Script.id).filter(Script.token_id == token_id).all()
    
    # 2. 해당 스크립트들의 내 분석 결과 삭제
    deleted_count = db.query(AnalysisResult).filter(
        AnalysisResult.script_id.in_(script_ids),
        AnalysisResult.user_id == user_id
    ).delete()
    
    return deleted_count
```

### 3. 점수 계산 로직

```sql
-- 토큰별 평균 점수
SELECT AVG((ar.result->'result'->>'overall_score')::float) as avg_score
FROM analysis_results ar
JOIN scripts s ON ar.script_id = s.id
WHERE s.token_id = {token_id} AND ar.user_id = {user_id};

-- 전체 평균 점수
SELECT AVG((result->'result'->>'overall_score')::float) as avg_score
FROM analysis_results
WHERE user_id = {user_id};
```

## 🛠️ 파일 변경사항

### 1. 새로 생성된 파일
- 없음 (기존 파일들 수정)

### 2. 수정된 파일

#### `models.py`
- `AnalysisResult` 모델에 `user_id` 컬럼 추가
- User ↔ AnalysisResult 관계 설정

#### `schemas.py`
- `MyDubbedTokenResponse` 스키마 추가
- `TokenAnalysisStatusResponse` 스키마 추가
- 기존 `TokenScore`, `UserScore` 스키마 유지

#### `router/mypage_router.py`
- `get_my_dubbed_tokens()` API 추가
- `get_token_analysis_status()` API 추가
- `delete_my_token_results()` API 추가

#### `router/score_router.py`
- 중복 기능 제거 (`/me/tokens/full` 삭제)
- 스크립트 기반 쿼리로 수정
- 점수 계산 기능만 유지

#### `main.py`
- `score_router` 임포트 및 등록 추가

## 🔒 보안 고려사항

### 1. 사용자 인증
- 모든 API에서 JWT 토큰 기반 사용자 인증 필수
- `get_current_user()` 의존성 주입으로 현재 사용자 식별

### 2. 데이터 접근 제어
- 사용자는 자신의 분석 결과만 조회/삭제 가능
- `user_id` 필터링으로 다른 사용자 데이터 접근 차단

### 3. 입력 검증
- Pydantic 스키마로 입력 데이터 검증
- SQL 인젝션 방지를 위한 ORM 사용

## 📊 성능 최적화

### 1. 데이터베이스 인덱스
```sql
-- 성능 향상을 위한 인덱스 추가 권장
CREATE INDEX idx_analysis_results_user_id ON analysis_results(user_id);
CREATE INDEX idx_analysis_results_script_id ON analysis_results(script_id);
CREATE INDEX idx_scripts_token_id ON scripts(token_id);
```

### 2. 쿼리 최적화
- JOIN 쿼리 최소화
- 필요한 컬럼만 SELECT
- LIMIT/OFFSET으로 페이지네이션 구현

## 🧪 테스트 방법

### 1. API 테스트
```bash
# 서버 실행
uvicorn main:app --reload

# 마이페이지 토큰 목록 조회
curl -H "Authorization: Bearer {jwt_token}" \
     http://localhost:8000/mypage/my-dubbed-tokens

# 토큰 분석 상태 확인
curl -H "Authorization: Bearer {jwt_token}" \
     http://localhost:8000/mypage/tokens/1/analysis-status

# 점수 조회
curl -H "Authorization: Bearer {jwt_token}" \
     http://localhost:8000/score/1/score
```

### 2. 데이터베이스 테스트
```sql
-- 테스트 데이터 확인
SELECT * FROM analysis_results WHERE user_id = 1;
SELECT * FROM scripts WHERE token_id = 1;
```

## 🚀 배포 고려사항

### 1. 마이그레이션
- `user_id` 컬럼 추가를 위한 Alembic 마이그레이션 필요
- 기존 데이터의 `user_id`는 NULL로 설정

### 2. 환경 변수
- 기존 환경 변수 그대로 사용
- 추가 설정 불필요

### 3. 의존성
- 새로운 Python 패키지 추가 없음
- 기존 requirements.txt 그대로 사용

## 📈 향후 개선 계획

### 1. 기능 확장
- 토큰별 상세 통계 (발음, 억양, 감정 점수 분석)
- 사용자 랭킹 시스템
- 학습 진도 추적 대시보드

### 2. 성능 개선
- Redis 캐싱 도입
- 비동기 처리 최적화
- 데이터베이스 쿼리 최적화

### 3. UX 개선
- 실시간 알림 시스템
- 소셜 기능 (친구 점수 비교)
- 개인화된 학습 추천

## 🐛 알려진 이슈

### 1. 현재 제한사항
- 대용량 데이터 처리 시 성능 저하 가능성
- 동시 접속자 수 증가 시 응답 시간 지연 가능

### 2. 해결 예정
- 데이터베이스 커넥션 풀 최적화
- 캐싱 레이어 도입
- 로드 밸런싱 구현

## 📞 문의 및 지원

- **개발팀**: [개발팀 연락처]
- **이슈 리포팅**: GitHub Issues
- **API 문서**: `/docs` 엔드포인트 참조

---

*이 문서는 2025년 7월 11일에 작성되었으며, 향후 기능 추가 시 업데이트될 예정입니다.*
