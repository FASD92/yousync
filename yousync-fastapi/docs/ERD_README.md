# 📊 FastAPI Database ERD Documentation

## 🎯 개요
이 문서는 FastAPI 프로젝트의 데이터베이스 스키마와 ERD(Entity Relationship Diagram) 설정에 대한 상세 설명을 제공합니다.

## 📁 파일 구조
```
fast-api/
├── complete_db_erd1.vuerd          # 최종 ERD 파일 ✅
├── db.json                         # 구 버전 ERD 파일 (사용 안함)
├── ERD_README.md                   # 이 문서
└── back-end/                       # FastAPI 백엔드 코드
```

## 🗄️ 데이터베이스 정보
- **데이터베이스명**: `fastapi_db`
- **DBMS**: PostgreSQL 15.12
- **호스트**: `yousync-db.cj60es4aa7pz.ap-northeast-2.rds.amazonaws.com`
- **포트**: 5432
- **리전**: ap-northeast-2 (서울)
- **사용자**: yousync

## 📋 테이블 구조 (10개)

### 1. 👤 users (사용자 정보)
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR,
    google_id VARCHAR UNIQUE,
    full_name VARCHAR,
    profile_picture VARCHAR,
    is_active BOOLEAN DEFAULT true,
    login_type VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```
**인덱스**: `ix_users_email`, `ix_users_google_id`, `ix_users_id`

### 2. 🔖 bookmarks (북마크 정보)
```sql
CREATE TABLE bookmarks (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_id INTEGER NOT NULL REFERENCES tokens(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 3. 🎬 tokens (토큰 정보)
```sql
CREATE TABLE tokens (
    id SERIAL PRIMARY KEY,
    token_name VARCHAR NOT NULL,
    actor_name VARCHAR NOT NULL,
    category VARCHAR,
    start_time DOUBLE PRECISION NOT NULL,
    end_time DOUBLE PRECISION NOT NULL,
    s3_textgrid_url TEXT,
    s3_pitch_url TEXT,
    s3_bgvoice_url TEXT,
    youtube_url TEXT NOT NULL REFERENCES urls(youtube_url),
    view_count INTEGER DEFAULT 0
);
```

### 4. 🔗 urls (URL 정보)
```sql
CREATE TABLE urls (
    youtube_url TEXT PRIMARY KEY,
    actor_id INTEGER NOT NULL REFERENCES actors(id)
);
```

### 5. 📝 scripts (스크립트 정보)
```sql
CREATE TABLE scripts (
    id SERIAL PRIMARY KEY,
    token_id INTEGER NOT NULL REFERENCES tokens(id),
    start_time DOUBLE PRECISION NOT NULL,
    end_time DOUBLE PRECISION NOT NULL,
    script TEXT NOT NULL,
    translation TEXT
);
```

### 6. 🎭 actors (배우 정보)
```sql
CREATE TABLE actors (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL
);
```

### 7. 🏷️ actor_aliases (배우 별명)
```sql
CREATE TABLE actor_aliases (
    id SERIAL PRIMARY KEY,
    actor_id INTEGER NOT NULL REFERENCES actors(id),
    name VARCHAR NOT NULL
);
```

### 8. 📖 words (단어 정보)
```sql
CREATE TABLE words (
    id SERIAL PRIMARY KEY,
    script_id INTEGER NOT NULL REFERENCES scripts(id),
    word VARCHAR NOT NULL,
    start_time DOUBLE PRECISION NOT NULL,
    end_time DOUBLE PRECISION NOT NULL
);
```

### 9. 🔄 token_actors (토큰-배우 관계, N:M 중간테이블)
```sql
CREATE TABLE token_actors (
    id SERIAL PRIMARY KEY,
    token_id INTEGER NOT NULL REFERENCES tokens(id),
    actor_id INTEGER NOT NULL REFERENCES actors(id)
);
```

### 10. 📊 analysis_results (분석 결과)
```sql
CREATE TABLE analysis_results (
    id SERIAL PRIMARY KEY,
    token_id INTEGER NOT NULL REFERENCES tokens(id),
    analysis_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🔗 관계 다이어그램

### 1:N 관계 (One-to-Many) - 10개
```
users(1) ──→ bookmarks(N)          # 한 사용자가 여러 북마크
tokens(1) ──→ bookmarks(N)         # 한 토큰이 여러 북마크에서 참조
urls(1) ──→ tokens(N)              # 한 URL이 여러 토큰에서 참조
actors(1) ──→ urls(N)              # 한 배우가 여러 URL 소유
actors(1) ──→ actor_aliases(N)     # 한 배우가 여러 별명 소유
tokens(1) ──→ scripts(N)           # 한 토큰이 여러 스크립트 소유
scripts(1) ──→ words(N)            # 한 스크립트가 여러 단어 포함
tokens(1) ──→ token_actors(N)      # 한 토큰이 여러 배우와 관계
actors(1) ──→ token_actors(N)      # 한 배우가 여러 토큰과 관계
tokens(1) ──→ analysis_results(N)  # 한 토큰이 여러 분석 결과
```

### N:M 관계 (Many-to-Many)
```
tokens ←→ actors (through token_actors)
```

## 🛠️ ERD 파일 사용법

### 1. ERD Editor (온라인) - 권장
1. https://erd-editor.io/ 접속
2. `File` → `Import` → `JSON`
3. `complete_db_erd1.vuerd` 파일 업로드
4. 자동으로 ERD 다이어그램 표시

### 2. VS Code
1. VS Code에서 `vuerd` 확장프로그램 설치
2. `complete_db_erd1.vuerd` 파일 열기
3. ERD 다이어그램 확인

### 3. 파일 정보
- **파일명**: `complete_db_erd1.vuerd`
- **크기**: 44KB
- **버전**: 3.0.0
- **스키마**: vuerd JSON 형식

## 🎨 ERD 시각적 요소

### 테이블 색상 구분
- 🟢 **users**: #4CAF50 (초록색) - 사용자 관련
- 🔵 **bookmarks**: #2196F3 (파란색) - 북마크 관련
- 🟠 **tokens**: #FF9800 (주황색) - 토큰 관련
- 🟣 **urls**: #9C27B0 (보라색) - URL 관련
- 🔘 **scripts**: #607D8B (회색) - 스크립트 관련
- 🟤 **actors**: #795548 (갈색) - 배우 관련
- 🟡 **actor_aliases**: #8BC34A (연두색) - 별명 관련
- 🟨 **words**: #CDDC39 (노란색) - 단어 관련
- 🟡 **token_actors**: #FFC107 (노란색) - 관계 테이블
- 🔴 **analysis_results**: #E91E63 (분홍색) - 분석 관련

### 관계선 표시
- **|---∞**: 1:N 관계 (One-to-Many)
- **점선**: Non-identifying 관계
- **화살표**: 관계 방향 표시

## 🔧 ERD 설정 상세

### 현재 설정값
```json
{
  "relationshipType": 16,        // 1:N 관계
  "identification": false,       // Non-identifying 관계
  "startRelationshipType": 2,    // 시작점 설정
  "zoomLevel": 0.7,             // 줌 레벨
  "databaseName": "fastapi_db"   // DB 이름
}
```

### relationshipType 코드
- `4`: 1:1 관계
- **`16`: 1:N 관계** ✅ (현재 사용)
- `2`: 0:1 관계
- `8`: 0:N 관계

### identification 설정
- **`false`: Non-identifying 관계** ✅ (현재 사용)
- `true`: Identifying 관계

## 📝 데이터베이스 연결

### 환경변수 설정
```bash
DATABASE_URL=postgresql://yousync:패스워드@yousync-db.cj60es4aa7pz.ap-northeast-2.rds.amazonaws.com:5432/fastapi_db
```

### psql 직접 연결
```bash
psql -h yousync-db.cj60es4aa7pz.ap-northeast-2.rds.amazonaws.com -U yousync -d fastapi_db
```

### Python 연결 (SQLAlchemy)
```python
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://yousync:패스워드@yousync-db.cj60es4aa7pz.ap-northeast-2.rds.amazonaws.com:5432/fastapi_db"
engine = create_engine(DATABASE_URL)
```

## 🔍 유용한 SQL 쿼리

### 테이블 목록 조회
```sql
\dt
```

### 외래키 관계 확인
```sql
SELECT
    tc.table_name AS child_table,
    kcu.column_name AS child_column,
    ccu.table_name AS parent_table,
    ccu.column_name AS parent_column,
    tc.constraint_name
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'
    AND tc.table_schema = 'public'
ORDER BY tc.table_name;
```

### 테이블별 레코드 수 확인
```sql
SELECT 
    schemaname,
    tablename,
    n_tup_ins - n_tup_del AS row_count
FROM pg_stat_user_tables
ORDER BY row_count DESC;
```

### 인덱스 정보 확인
```sql
SELECT
    t.relname AS table_name,
    i.relname AS index_name,
    a.attname AS column_name
FROM pg_class t,
     pg_class i,
     pg_index ix,
     pg_attribute a
WHERE t.oid = ix.indrelid
    AND i.oid = ix.indexrelid
    AND a.attrelid = t.oid
    AND a.attnum = ANY(ix.indkey)
    AND t.relkind = 'r'
ORDER BY t.relname, i.relname;
```

## 📈 데이터 흐름 시나리오

### 1. 사용자 북마크 생성
```
1. 사용자 로그인 (users)
2. 토큰 선택 (tokens)
3. 북마크 생성 (bookmarks)
```

### 2. 영상 분석 프로세스
```
1. 배우 정보 등록 (actors)
2. URL 등록 (urls)
3. 토큰 생성 (tokens)
4. 스크립트 추출 (scripts)
5. 단어 분석 (words)
6. 결과 저장 (analysis_results)
```

### 3. 배우-토큰 관계 설정
```
1. 배우 등록 (actors)
2. 토큰 생성 (tokens)
3. 관계 설정 (token_actors)
```

## 🚀 ERD 버전 히스토리

### v3.0 (2025-07-05) - complete_db_erd1.vuerd ✅
- ✅ **1:N 관계 정확히 설정** (relationshipType: 16)
- ✅ **Non-identifying 관계** (identification: false)
- ✅ **10개 테이블 완전 구현**
- ✅ **모든 외래키 관계 반영**
- ✅ **색상별 테이블 구분**
- ✅ **44KB 완전한 스키마**

### v2.0 (2025-07-04) - db.json
- ✅ 기본 ERD 구조 생성
- ❌ 영화 관련 테이블 (현재 사용 안함)

## 📊 현재 데이터베이스 상태

### 테이블 현황 (2025-07-05 기준)
```
users: 0 rows (빈 테이블)
bookmarks: 0 rows (빈 테이블)
tokens: 0 rows (빈 테이블)
urls: 0 rows (빈 테이블)
scripts: 0 rows (빈 테이블)
actors: 0 rows (빈 테이블)
actor_aliases: 0 rows (빈 테이블)
words: 0 rows (빈 테이블)
token_actors: 0 rows (빈 테이블)
analysis_results: 0 rows (빈 테이블)
```

### 스키마 상태
- ✅ **테이블 구조**: 완성
- ✅ **외래키 제약조건**: 설정됨
- ✅ **인덱스**: 설정됨
- ⚠️ **데이터**: 비어있음 (테스트 데이터 필요)

## 🔧 개발 가이드

### ERD 수정 시 주의사항
1. **complete_db_erd1.vuerd** 파일만 수정
2. 수정 후 JSON 유효성 검사 필수
3. 관계 타입 변경 시 팀원들과 협의
4. 백업 파일 생성 권장

### 새로운 테이블 추가 시
1. ERD에서 테이블 추가
2. 관계 설정
3. 실제 DB에 마이그레이션 적용
4. README 업데이트

## 📞 문의사항
ERD 관련 문의사항이 있으시면 개발팀에 연락해주세요.

---
**마지막 업데이트**: 2025-07-05  
**ERD 파일**: complete_db_erd1.vuerd (44KB)  
**데이터베이스**: PostgreSQL 15.12 on AWS RDS  
**관계 타입**: 1:N (One-to-Many) × 10개  
**상태**: ✅ 완성, 테스트 데이터 필요
