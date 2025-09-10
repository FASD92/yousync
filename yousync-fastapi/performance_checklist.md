# YouSync FastAPI 성능 개선 체크리스트

## 🚀 확실한 성능 개선 효과가 있는 것들

### 1. FastAPI 서버 최적화
```python
# uvicorn 실행 시 워커 수 증가
uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000

# 또는 gunicorn 사용
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### 2. 데이터베이스 연결 풀 최적화
```python
# database.py에서
engine = create_engine(
    DATABASE_URL,
    pool_size=20,          # 기본 5 → 20으로 증가
    max_overflow=30,       # 추가 연결 허용
    pool_pre_ping=True,    # 연결 상태 확인
    pool_recycle=3600      # 1시간마다 연결 재생성
)
```

### 3. S3 업로드 최적화
```python
# 멀티파트 업로드 + 비동기 처리
import aioboto3

# 현재보다 더 빠른 업로드
async def upload_to_s3_optimized(file_content, key):
    session = aioboto3.Session()
    async with session.client('s3') as s3:
        await s3.upload_fileobj(file_content, bucket, key)
```

### 4. 캐싱 도입
```python
# Redis 캐싱으로 반복 쿼리 최적화
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

# 스크립트 데이터, 배우 정보 등 캐싱
@cache(expire=3600)  # 1시간 캐싱
async def get_script_data(script_id: int):
    # DB 쿼리 대신 캐시에서 반환
```

## 📊 성능 측정 방법

### 1. API 응답 시간 측정
```bash
# 실제 API 성능 테스트
curl -w "@curl-format.txt" -o /dev/null -s "http://your-server/tokens/1"

# 또는 ab 테스트
ab -n 100 -c 10 http://your-server/health
```

### 2. 메모리 사용량 실시간 모니터링
```bash
# YouSync 프로세스만 모니터링
watch -n 1 'ps aux | grep -E "(python|uvicorn)" | grep -v grep'
```

### 3. 데이터베이스 쿼리 성능
```sql
-- PostgreSQL 슬로우 쿼리 확인
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC LIMIT 10;
```

## 🎯 우선순위별 개선 방안

### 🔥 즉시 효과 (High Impact)
1. **FastAPI 워커 수 증가** → 동시 요청 처리 능력 ↑
2. **DB 연결 풀 최적화** → 쿼리 응답 속도 ↑
3. **S3 업로드 최적화** → 파일 업로드 속도 ↑

### 🚀 중장기 효과 (Medium Impact)  
1. **Redis 캐싱 도입** → 반복 쿼리 속도 ↑
2. **CDN 도입** → 정적 파일 로딩 속도 ↑
3. **로드 밸런서** → 트래픽 분산

### 🔧 미세 최적화 (Low Impact)
1. **netdata 경량화** → 메모리 100MB 절약
2. **로그 로테이션** → 디스크 공간 절약
3. **불필요한 서비스 제거** → 약간의 리소스 절약

## 💡 결론
netdata 최적화는 "안 하는 것보다는 낫지만", 
실제 체감 성능 개선을 원한다면 위의 1~3번을 먼저 적용하세요!
