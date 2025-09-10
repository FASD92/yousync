# Redis + RQ 구현 가이드

## 🚀 빠른 시작 가이드

### 1. 의존성 설치

```bash
cd back-end
pip install redis rq rq-dashboard boto3
```

### 2. Redis 설치 및 실행

#### macOS (Homebrew)
```bash
brew install redis
brew services start redis
```

#### Docker
```bash
docker run -d -p 6379:6379 --name redis redis:7-alpine
```

### 3. 환경 변수 설정

`.env` 파일에 추가:
```bash
# Redis 설정
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# SQS 설정
SQS_QUEUE_URL=https://sqs.ap-northeast-2.amazonaws.com/YOUR_ACCOUNT/audio-analysis-queue
SQS_RESULT_QUEUE_URL=https://sqs.ap-northeast-2.amazonaws.com/YOUR_ACCOUNT/analysis-results-queue

# 큐 시스템 활성화
USE_QUEUE_SYSTEM=true
```

## 📁 새로운 파일 구조

```
back-end/
├── redis_client.py          # Redis 연결 및 큐 설정
├── tasks/
│   ├── __init__.py
│   ├── audio_analysis.py    # 오디오 분석 작업 함수
│   └── sqs_handler.py       # SQS 메시지 처리
├── services/
│   ├── queue_service.py     # 큐 관련 서비스
│   └── analysis_service.py  # 분석 관련 서비스
└── workers/
    ├── __init__.py
    └── rq_worker.py         # RQ 워커 설정
```

## 🔧 구현 코드

### 1. Redis 클라이언트 설정

```python
# redis_client.py
import redis
import os
from rq import Queue
from typing import Optional

class RedisClient:
    _instance: Optional[redis.Redis] = None
    
    @classmethod
    def get_client(cls) -> redis.Redis:
        if cls._instance is None:
            cls._instance = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=int(os.getenv('REDIS_DB', 0)),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
        return cls._instance

# 큐 인스턴스들
redis_client = RedisClient.get_client()
audio_analysis_queue = Queue('audio_analysis', connection=redis_client)
high_priority_queue = Queue('high_priority', connection=redis_client)
low_priority_queue = Queue('low_priority', connection=redis_client)
```

### 2. 작업 함수 정의

```python
# tasks/audio_analysis.py
import json
import boto3
import logging
from typing import Dict, Any
from database import SessionLocal
from models import AnalysisResult

logger = logging.getLogger(__name__)

def process_audio_analysis(job_id: str, s3_url: str, token_id: int, token_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    RQ 작업 함수: 오디오 분석 처리
    
    Args:
        job_id: 작업 ID
        s3_url: S3 오디오 파일 URL
        token_id: 토큰 ID
        token_info: 토큰 관련 정보
    
    Returns:
        처리 결과 딕셔너리
    """
    db = SessionLocal()
    
    try:
        logger.info(f"[작업 시작] job_id={job_id}, token_id={token_id}")
        
        # 1. 상태 업데이트: 처리 시작
        update_analysis_status(db, job_id, "processing", 50, "SQS 메시지 전송 중...")
        
        # 2. SQS 메시지 생성
        sqs_message = {
            "job_id": job_id,
            "s3_audio_url": s3_url,
            "token_id": token_id,
            "s3_textgrid_url": token_info.get('s3_textgrid_url'),
            "s3_pitch_url": token_info.get('s3_pitch_url'),
            "callback_url": f"{os.getenv('WEBHOOK_URL')}?job_id={job_id}",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # 3. SQS에 메시지 전송
        message_id = send_to_sqs(sqs_message)
        logger.info(f"[SQS 전송 완료] job_id={job_id}, message_id={message_id}")
        
        # 4. 상태 업데이트: SQS 전송 완료
        update_analysis_status(
            db, job_id, 
            status="queued_for_analysis", 
            progress=80, 
            message="분석 서버 대기열에 추가됨",
            metadata={"sqs_message_id": message_id}
        )
        
        return {
            "status": "success",
            "job_id": job_id,
            "sqs_message_id": message_id,
            "message": "SQS 큐에 성공적으로 추가됨"
        }
        
    except Exception as e:
        logger.error(f"[작업 실패] job_id={job_id}, error={str(e)}")
        update_analysis_status(db, job_id, "failed", 0, f"처리 실패: {str(e)}")
        raise
    finally:
        db.close()

def send_to_sqs(message: Dict[str, Any]) -> str:
    """SQS에 메시지 전송"""
    try:
        sqs = boto3.client(
            'sqs',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'ap-northeast-2')
        )
        
        queue_url = os.getenv('SQS_QUEUE_URL')
        if not queue_url:
            raise ValueError("SQS_QUEUE_URL 환경 변수가 설정되지 않았습니다.")
        
        response = sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(message, ensure_ascii=False),
            MessageAttributes={
                'job_type': {
                    'StringValue': 'audio_analysis',
                    'DataType': 'String'
                },
                'priority': {
                    'StringValue': 'normal',
                    'DataType': 'String'
                }
            }
        )
        
        return response['MessageId']
        
    except Exception as e:
        logger.error(f"SQS 전송 실패: {str(e)}")
        raise

def update_analysis_status(db, job_id: str, status: str, progress: int, message: str, metadata: Dict = None):
    """분석 상태 업데이트"""
    try:
        analysis_result = db.query(AnalysisResult).filter(AnalysisResult.job_id == job_id).first()
        if analysis_result:
            analysis_result.status = status
            analysis_result.progress = progress
            analysis_result.message = message
            if metadata:
                # 기존 메타데이터와 병합
                existing_metadata = analysis_result.metadata or {}
                existing_metadata.update(metadata)
                analysis_result.metadata = existing_metadata
            db.commit()
            logger.info(f"[상태 업데이트] job_id={job_id}, status={status}, progress={progress}")
    except Exception as e:
        logger.error(f"상태 업데이트 실패: {str(e)}")
        db.rollback()
```

### 3. 큐 서비스 클래스

```python
# services/queue_service.py
from typing import Dict, Any, Optional
from rq import Queue
from rq.job import Job
from redis_client import audio_analysis_queue, high_priority_queue, low_priority_queue
import logging

logger = logging.getLogger(__name__)

class QueueService:
    """큐 관리 서비스"""
    
    @staticmethod
    def enqueue_audio_analysis(
        job_id: str, 
        s3_url: str, 
        token_id: int, 
        token_info: Dict[str, Any],
        priority: str = 'normal'
    ) -> Job:
        """오디오 분석 작업을 큐에 추가"""
        
        # 우선순위에 따른 큐 선택
        queue_map = {
            'high': high_priority_queue,
            'normal': audio_analysis_queue,
            'low': low_priority_queue
        }
        
        selected_queue = queue_map.get(priority, audio_analysis_queue)
        
        job = selected_queue.enqueue(
            'tasks.audio_analysis.process_audio_analysis',
            job_id=job_id,
            s3_url=s3_url,
            token_id=token_id,
            token_info=token_info,
            job_timeout='15m',  # 15분 타임아웃
            retry=3,  # 3회 재시도
            job_id=f"audio_analysis_{job_id}"  # RQ job ID 설정
        )
        
        logger.info(f"[큐 추가] job_id={job_id}, rq_job_id={job.id}, priority={priority}")
        return job
    
    @staticmethod
    def get_job_status(rq_job_id: str) -> Optional[Dict[str, Any]]:
        """RQ 작업 상태 조회"""
        try:
            job = Job.fetch(rq_job_id, connection=audio_analysis_queue.connection)
            return {
                "id": job.id,
                "status": job.get_status(),
                "created_at": job.created_at,
                "started_at": job.started_at,
                "ended_at": job.ended_at,
                "result": job.result,
                "exc_info": job.exc_info
            }
        except Exception as e:
            logger.error(f"RQ 작업 상태 조회 실패: {str(e)}")
            return None
    
    @staticmethod
    def cancel_job(rq_job_id: str) -> bool:
        """작업 취소"""
        try:
            job = Job.fetch(rq_job_id, connection=audio_analysis_queue.connection)
            job.cancel()
            logger.info(f"[작업 취소] rq_job_id={rq_job_id}")
            return True
        except Exception as e:
            logger.error(f"작업 취소 실패: {str(e)}")
            return False
    
    @staticmethod
    def get_queue_info() -> Dict[str, Any]:
        """큐 상태 정보 조회"""
        return {
            "audio_analysis": {
                "length": len(audio_analysis_queue),
                "failed_count": audio_analysis_queue.failed_job_registry.count,
                "started_count": audio_analysis_queue.started_job_registry.count
            },
            "high_priority": {
                "length": len(high_priority_queue),
                "failed_count": high_priority_queue.failed_job_registry.count,
                "started_count": high_priority_queue.started_job_registry.count
            },
            "low_priority": {
                "length": len(low_priority_queue),
                "failed_count": low_priority_queue.failed_job_registry.count,
                "started_count": low_priority_queue.started_job_registry.count
            }
        }
```

### 4. 수정된 API 엔드포인트

```python
# router/user_audio_router.py (수정된 부분)
from services.queue_service import QueueService
import os

@router.post("/{token_id}/upload-audio")
async def upload_audio_by_token_id(
    request: Request,
    token_id: str = Path(...),
    file: UploadFile = File(...),
    priority: str = "normal",  # 우선순위 파라미터 추가
    db: Session = Depends(get_db)
):
    try:
        s3_client = request.app.state.s3_client
        job_id = str(uuid4())
        file_data = await file.read()
        token_info = await get_token_by_id(token_id, db)
        
        # DB에 초기 상태 저장
        analysis_result = create_analysis_result(
            db, job_id, int(token_id), 
            status="uploading", 
            progress=10, 
            message="S3 업로드 중..."
        )
        
        # S3 업로드
        update_analysis_result(db, job_id, progress=30, message="S3 업로드 중...")
        s3_key = await upload_to_s3_async(s3_client, file_data, file.filename)
        s3_url = f"s3://{S3_BUCKET}/{s3_key}"
        
        # 큐 시스템 사용 여부 확인
        if os.getenv('USE_QUEUE_SYSTEM', 'false').lower() == 'true':
            # 새로운 큐 방식
            update_analysis_result(db, job_id, status="queuing", progress=40, message="작업 큐에 추가 중...")
            
            # RQ 큐에 작업 추가
            rq_job = QueueService.enqueue_audio_analysis(
                job_id=job_id,
                s3_url=s3_url,
                token_id=int(token_id),
                token_info={
                    'id': token_info.id,
                    's3_textgrid_url': getattr(token_info, 's3_textgrid_url', None),
                    's3_pitch_url': getattr(token_info, 's3_pitch_url', None)
                },
                priority=priority
            )
            
            update_analysis_result(
                db, job_id, 
                status="queued", 
                progress=50, 
                message="작업 큐에 추가됨",
                metadata={"rq_job_id": rq_job.id}
            )
            
            return {
                "message": "업로드 완료, 작업 큐에 추가되었습니다.",
                "job_id": job_id,
                "rq_job_id": rq_job.id,
                "status": "queued",
                "priority": priority,
                "queue_position": len(audio_analysis_queue) if priority == 'normal' else 0
            }
        else:
            # 기존 httpx 방식 (하위 호환성)
            return await upload_with_httpx_legacy(...)
            
    except Exception as e:
        logging.error(f"업로드 실패: {str(e)}")
        if 'analysis_result' in locals():
            update_analysis_result(db, job_id, status="failed", progress=0, message=f"업로드 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"에러: {str(e)}")

# 큐 상태 조회 API 추가
@router.get("/queue/status")
def get_queue_status():
    """큐 상태 조회"""
    return QueueService.get_queue_info()

# 작업 취소 API 추가
@router.delete("/jobs/{job_id}/cancel")
def cancel_analysis_job(job_id: str, db: Session = Depends(get_db)):
    """분석 작업 취소"""
    analysis_result = get_analysis_result(db, job_id)
    if not analysis_result:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    
    if analysis_result.status in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="이미 완료된 작업은 취소할 수 없습니다.")
    
    # RQ 작업 취소
    metadata = analysis_result.metadata or {}
    rq_job_id = metadata.get('rq_job_id')
    
    if rq_job_id:
        success = QueueService.cancel_job(rq_job_id)
        if success:
            update_analysis_result(db, job_id, status="cancelled", message="사용자에 의해 취소됨")
            return {"message": "작업이 취소되었습니다.", "job_id": job_id}
    
    raise HTTPException(status_code=500, detail="작업 취소에 실패했습니다.")
```

### 5. 워커 실행 스크립트

```python
# workers/rq_worker.py
import os
import sys
import logging
from rq import Worker
from redis_client import redis_client

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_worker():
    """RQ 워커 실행"""
    # 큐 이름들
    queue_names = ['high_priority', 'audio_analysis', 'low_priority']
    
    # 워커 생성
    worker = Worker(
        queue_names,
        connection=redis_client,
        name=f"worker-{os.getpid()}",
        default_result_ttl=3600  # 결과 1시간 보관
    )
    
    logger.info(f"RQ Worker 시작: PID={os.getpid()}, Queues={queue_names}")
    
    try:
        worker.work(with_scheduler=True)
    except KeyboardInterrupt:
        logger.info("워커 종료 중...")
        worker.stop()

if __name__ == '__main__':
    run_worker()
```

### 6. 실행 명령어

```bash
# 1. FastAPI 서버 실행
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 2. RQ 워커 실행 (별도 터미널)
python workers/rq_worker.py

# 3. RQ 대시보드 실행 (선택사항, 별도 터미널)
rq-dashboard --redis-url redis://localhost:6379

# 4. 여러 워커 실행 (성능 향상)
python workers/rq_worker.py &
python workers/rq_worker.py &
python workers/rq_worker.py &
```

## 🔍 모니터링 및 디버깅

### RQ 대시보드 접속
- URL: http://localhost:9181
- 실시간 큐 상태, 작업 진행 상황, 실패한 작업 확인 가능

### 로그 확인
```bash
# FastAPI 로그
tail -f server.log

# RQ 워커 로그
tail -f worker.log
```

### Redis 상태 확인
```bash
redis-cli info
redis-cli monitor  # 실시간 명령어 모니터링
```

## 🚨 트러블슈팅

### 1. Redis 연결 실패
```bash
# Redis 서비스 상태 확인
brew services list | grep redis

# Redis 재시작
brew services restart redis
```

### 2. 작업이 처리되지 않음
- 워커 프로세스가 실행 중인지 확인
- 큐에 작업이 쌓여있는지 RQ 대시보드에서 확인
- Redis 메모리 사용량 확인

### 3. SQS 권한 오류
```bash
# AWS 자격 증명 확인
aws sts get-caller-identity

# SQS 큐 권한 확인
aws sqs get-queue-attributes --queue-url YOUR_QUEUE_URL --attribute-names All
```

이제 단계별로 구현하시면 됩니다! 궁금한 점이 있으면 언제든 물어보세요.
