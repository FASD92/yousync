import os, logging, asyncio, json
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, Path, Request, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from database import get_db, SessionLocal
from models import YoutubeProcessJob, Token
from schemas import YoutubeProcessRequest, YoutubeProcessResponse, YoutubeProcessStatusResponse
import httpx

# ────────────── 환경 변수 ──────────────
PREPROCESS_SERVER_URL = os.getenv("PREPROCESS_SERVER_URL")
YOUTUBE_WEBHOOK_URL = os.getenv("YOUTUBE_WEBHOOK_URL")

# ────────────── DB 헬퍼 ──────────────
def create_youtube_process_job(db: Session, job_id: str, user_id: int = None, status: str = "processing", progress: int = 10, message: str = "요청 접수"):
    job = YoutubeProcessJob(
        job_id=job_id,
        user_id=user_id, # 현재는 사용하지 않지만, 추후 사용자 인증 추가 시 활용
        status=status,
        progress=progress,
        message=message,
        token_id=None # 초기에는 token_id가 없음
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job

def update_youtube_process_job(db: Session, job_id: str, **kw):
    job = db.query(YoutubeProcessJob).filter_by(job_id=job_id).first()
    if job:
        for k, v in kw.items(): setattr(job, k, v)
        db.commit()
        db.refresh(job)
    return job

def get_youtube_process_job(db: Session, job_id: str):
    return db.query(YoutubeProcessJob).filter_by(job_id=job_id).first()

# ────────────── 비동기 유틸 ──────────────
async def send_preprocess_request_async(
    youtube_url: str, movie_name: str, actor_name: str, webhook_url: str, job_id: str
):
    payload = {
        "youtube_url": youtube_url,
        "movie_name": movie_name,
        "actor_name": actor_name,
        "webhook_url": webhook_url,
        "job_id": job_id
    }
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client: # 전처리 시간 고려하여 타임아웃 5분 설정
            resp = await client.post(
                f"{PREPROCESS_SERVER_URL}/process",
                json=payload
            )
            resp.raise_for_status()
            logging.info(f"[전처리 서버 요청 성공] job_id={job_id}, 응답: {resp.json()}")
            return resp.json()
    except httpx.HTTPError as e:
        logging.error(f"[전처리 서버 요청 실패] job_id={job_id} - {e}, 응답: {e.response.text if e.response else 'N/A'}")
        raise

# ────────────── Router ──────────────
router = APIRouter(prefix="/youtube", tags=["youtube_process"])

# 1) 유튜브 URL 전처리 요청
@router.post("/process", response_model=YoutubeProcessResponse)
async def process_youtube_url(
    request_data: YoutubeProcessRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    job_id = uuid4().hex
    
    # DB에 초기 상태 저장
    create_youtube_process_job(db, job_id, status="processing", progress=10, message="전처리 요청 접수")

    # 백그라운드에서 전처리 서버에 요청
    async def bg_work():
        bg_db = SessionLocal()
        try:
            update_youtube_process_job(bg_db, job_id, progress=40, message="전처리 서버 호출 중...")
            
            webhook_url = f"{YOUTUBE_WEBHOOK_URL}?job_id={job_id}"
            
            await send_preprocess_request_async(
                request_data.youtube_url,
                request_data.movie_name,
                request_data.actor_name,
                webhook_url,
                job_id
            )
            
            update_youtube_process_job(bg_db, job_id, progress=70, message="전처리 서버 응답 대기 중...")
            
        except Exception as e:
            logging.error(f"유튜브 전처리 백그라운드 작업 실패: {str(e)}")
            update_youtube_process_job(bg_db, job_id, status="failed", message=str(e), progress=0)
        finally:
            bg_db.close()

    background_tasks.add_task(bg_work)
    
    return YoutubeProcessResponse(
        job_id=job_id,
        status="processing",
        message="유튜브 전처리 요청이 접수되었습니다. 백그라운드에서 처리됩니다."
    )

# 2) 전처리 서버 웹훅 (결과 수신)
@router.post("/webhook/process-complete")
async def process_complete_webhook(request: Request, db: Session = Depends(get_db)):
    logging.info("=" * 50)
    logging.info("[🔔 웹훅 호출됨] 유튜브 전처리 결과 웹훅 수신")
    logging.info(f"[웹훅 요청 IP] {request.client.host if request.client else 'Unknown'}")
    logging.info(f"[웹훅 헤더] {dict(request.headers)}")
    
    job_id = request.query_params.get("job_id")
    logging.info(f"[웹훅 파라미터] job_id={job_id}")
    
    if not job_id:
        logging.warning("[❗경고] 웹훅에 job_id 없음")
        raise HTTPException(400, "job_id missing")
        
    payload = await request.json()
    logging.info(f"[웹훅 데이터] 받은 결과: {payload}")
    
    token_ids = payload.get("token_ids")
    status = payload.get("status", "completed")
    message = payload.get("message", "전처리 완료")
    
    if token_ids:
        # token_ids가 있으면 성공으로 간주하고 업데이트
        update_youtube_process_job(db, job_id, 
                               status=status, 
                               progress=100, 
                               result=payload, 
                               message=message,
                               token_id=token_ids[0] if token_ids else None)
        logging.info(f"[✅ 웹훅 처리 완료] job_id={job_id}, token_ids={token_ids}")
    else:
        # token_ids가 없으면 실패로 간주
        update_youtube_process_job(db, job_id, 
                               status="failed", 
                               progress=0, 
                               result=payload, 
                               message=message or "전처리 실패: token_ids 없음")
        logging.error(f"[❌ 웹훅 처리 실패] job_id={job_id}, token_ids 없음")

    logging.info("=" * 50)
    return {"received": True, "job_id": job_id}

# 3) 결과 조회
@router.get("/process-status/{job_id}", response_model=YoutubeProcessStatusResponse)
def get_process_status(job_id: str, db: Session = Depends(get_db)):
    ar = get_youtube_process_job(db, job_id)
    if not ar:
        raise HTTPException(404, "결과를 찾을 수 없습니다.")
    
    token_ids = ar.result.get("token_ids") if ar.result else None

    return YoutubeProcessStatusResponse(
        job_id=ar.job_id,
        status=ar.status,
        progress=ar.progress,
        message=ar.message,
        token_id=ar.token_id,
        token_ids=token_ids,
        result=ar.result
    )

# 4) SSE 진행 스트림
@router.get("/process-progress/{job_id}")
async def stream_process_progress(job_id: str):
    async def gen():
        while True:
            db = SessionLocal()
            try:
                ar = get_youtube_process_job(db, job_id)
                if not ar:
                    yield "data: {\"error\":\"Job not found\"}\n\n"
                    break
                
                token_ids = ar.result.get("token_ids") if ar.result else None
                
                data = {
                    "job_id":   ar.job_id,
                    "status":   ar.status,
                    "progress": ar.progress,
                    "message":  ar.message,
                    "token_id": ar.token_id,
                    "token_ids": token_ids,
                    "result":   ar.result,
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                if ar.status in ("completed", "failed"): break
            finally:
                db.close()
            await asyncio.sleep(5)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache",
                                      "Connection":"keep-alive",
                                      "Access-Control-Allow-Origin":"*"})