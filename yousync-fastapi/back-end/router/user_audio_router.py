# router/user_audio_router.py

import os, logging, boto3, asyncio, json
from uuid import uuid4
from typing import List
from fastapi import APIRouter, UploadFile, File, Path, HTTPException, Request, BackgroundTasks, Depends, Form
from fastapi.responses import StreamingResponse
from concurrent.futures import ThreadPoolExecutor
import httpx
from database import get_db
from sqlalchemy.orm import Session
from models import Token, AnalysisResult, User
from services.sqs_service import sqs_service
from router.auth_router import get_current_user  # 인증 함수 import


async def get_token_by_id(token_id: str, db:Session):
    token = db.query(Token).filter(Token.id == int(token_id)).first()
    if not token:
        raise HTTPException(status_code=404, detail="Token not found")
    return token

# DB 헬퍼 함수들
def create_analysis_result(db: Session, job_id: str, token_id: int, user_id: int = None, status: str = "processing", progress: int = 10, message: str = "업로드 시작"):
    analysis_result = AnalysisResult(
        job_id=job_id,
        token_id=token_id,
        user_id=user_id,  # 사용자 ID 추가
        status=status,
        progress=progress,
        message=message
    )
    db.add(analysis_result)
    db.commit()
    db.refresh(analysis_result)
    return analysis_result

def update_analysis_result(db: Session, job_id: str, **kwargs):
    analysis_result = db.query(AnalysisResult).filter(AnalysisResult.job_id == job_id).first()
    if analysis_result:
        for key, value in kwargs.items():
            setattr(analysis_result, key, value)
        db.commit()
        db.refresh(analysis_result)
    return analysis_result

def get_analysis_result(db: Session, job_id: str):
    return db.query(AnalysisResult).filter(AnalysisResult.job_id == job_id).first()



# ===============================================

router = APIRouter(
    prefix="/tokens",
    tags=["tokens"]
)

S3_BUCKET = os.getenv("S3_BUCKET_NAME")
TARGET_URL = os.getenv("TARGET_SERVER_URL")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

# 비동기 S3 업로드 함수
async def upload_to_s3_async(s3_client, file_data: bytes, filename: str) -> str:
    """비동기 S3 업로드"""
    def sync_upload():
        import io
        key = f"audio/{uuid4().hex}_{filename}"
        s3_client.upload_fileobj(io.BytesIO(file_data), S3_BUCKET, key)
        return key
    
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(executor, sync_upload)

# SQS 메시지 전송 함수
async def send_to_sqs_async(s3_url: str, token_id: str, webhook_url: str, job_id: str, token_info: Token):
    """SQS를 사용한 분석 요청"""
    try:
        # 토큰 정보를 딕셔너리로 변환
        token_info_dict = {
            's3_textgrid_url': getattr(token_info, 's3_textgrid_url', None),
            's3_pitch_url': getattr(token_info, 's3_pitch_url', None)
        }
        
        # SQS에 메시지 전송
        message_id = sqs_service.send_analysis_message(
            job_id=job_id,
            s3_audio_url=s3_url,
            token_id=str(token_id),
            webhook_url=webhook_url,
            token_info=token_info_dict
        )
        
        if message_id:
            logging.info(f"[SQS 전송 성공] job_id={job_id}, message_id={message_id}")
            return {"message_id": message_id, "status": "queued"}
        else:
            raise Exception("SQS 메시지 전송 실패")
            
    except Exception as e:
        logging.error(f"[SQS 전송 실패] job_id={job_id}, error={str(e)}")
        raise

# 비동기 HTTP 요청 함수 (기존 방식 - 하위 호환용)
async def send_analysis_request_async(s3_url: str, token_id: str, webhook_url: str, job_id: str, token_info: Token):
    """httpx를 사용한 완전 비동기 분석 요청"""
    try:
        async with httpx.AsyncClient(timeout=70.0) as client:
            response = await client.post(
                TARGET_URL,
                data={
                    "s3_audio_url": s3_url,
                    "video_id": token_id,
                    "webhook_url": webhook_url,
                    "s3_textgrid_url": f"s3://testgrid-pitch-bgvoice-yousync/{token_info.s3_textgrid_url}" if token_info.s3_textgrid_url else None,
                    "s3_pitch_url": f"s3://testgrid-pitch-bgvoice-yousync/{token_info.s3_pitch_url}" if token_info.s3_pitch_url else None
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()
            logging.info(f"[분석 요청 성공] job_id={job_id}")
            
            # 응답 데이터 확인 및 반환
            if response.text and response.text.strip():
                try:
                    response_data = response.json()
                    logging.info(f"[분석 서버 응답] {len(str(response_data))} 문자")
                    return response_data
                except:
                    logging.info(f"[분석 서버 텍스트 응답] {response.text}")
                    
    except httpx.HTTPStatusError as e:
        logging.error(f"[분석 요청 실패] job_id={job_id}, status={e.response.status_code}, body={e.response.text}")
        raise
    except Exception as e:
        logging.error(f"[분석 요청 실패] job_id={job_id}, error={e}")
        raise

# 1. 오디오 업로드 및 분석 요청 (완전 비동기 처리)
@router.post("/{token_id}/upload-audio/")
async def upload_audio_by_token_id(
    request: Request,
    token_id: str = Path(...),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)  # 🔐 로그인한 사용자만 호출 가능
    
):
    try:
        s3_client = request.app.state.s3_client
        job_id = str(uuid4())
        file_data = await file.read()
        token_info = await get_token_by_id(token_id, db)
        
        # DB에 초기 상태 저장
        analysis_result = create_analysis_result(db, job_id, int(token_id), current_user.id)

        # 백그라운드에서 완전 비동기 처리
        async def process_in_background(s3_client_bg):
            # 새로운 DB 세션 생성 (백그라운드 작업용)
            from database import SessionLocal
            bg_db = SessionLocal()
            try:
                # S3 업로드
                update_analysis_result(bg_db, job_id, progress=40, message="S3 업로드 중...")
                
                s3_key = await upload_to_s3_async(s3_client_bg, file_data, file.filename)
                s3_url = f"s3://{S3_BUCKET}/{s3_key}"
                
                webhook_url = f"{WEBHOOK_URL}?job_id={job_id}"
                
                # 환경 변수로 SQS 사용 여부 결정
                use_sqs = os.getenv('USE_SQS_QUEUE', 'false').lower() == 'true'
                
                if use_sqs:
                    # SQS 방식
                    update_analysis_result(bg_db, job_id, progress=70, message="SQS 큐에 메시지 전송 중...")
                    
                    sqs_result = await send_to_sqs_async(
                        s3_url, 
                        token_info.id, 
                        webhook_url, 
                        job_id,
                        token_info
                    )
                    
                    update_analysis_result(
                        bg_db, job_id, 
                        status="queued_for_analysis", 
                        progress=90, 
                        message="SQS 큐에 전송 완료, 분석 대기 중...",
                        metadata={"sqs_message_id": sqs_result.get("message_id")}
                    )
                    logging.info(f"[SQS 방식 완료] job_id={job_id}")
                    
                else:
                    # 기존 HTTP 방식
                    update_analysis_result(bg_db, job_id, progress=70, message="분석 서버 요청 중...")
                    
                    response_data = await send_analysis_request_async(
                        s3_url, 
                        token_info.id, 
                        webhook_url, 
                        job_id,
                        token_info
                    )
                    
                    # POST 응답에서 실제 분석 결과가 있는지 확인
                    if response_data and isinstance(response_data, dict) and 'scores' in response_data:
                        # 실제 분석 결과를 받은 경우
                        update_analysis_result(bg_db, job_id, 
                                              status="completed", 
                                              progress=100, 
                                              result=response_data, 
                                              message="분석 완료")
                        logging.info(f"[HTTP POST 응답으로 분석 완료] job_id={job_id}")
                    else:
                        # 웹훅 대기 상태로 설정
                        update_analysis_result(bg_db, job_id, progress=90, message="분석 중... 결과 대기")
                        logging.info(f"[HTTP 웹훅 대기] job_id={job_id}")
                
            except Exception as e:
                    # 웹훅 대기 상태로 설정
                    update_analysis_result(bg_db, job_id, progress=90, message="분석 중... 결과 대기")
                    logging.info(f"[웹훅 대기] job_id={job_id}")
                
            except Exception as e:
                logging.error(f"백그라운드 처리 실패: {str(e)}")
                update_analysis_result(bg_db, job_id, status="failed", message=str(e), progress=0)
            finally:
                bg_db.close()

        background_tasks.add_task(process_in_background, s3_client)
        
        return {
            "message": "업로드 완료, 백그라운드에서 처리됩니다.",
            "job_id": job_id,
            "status": "processing",
            "token_info": {
                "id": token_info.id,
                "s3_textgrid_url": getattr(token_info, 's3_textgrid_url', None),
                "s3_pitch_url": getattr(token_info, 's3_pitch_url', None)
            }
        }
        
    except Exception as e:
        logging.error(f"업로드 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"에러: {str(e)}")


# 2. 분석 결과를 수신할 웹훅 엔드포인트
@router.post("/webhook/analysis-complete/")
async def receive_analysis(request: Request, db: Session = Depends(get_db)):
    from fastapi.responses import JSONResponse
    
    # 웹훅 호출 로깅 추가
    logging.info("=" * 50)
    logging.info("[🔔 웹훅 호출됨] Token 분석 결과 웹훅 수신")
    logging.info(f"[웹훅 요청 IP] {request.client.host if request.client else 'Unknown'}")
    logging.info(f"[웹훅 헤더] {dict(request.headers)}")
    
    job_id = request.query_params.get("job_id")
    task_id = request.query_params.get("task_id")
    
    logging.info(f"[웹훅 파라미터] job_id={job_id}, task_id={task_id}")

    if not job_id:
        logging.warning("[❗경고] job_id 없이 webhook 도착. 무시됨")
        return JSONResponse(status_code=400, content={"error": "job_id is required"})

    # 이미 완료된 job_id인지 확인
    existing_result = get_analysis_result(db, job_id)
    if existing_result and existing_result.status == "completed":
        logging.info(f"[중복 요청 무시] job_id={job_id}는 이미 완료된 상태")
        return {"received": True, "job_id": job_id, "message": "이미 완료된 작업"}

    data = await request.json()
    # analysis_results 키가 있으면 그것을, 없으면 전체 데이터를 사용
    if "analysis_results" in data:
        results = data.get("analysis_results", {})
    else:
        results = data  # 전체 데이터를 결과로 사용
    
    logging.info(f"[웹훅 데이터] 받은 결과 크기: {len(str(results))} 문자")
    logging.info(f"[웹훅 데이터] 결과 키들: {list(results.keys()) if isinstance(results, dict) else 'Not dict'}")
    
    # 분석 완료 상태로 업데이트
    update_analysis_result(db, job_id, 
                          status="completed", 
                          progress=100, 
                          result=results, 
                          message="분석 완료")

    logging.info(f"[✅ 웹훅 처리 완료] job_id={job_id}, task_id={task_id}")
    logging.info("=" * 50)
    return {"received": True, "job_id": job_id, "task_id": task_id}


# 3. 클라이언트가 조회할 수 있는 결과 조회 API
@router.get("/analysis-result/{job_id}/")
def get_analysis_result_api(job_id: str, db: Session = Depends(get_db)):
    """분석 결과 조회"""
    stored_data = get_analysis_result(db, job_id)
    if not stored_data:
        raise HTTPException(status_code=404, detail="분석 결과를 찾을 수 없습니다.")
    
    return {
        "job_id": stored_data.job_id,
        "token_id": stored_data.token_id,
        "status": stored_data.status,
        "progress": stored_data.progress,
        "result": stored_data.result,
        "message": stored_data.message,
        "created_at": stored_data.created_at
    }


# 4. 실시간 진행 상황을 위한 Server-Sent Events 엔드포인트
@router.get("/analysis-progress/{job_id}/")
async def stream_analysis_progress(job_id: str):
    """실시간 진행 상황 스트리밍 (SSE)"""
    async def event_generator():
        from database import SessionLocal
        
        while True:
            try:
                # 새로운 DB 세션으로 현재 상태 조회
                db = SessionLocal()
                try:
                    current_data = get_analysis_result(db, job_id)
                    if not current_data:
                        yield f"data: {json.dumps({'error': 'Job not found'}, ensure_ascii=False)}\n\n"
                        break
                    
                    data_dict = {
                        "job_id": current_data.job_id,
                        "status": current_data.status,
                        "progress": current_data.progress,
                        "message": current_data.message,
                        "result": current_data.result
                    }
                    
                    # 완료된 경우 마지막 데이터 전송 후 종료
                    if current_data.status in ["completed", "failed"]:
                        yield f"data: {json.dumps(data_dict, ensure_ascii=False)}\n\n"
                        break
                    
                    yield f"data: {json.dumps(data_dict, ensure_ascii=False)}\n\n"
                finally:
                    db.close()
                    
                await asyncio.sleep(3)  # 1초마다 업데이트
                
            except Exception as e:
                logging.error(f"SSE 스트림 오류: {e}")
                error_data = {
                    "status": "error", 
                    "error": str(e),
                    "job_id": job_id
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                break
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )

# SQS 큐 상태 조회 API
@router.get("/queue/status/")
def get_sqs_queue_status():
    """SQS 큐 상태 조회"""
    try:
        queue_attributes = sqs_service.get_queue_attributes()
        
        if queue_attributes:
            return {
                "status": "success",
                "queue_info": {
                    "messages_available": queue_attributes.get('ApproximateNumberOfMessages', '0'),
                    "messages_in_flight": queue_attributes.get('ApproximateNumberOfMessagesNotVisible', '0'),
                    "queue_url": os.getenv('SQS_QUEUE_URL', 'Not configured')
                },
                "sqs_enabled": os.getenv('USE_SQS_QUEUE', 'false').lower() == 'true'
            }
        else:
            return {
                "status": "error",
                "message": "SQS 큐 정보를 가져올 수 없습니다.",
                "sqs_enabled": os.getenv('USE_SQS_QUEUE', 'false').lower() == 'true'
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"SQS 상태 조회 실패: {str(e)}",
            "sqs_enabled": os.getenv('USE_SQS_QUEUE', 'false').lower() == 'true'
        }

# ═══════════════════════════════════════════════════════════════
# 🚀 배치 처리 기능 (여러 파일 동시 처리)
# ═══════════════════════════════════════════════════════════════

@router.post("/batch-upload/")
async def batch_upload_audio(
    request: Request,
    files: List[UploadFile] = File(...),
    token_ids: str = Form(...),  # 쉼표로 구분된 문자열로 받기
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    여러 오디오 파일을 동시에 업로드하고 병렬로 분석 요청
    
    Args:
        files: 업로드할 오디오 파일들
        token_ids: 쉼표로 구분된 token_id 문자열 (예: "1,2,3")
    
    Returns:
        job_ids: 각 파일의 작업 ID 리스트
    """
    # 쉼표로 구분된 문자열을 리스트로 변환
    token_id_list = [tid.strip() for tid in token_ids.split(",") if tid.strip()]
    
    if len(files) != len(token_id_list):
        raise HTTPException(400, f"파일 수({len(files)})와 token_id 수({len(token_id_list)})가 일치하지 않습니다")
    
    s3_client = request.app.state.s3_client
    job_ids = []
    
    # 각 파일에 대해 job_id 생성 및 초기 상태 저장
    for i, (file, token_id) in enumerate(zip(files, token_id_list)):
        job_id = str(uuid4())
        job_ids.append(job_id)
        
        # DB에 초기 상태 저장
        create_analysis_result(db, job_id, int(token_id), current_user.id)
        logging.info(f"[배치 작업 생성] job_id={job_id}, file={file.filename}, token_id={token_id}")
    
    # 백그라운드에서 병렬 처리 시작
    background_tasks.add_task(
        process_batch_files_parallel,
        s3_client,
        [(await file.read(), file.filename) for file in files],  # 파일 데이터 미리 읽기
        token_id_list,
        job_ids,
        current_user.id
    )
    
    return {
        "message": f"{len(files)}개 파일 배치 처리 시작",
        "job_ids": job_ids,
        "total_files": len(files)
    }

async def process_batch_files_parallel(
    s3_client,
    file_data_list: List[tuple],  # (file_data, filename) 튜플 리스트
    token_ids: List[str],
    job_ids: List[str],
    user_id: int
):
    """여러 파일을 병렬로 처리하는 백그라운드 함수"""
    
    async def process_single_file_async(file_data: bytes, filename: str, token_id: str, job_id: str):
        """단일 파일 비동기 처리"""
        from database import SessionLocal
        bg_db = SessionLocal()
        
        try:
            logging.info(f"[배치 처리 시작] job_id={job_id}, file={filename}")
            
            # 토큰 정보 조회
            token_info = await get_token_by_id(token_id, bg_db)
            
            # 1. S3 업로드
            update_analysis_result(bg_db, job_id, progress=20, message="S3 업로드 중...")
            s3_key = await upload_to_s3_async(s3_client, file_data, filename)
            s3_url = f"s3://{S3_BUCKET}/{s3_key}"
            
            # 2. 분석 서버 요청 준비
            update_analysis_result(bg_db, job_id, progress=50, message="분석 서버 요청 중...")
            webhook_url = f"{WEBHOOK_URL}?job_id={job_id}"
            
            # 3. 분석 서버에 비동기 요청
            await send_analysis_request_async(
                s3_url, token_id, webhook_url, job_id, token_info
            )
            
            # 4. 요청 완료 상태 업데이트
            update_analysis_result(
                bg_db, job_id, 
                status="processing", 
                progress=80, 
                message="분석 서버에서 처리 중..."
            )
            
            logging.info(f"[배치 처리 성공] job_id={job_id}, file={filename}")
            
        except Exception as e:
            logging.error(f"[배치 처리 실패] job_id={job_id}, file={filename}, error={e}")
            update_analysis_result(
                bg_db, job_id, 
                status="failed", 
                progress=0, 
                message=f"처리 실패: {str(e)}"
            )
        finally:
            bg_db.close()
    
    # 🚀 핵심: asyncio.gather로 모든 파일을 동시에 처리
    tasks = [
        process_single_file_async(file_data, filename, token_id, job_id)
        for (file_data, filename), token_id, job_id in zip(file_data_list, token_ids, job_ids)
    ]
    
    try:
        # 모든 작업을 병렬로 실행
        await asyncio.gather(*tasks, return_exceptions=True)
        logging.info(f"[배치 처리 완료] 총 {len(file_data_list)}개 파일 처리 완료")
    except Exception as e:
        logging.error(f"[배치 처리 오류] {e}")

@router.get("/batch-progress/")
async def get_batch_progress(
    job_ids: str,  # 쉼표로 구분된 job_id 문자열
    db: Session = Depends(get_db)
):
    """
    배치 작업들의 진행 상황 조회
    
    Args:
        job_ids: 쉼표로 구분된 job_id 문자열 (예: "job1,job2,job3")
    """
    job_id_list = [jid.strip() for jid in job_ids.split(",") if jid.strip()]
    
    if not job_id_list:
        raise HTTPException(400, "job_ids가 비어있습니다")
    
    results = []
    completed_count = 0
    failed_count = 0
    
    for job_id in job_id_list:
        result = get_analysis_result(db, job_id)
        if result:
            status_info = {
                "job_id": result.job_id,
                "status": result.status,
                "progress": result.progress,
                "message": result.message,
                "token_id": result.token_id
            }
            
            if result.status == "completed":
                completed_count += 1
            elif result.status == "failed":
                failed_count += 1
                
            results.append(status_info)
        else:
            results.append({
                "job_id": job_id,
                "status": "not_found",
                "progress": 0,
                "message": "작업을 찾을 수 없습니다"
            })
            failed_count += 1
    
    # 전체 진행률 계산
    total_jobs = len(job_id_list)
    overall_progress = (completed_count / total_jobs) * 100 if total_jobs > 0 else 0
    
    return {
        "batch_results": results,
        "summary": {
            "total_jobs": total_jobs,
            "completed": completed_count,
            "failed": failed_count,
            "in_progress": total_jobs - completed_count - failed_count,
            "overall_progress": round(overall_progress, 1)
        }
    }