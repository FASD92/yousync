# """
# 영상 분석 관련 API 엔드포인트

# 유튜브 링크 분석, 얼굴 인식, 음성 전사, Celery 비동기 분석 등의 기능을 제공합니다.
# """

# from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File, Form
# from pydantic import BaseModel, HttpUrl
# from typing import List, Optional, Dict, Any
# from sqlalchemy.orm import Session
# import logging
# import os
# import json
# from datetime import datetime

# from database import get_db
# from auth import get_current_active_user
# from models import User as UserModel, Script, Result
# from services.youtube_service import YouTubeService
# from services.face_service_dummy import FaceService
# from services.whisper_service import WhisperService

# # Celery tasks import
# try:
#     from celery_tasks.voice_analysis_task import analyze_voice, get_analysis_status
#     from celery_tasks.shorts_generation_task import generate_daily_shorts, get_daily_rankings
#     CELERY_AVAILABLE = True
# except ImportError:
#     CELERY_AVAILABLE = False
#     logging.warning("Celery tasks를 import할 수 없습니다. 비동기 분석이 비활성화됩니다.")

# logger = logging.getLogger(__name__)

# # API 라우터 설정
# router = APIRouter(
#     prefix="/analysis",
#     tags=["analysis"],
#     responses={404: {"description": "Not found"}}
# )

# # 서비스 인스턴스
# youtube_service = YouTubeService()
# face_service = FaceService()
# whisper_service = WhisperService()

# # === Pydantic 스키마 정의 ===

# # === Pydantic 스키마 정의 ===

# class YouTubeAnalysisRequest(BaseModel):
#     """유튜브 분석 요청 스키마"""
#     youtube_url: HttpUrl
#     script_id: Optional[int] = None  # 선택된 스크립트 ID

# class VoiceAnalysisRequest(BaseModel):
#     """음성 분석 요청 스키마"""
#     script_id: int
#     audio_file_path: Optional[str] = None
#     video_file_path: Optional[str] = None

# class AnalysisStatusResponse(BaseModel):
#     """분석 상태 응답 스키마"""
#     task_id: str
#     status: str
#     result: Optional[Dict[str, Any]] = None
#     progress: Optional[int] = None

# class AnalysisResultResponse(BaseModel):
#     """분석 결과 응답 스키마"""
#     id: int
#     user_id: int
#     script_id: int
#     total_score: float
#     pronunciation_score: float
#     intonation_score: float
#     rhythm_score: float
#     expression_score: float
#     transcribed_text: str
#     created_at: datetime

# # === API 엔드포인트 ===

# @router.post("/youtube/metadata")
# async def extract_youtube_metadata(
#     request: YouTubeAnalysisRequest,
#     current_user: User = Depends(get_current_active_user)
# ):
#     """
#     유튜브 URL에서 메타데이터를 추출합니다.
#     """
#     try:
#         logger.info(f"유튜브 메타데이터 추출 요청: {request.youtube_url}")
        
#         # URL 유효성 검증
#         if not youtube_service.validate_youtube_url(str(request.youtube_url)):
#             raise HTTPException(status_code=400, detail="유효하지 않은 유튜브 URL입니다")
        
#         # 메타데이터 추출
#         metadata = youtube_service.extract_metadata(str(request.youtube_url))
        
#         return {
#             "status": "success",
#             "metadata": metadata,
#             "extracted_at": datetime.utcnow()
#         }
        
#     except Exception as e:
#         logger.error(f"메타데이터 추출 실패: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"메타데이터 추출 중 오류가 발생했습니다: {str(e)}")

# @router.post("/youtube/thumbnail")
# async def extract_youtube_thumbnail(
#     request: YouTubeAnalysisRequest,
#     current_user: User = Depends(get_current_active_user)
# ):
#     """
#     유튜브 비디오의 썸네일을 다운로드합니다.
#     """
#     try:
#         logger.info(f"유튜브 썸네일 추출 요청: {request.youtube_url}")
        
#         # 썸네일 추출
#         thumbnail_info = youtube_service.extract_thumbnails(str(request.youtube_url))
        
#         return {
#             "status": "success",
#             "thumbnail": thumbnail_info,
#             "extracted_at": datetime.utcnow()
#         }
        
#     except Exception as e:
#         logger.error(f"썸네일 추출 실패: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"썸네일 추출 중 오류가 발생했습니다: {str(e)}")

# @router.post("/youtube/audio")
# async def extract_youtube_audio(
#     request: YouTubeAnalysisRequest,
#     current_user: User = Depends(get_current_active_user)
# ):
#     """
#     유튜브 비디오에서 오디오를 추출합니다.
#     """
#     try:
#         logger.info(f"유튜브 오디오 추출 요청: {request.youtube_url}")
        
#         # 오디오 추출
#         audio_info = youtube_service.extract_audio(str(request.youtube_url))
        
#         return {
#             "status": "success",
#             "audio": audio_info,
#             "extracted_at": datetime.utcnow()
#         }
        
#     except Exception as e:
#         logger.error(f"오디오 추출 실패: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"오디오 추출 중 오류가 발생했습니다: {str(e)}")

# @router.post("/voice/upload")
# async def upload_voice_file(
#     script_id: int = Form(...),
#     audio_file: UploadFile = File(...),
#     video_file: Optional[UploadFile] = File(None),
#     current_user: User = Depends(get_current_active_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     사용자 음성 파일을 업로드하고 분석을 시작합니다.
#     """
#     try:
#         logger.info(f"음성 파일 업로드 및 분석 시작: 사용자 {current_user.id}, 스크립트 {script_id}")
        
#         # 스크립트 존재 확인
#         script = db.query(Script).filter(Script.id == script_id).first()
#         if not script:
#             raise HTTPException(status_code=404, detail="스크립트를 찾을 수 없습니다")
        
#         # 파일 저장 디렉토리 생성
#         upload_dir = f"storage/uploads/user_{current_user.id}"
#         os.makedirs(upload_dir, exist_ok=True)
        
#         # 오디오 파일 저장
#         audio_filename = f"audio_{script_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
#         audio_path = os.path.join(upload_dir, audio_filename)
        
#         with open(audio_path, "wb") as f:
#             content = await audio_file.read()
#             f.write(content)
        
#         # 비디오 파일 저장 (있는 경우)
#         video_path = None
#         if video_file:
#             video_filename = f"video_{script_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
#             video_path = os.path.join(upload_dir, video_filename)
            
#             with open(video_path, "wb") as f:
#                 content = await video_file.read()
#                 f.write(content)
        
#         # Celery를 사용한 비동기 분석 시작
#         if CELERY_AVAILABLE:
#             task = analyze_voice.delay(current_user.id, script_id, audio_path, video_path)
            
#             return {
#                 "status": "processing",
#                 "task_id": task.id,
#                 "message": "분석이 시작되었습니다. 상태를 확인해주세요.",
#                 "audio_path": audio_path,
#                 "video_path": video_path
#             }
#         else:
#             # Celery가 없는 경우 동기 분석 수행
#             result = await analyze_voice_sync(current_user.id, script_id, audio_path, video_path, db)
#             return result
        
#     except Exception as e:
#         logger.error(f"음성 파일 업로드 실패: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"파일 업로드 중 오류가 발생했습니다: {str(e)}")

# async def analyze_voice_sync(user_id: int, script_id: int, audio_path: str, video_path: str, db: Session):
#     """동기 음성 분석 (Celery가 없는 경우)"""
#     try:
#         # 스크립트 조회
#         script = db.query(Script).filter(Script.id == script_id).first()
        
#         # 음성 전사
#         transcription = whisper_service.transcribe_audio(audio_path)
        
#         # 정확도 계산
#         accuracy = whisper_service.calculate_accuracy(script.content, transcription.get('text', ''))
        
#         # 얼굴 분석 (비디오가 있는 경우)
#         face_analysis = {}
#         if video_path:
#             face_analysis = face_service.analyze_dummy_face(video_path)
        
#         # 종합 점수 계산
#         total_score = (
#             accuracy * 0.4 +
#             85.0 * 0.3 +  # 억양 점수 (더미)
#             80.0 * 0.2 +  # 리듬 점수 (더미)
#             face_analysis.get("expression_score", 75.0) * 0.1
#         )
        
#         # 결과 저장
#         analysis_result = AnalysisResult(
#             user_id=user_id,
#             script_id=script_id,
#             audio_file_path=audio_path,
#             video_file_path=video_path,
#             transcribed_text=transcription.get('text', ''),
#             pronunciation_score=accuracy,
#             intonation_score=85.0,
#             rhythm_score=80.0,
#             expression_score=face_analysis.get("expression_score", 75.0),
#             total_score=total_score,
#             analysis_data=json.dumps({
#                 "transcription": transcription,
#                 "face_analysis": face_analysis
#             }),
#             created_at=datetime.utcnow()
#         )
        
#         db.add(analysis_result)
#         db.commit()
#         db.refresh(analysis_result)
        
#         return {
#             "status": "completed",
#             "result_id": analysis_result.id,
#             "total_score": total_score,
#             "pronunciation_score": accuracy,
#             "intonation_score": 85.0,
#             "rhythm_score": 80.0,
#             "expression_score": face_analysis.get("expression_score", 75.0)
#         }
        
#     except Exception as e:
#         logger.error(f"동기 음성 분석 실패: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"분석 중 오류가 발생했습니다: {str(e)}")

# @router.get("/status/{task_id}")
# async def get_analysis_task_status(
#     task_id: str,
#     current_user: User = Depends(get_current_active_user)
# ):
#     """
#     분석 태스크의 상태를 조회합니다.
#     """
#     try:
#         if CELERY_AVAILABLE:
#             status_info = get_analysis_status.delay(task_id).get(timeout=10)
#             return status_info
#         else:
#             return {
#                 "task_id": task_id,
#                 "status": "UNAVAILABLE",
#                 "message": "Celery가 설치되지 않았습니다"
#             }
            
#     except Exception as e:
#         logger.error(f"태스크 상태 조회 실패: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"상태 조회 중 오류가 발생했습니다: {str(e)}")

# @router.get("/results")
# async def get_user_analysis_results(
#     limit: int = 10,
#     offset: int = 0,
#     current_user: User = Depends(get_current_active_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     사용자의 분석 결과 목록을 조회합니다.
#     """
#     try:
#         results = db.query(AnalysisResult)\
#             .filter(AnalysisResult.user_id == current_user.id)\
#             .order_by(AnalysisResult.created_at.desc())\
#             .offset(offset)\
#             .limit(limit)\
#             .all()
        
#         return {
#             "status": "success",
#             "results": [
#                 {
#                     "id": result.id,
#                     "script_id": result.script_id,
#                     "total_score": result.total_score,
#                     "pronunciation_score": result.pronunciation_score,
#                     "intonation_score": result.intonation_score,
#                     "rhythm_score": result.rhythm_score,
#                     "expression_score": result.expression_score,
#                     "transcribed_text": result.transcribed_text,
#                     "created_at": result.created_at
#                 }
#                 for result in results
#             ],
#             "total_count": len(results),
#             "offset": offset,
#             "limit": limit
#         }
        
#     except Exception as e:
#         logger.error(f"분석 결과 조회 실패: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"결과 조회 중 오류가 발생했습니다: {str(e)}")

# @router.get("/results/{result_id}")
# async def get_analysis_result_detail(
#     result_id: int,
#     current_user: User = Depends(get_current_active_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     특정 분석 결과의 상세 정보를 조회합니다.
#     """
#     try:
#         result = db.query(AnalysisResult)\
#             .filter(
#                 AnalysisResult.id == result_id,
#                 AnalysisResult.user_id == current_user.id
#             )\
#             .first()
        
#         if not result:
#             raise HTTPException(status_code=404, detail="분석 결과를 찾을 수 없습니다")
        
#         # 분석 데이터 파싱
#         analysis_data = {}
#         if result.analysis_data:
#             try:
#                 analysis_data = json.loads(result.analysis_data)
#             except json.JSONDecodeError:
#                 pass
        
#         return {
#             "status": "success",
#             "result": {
#                 "id": result.id,
#                 "script_id": result.script_id,
#                 "total_score": result.total_score,
#                 "pronunciation_score": result.pronunciation_score,
#                 "intonation_score": result.intonation_score,
#                 "rhythm_score": result.rhythm_score,
#                 "expression_score": result.expression_score,
#                 "transcribed_text": result.transcribed_text,
#                 "audio_file_path": result.audio_file_path,
#                 "video_file_path": result.video_file_path,
#                 "analysis_data": analysis_data,
#                 "created_at": result.created_at
#             }
#         }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"분석 결과 상세 조회 실패: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"상세 조회 중 오류가 발생했습니다: {str(e)}")

# @router.post("/shorts/generate")
# async def trigger_daily_shorts_generation(
#     target_date: Optional[str] = None,
#     current_user: User = Depends(get_current_active_user)
# ):
#     """
#     일일 숏츠 생성을 트리거합니다. (관리자 전용)
#     """
#     try:
#         # 관리자 권한 확인 (실제로는 User 모델에 is_admin 필드 추가 필요)
#         if not getattr(current_user, 'is_admin', False):
#             raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다")
        
#         if CELERY_AVAILABLE:
#             task = generate_daily_shorts.delay(target_date)
#             return {
#                 "status": "processing",
#                 "task_id": task.id,
#                 "message": "숏츠 생성이 시작되었습니다"
#             }
#         else:
#             return {
#                 "status": "unavailable",
#                 "message": "Celery가 설치되지 않았습니다"
#             }
            
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"숏츠 생성 트리거 실패: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"숏츠 생성 트리거 중 오류가 발생했습니다: {str(e)}")

# @router.get("/rankings/daily")
# async def get_daily_rankings_api(
#     date: Optional[str] = None,
#     limit: int = 10,
#     current_user: User = Depends(get_current_active_user)
# ):
#     """
#     일일 랭킹을 조회합니다.
#     """
#     try:
#         if CELERY_AVAILABLE:
#             rankings = get_daily_rankings.delay(date, limit).get(timeout=30)
#             return rankings
#         else:
#             return {
#                 "status": "unavailable",
#                 "message": "Celery가 설치되지 않았습니다"
#             }
            
#     except Exception as e:
#         logger.error(f"일일 랭킹 조회 실패: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"랭킹 조회 중 오류가 발생했습니다: {str(e)}")

# @router.post("/face/analyze")
# async def analyze_face_image(
#     image_file: UploadFile = File(...),
#     current_user: User = Depends(get_current_active_user)
# ):
#     """
#     업로드된 이미지에서 얼굴을 분석합니다.
#     """
#     try:
#         logger.info(f"얼굴 분석 요청: 사용자 {current_user.id}")
        
#         # 파일 저장
#         upload_dir = f"storage/face_analysis/user_{current_user.id}"
#         os.makedirs(upload_dir, exist_ok=True)
        
#         filename = f"face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#         filepath = os.path.join(upload_dir, filename)
        
#         with open(filepath, "wb") as f:
#             content = await image_file.read()
#             f.write(content)
        
#         # 얼굴 분석 수행
#         analysis_result = face_service.analyze_face_from_image(filepath)
        
#         return {
#             "status": "success",
#             "analysis": analysis_result,
#             "image_path": filepath,
#             "analyzed_at": datetime.utcnow()
#         }
        
#     except Exception as e:
#         logger.error(f"얼굴 분석 실패: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"얼굴 분석 중 오류가 발생했습니다: {str(e)}")

# class CrawlRequest(BaseModel):
#     """배우 이미지 크롤링 요청 스키마"""
#     actors: List[str]
#     max_images_per_actor: Optional[int] = 10

# class CrawlResponse(BaseModel):
#     """크롤링 응답 스키마"""
#     status: str
#     message: str
#     results: Dict[str, int]  # 배우별 다운로드된 이미지 수

# class FaceMatchRequest(BaseModel):
#     """얼굴 매칭 요청 스키마"""
#     video_url: HttpUrl
#     actors: List[str]
#     frame_interval: Optional[int] = 30  # 프레임 추출 간격 (초)

# class FaceMatchResponse(BaseModel):
#     """얼굴 매칭 응답 스키마"""
#     status: str
#     message: str
#     matches: Dict[str, List[Dict]]  # 배우별 매칭된 프레임 정보

# class TranscribeRequest(BaseModel):
#     """음성 전사 요청 스키마"""
#     video_url: HttpUrl
#     language: Optional[str] = "ko"  # 언어 코드

# class TranscribeResponse(BaseModel):
#     """음성 전사 응답 스키마"""
#     status: str
#     message: str
#     transcript: str
#     segments: List[Dict]  # 타임스탬프별 구간

# class AnalysisStatusResponse(BaseModel):
#     """분석 상태 응답 스키마"""
#     status: str
#     message: str
#     data: Optional[Dict] = None

# # === API 엔드포인트 ===

# @router.post("/youtube/metadata", response_model=YouTubeMetadataResponse)
# async def extract_youtube_metadata(request: YouTubeAnalysisRequest):
#     """
#     유튜브 URL에서 메타데이터를 추출합니다.
    
#     - **youtube_url**: 분석할 유튜브 비디오 URL
#     - **movie_title**: 사용자가 입력한 영화 제목
#     """
#     try:
#         # URL 유효성 검사
#         url_str = str(request.youtube_url)
#         if not youtube_service.validate_youtube_url(url_str):
#             raise HTTPException(status_code=400, detail="유효하지 않은 유튜브 URL입니다.")
        
#         # 메타데이터 추출
#         metadata = youtube_service.extract_metadata(url_str)
        
#         return YouTubeMetadataResponse(**metadata)
        
#     except Exception as e:
#         logger.error(f"메타데이터 추출 실패: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"메타데이터 추출 중 오류가 발생했습니다: {str(e)}")

# @router.post("/full-analysis", response_model=AnalysisStatusResponse)
# async def start_full_analysis(
#     request: YouTubeAnalysisRequest, 
#     background_tasks: BackgroundTasks,
#     db: Session = Depends(get_db)
# ):
#     """
#     유튜브 영상에 대한 전체 분석을 시작합니다.
    
#     1. 메타데이터 추출
#     2. 영상 분석 및 처리 (백그라운드)
    
#     - **youtube_url**: 분석할 유튜브 비디오 URL
#     - **movie_title**: 사용자가 입력한 영화 제목
#     """
#     try:
#         url_str = str(request.youtube_url)
        
#         # 1. 메타데이터 추출
#         metadata = youtube_service.extract_metadata(url_str)
        
#         # 2. 백그라운드 작업 시작 (추후 구현)
#         # background_tasks.add_task(process_video_analysis, url_str, request.movie_title)
        
#         return AnalysisStatusResponse(
#             status="started",
#             message="분석이 시작되었습니다. 백그라운드에서 처리됩니다.",
#             data={
#                 "metadata": metadata,
#                 "movie_title": request.movie_title
#             }
#         )
        
#     except Exception as e:
#         logger.error(f"전체 분석 시작 실패: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"분석 시작 중 오류가 발생했습니다: {str(e)}")

# @router.post("/actors/crawl", response_model=CrawlResponse)
# async def crawl_actor_images(
#     request: CrawlRequest,
#     background_tasks: BackgroundTasks
# ):
#     """
#     배우 이미지를 크롤링합니다.
    
#     - **actors**: 크롤링할 배우 이름 목록
#     - **max_images_per_actor**: 배우당 최대 이미지 수 (기본: 10)
#     """
#     try:
#         # 배경 작업으로 크롤링 실행
#         def crawl_task():
#             results = {}
#             for actor in request.actors:
#                 try:
#                     downloaded_count = crawl_service.crawl_actor_images(
#                         actor, 
#                         max_images=request.max_images_per_actor
#                     )
#                     results[actor] = downloaded_count
#                     logger.info(f"배우 {actor}: {downloaded_count}개 이미지 크롤링 완료")
#                 except Exception as e:
#                     logger.error(f"배우 {actor} 크롤링 실패: {str(e)}")
#                     results[actor] = 0
#             return results
        
#         background_tasks.add_task(crawl_task)
        
#         return CrawlResponse(
#             status="started",
#             message=f"{len(request.actors)}명의 배우 이미지 크롤링을 시작했습니다.",
#             results={actor: 0 for actor in request.actors}  # 초기값
#         )
        
#     except Exception as e:
#         logger.error(f"크롤링 시작 실패: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"크롤링 시작 중 오류가 발생했습니다: {str(e)}")

# @router.post("/face/match", response_model=FaceMatchResponse)
# async def match_faces_in_video(
#     request: FaceMatchRequest,
#     background_tasks: BackgroundTasks
# ):
#     """
#     비디오에서 배우 얼굴을 매칭합니다.
    
#     - **video_url**: 분석할 비디오 URL
#     - **actors**: 매칭할 배우 목록
#     - **frame_interval**: 프레임 추출 간격 (초, 기본: 30)
#     """
#     try:
#         url_str = str(request.video_url)
        
#         # 배경 작업으로 얼굴 매칭 실행
#         def face_match_task():
#             matches = {}
#             try:
#                 # 1. 비디오에서 프레임 추출
#                 frames = face_service.extract_frames_from_video(
#                     url_str, 
#                     interval=request.frame_interval
#                 )
                
#                 # 2. 각 배우별로 얼굴 매칭
#                 for actor in request.actors:
#                     actor_matches = face_service.match_faces_in_frames(actor, frames)
#                     matches[actor] = actor_matches
#                     logger.info(f"배우 {actor}: {len(actor_matches)}개 매칭 완료")
                
#             except Exception as e:
#                 logger.error(f"얼굴 매칭 실패: {str(e)}")
                
#             return matches
        
#         background_tasks.add_task(face_match_task)
        
#         return FaceMatchResponse(
#             status="started",
#             message=f"비디오에서 {len(request.actors)}명의 배우 얼굴 매칭을 시작했습니다.",
#             matches={actor: [] for actor in request.actors}  # 초기값
#         )
        
#     except Exception as e:
#         logger.error(f"얼굴 매칭 시작 실패: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"얼굴 매칭 시작 중 오류가 발생했습니다: {str(e)}")

# @router.post("/audio/transcribe", response_model=TranscribeResponse)
# async def transcribe_audio(
#     request: TranscribeRequest,
#     background_tasks: BackgroundTasks
# ):
#     """
#     비디오에서 음성을 추출하고 텍스트로 변환합니다.
    
#     - **video_url**: 분석할 비디오 URL
#     - **language**: 언어 코드 (기본: ko)
#     """
#     try:
#         url_str = str(request.video_url)
        
#         # 배경 작업으로 음성 전사 실행
#         def transcribe_task():
#             try:
#                 # 1. 비디오에서 오디오 추출
#                 audio_path = whisper_service.extract_audio_from_video(url_str)
                
#                 # 2. 음성을 텍스트로 변환
#                 transcript, segments = whisper_service.transcribe_audio(
#                     audio_path, 
#                     language=request.language
#                 )
                
#                 logger.info(f"음성 전사 완료: {len(segments)}개 구간")
#                 return transcript, segments
                
#             except Exception as e:
#                 logger.error(f"음성 전사 실패: {str(e)}")
#                 return "", []
        
#         background_tasks.add_task(transcribe_task)
        
#         return TranscribeResponse(
#             status="started",
#             message="음성 추출 및 전사를 시작했습니다.",
#             transcript="",  # 초기값
#             segments=[]     # 초기값
#         )
        
#     except Exception as e:
#         logger.error(f"음성 전사 시작 실패: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"음성 전사 시작 중 오류가 발생했습니다: {str(e)}")

# @router.get("/health")
# async def analysis_health_check():
#     """분석 서비스 상태 확인"""
#     return {"status": "healthy", "service": "analysis", "message": "분석 서비스가 정상 작동 중입니다."}
