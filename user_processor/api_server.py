from typing import Optional
from fastapi import FastAPI, HTTPException, Form, BackgroundTasks, Request
from fastapi.responses import JSONResponse, PlainTextResponse
import requests
import uuid
import boto3
import os
import json
import logging
import sys
from datetime import datetime
from urllib.parse import urlparse
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log', mode='a', encoding='utf-8'),
        # logging.StreamHandler()  # 중복 출력 방지를 위해 주석 처리
    ]
)

# print 함수를 오버라이드하여 로그 파일에도 기록
original_print = print

def enhanced_print(*args, **kwargs):
    """print 함수를 확장하여 로그 파일에도 기록"""
    # 원래 print 함수 호출 (터미널 출력)
    original_print(*args, **kwargs)
    
    # 로그 파일에도 기록
    message = ' '.join(str(arg) for arg in args)
    if message.strip():  # 빈 줄이 아닌 경우만
        logging.info(message)

# print 함수 교체
print = enhanced_print

# 기존 파이프라인 import
from main import process_user_audio, process_user_audio_with_s3_reference

app = FastAPI(
    title="Voice Analysis API",
    description="음성 분석 및 평가 자동화 API",
    version="1.0.0"
)

# S3 클라이언트 초기화 (.env에서 자동으로 읽어옴)
s3_client = boto3.client('s3')
BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# 환경변수 검증
if not BUCKET_NAME:
    raise ValueError("S3_BUCKET_NAME 환경변수가 설정되지 않았습니다.")

# 작업 상태 저장
job_status = {}

@app.get("/")
async def root():
    return {"message": "Voice Analysis API", "status": "running"}

@app.post("/analyze-voice")
async def analyze_voice(
    background_tasks: BackgroundTasks,
    s3_audio_url: str = Form(...),
    video_id: Optional[str] = Form(None),
    s3_textgrid_url: Optional[str] = Form(None),
    s3_pitch_url: Optional[str] = Form(None),
    webhook_url: str = Form(...)
):
    """
    S3 음성 파일을 분석하여 웹훅으로 결과 전송
    """
    job_id = str(uuid.uuid4())
    
    try:
        # S3 URL에서 키 추출
        s3_key = extract_s3_key_from_url(s3_audio_url)
        
        # 작업 상태 초기화
        job_status[job_id] = {
            "status": "processing",
            "s3_audio_url": s3_audio_url,
            "video_id": video_id,
            "webhook_url": webhook_url,
            "started_at": datetime.now().isoformat()
        }
        
        # 백그라운드에서 처리
        background_tasks.add_task(
            process_s3_to_webhook,
            job_id, s3_key, video_id, webhook_url, s3_textgrid_url, s3_pitch_url, s3_audio_url
        )
        
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "음성 분석이 시작되었습니다."
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"요청 처리 실패: {str(e)}"}
        )
        job_status[job_id] = {
            "status": "processing",
            "s3_audio_url": s3_audio_url,
            "video_id": video_id,
            "webhook_url": webhook_url,
            "started_at": datetime.now().isoformat()
        }
        
        # 백그라운드에서 파이프라인 실행
        background_tasks.add_task(
            process_s3_to_webhook,
            job_id,
            s3_key,
            video_id,
            webhook_url,
            s3_textgrid_url,
            s3_pitch_url,
            s3_audio_url
        )
        
        # 즉시 응답 반환
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "음성 분석이 시작되었습니다."
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"요청 처리 실패: {str(e)}"}
        )

def extract_s3_key_from_url(s3_url: str) -> str:
    """S3 URL에서 키 추출"""
    if s3_url.startswith('s3://'):
        # s3://bucket/key 형태
        return s3_url.split('/', 3)[3]
    elif 'amazonaws.com' in s3_url:
        # https://bucket.s3.amazonaws.com/key 형태
        parsed = urlparse(s3_url)
        return parsed.path.lstrip('/')
    else:
        raise ValueError("올바르지 않은 S3 URL 형식입니다.")

async def process_s3_to_webhook(job_id: str, s3_key: str, video_id: str, webhook_url: str, s3_textgrid_url: str = None, s3_pitch_url: str = None, s3_audio_url: str = None):
    """S3에서 음성 파일을 가져와 분석 후 웹훅으로 결과 전송"""
    local_temp_path = None
    
    try:
        print(f"🚀 작업 시작: {job_id}")
        
        # S3에서 음성 파일 다운로드
        local_temp_path = f"/tmp/{job_id}_input.mp4"
        s3_client.download_file(BUCKET_NAME, s3_key, local_temp_path)
        print(f"📥 S3 다운로드 완료: {s3_key}")
        
        # S3 URL 필수 검증
        if not s3_textgrid_url or not s3_pitch_url:
            raise ValueError("s3_textgrid_url과 s3_pitch_url은 필수 파라미터입니다.")
        
        # S3 기반 파이프라인 실행
        #print(f"🌐 S3 기반 처리 모드")
        print(f"   사용자 음성: {s3_audio_url}")
        print(f"   TextGrid: {s3_textgrid_url}")
        print(f"   Pitch: {s3_pitch_url}")
        
        result = process_user_audio_with_s3_reference(
            s3_textgrid_url=s3_textgrid_url,
            s3_pitch_url=s3_pitch_url,
            s3_user_audio_url=s3_audio_url,
            remove_background=False
        )
        
        # 작업 완료 상태 업데이트
        job_status[job_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "result": result["analysis_results"]
        })
        
        # 결과 파일에서 데이터 추출 (S3 방식이면 final_result_s3.json)
        if s3_textgrid_url and s3_pitch_url:
            final_result_path = "../shared_data/final_result_s3.json"
        else:
            final_result_path = "../shared_data/final_result.json"
            
        analysis_results = None
        
        try:
            with open(final_result_path, 'r', encoding='utf-8') as f:
                final_data = json.load(f)
                
                # S3 방식과 기존 방식 구조 차이 처리
                if "analysis_results" in final_data:
                    # 기존 방식
                    analysis_results = final_data["analysis_results"]
                else:
                    # S3 방식 - 전체 데이터를 analysis_results로 사용
                    analysis_results = final_data
                    
                print(f"📊 분석 결과 추출 완료: {final_result_path}")
        except Exception as e:
            print(f"⚠️ 결과 파일 읽기 실패 ({final_result_path}): {e}")
            # 백업으로 result에서 가져오기
            analysis_results = result.get("analysis_results", {})
        
        # 웹훅으로 결과 전송
        webhook_payload = {
            "job_id": job_id,
            "status": "completed",
            "video_id": video_id,
            "s3_audio_url": f"s3://{BUCKET_NAME}/{s3_key}",
            "analysis_results": analysis_results
        }
        
        # 백엔드에서 이미 완전한 URL을 보내므로 그대로 사용
        webhook_url_with_job = webhook_url
        webhook_success = await send_webhook(webhook_url_with_job, webhook_payload)
        
        if webhook_success:
            print(f"✅ 작업 완료 및 웹훅 전송 성공: {job_id}")
        else:
            print(f"⚠️ 작업 완료되었으나 웹훅 전송 실패: {job_id}")
        
    except Exception as e:
        print(f"❌ 작업 실패: {job_id} - {str(e)}")
        
        # 실패 상태 업데이트
        job_status[job_id].update({
            "status": "failed",
            "job_id": job_id,
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })
        
        # 실패 웹훅 알림
        webhook_payload = {
            "job_id": job_id,
            "status": "failed",
            "job_id": job_id,
            "video_id": video_id,
            "s3_audio_url": f"s3://{BUCKET_NAME}/{s3_key}",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        }
        
        # 백엔드에서 이미 완전한 URL을 보내므로 그대로 사용
        webhook_url_with_job = webhook_url
        await send_webhook(webhook_url_with_job, webhook_payload)
        
    finally:
        # 로컬 임시 파일 정리
        if local_temp_path and os.path.exists(local_temp_path):
            os.unlink(local_temp_path)
            print(f"🗑️ 임시 파일 정리: {local_temp_path}")

async def send_webhook(webhook_url: str, payload: dict):
    """웹훅 전송 (개선된 버전)"""
    try:
        import requests
        
        print(f"📤 웹훅 전송 시도: {webhook_url}")
        print(f"📦 전송 데이터: {payload.get('job_id', 'N/A')} - {payload.get('status', 'N/A')}")
        
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=30,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Voice-Analysis-API/1.0"
            },
            # 재시도 설정
            verify=True
        )
        
        print(f"📡 웹훅 응답: {response.status_code} - {response.text[:200]}")
        
        if response.status_code in [200, 201, 202]:
            print(f"✅ 웹훅 전송 성공: {webhook_url} (상태: {response.status_code})")
            return True
        else:
            print(f"⚠️ 웹훅 응답 오류: {response.status_code} - {response.text[:100]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏰ 웹훅 타임아웃 (60초): {webhook_url}")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"🔌 웹훅 연결 오류: {webhook_url} - {str(e)}")
        return False
    except Exception as e:
        print(f"❌ 웹훅 전송 실패: {webhook_url} - {str(e)}")
        return False

@app.post("/webhook")
async def receive_webhook(request: Request):
    """웹훅 수신 엔드포인트 - 분석 결과를 받아서 표시"""
    try:
        data = await request.json()
        
        print("\n" + "="*60)
        print(f"🔔 웹훅 수신! - {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)
        
        print("📦 받은 데이터:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        if data.get('status') == 'completed':
            results = data.get('analysis_results', {})
            print(f"\n🎉 분석 완료!")
            print(f"📊 최종 점수: {results.get('final_score', 'N/A')}")
            print(f"🗣️  발음 점수: {results.get('pronunciation_score', 'N/A')}")
            print(f"⏰ 타이밍 점수: {results.get('timing_score', 'N/A')}")
            print(f"🎵 피치 점수: {results.get('pitch_score', 'N/A')}")
        elif data.get('status') == 'failed':
            print(f"❌ 분석 실패: {data.get('error', 'Unknown error')}")
        
        print("="*60)
        
        return {"received": True, "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        print(f"❌ 웹훅 처리 오류: {e}")
        return {"error": str(e)}

@app.get("/logs")
async def get_logs():
    """서버 로그 조회"""
    try:
        log_file = "api_server.log"
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                logs = f.read()
            return PlainTextResponse(logs)
        else:
            return PlainTextResponse("로그 파일이 없습니다.")
    except Exception as e:
        return PlainTextResponse(f"로그 읽기 실패: {str(e)}")

@app.get("/logs/tail")
async def get_recent_logs():
    """최근 로그만 조회 (마지막 100줄)"""
    try:
        log_file = "api_server.log"
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                recent_lines = lines[-100:]  # 마지막 100줄
            return PlainTextResponse("".join(recent_lines))
        else:
            return PlainTextResponse("로그 파일이 없습니다.")
    except Exception as e:
        return PlainTextResponse(f"로그 읽기 실패: {str(e)}")

@app.get("/status")
async def server_status():
    """서버 상태 확인"""
    return {
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "message": "Voice Analysis API 정상 작동 중"
    }

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """작업 상태 조회"""
    if job_id in job_status:
        return job_status[job_id]
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "Job not found"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
