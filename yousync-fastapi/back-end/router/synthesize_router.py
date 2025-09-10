from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from models import User
from router.auth_router import get_current_user
from router.utils_s3 import generate_presigned_url
from utils.synthesize import run_synthesis_async

router = APIRouter(tags=["synthesize"])

@router.post("/synthesize/{token_id}")
async def synthesize_audio_endpoint(
    token_id: int,
    current_user: User = Depends(get_current_user),
):
    user_id = current_user.id

    try:
        # 1. 비동기 합성 함수 호출
        s3_key = await run_synthesis_async(token_id, user_id)

        # 2. Pre-signed URL 생성 (동기 함수, 매우 빠르므로 그냥 호출)
        presigned_url = generate_presigned_url(s3_key)

        return {
            "status": "success",
            "message": "합성 및 업로드 완료",
            "dubbing_audio_url": presigned_url
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"합성 중 오류 발생: {e}")
