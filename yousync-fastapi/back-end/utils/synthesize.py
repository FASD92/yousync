# step1_get_scripts.py
from sqlalchemy.orm import Session
from models import Script, DubbingResult, Token # DubbingResult, Token 모델 추가
from pydub import AudioSegment
from router.utils_s3 import load_user_audio_from_s3, upload_audio_to_s3, load_main_audio_from_s3
from database import get_db
import asyncio

import os
import re
import uuid # uuid 모듈 추가


def extract_youtube_video_id(url: str) -> str:

    match = re.search(r"(?:youtu\.be/|v=)([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    raise ValueError(f"유효한 YouTube URL 아닙니다: {url}")


def get_token_info(session: Session, token_id: int):
    from models import Token
    token = session.query(Token).filter(Token.id == token_id).first()
    if token is None:
        raise ValueError(f"token_id={token_id}에 해당하는 Token이 없습니다.")

    actor_name = token.actor_name
    youtube_url = token.youtube_url

    # YouTube ID 추출
    video_id = extract_youtube_video_id(youtube_url)
    start_time = token.start_time  # 추가

    return actor_name, video_id, start_time
    

def get_scripts_by_token(session: Session, token_id: int):
    scripts = (
        session.query(Script)
        .filter(Script.token_id == token_id)
        .order_by(Script.id)
        .all()
    )

    return scripts


def prepare_dub_segments(user_id: int, token_id: int, scripts: list,  token_start_time: float) -> list[dict]:
    segments = []

    for script in scripts:
        audio = load_user_audio_from_s3(user_id, token_id, script.id)
        if audio is None:
            continue

        segments.append({
            # 절대 시간 기준을 만들기 위해서 필요함
            "start": script.start_time - token_start_time,
            "end": script.end_time - token_start_time,
            "audio": audio  # 경로가 아닌 실제 AudioSegment
        })

    return segments



def synthesize_audio_from_segments(
    background: AudioSegment, 
    original: AudioSegment, 
    segments: list, 
    user_id: int, 
    token_id: int
) -> str:
    result = background[:]

    # 1. 내 음성 덮어쓰기
    for seg in segments:
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        print(f"내 더빙 구간: {seg['start']}s ~ {seg['end']}s")

        # 원본 오디오의 해당 부분을 먼저 잘라내고, 그 다음에 사용자 음성을 오버레이합니다.
        # 이 방식은 원본의 소리를 완전히 대치합니다.
        result = result.overlay(AudioSegment.silent(duration=end_ms - start_ms), position=start_ms)
        result = result.overlay(seg["audio"], position=start_ms)

    # 2. 상대방 음성(원본 보컬) 삽입
    print("상대방 음성 덮어쓰기 중...")
    
    current_position_ms = 0
    for seg in sorted(segments, key=lambda x: x['start']):
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)

        # 현재 위치와 내 더빙 시작점 사이에 비어있는 공간(상대방 목소리)을 원본으로 채웁니다.
        if current_position_ms < start_ms:
            original_part = original[current_position_ms:start_ms]
            result = result.overlay(original_part, position=current_position_ms)
        
        current_position_ms = end_ms

    # 마지막 더빙 이후의 남은 부분을 원본으로 채웁니다.
    if current_position_ms < len(original):
        remaining_part = original[current_position_ms:]
        result = result.overlay(remaining_part, position=current_position_ms)

    # S3에 업로드하고 S3 키를 반환
    s3_key = upload_audio_to_s3(result, user_id, token_id)
    
    return s3_key


async def run_synthesis_async(token_id: int, user_id: int) -> str:
    db: Session = next(get_db())
    try:
        print(f"비동기 합성 작업 시작: user_id={user_id}, token_id={token_id}")

        # DB 접근은 동기적으로 수행 (일반적으로 매우 빠름)
        actor_name, video_id, token_start_time = get_token_info(db, token_id)
        scripts = get_scripts_by_token(db, token_id)

        # I/O 바운드 작업을 별도 스레드에서 실행
        background, original = await asyncio.to_thread(load_main_audio_from_s3, actor_name, video_id)
        segments = await asyncio.to_thread(prepare_dub_segments, user_id, token_id, scripts, token_start_time)

        if not segments:
            raise ValueError(f"사용자 더빙 음성이 없어 작업을 중단합니다.")

        # CPU 바운드 작업(합성) 및 I/O 바운드 작업(업로드)을 별도 스레드에서 실행
        s3_key = await asyncio.to_thread(
            synthesize_audio_from_segments,
            background,
            original,
            segments,
            user_id,
            token_id
        )
        
        print(f"비동기 작업 완료, 결과물 S3 Key: {s3_key}")

        # 더빙 결과 DB에 저장
        try:
            # 토큰 정보를 다시 조회하여 actor_id를 가져옵니다.
            token_obj = db.query(Token).filter(Token.id == token_id).first()
            if not token_obj:
                raise ValueError("Token not found when trying to save dubbing result")

            new_dubbing_result = DubbingResult(
                user_id=user_id,
                token_id=token_id,
                # actor_id=token_obj.actor_id, # Token에서 actor_id 가져오기
                s3_key=s3_key
            )
            db.add(new_dubbing_result)
            db.commit()
            print(f"더빙 결과 저장 완료: user_id={user_id}, token_id={token_id}, s3_key={s3_key}")
        except Exception as e:
            db.rollback()
            print(f"더빙 결과 DB 저장 중 오류 발생: {e}")
            # DB 저장 실패 시에도 S3 업로드는 성공했으므로, 예외를 다시 발생시키지 않고 로그만 남길 수 있습니다.
            # 하지만 여기서는 명확한 오류 처리를 위해 다시 발생시킵니다.
            raise

        return s3_key

    except Exception as e:
        print(f"비동기 작업 중 오류 발생: {e}")
        raise
    finally:
        db.close()

