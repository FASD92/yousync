import os, boto3, json, urllib.parse, httpx, logging
from typing import Any, Optional
import io
from pydub import AudioSegment
import uuid

# 기본 버킷 이름을 환경 변수 또는 상수로 설정
DEFAULT_BUCKET = os.getenv("S3_BUCKET_NAME")

def _parse_s3(url: str) -> Optional[tuple[str, str]]:
    if not url:
        return None
    # 1) s3:// 접두사가 있는 경우 처리
    if url.startswith("s3://"):
        p = urllib.parse.urlparse(url)
        return p.netloc, p.path.lstrip("/")
    # 2) 버킷 가상 호스트 형식 URL ( .s3. 포함 ) 처리
    if ".s3." in url:
        p = urllib.parse.urlparse(url)
        bucket_name = p.netloc.split(".s3.")[0]
        return bucket_name, p.path.lstrip("/")
    # 3) 별도 프로토콜 없는 경우 -> S3 키로 간주
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "" and parsed.netloc == "":
        # 기본 버킷과 조합하여 반환
        return DEFAULT_BUCKET, parsed.path.lstrip("/")
    return None

async def load_json(s3_client, url: Optional[str]) -> Any:
    if not url:
        return None
    bk = _parse_s3(url)
    try:
        if bk:
            # _parse_s3가 (bucket, key)를 반환한 경우 S3에서 객체 가져오기
            b, k = bk
            obj = s3_client.get_object(Bucket=b, Key=k)
            return json.load(obj["Body"])
        # _parse_s3가 None이면 (예: 일반 http URL인 경우) httpx로 요청
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(url)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logging.warning("pitch.json load error %s", e)
        return None

def presign(s3_client, url: Optional[str], exp: int = 900) -> Optional[str]:
    if not url:
        return None
    bk = _parse_s3(url)
    if bk:
        # S3 경로로 인식되면 프리사인 URL 생성
        b, k = bk
        return s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": b, "Key": k},
            ExpiresIn=exp,
        )
    # S3 경로가 아니라고 판단된 경우 원본 그대로 반환 (이미 퍼블릭 URL 등)
    return url

#==========================# 병합 더빙 음성 제공용

s3 = boto3.client("s3", region_name='ap-northeast-2')

def load_main_audio_from_s3(actor_name: str, video_id: str):
    base_prefix = f"{actor_name}/{video_id}/0/"

    def load_file(key):
        print(f"☁️ S3에서 로딩 중: {key}")
        response = s3.get_object(Bucket=DEFAULT_BUCKET, Key=key)
        return AudioSegment.from_file(io.BytesIO(response["Body"].read()), format="wav")

    vocal = load_file(f"{base_prefix}vocal.wav")
    bgvoice = load_file(f"{base_prefix}bgvoice.wav")

    return bgvoice, vocal  # background, original


from typing import Optional
from pydub import AudioSegment

# AWS_DEFAULT_BUCKET = os.getenv("AWS_S3_BUCKET_NAME")


def load_user_audio_from_s3(user_id: int, token_id: int, script_id: int) -> Optional[AudioSegment]:
    key = f"user_audio/{user_id}/{token_id}/{script_id}.wav"
    try:
        print(f"☁️ S3에서 사용자 음성 로딩 중: s3://{DEFAULT_BUCKET}/{key}")
        response = s3.get_object(Bucket=DEFAULT_BUCKET, Key=key)
        audio_bytes = io.BytesIO(response["Body"].read())
        return AudioSegment.from_file(audio_bytes, format="wav")
    except Exception as e:
        logging.warning(f"⚠️ 사용자 음성 로딩 실패: {key} → {e}")
        return None


def upload_audio_to_s3(audio_segment: AudioSegment, user_id: int, token_id: int) -> str:
    # 고유한 파일명 생성을 위해 UUID 사용
    unique_filename = f"dubbing_audio_{uuid.uuid4()}.wav"
    key = f"user_Dubbing_auido/{user_id}/{token_id}/{unique_filename}"
    
    # AudioSegment를 메모리 내 바이트 버퍼로 내보내기
    buffer = io.BytesIO()
    audio_segment.export(buffer, format="wav")
    buffer.seek(0)  # 버퍼의 시작으로 포인터 이동

    try:
        print(f"☁️ S3에 합성 음성 업로드 중: s3://{DEFAULT_BUCKET}/{key}")
        s3.put_object(Bucket=DEFAULT_BUCKET, Key=key, Body=buffer, ContentType="audio/wav")
        print(f"✅ S3 업로드 성공: {key}")
        return key
    except Exception as e:
        logging.error(f"❌ S3 업로드 실패: {key} → {e}")
        raise

def generate_presigned_url(key: str, expiration: int = 3600) -> str:
    try:
        url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': DEFAULT_BUCKET, 'Key': key},
            ExpiresIn=expiration
        )
        return url
    except Exception as e:
        logging.error(f"❌ Pre-signed URL 생성 실패: {key} → {e}")
        raise


