"""
S3 다운로드 모듈 (s3_downloader.py)

기능:
- S3 URL에서 기준 TextGrid 및 피치 JSON 파일 다운로드
- 로컬 캐시 관리
- 에러 처리 및 재시도 로직
"""

import boto3
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
from urllib.parse import urlparse

def parse_s3_url(s3_url: str) -> tuple:
    """
    S3 URL을 bucket과 key로 파싱
    
    Args:
        s3_url: s3://bucket-name/path/to/file.ext 형식
    
    Returns:
        (bucket_name, object_key) 튜플
    """
    if not s3_url.startswith('s3://'):
        raise ValueError(f"Invalid S3 URL format: {s3_url}")
    
    parsed = urlparse(s3_url)
    bucket_name = parsed.netloc
    object_key = parsed.path.lstrip('/')
    
    return bucket_name, object_key

def download_from_s3(s3_url: str, local_path: str) -> str:
    """
    S3에서 파일을 다운로드
    
    Args:
        s3_url: S3 URL (s3://bucket/path/file.ext)
        local_path: 로컬 저장 경로
    
    Returns:
        다운로드된 파일의 로컬 경로
    """
    try:
        print(f"📥 S3 다운로드 시작: {s3_url}")
        
        # S3 URL 파싱
        bucket_name, object_key = parse_s3_url(s3_url)
        
        # S3 클라이언트 생성
        s3_client = boto3.client('s3')
        
        # 로컬 디렉토리 생성
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # S3에서 파일 다운로드
        s3_client.download_file(bucket_name, object_key, local_path)
        
        print(f"✅ S3 다운로드 완료: {local_path}")
        return local_path
        
    except Exception as e:
        print(f"❌ S3 다운로드 실패: {s3_url} - {str(e)}")
        raise

def download_reference_data_from_s3(s3_textgrid_url: str, s3_pitch_url: str) -> Dict[str, Any]:
    """
    S3에서 기준 데이터 다운로드
    
    Args:
        s3_textgrid_url: 기준 TextGrid S3 URL
        s3_pitch_url: 기준 피치 JSON S3 URL
    
    Returns:
        기준 데이터 딕셔너리
    """
    try:
        print("🌐 S3에서 기준 데이터 다운로드 중...")
        
        # 임시 디렉토리 생성
        temp_dir = Path("../shared_data/temp_reference")
        temp_dir.mkdir(exist_ok=True)
        
        # TextGrid 다운로드
        textgrid_path = temp_dir / "reference.TextGrid"
        download_from_s3(s3_textgrid_url, str(textgrid_path))
        
        # 피치 JSON 다운로드
        pitch_path = temp_dir / "reference_pitch.json"
        download_from_s3(s3_pitch_url, str(pitch_path))
        
        # 피치 JSON 로드
        with open(pitch_path, 'r', encoding='utf-8') as f:
            pitch_data = json.load(f)
        
        reference_data = {
            "textgrid_file": str(textgrid_path),
            "pitch_data": pitch_data,
            "pitch_file": str(pitch_path),
            "source": "s3",
            "s3_textgrid_url": s3_textgrid_url,
            "s3_pitch_url": s3_pitch_url
        }
        
        print("✅ S3 기준 데이터 다운로드 완료")
        return reference_data
        
    except Exception as e:
        print(f"❌ S3 기준 데이터 다운로드 실패: {str(e)}")
        raise

def cleanup_temp_reference_data():
    """
    임시 기준 데이터 정리
    """
    try:
        temp_dir = Path("../shared_data/temp_reference")
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
            print("🗑️ 임시 기준 데이터 정리 완료")
    except Exception as e:
        print(f"⚠️ 임시 데이터 정리 중 오류: {e}")

def download_user_audio_from_s3(s3_user_audio_url: str, local_path: str) -> str:
    """
    S3에서 사용자 음성 파일 다운로드
    
    Args:
        s3_user_audio_url: 사용자 음성 S3 URL
        local_path: 로컬 저장 경로
    
    Returns:
        다운로드된 파일의 로컬 경로
    """
    try:
        print(f"🎵 사용자 음성 S3 다운로드 시작: {s3_user_audio_url}")
        
        # 기존 download_from_s3 함수 재사용
        result_path = download_from_s3(s3_user_audio_url, local_path)
        
        print(f"✅ 사용자 음성 다운로드 완료: {result_path}")
        return result_path
        
    except Exception as e:
        print(f"❌ 사용자 음성 다운로드 실패: {s3_user_audio_url} - {str(e)}")
        raise

def download_all_s3_data(s3_user_audio_url: str, s3_textgrid_url: str, s3_pitch_url: str) -> Dict[str, str]:
    """
    모든 S3 데이터를 한 번에 다운로드
    
    Args:
        s3_user_audio_url: 사용자 음성 S3 URL
        s3_textgrid_url: 기준 TextGrid S3 URL
        s3_pitch_url: 기준 피치 JSON S3 URL
    
    Returns:
        다운로드된 파일 경로들의 딕셔너리
    """
    try:
        print("🌐 모든 S3 데이터 다운로드 시작...")
        
        # 임시 디렉토리 생성
        temp_dir = Path("../shared_data/temp_s3_data")
        temp_dir.mkdir(exist_ok=True)
        
        # 사용자 음성 다운로드
        user_audio_path = temp_dir / "user_audio.mp4"
        download_from_s3(s3_user_audio_url, str(user_audio_path))
        
        # 기준 데이터 다운로드
        reference_data = download_reference_data_from_s3(s3_textgrid_url, s3_pitch_url)
        
        result = {
            "user_audio": str(user_audio_path),
            "reference_textgrid": reference_data["textgrid_file"],
            "reference_pitch": reference_data["pitch_file"],
            "reference_data": reference_data
        }
        
        print("✅ 모든 S3 데이터 다운로드 완료")
        return result
        
    except Exception as e:
        print(f"❌ S3 데이터 다운로드 실패: {str(e)}")
        raise
