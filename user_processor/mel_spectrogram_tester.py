"""
Mel Spectrogram 기반 음성 유사도 테스트 모듈

기능:
- S3 URL에서 음성 파일 다운로드
- 기준 음성과 유저 음성의 mel spectrogram 추출 및 비교
- 코사인 유사도 계산
"""

import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import tempfile
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional
import json

def download_audio_from_s3(s3_url: str, temp_dir: str = None) -> str:
    """
    S3 URL에서 음성 파일 다운로드
    
    Args:
        s3_url: S3 오디오 파일 URL
        temp_dir: 임시 저장 디렉토리 (None이면 자동 생성)
    
    Returns:
        다운로드된 로컬 파일 경로
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    
    # URL에서 파일명 추출
    filename = s3_url.split("/")[-1]
    if not filename or "." not in filename:
        filename = f"audio_{hash(s3_url) % 10000}.wav"
    
    local_path = os.path.join(temp_dir, filename)
    
    print(f"📥 S3에서 다운로드 중: {s3_url}")
    
    try:
        response = requests.get(s3_url, stream=True)
        response.raise_for_status()
        
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ 다운로드 완료: {local_path}")
        return local_path
        
    except Exception as e:
        print(f"❌ 다운로드 실패: {str(e)}")
        raise

def compute_similarity(audio_path1, audio_path2):
    """기존 코드 기반 유사도 계산"""
    y1, sr1 = librosa.load(audio_path1)
    y2, sr2 = librosa.load(audio_path2)

    # 멜 스펙트로그램 추출
    mel1 = librosa.feature.melspectrogram(y=y1, sr=sr1)
    mel2 = librosa.feature.melspectrogram(y=y2, sr=sr2)

    # 로그 스케일 변환
    log_mel1 = librosa.power_to_db(mel1)
    log_mel2 = librosa.power_to_db(mel2)

    # 벡터화 및 평균
    vec1 = np.mean(log_mel1, axis=1).reshape(1, -1)
    vec2 = np.mean(log_mel2, axis=1).reshape(1, -1)

    # 코사인 유사도 계산
    similarity = cosine_similarity(vec1, vec2)
    return similarity[0][0]

def test_mel_similarity_from_s3(reference_s3_url: str, user_s3_url: str) -> dict:
    """
    S3 URL로부터 음성 파일을 받아 mel spectrogram 유사도 테스트
    """
    print("🚀 Mel Spectrogram 유사도 테스트 시작")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # S3에서 파일 다운로드
        print("\n=== 파일 다운로드 ===")
        reference_path = download_audio_from_s3(reference_s3_url, temp_dir)
        user_path = download_audio_from_s3(user_s3_url, temp_dir)
        
        # 유사도 계산
        print("\n=== 유사도 계산 ===")
        similarity_score = compute_similarity(reference_path, user_path)
        
        results = {
            "reference_url": reference_s3_url,
            "user_url": user_s3_url,
            "similarity_score": float(similarity_score)
        }
        
        print(f"\n✅ 유사도 점수: {similarity_score:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")
        raise
    
    finally:
        # 임시 파일 정리
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    # 테스트 실행
    ref_url = input("기준 음성 S3 URL: ")
    user_url = input("유저 음성 S3 URL: ")
    
    result = test_mel_similarity_from_s3(ref_url, user_url)
    print(f"최종 유사도: {result[similarity_score]:.4f}")
