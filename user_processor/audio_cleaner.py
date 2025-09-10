"""
사용자 음성 전처리 모듈 (audio_cleaner.py)

기능:
- 배경 소음 제거 (Demucs 활용)
- 16kHz 모노 WAV 파일로 변환
- 음성 품질 향상 및 검증

입력: 사용자가 녹음한 원본 음성 파일
출력: 전처리된 16kHz 모노 WAV 파일
"""

import subprocess
import sys
import os
import tempfile
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

# 기존 youtube_processor 모듈 import
sys.path.append('../youtube_downloader/Youtube_Downloader/youtube_processor')
try:
    from demucs_wrapper import separate_vocals
    print("✅ demucs_wrapper 연동 성공")
except ImportError as e:
    print(f"⚠️ demucs_wrapper를 찾을 수 없습니다: {e}")
    print("배경 제거 기능이 제한됩니다.")
    separate_vocals = None

def clean_user_audio(input_path: str, output_path: str, 
                    remove_background: bool = True, target_sr: int = 16000) -> str:
    """
    사용자 음성을 전처리하여 분석에 적합한 형태로 변환
    
    Args:
        input_path: 원본 음성 파일 경로
        output_path: 전처리된 파일 저장 경로
        remove_background: 배경 제거 여부
        target_sr: 목표 샘플링 레이트 (기본값: 16kHz)
    
    Returns:
        전처리된 파일 경로
    """
    print(f"🧹 음성 전처리 시작: {input_path}")
    
    try:
        # 1. 파일 존재 확인
        if not Path(input_path).exists():
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")
        
        # 2. 배경 제거 (선택적)
        if remove_background and separate_vocals:
            print("🎵 배경 소음 제거 중...")
            vocal_path = remove_background_noise(input_path)
        else:
            vocal_path = input_path
            print("⏭️ 배경 제거 건너뜀")
        
        # 3. 16kHz 모노로 변환
        print("🔄 16kHz 모노로 변환 중...")
        converted_path = convert_to_16khz_mono(vocal_path, output_path, target_sr)
        
        print(f"✨ 전처리 완료!")
        print(f"📁 저장 위치: {converted_path}")
        
        return converted_path
        
    except Exception as e:
        print(f"❌ 전처리 실패: {str(e)}")
        raise

def remove_background_noise(input_path: str) -> str:
    """
    Demucs를 사용하여 배경 소음 제거
    
    Args:
        input_path: 원본 음성 파일 경로
    
    Returns:
        배경이 제거된 음성 파일 경로
    """
    try:
        # 임시 디렉토리 생성
        temp_dir = Path(tempfile.mkdtemp(prefix="audio_cleaner_"))
        temp_output_dir = temp_dir / f"vocal_{Path(input_path).stem}"
        
        # Demucs로 보컬 분리
        vocal_path = separate_vocals(input_path, str(temp_output_dir))
        
        if Path(vocal_path).exists():
            print(f"✅ 배경 제거 성공: {vocal_path}")
            return vocal_path
        else:
            print("⚠️ 배경 제거 실패, 원본 파일 사용")
            return input_path
            
    except Exception as e:
        print(f"⚠️ 배경 제거 중 오류: {str(e)}, 원본 파일 사용")
        return input_path

def convert_to_16khz_mono(input_path: str, output_path: str, target_sr: int = 16000) -> str:
    """
    음성 파일을 16kHz 모노로 변환
    
    Args:
        input_path: 입력 음성 파일 경로
        output_path: 출력 파일 경로
        target_sr: 목표 샘플링 레이트
    
    Returns:
        변환된 파일 경로
    """
    try:
        # librosa로 오디오 로드 (자동으로 모노 변환)
        audio, sr = librosa.load(input_path, sr=target_sr, mono=True)
        
        # 정규화 (음량 조절)
        audio = librosa.util.normalize(audio)
        
        # 출력 디렉토리 생성
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # WAV 파일로 저장
        sf.write(output_path, audio, target_sr)
        
        print(f"🎯 변환 완료: {sr}Hz → {target_sr}Hz, 모노")
        return output_path
        
    except Exception as e:
        print(f"❌ 변환 실패: {str(e)}")
        raise

def validate_audio_file(file_path: str) -> dict:
    """
    음성 파일 유효성 검사 및 정보 추출
    
    Args:
        file_path: 음성 파일 경로
    
    Returns:
        파일 정보 딕셔너리
    """
    try:
        audio, sr = librosa.load(file_path, sr=None)
        duration = len(audio) / sr
        
        return {
            "valid": True,
            "duration": round(duration, 2),
            "sample_rate": sr,
            "channels": 1 if audio.ndim == 1 else audio.shape[0],
            "samples": len(audio)
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }
