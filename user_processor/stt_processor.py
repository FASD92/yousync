"""
STT 처리 모듈 (stt_processor.py)

기능:
- 사용자 음성을 텍스트로 변환 (Whisper 사용)
- MFA 호환 .lab 파일 생성
- 자동 언어 감지

입력: 전처리된 사용자 음성 파일
출력: .lab 텍스트 파일 (MFA 형식)
"""

import whisper
import sys
import os
from pathlib import Path
import json
import re

# 전역 모델 캐시
_whisper_model = None

def load_whisper_model(model_size: str = "base.en"):
    """
    Whisper 모델 로드 (전역 캐시 사용)
    
    Args:
        model_size: 모델 크기 ("tiny", "base", "small", "medium", "large")
    
    Returns:
        로드된 Whisper 모델
    """
    global _whisper_model
    
    if _whisper_model is None:
        print(f"🤖 Whisper {model_size} 모델 로드 중...")
        _whisper_model = whisper.load_model(model_size)
        print("✅ 모델 로드 완료")
    
    return _whisper_model

def transcribe_user_audio(audio_path: str, output_lab_path: str = None, 
                         model_size: str = "base.en") -> dict:
    """
    사용자 음성을 STT로 변환하여 .lab 파일 생성
    
    Args:
        audio_path: 전처리된 음성 파일 경로
        output_lab_path: .lab 파일 저장 경로 (None이면 자동 생성)
        model_size: Whisper 모델 크기
    
    Returns:
        STT 결과 딕셔너리 (텍스트, 세그먼트 등)
    """
    print(f"🎯 STT 처리 시작: {audio_path}")
    
    try:
        # 모델 로드
        model = load_whisper_model(model_size)
        
        # Whisper로 음성 인식
        print("🎤 음성 인식 중...")
        result = model.transcribe(
            audio_path,
            language="en",  # 영어 고정
            word_timestamps=True,  # 단어별 타임스탬프
            verbose=False
        )
        
        # 결과 정리
        transcription_result = {
            "text": result["text"].strip(),
            "language": result["language"],
            "segments": result["segments"]
        }
        
        print(f"📝 인식 결과: {transcription_result['text']}")
        
        # .lab 파일 생성
        if output_lab_path is None:
            audio_name = Path(audio_path).stem
            output_lab_path = f"../shared_data/user/lab/{audio_name}.lab"
        
        lab_path = create_lab_file(transcription_result["text"], output_lab_path)
        transcription_result["lab_file"] = str(lab_path)
        
        print(f"✅ STT 처리 완료: {lab_path}")
        return transcription_result
        
    except Exception as e:
        print(f"❌ STT 처리 실패: {str(e)}")
        raise

def create_lab_file(text: str, output_path: str) -> str:
    """
    텍스트를 MFA 호환 .lab 파일로 저장
    
    Args:
        text: 변환된 텍스트
        output_path: .lab 파일 저장 경로
    
    Returns:
        생성된 .lab 파일 경로
    """
    # 출력 디렉토리 생성
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 텍스트 정리 (MFA 호환)
    cleaned_text = clean_text_for_mfa(text)
    
    # .lab 파일 저장 (단순 텍스트 형식)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    print(f"📄 .lab 파일 생성: {output_path}")
    return str(output_path)

def clean_text_for_mfa(text: str) -> str:
    """
    MFA 호환을 위한 텍스트 정리
    
    Args:
        text: 원본 텍스트
    
    Returns:
        정리된 텍스트
    """
    # 기본 정리
    text = text.strip()
    
    # 여러 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    
    return text

def extract_segments_info(transcription_result: dict) -> list:
    """
    STT 결과에서 세그먼트 정보 추출
    
    Args:
        transcription_result: transcribe_user_audio 결과
    
    Returns:
        세그먼트 정보 리스트
    """
    segments = []
    
    for segment in transcription_result.get("segments", []):
        segments.append({
            "start": segment.get("start", 0),
            "end": segment.get("end", 0),
            "text": segment.get("text", "").strip()
        })
    
    return segments

def get_detailed_word_timestamps(transcription_result: dict) -> list:
    """
    STT 결과에서 단어별 상세 타임스탬프 추출
    
    Args:
        transcription_result: transcribe_user_audio 결과
    
    Returns:
        단어별 상세 정보 리스트
    """
    detailed_segments = []
    
    for segment in transcription_result.get("segments", []):
        # word-level timestamps가 있는 경우 활용
        if "words" in segment and segment["words"]:
            for word in segment["words"]:
                if word.get("start") is not None and word.get("end") is not None:
                    detailed_segments.append({
                        "start": word.get("start", 0),
                        "end": word.get("end", 0),
                        "text": word.get("word", "").strip()
                    })
        else:
            # word-level이 없으면 segment-level 사용
            detailed_segments.append({
                "start": segment.get("start", 0),
                "end": segment.get("end", 0),
                "text": segment.get("text", "").strip()
            })
    
    return detailed_segments

def get_optimized_segments_for_pitch(transcription_result: dict, min_duration: float = 0.5) -> list:
    """
    피치 분석에 최적화된 세그먼트 추출
    
    Args:
        transcription_result: transcribe_user_audio 결과
        min_duration: 최소 세그먼트 길이 (초)
    
    Returns:
        피치 분석용 세그먼트 리스트
    """
    segments = []
    
    for segment in transcription_result.get("segments", []):
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        duration = end - start
        
        # 너무 짧은 세그먼트는 피치 분석에 부적합하므로 필터링
        if duration >= min_duration:
            segments.append({
                "start": start,
                "end": end,
                "text": segment.get("text", "").strip(),
                "duration": duration
            })
    
    print(f"📊 피치 분석용 세그먼트: {len(segments)}개 (최소 {min_duration}초 이상)")
    
    return segments

def save_transcription_result(result: dict, output_path: str) -> str:
    """
    STT 결과를 JSON 파일로 저장
    
    Args:
        result: transcribe_user_audio 결과
        output_path: JSON 파일 저장 경로
    
    Returns:
        저장된 파일 경로
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f" STT 결과 저장: {output_path}")
    return output_path
