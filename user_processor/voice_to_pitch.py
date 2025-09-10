"""
사용자 음성 피치 분석 모듈 (voice_to_pitch.py)

기능:
- 사용자 음성에서 피치 데이터 추출
- JSON 형식으로 피치 정보 저장
- youtube_processor/voice_to_pitch.py와 동일한 형식

입력: 사용자 음성 WAV 파일
출력: 피치 데이터 JSON 파일
"""

import json
import numpy as np
from pathlib import Path

def create_user_pitch_json(audio_path: str, output_path: str = None) -> str:
    """
    사용자 음성에서 피치 데이터를 추출하여 JSON으로 저장
    
    Args:
        audio_path: 사용자 음성 파일 경로
        output_path: 피치 JSON 저장 경로 (None이면 자동 생성)
    
    Returns:
        생성된 JSON 파일 경로
    """
    try:
        import parselmouth
        
        print(f"🎵 사용자 음성 피치 분석: {audio_path}")
        
        # 출력 경로 설정
        if output_path is None:
            audio_name = Path(audio_path).stem
            output_path = f"../shared_data/pitch_data/user/{audio_name}_pitch.json"
        
        # 출력 디렉토리 생성
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 음성 파일 로드
        sound = parselmouth.Sound(audio_path)
        
        # 피치 추출 (youtube_processor와 동일한 설정)
        pitch = sound.to_pitch(
            time_step=0.01,      # 10ms 간격
            pitch_floor=75.0,    # 최소 피치
            pitch_ceiling=600.0  # 최대 피치
        )
        
        # JSON 형식으로 변환
        pitch_data = []
        for i, time in enumerate(pitch.xs()):
            hz_value = pitch.get_value_at_time(time)
            pitch_data.append({
                "time": round(time, 3),
                "hz": round(hz_value, 2) if not np.isnan(hz_value) else None
            })
        
        # JSON 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pitch_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 피치 데이터 저장 완료: {output_path}")
        print(f"📊 총 {len(pitch_data)}개 피치 포인트 추출")
        
        return output_path
        
    except ImportError:
        print("❌ parselmouth 라이브러리가 필요합니다: pip install praat-parselmouth")
        raise
    except Exception as e:
        print(f"❌ 피치 분석 실패: {str(e)}")
        raise

def load_pitch_data(json_path: str) -> list:
    """
    피치 JSON 파일 로드
    
    Args:
        json_path: 피치 JSON 파일 경로
    
    Returns:
        피치 데이터 리스트
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 피치 데이터 로드 실패: {str(e)}")
        raise

def extract_pitch_segment(pitch_data: list, start_time: float, end_time: float) -> list:
    """
    특정 시간 구간의 피치 데이터 추출
    
    Args:
        pitch_data: 전체 피치 데이터
        start_time: 시작 시간 (초)
        end_time: 끝 시간 (초)
    
    Returns:
        해당 구간의 피치 값 리스트 (null 제외)
    """
    segment_pitch = []
    for point in pitch_data:
        if start_time <= point["time"] <= end_time and point["hz"] is not None:
            segment_pitch.append(point["hz"])
    
    return segment_pitch
