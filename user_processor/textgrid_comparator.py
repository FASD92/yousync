"""
TextGrid 비교 분석 모듈 (textgrid_comparator.py)

기능:
- 기준 영상과 사용자 음성의 TextGrid 비교
- 피치 패턴 유사도 분석
- 음소 단위 발음 정확도 측정
- 타이밍 정확도 분석

입력:
- 기준 TextGrid 파일
- 사용자 TextGrid 파일
- 기준/사용자 음성 파일 (피치 분석용)

출력: 종합 비교 결과 JSON
"""

import json
import sys
import os
from pathlib import Path
import numpy as np

# voice_to_pitch 모듈 import
from voice_to_pitch import create_user_pitch_json, load_pitch_data, extract_pitch_segment

# textgrid 및 피치 유사도 분석 병렬 실행 위한 라이브러리 import
from concurrent.futures import ThreadPoolExecutor

def parse_textgrid(textgrid_path: str) -> dict:
    """
    TextGrid 파일 파싱 (간단한 구현)
    
    Args:
        textgrid_path: TextGrid 파일 경로
    
    Returns:
        파싱된 데이터 딕셔너리
    """
    try:
        with open(textgrid_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 간단한 TextGrid 파싱 (phones tier 추출)
        lines = content.split('\n')
        intervals = []
        
        in_phones_tier = False
        current_interval = {}
        
        for line in lines:
            line = line.strip()
            
            if 'name = "phones"' in line:
                in_phones_tier = True
                continue
            
            if in_phones_tier and 'intervals [' in line:
                if current_interval:
                    intervals.append(current_interval)
                current_interval = {}
                continue
            
            if in_phones_tier and 'xmin =' in line:
                current_interval['start'] = float(line.split('=')[1].strip())
            elif in_phones_tier and 'xmax =' in line:
                current_interval['end'] = float(line.split('=')[1].strip())
            elif in_phones_tier and 'text =' in line:
                text = line.split('=')[1].strip().strip('"')
                current_interval['phone'] = text
            
            # 다른 tier 시작하면 phones tier 종료
            if in_phones_tier and 'name =' in line and 'phones' not in line:
                if current_interval:
                    intervals.append(current_interval)
                break
        
        # 마지막 interval 추가
        if current_interval:
            intervals.append(current_interval)
        
        return {
            "file": textgrid_path,
            "intervals": intervals,
            "total_duration": intervals[-1]['end'] if intervals else 0
        }
        
    except Exception as e:
        print(f"❌ TextGrid 파싱 실패: {e}")
        return {"file": textgrid_path, "intervals": [], "total_duration": 0}

def compare_textgrids(reference_textgrid: str, user_textgrid: str) -> dict:
    """
    두 TextGrid 파일을 비교하여 발음 정확도 분석
    
    Args:
        reference_textgrid: 기준 TextGrid 파일 경로
        user_textgrid: 사용자 TextGrid 파일 경로
    
    Returns:
        음소별 비교 결과 딕셔너리
    """
    #print(f"📊 TextGrid 비교 분석 중...")
    
    # TextGrid 파싱
    ref_data = parse_textgrid(reference_textgrid)
    user_data = parse_textgrid(user_textgrid)
    
    ref_phones = [interval['phone'] for interval in ref_data['intervals'] if interval.get('phone')]
    user_phones = [interval['phone'] for interval in user_data['intervals'] if interval.get('phone')]
    
    # 발음 정확도 계산 (간단한 매칭)
    total_phones = len(ref_phones)
    matched_phones = 0
    
    min_length = min(len(ref_phones), len(user_phones))
    for i in range(min_length):
        if ref_phones[i] == user_phones[i]:
            matched_phones += 1
    
    pronunciation_accuracy = matched_phones / total_phones if total_phones > 0 else 0
    
    # 타이밍 정확도 계산
    timing_accuracy = calculate_timing_accuracy(ref_data['intervals'], user_data['intervals'])
    
    return {
        "pronunciation_accuracy": round(pronunciation_accuracy, 3),
        "timing_accuracy": round(timing_accuracy, 3),
        "reference_phones": len(ref_phones),
        "user_phones": len(user_phones),
        "matched_phones": matched_phones,
        "reference_duration": ref_data['total_duration'],
        "user_duration": user_data['total_duration']
    }

def calculate_timing_accuracy(ref_intervals: list, user_intervals: list) -> float:
    """
    타이밍 정확도 계산
    """
    if not ref_intervals or not user_intervals:
        return 0.0
    
    # 전체 발화 시간 비교
    ref_duration = ref_intervals[-1]['end'] - ref_intervals[0]['start']
    user_duration = user_intervals[-1]['end'] - user_intervals[0]['start']
    
    # 시간 차이 비율로 정확도 계산
    time_diff_ratio = abs(ref_duration - user_duration) / ref_duration if ref_duration > 0 else 1
    timing_accuracy = max(0, 1 - time_diff_ratio)
    
    return timing_accuracy

def create_reference_pitch_json(audio_path: str, output_path: str) -> str:
    """
    기준 음성 피치 데이터 생성 (간단한 버전)
    """
    try:
        import parselmouth
        
        # 출력 디렉토리 생성
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 음성 파일 로드
        sound = parselmouth.Sound(audio_path)
        
        # 피치 추출
        pitch = sound.to_pitch(time_step=0.01)
        
        # JSON 형식으로 변환
        pitch_data = []
        for i, time in enumerate(pitch.xs()):
            hz_value = pitch.get_value_at_time(time)
            pitch_data.append({
                "time": round(time, 3),
                "hz": round(hz_value, 2) if not np.isnan(hz_value) else None
            })
        
        # 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pitch_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 기준 피치 데이터 저장: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ 기준 피치 추출 실패: {e}")
        return ""

def extract_pitch_segment_safe(pitch_data: list, start: float, end: float) -> list:
    """안전한 피치 세그먼트 추출"""
    segment = []
    for p in pitch_data:
        if start <= p["time"] <= end and p["hz"] is not None and p["hz"] > 0:
            segment.append(p["hz"])
    return segment

def calculate_dtw_pitch_similarity(ref_pitch: list, user_pitch: list) -> float:
    """DTW 알고리즘을 사용한 고급 피치 유사도 계산"""
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        
        # 최소 길이 확보
        if len(ref_pitch) < 3 or len(user_pitch) < 3:
            return 0.0
        
        # Z-score 정규화
        ref_norm = zscore_normalize_pitch(ref_pitch)
        user_norm = zscore_normalize_pitch(user_pitch)
        
        # DTW 거리 계산
        distance, _ = fastdtw(ref_norm, user_norm, dist=euclidean)
        
        # 거리를 유사도로 변환 (0~1)
        max_len = max(len(ref_norm), len(user_norm))
        similarity = max(0.0, 1.0 - distance / max_len)
        
        return similarity
        
    except ImportError:
        print("⚠️ fastdtw 라이브러리가 없습니다. 기본 코사인 유사도를 사용합니다.")
        return calculate_simple_pitch_similarity(ref_pitch, user_pitch)
    except Exception as e:
        print(f"⚠️ DTW 피치 유사도 계산 오류: {e}")
        return calculate_simple_pitch_similarity(ref_pitch, user_pitch)

def zscore_normalize_pitch(pitch_values: list) -> list:
    """Z-score 정규화"""
    if not pitch_values:
        return []
    
    mean_val = np.mean(pitch_values)
    std_val = np.std(pitch_values)
    
    if std_val == 0:
        return [0.0] * len(pitch_values)
    
    return [(val - mean_val) / std_val for val in pitch_values]

def calculate_simple_pitch_similarity(ref_pitch: list, user_pitch: list) -> float:
    """간단한 피치 유사도 계산 (코사인 유사도)"""
    try:
        # 길이 맞추기 (선형 보간)
        min_len = min(len(ref_pitch), len(user_pitch))
        if min_len < 3:
            return 0.0
        
        # 리샘플링으로 길이 맞추기
        ref_resampled = np.interp(np.linspace(0, 1, min_len), 
                                np.linspace(0, 1, len(ref_pitch)), ref_pitch)
        user_resampled = np.interp(np.linspace(0, 1, min_len), 
                                 np.linspace(0, 1, len(user_pitch)), user_pitch)
        
        # Z-score 정규화
        ref_norm = (ref_resampled - np.mean(ref_resampled)) / (np.std(ref_resampled) + 1e-8)
        user_norm = (user_resampled - np.mean(user_resampled)) / (np.std(user_resampled) + 1e-8)
        
        # 코사인 유사도 계산
        dot_product = np.dot(ref_norm, user_norm)
        norm_ref = np.linalg.norm(ref_norm)
        norm_user = np.linalg.norm(user_norm)
        
        if norm_ref == 0 or norm_user == 0:
            return 0.0
        
        similarity = dot_product / (norm_ref * norm_user)
        return max(0.0, (similarity + 1) / 2)  # -1~1을 0~1로 변환
        
    except Exception as e:
        print(f"⚠️ 피치 유사도 계산 오류: {e}")
        return 0.0

def comprehensive_analysis(reference_textgrid: str, user_textgrid: str,
                         reference_audio: str, user_audio: str,
                         reference_segments: list) -> dict:
    """
    종합 비교 분석
    
    Args:
        reference_textgrid: 기준 TextGrid 파일
        user_textgrid: 사용자 TextGrid 파일
        reference_audio: 기준 음성 파일
        user_audio: 사용자 음성 파일
        reference_segments: 기준 세그먼트 정보
    
    Returns:
        종합 분석 결과
    """
    print("🔍 종합 비교 분석 시작...")
    
    # 1, 2. textgrid 비교와 피치 분석을 병렬 처리로 시간 단축
    print("📊 textgrid 및 피치 유사도 비교 분석 시작...")

    with ThreadPoolExecutor(max_workers=2) as executor:
        textgrid_future = executor.submit(compare_textgrids, reference_textgrid, user_textgrid)
        pitch_future = executor.submit(analyze_pitch_similarity, reference_audio, user_audio, reference_segments)

        textgrid_results = textgrid_future.result()
        pitch_results = pitch_future.result()
    
    # 3. 종합 결과
    comprehensive_result = {
        "textgrid_analysis": textgrid_results,
        "pitch_analysis": pitch_results,
        "overall_scores": {
            "pronunciation_score": textgrid_results.get("pronunciation_accuracy", 0) * 100,
            "timing_score": textgrid_results.get("timing_accuracy", 0) * 100,
            "pitch_score": pitch_results.get("pitch_similarity", 0) * 100
        },
        "file_paths": {
            "reference_textgrid": reference_textgrid,
            "user_textgrid": user_textgrid,
            "reference_audio": reference_audio,
            "user_audio": user_audio
        }
    }
    
    # 결과 저장
    shared_data_path = Path("../shared_data")
    result_path = shared_data_path / "comparison_result.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_result, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 종합 분석 완료: {result_path}")
    return comprehensive_result

def analyze_pitch_similarity(reference_audio: str, user_audio: str, segments: list) -> dict:
    """
    피치 유사도 분석
    """
    #print("🎵 피치 유사도 분석 중...")
    
    try:
        # 피치 데이터 생성
        shared_data_path = Path("../shared_data")
        pitch_data_path = shared_data_path / "pitch_data"
        pitch_data_path.mkdir(exist_ok=True)
        
        ref_pitch_path = pitch_data_path / "reference" / "pitch.json"
        user_pitch_path = pitch_data_path / "user" / "pitch.json"
        segments_path = pitch_data_path / "segments.json"
        
        # 기준 음성 피치 생성
        create_reference_pitch_json(reference_audio, str(ref_pitch_path))
        
        # 사용자 음성 피치 생성
        create_user_pitch_json(user_audio, str(user_pitch_path))
        
        # 세그먼트 정보 저장
        with open(segments_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        
        # 피치 데이터 로드
        ref_pitch_data = load_pitch_data(str(ref_pitch_path))
        user_pitch_data = load_pitch_data(str(user_pitch_path))
        
        # 세그먼트별 유사도 계산
        segment_results = []
        similarities = []
        
        for segment in segments:
            start, end = segment["start"], segment["end"]
            
            # 세그먼트별 피치 추출
            ref_segment = extract_pitch_segment_safe(ref_pitch_data, start, end)
            user_segment = extract_pitch_segment_safe(user_pitch_data, start, end)
            
            if len(ref_segment) > 5 and len(user_segment) > 5:  # 최소 길이 확보
                # DTW 알고리즘 우선 시도, 실패 시 코사인 유사도
                similarity = calculate_dtw_pitch_similarity(ref_segment, user_segment)
            else:
                similarity = None
            
            segment_results.append({
                "text": segment["text"],
                "start": start,
                "end": end,
                "similarity": round(similarity, 3) if similarity is not None else None
            })
            
            if similarity is not None:
                similarities.append(similarity)
        
        # 전체 유사도 평균
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        return {
            "pitch_similarity": round(avg_similarity, 3),
            "segment_details": segment_results
        }
        
    except Exception as e:
        print(f"❌ 피치 유사도 분석 실패: {e}")
        return {"pitch_similarity": 0.0, "segment_details": []}
