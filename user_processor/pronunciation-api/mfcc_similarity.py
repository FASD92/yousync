import librosa
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# 로그 출력 설정 (필요시 활성화)
DEBUG_MODE = True

def debug_print(message):
    """디버그 메시지 출력 (DEBUG_MODE가 True일 때만)"""
    if DEBUG_MODE:
        print(f"[MFCC_DEBUG] {message}")

# 연속 선형 보간을 위한 점수 매핑 포인트
# 데이터셋 분석 결과를 바탕으로 설정:
# - 정답 음성: ≃0.45 (90-100점)
# - 팀원(정상 발음): ≃0.10 (70점)  
# - 단어 2개 누락: ≃0.08 (50점)
# - 액센트 없이 발음: ≃0.09 (60점)
# - 침묵: ≃0.02 (0점)
# 0.08~0.10 구간에서 세밀한 점수 구분이 중요함
sim_points = [0.00, 0.02, 0.05, 0.08, 0.09, 0.10, 0.30, 0.40, 0.53, 1.00]
score_points = [0.0,  0.0,  40.0,  50.0,  60.0,  70.0,   80.0,   90.0,   100.0, 100.0]

def continuous_score(similarity: float) -> float:
    """
    연속 선형 보간을 사용한 0~100점 매핑
    
    기존 if-elif 구조를 대체:
    - 0.4~0.53: 90~100점 (정답 음성 수준)
    - 0.3~0.39: 80~89점 (우수한 발음)
    - 0.1~0.29: 70~79점 (정상 발음)
    - 0.09~0.099: 60~69점 (액센트 없는 발음)
    - 0.08~0.089: 50~59점 (단어 누락)
    - 0.05~0.079: 40~49점 (부족한 발음)
    - 0.02 이하: 0점 (침묵/무음성)
    """
    return float(np.interp(similarity, sim_points, score_points))

def _apply_cmvn(mfcc):
    """
    MFCC에 CMVN(Cepstral Mean and Variance Normalization) 적용
    
    Parameters:
    -----------
    mfcc : ndarray
        (시간, 특성) 형태의 MFCC 행렬
        
    Returns:
    --------
    ndarray
        CMVN이 적용된 MFCC 행렬
    """
    if mfcc.shape[0] == 0:
        return mfcc
    
    # 평균과 표준편차 계산
    mean = np.mean(mfcc, axis=0)
    std = np.std(mfcc, axis=0) + 1e-9  # 0으로 나누기 방지
    
    # 정규화 적용
    normalized = (mfcc - mean) / std
    
    return normalized

def cmvn_with_c0_clipping(mfcc):
    """
    CMVN + C0 적응적 클리핑 정규화
    mfcc: (N_frames, 13)
    """
    if mfcc.shape[0] == 0:
        return mfcc
    
    # 1) C1–C12 CMVN (스펙트럼 형태 정보)
    rest = mfcc[:, 1:]                   # (N,12)
    rest_mean = rest.mean(axis=0)        # (12,)
    rest_std = rest.std(axis=0) + 1e-9   # (12,) - 0으로 나누기 방지
    rest_norm = (rest - rest_mean) / rest_std  # (N,12)

    # 2) C0 클리핑 (에너지 정보)
    c0 = mfcc[:, 0]                      # (N,)
    c0_min = np.percentile(c0, 5)
    c0_max = np.percentile(c0, 95)

    if c0_max - c0_min < 1e-6:
        c0_norm = np.full_like(c0, 0.5)
    else:
        c0_clipped = np.clip(c0, c0_min, c0_max)
        c0_norm = (c0_clipped - c0_min) / (c0_max - c0_min)

    # 3) 다시 합치기
    mfcc_norm = np.concatenate(
        [c0_norm[:, None], rest_norm],
        axis=1
    )  # (N,13)
    return mfcc_norm

def _add_delta_features(mfcc):
    """
    MFCC에 델타 및 델타-델타 특성 추가
    
    Parameters:
    -----------
    mfcc : ndarray
        (시간, 특성) 형태의 MFCC 행렬
        
    Returns:
    --------
    ndarray
        델타 및 델타-델타 특성이 추가된 MFCC 행렬
    """
    if mfcc.shape[0] == 0:
        return np.zeros((0, 39))
    
    # 프레임 수에 따라 처리 방식 결정
    n_frames = mfcc.shape[0]
    
    # 매우 짧은 세그먼트 처리 (5프레임 미만)
    if n_frames < 5:
        debug_print(f"매우 짧은 세그먼트 감지 - 프레임 수: {n_frames}, 패딩 적용")
        # 패딩 추가 (양쪽에 원본 데이터 반복)
        pad_size = max(0, 5 - n_frames)
        padded_mfcc = np.pad(mfcc, ((pad_size, pad_size), (0, 0)), mode='edge')
        
        # 패딩된 데이터로 델타 계산 (최소 윈도우 크기 사용)
        delta_mfcc = librosa.feature.delta(padded_mfcc.T, width=3).T
        delta2_mfcc = librosa.feature.delta(padded_mfcc.T, order=2, width=3).T
        
        # 패딩 제거하고 원래 크기로 복원
        if pad_size > 0:
            delta_mfcc = delta_mfcc[pad_size:pad_size+n_frames]
            delta2_mfcc = delta2_mfcc[pad_size:pad_size+n_frames]
    
    # 짧은 세그먼트 처리 (5~9프레임)
    elif n_frames < 9:
        debug_print(f"짧은 세그먼트 감지 - 프레임 수: {n_frames}")
        # 윈도우 크기를 프레임 수에 맞게 조정 (홀수여야 함)
        width = n_frames if n_frames % 2 == 1 else n_frames - 1
        width = max(3, width)  # 최소 3 프레임 필요
        
        delta_mfcc = librosa.feature.delta(mfcc.T, width=width).T
        delta2_mfcc = librosa.feature.delta(mfcc.T, order=2, width=width).T
    
    # 일반적인 경우 (9프레임 이상)
    else:
        # 기본 윈도우 크기 사용
        delta_mfcc = librosa.feature.delta(mfcc.T).T
        delta2_mfcc = librosa.feature.delta(mfcc.T, order=2).T
    
    # 특성 결합
    mfcc_with_delta = np.concatenate([mfcc, delta_mfcc, delta2_mfcc], axis=1)
    
    return mfcc_with_delta

def _pad_reference_mfcc(mfcc, target_length):
    """
    MFCC에 패딩을 추가하여 목표 길이에 맞춤
    
    Parameters:
    -----------
    mfcc : ndarray
        MFCC 벡터 (N, D) 형태
    target_length : int
        목표 프레임 수
        
    Returns:
    --------
    ndarray
        패딩된 MFCC 벡터 (target_length, D) 형태
    """
    if len(mfcc) >= target_length:
        return mfcc  # 이미 충분히 길면 그대로 반환
        
    # 필요한 패딩 크기 계산
    pad_size = target_length - len(mfcc)
    
    # 엣지 패딩 (가장자리 값 반복)
    padded_mfcc = np.pad(mfcc, ((0, pad_size), (0, 0)), mode='edge')
    
    debug_print(f"MFCC 패딩 적용: {len(mfcc)} → {len(padded_mfcc)} 프레임")
    return padded_mfcc



def frame_wise_euclidean_similarity(ref_mfcc, user_mfcc):
    """
    프레임별 유클리디언 거리 기반 유사도 계산
    표준편차 기반 단어 누락 감지 로직 포함
    
    Parameters:
    -----------
    ref_mfcc : ndarray
        기준 MFCC 특성
    user_mfcc : ndarray
        사용자 MFCC 특성
        
    Returns:
    --------
    float
        유사도 점수 (0~1)
    """
    # 빈 배열 체크
    if ref_mfcc.shape[0] == 0 or user_mfcc.shape[0] == 0:
        debug_print("빈 MFCC 배열 감지됨")
        return 0.0
    
    # 최소 프레임 수 설정 (이 값 미만인 경우 0점 처리)
    MIN_FRAMES = 5
    
    # 프레임 수가 최소 기준 미만인 경우 0점 처리
    if ref_mfcc.shape[0] < MIN_FRAMES or user_mfcc.shape[0] < MIN_FRAMES:
        debug_print(f"프레임 수 부족 - ref: {ref_mfcc.shape[0]}, user: {user_mfcc.shape[0]} < 최소 기준: {MIN_FRAMES}")
        return 0.0
    
    try:
        # 표준편차 기반 단어 누락 감지
        ref_std = _calculate_std_deviation(ref_mfcc)
        user_std = _calculate_std_deviation(user_mfcc)
        
        # 표준편차 비율 계산 (사용자/기준)
        std_ratio = user_std / ref_std if ref_std > 0 else 0.0
        debug_print(f"표준편차 비교 - 기준: {ref_std:.4f}, 사용자: {user_std:.4f}, 비율: {std_ratio:.4f}")
        
        # 표준편차 차이가 크면 단어 누락 가능성 높음
        # 사용자 표준편차가 기준 표준편차의 50% 미만이면 단어 누락으로 간주
        STD_THRESHOLD = 0.5
        if std_ratio < STD_THRESHOLD:
            debug_print(f"표준편차 비율이 너무 낮음 ({std_ratio:.4f} < {STD_THRESHOLD}) - 단어 누락 가능성 높음")
            
            # 표준편차 비율에 따라 패널티 적용 (비율이 낮을수록 더 큰 패널티)
            penalty_factor = 1.0 - (std_ratio / STD_THRESHOLD)
            similarity = 0.05 * (1.0 - penalty_factor)  # 최대 0.05의 유사도
            
            debug_print(f"표준편차 기반 단어 누락 감지 - 유사도: {similarity:.4f}")
            return similarity
        
        # 델타 특성이 없는 경우 추가
        if ref_mfcc.shape[1] == 13:
            ref_mfcc = _add_delta_features(ref_mfcc)
            debug_print(f"기준 MFCC에 델타 특성 추가 - shape: {ref_mfcc.shape}")
            
        if user_mfcc.shape[1] == 13:
            user_mfcc = _add_delta_features(user_mfcc)
            debug_print(f"사용자 MFCC에 델타 특성 추가 - shape: {user_mfcc.shape}")
        
        # CMVN 적용
        ref_norm = _apply_cmvn(ref_mfcc)
        user_norm = _apply_cmvn(user_mfcc)
        
        # 가중치 적용 (MFCC, 델타, 델타-델타에 다른 가중치 부여)
        dimension_weights = [1.5, 0.8, 0.3]  # MFCC, 델타, 델타-델타 가중치
        d = ref_norm.shape[1] // 3
        w = np.concatenate([
            np.full(d, dimension_weights[0]),
            np.full(d, dimension_weights[1]),
            np.full(d, dimension_weights[2])
        ])
        
        # 가중치 적용
        ref_norm = ref_norm * w
        user_norm = user_norm * w
        
        # 프레임 수가 다른 경우 패딩 적용
        if len(ref_norm) != len(user_norm):
            debug_print(f"프레임 수 불일치 - ref: {len(ref_norm)}, user: {len(user_norm)}")
            
            # 더 짧은 쪽에 패딩 적용
            if len(ref_norm) > len(user_norm):
                user_norm = _pad_reference_mfcc(user_norm, len(ref_norm))
            else:
                ref_norm = _pad_reference_mfcc(ref_norm, len(user_norm))
        
        # 유클리디언 거리 계산
        frame_distances = np.sqrt(np.sum((ref_norm - user_norm) ** 2, axis=1))
        mean_distance = np.mean(frame_distances)
        
        # 유사도로 변환
        similarity = 1.0 / (1.0 + mean_distance)
        
        # 표준편차 비율에 따른 유사도 조정
        # 표준편차 비율이 1에 가까울수록 패널티 없음, 0.5에 가까울수록 최대 30% 패널티
        if std_ratio < 1.0:
            std_penalty = (1.0 - std_ratio) * 0.3  # 최대 30% 패널티
            similarity = similarity * (1.0 - std_penalty)
            debug_print(f"표준편차 비율에 따른 패널티 적용: {std_penalty:.4f}, 조정된 유사도: {similarity:.4f}")
        
        debug_print(f"유사도 계산 완료: {similarity:.4f}")
        return similarity
        
    except Exception as e:
        print(f"[MFCC_ERROR] 유사도 계산 중 오류 발생: {str(e)}")
        return 0.0

def extract_mfcc_from_audio(audio_path: str, sr: int = 16000) -> tuple[np.ndarray, np.ndarray]:
    """
    음성에서 mfcc를 추출하는 함수
    개선된 버전: 25ms 윈도우, 10ms 홉 크기 사용
    """
    debug_print(f"extract_mfcc_from_audio 시작 - 파일: {audio_path}")
    
    # 오디오 로드
    y, _ = librosa.load(audio_path, sr=sr)
    
    # MFCC 추출 (25ms 윈도우, 10ms 홉 크기)
    n_fft = 512  # FFT window size
    win_length = int(0.025 * sr)  # 25ms
    hop_length = int(0.010 * sr)  # 10ms
    
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length
    )
    mfcc = mfcc.T  # (N,13) 형태로 변환. 즉, 프레임마다 13차원 벡터가 하나씩 존재하는 형태로 변환

    duration = librosa.get_duration(y=y, sr=sr) # 전체 오디오 길이
    n_frames = mfcc.shape[0]    #   프레임 수 n_frames만큼
    frame_times = np.linspace(0, duration, num=n_frames)    # 0초~duration초를 균등하게 나눔. 즉, 각 MFCC 벡터가 '몇 초쯤에 해당하는 소리인가'를 알 수 있다.

    debug_print(f"MFCC 추출 완료 - shape: {mfcc.shape}, 길이: {duration:.3f}초")
    return mfcc, frame_times

def extract_mfcc_segment(mfcc: np.ndarray, frame_times: np.ndarray, start_time: float, end_time: float) -> np.ndarray:
    """
    시작, 끝 시간을 입력 받아 mfcc 행렬을 추출
    """
    debug_print(f"segment 추출 - 시간: {start_time:.3f}~{end_time:.3f}초")
    
    # frame_times: 각 프레임이 오디오의 몇 초 시점에 해당하는지 담은 배열
    # np.serachsorted: start_time 또는 end_time에 해당하는 프레임 인덱스를 찾아줌
    start_idx = np.searchsorted(frame_times, start_time, side = "left")
    end_idx = np.searchsorted(frame_times, end_time, side = "right")
    
    # 인덱스 범위 확인
    start_idx = max(0, start_idx)
    end_idx = min(len(mfcc), end_idx)
    
    # 최소 프레임 수 확인 (너무 짧으면 의미 없는 세그먼트)
    MIN_FRAMES = 5
    if end_idx - start_idx < MIN_FRAMES:
        debug_print(f"프레임 수 부족 - {end_idx - start_idx} < 최소 기준: {MIN_FRAMES}")
        return np.array([])  # 빈 배열 반환하여 누락으로 처리
    
    debug_print(f"인덱스: {start_idx}~{end_idx}, 결과 shape: {mfcc[start_idx:end_idx].shape}")
    # MFCC 행렬 중에서 [start_time, end_time]에 해당하는 구간만 추출
    return mfcc[start_idx:end_idx]

def compare_mfcc_segments(cached_segments: list[dict], user_mfcc: np.ndarray, user_frame_times: np.ndarray, job_id: str = None) -> list[dict]:
    """
    기준 음성의 mfcc 행렬과 유저 음성의 mfcc 행렬을 서로 비교하여 유사도 점수를 계산
    개선된 버전: 표준편차 기반 단어 누락 감지 로직 포함
    """
    debug_print(f"compare_mfcc_segments 시작 - 세그먼트 수: {len(cached_segments)}")
    debug_print(f"user_mfcc shape: {user_mfcc.shape}")
    
    # 유저 음성 전체 길이 계산
    user_duration = user_frame_times[-1] if len(user_frame_times) > 0 else 0
    
    results = []

    # 기준 음성 세그먼트들의 시작 지점을 0으로 맞추는 정규화를 위한 offset
    # 모든 segment들의 시간에 offset만큼 빼서 0 기준으로 맞춘다.
    offset = cached_segments[0]['start_time'] if cached_segments else 0

    # 모든 세그먼트에 대해 반복     
    for i, segment in enumerate(cached_segments):
        word = segment["word"]
        start = segment["start_time"] - offset
        end = segment["end_time"] - offset
        
        debug_print(f"=== {i+1}번째 단어: '{word}' ({start:.3f}~{end:.3f}초) ===")
        
        # 🔍 기준 MFCC 상세 정보 출력
        ref_mfcc_data = segment.get("mfcc")
        debug_print(f"'{word}': 기준 MFCC 원본 타입: {type(ref_mfcc_data)}")
        
        if ref_mfcc_data is None:
            debug_print(f"'{word}': ❌ 기준 MFCC가 None → similarity = 0.0")
            similarity = 0.0
        elif ref_mfcc_data == []:
            debug_print(f"'{word}': ❌ 기준 MFCC가 빈 배열 → similarity = 0.0")
            similarity = 0.0
        else:
            try:
                ref_mfcc = np.array(ref_mfcc_data)
                debug_print(f"'{word}': ✅ 기준 MFCC shape = {ref_mfcc.shape}")
                debug_print(f"'{word}': 기준 MFCC 샘플 값: {ref_mfcc.flatten()[:5] if ref_mfcc.size > 0 else 'empty'}")
                
                user_segment = extract_mfcc_segment(user_mfcc, user_frame_times, start, end)
                debug_print(f"'{word}': ✅ 유저 segment shape = {user_segment.shape}")
                debug_print(f"'{word}': 유저 segment 샘플 값: {user_segment.flatten()[:5] if user_segment.size > 0 else 'empty'}")

                if ref_mfcc.shape[0] == 0:
                    debug_print(f"'{word}': ❌ 기준 MFCC가 빈 배열 (shape[0]=0) → similarity = 0.0")
                    similarity = 0.0
                elif user_segment.shape[0] == 0:
                    debug_print(f"'{word}': ❌ 유저 segment가 빈 배열 (shape[0]=0) → similarity = 0.0")
                    similarity = 0.0
                else:
                    # 🚀 개선된 방식: 표준편차 기반 단어 누락 감지 + 유클리디언 거리
                    similarity = frame_wise_euclidean_similarity(ref_mfcc, user_segment)
                    
                    if job_id:
                        # 표준편차 계산
                        ref_std = _calculate_std_deviation(ref_mfcc)
                        user_std = _calculate_std_deviation(user_segment)
                        std_ratio = user_std / ref_std if ref_std > 0 else 0.0
                        
                        # 정규화 전 원본 통계
                        ref_raw_mean = np.mean(ref_mfcc, axis=0)
                        user_raw_mean = np.mean(user_segment, axis=0)

                        # 디버그 출력을 위한 정규화된 평균값 계산
                        ref_norm = _apply_cmvn(ref_mfcc)
                        user_norm = _apply_cmvn(user_segment)

                        # 프레임별 거리 통계
                        min_frames = min(len(ref_norm), len(user_norm))
                        frame_distances = []
                        if min_frames > 0:
                            ref_aligned = ref_norm[:min_frames]
                            user_aligned = user_norm[:min_frames]
                            frame_distances = np.linalg.norm(ref_aligned - user_aligned, axis=1)
                        
                        print(f"[{job_id}] 🎵 MFCC 비교: '{word}'")
                        print(f"[{job_id}]   기준 원본 평균: {np.round(ref_raw_mean[:5], 2)}...")  # 처음 5개만
                        print(f"[{job_id}]   유저 원본 평균: {np.round(user_raw_mean[:5], 2)}...")
                        print(f"[{job_id}]   표준편차 비율: {std_ratio:.2f} (기준: {ref_std:.2f}, 유저: {user_std:.2f})")
                        if len(frame_distances) > 0:
                            print(f"[{job_id}]   거리 min/mean/max: {frame_distances.min():.2f}/{frame_distances.mean():.2f}/{frame_distances.max():.2f}")
                        print(f"[{job_id}]   유사도: {similarity:.3f}")
                    
                    # 유사도 값 검증
                    if np.isnan(similarity):
                        debug_print(f"'{word}': ⚠️ similarity가 NaN! → 0.0으로 설정")
                        similarity = 0.0
                        
            except Exception as e:
                debug_print(f"'{word}': ❌ 처리 중 에러: {str(e)} → similarity = 0.0")
                similarity = 0.0
        
        # 연속 선형 보간을 사용한 점수 환산 (0~100 범위)
        # 
        # [기존 if-elif 구조의 설정 근거]
        # 데이터셋 결과 기반 점수 환산:
        # - 정답 음성 ≃0.45 → 90-100점 구간
        # - 팀원(정상 발음) ≃0.10 → 70점 
        # - 단어 2개 누락 ≃0.08 → 50점
        # - 액센트 없이 발음 ≃0.09 → 60점  
        # - 침묵 ≃0.02 → 0점
        #
        # [기존 구간별 매핑 로직]
        # if similarity >= 0.4:    # 0.4~0.53 => 90~100점 (정답 수준)
        # elif similarity >= 0.3:  # 0.3~0.39 => 80~89점 (우수)
        # elif similarity >= 0.1:  # 0.1~0.29 => 70~79점 (정상)
        # elif similarity >= 0.09: # 0.09~0.099 => 60~69점 (액센트 없음)
        # elif similarity >= 0.08: # 0.08~0.089 => 50~59점 (단어 누락)
        # elif similarity >= 0.05: # 0.05~0.079 => 40~49점 (부족)
        # else: 0.02 이하 => 0점 (침묵), 0.02~0.049 => 0~39점 (매우 부족)
        #
        # 위 로직을 연속 선형 보간으로 단순화:
        adjusted_score = continuous_score(similarity)
        
        # 정규화된 점수 (0~1 범위)
        normalized_score = adjusted_score / 100.0
        
        if job_id:
            print(f"[{job_id}]   환산 점수: {adjusted_score:.1f}/100 (원시 유사도: {similarity:.3f})")
        
        results.append({
            "word": word,
            "similarity": similarity,
            "adjusted_score": normalized_score  # 0~1 범위로 정규화된 점수 추가
        })
        
        debug_print(f"'{word}': 최종 similarity = {similarity:.6f}, 환산 점수 = {adjusted_score:.1f}/100")

    debug_print(f"compare_mfcc_segments 완료 - 결과 수: {len(results)}")
    return results

def _calculate_std_deviation(mfcc):
    """
    MFCC 특성의 표준편차 계산
    
    Parameters:
    -----------
    mfcc : ndarray
        MFCC 특성 (시간, 특성)
        
    Returns:
    --------
    float
        MFCC 계수의 평균 표준편차
    """
    if mfcc.shape[0] == 0:
        return 0.0
        
    # 각 MFCC 계수의 표준편차 계산
    if mfcc.shape[1] >= 13:  # 기본 MFCC (델타 특성 포함 가능)
        # 첫 13개 계수만 사용 (델타 특성 제외)
        std_dev = np.mean(np.std(mfcc[:, :13], axis=0))
    else:
        std_dev = np.mean(np.std(mfcc, axis=0))
    
    return std_dev