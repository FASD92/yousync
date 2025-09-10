import re
import json

"""
텍스트를 정규화 및 토크나이징 및
기준 음성과 유저 음성을 정렬하여 텍스트 유사도를 확인
"""

def normalize_and_tokenize(text: str) -> list[str]:
    text = text.lower()                     #1. 모든 문자를 소문자로
    text = text.replace("’", "'")           #2. 유니코드 아포스트로피 -> 일반 아포스트로피 정규화(아포스트로피가 종류가 2개라고 함...)
    text = re.sub(r"[^\w\s']", "", text)    #2. \w 단어 문자, \s 공백 문자, ' 아포스트로피 제외 모든 문자 제거
    text = re.sub(r"\s+", " ", text)        #3. 연속된 공백들을 하나의 공백으로 통합
    text = text.strip()                     #4. 문자열 앞뒤 공백 제거
    return text.split()                     #5. 공백 기준으로 분할 -> 문자열 리스트로 반환

def parse_time(time_str: str) -> float:
    """
    '00:00:01,500' → 1.5 (초)
    whisper.cpp json의 시간 단위가 저런 방식이라 계산하기 쉽게 parse 해주는 헬퍼 함수
    """
    parts = time_str.replace(',','.').split(':')
    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])

def parse_whisper_cpp_result(whisper_json: dict) -> list[dict]:
    """whisper.cpp JSON에서 단어별 타임스탬프와 텍스트를 추출하여 세그먼트 리스트로 변환"""
    user_segments = []
    for segment in whisper_json.get('transcription', []):
        for token in segment.get('tokens', []):
            if 'text' in token and not token['text'].startswith('['):               # token에 'text' 키가 있고, '[music]' 같은 특수 토큰이 아닌지 확인
                word = token['text'].strip()                                        # token의 'text'에서 앞뒤 공백 제거 
                if word:                                                            # 빈 문자열이 아닌 경우에만 처리
                    user_segments.append({                                          
                        'word': word,
                        'start_time': parse_time(token['timestamps']['from']),
                        'end_time': parse_time(token['timestamps']['to']),
                        'confidence': token.get('p', 0.0)
                        })
    return user_segments

def normalize_segments_to_zero(segments: list[dict]) -> list[dict]:
    """세그먼트들을 0초 기준으로 정규화"""
    if not segments:
        return segments # 빈 리스트인 경우 그대로 반환

    offset = segments[0]['start_time']                                              # 첫 번째 세그먼트의 시작 시간을 오프셋으로 설정
    normalized_segments = []
    for seg in segments:                                                            # 각 세그먼트를 순회하며 start_time과 end_time을 0초 기준으로 정규화
        new_seg = seg.copy()
        new_seg['start_time'] = seg['start_time'] - offset
        new_seg['end_time'] = seg['end_time'] - offset
        normalized_segments.append(new_seg)
    return normalized_segments

def word_match_with_normalization(ref_word: str, user_word: str) -> bool:
    """정규화된 두 단어가 일치하는지 비교해서 bool 값 반환"""
    # 0초 정규화
    ref_normalized = normalize_and_tokenize(ref_word)
    user_normalized = normalize_and_tokenize(user_word)

    ref_clean = ref_normalized[0] if ref_normalized else ""
    user_clean = user_normalized[0] if user_normalized else ""

    return ref_clean == user_clean

# 문자 단위 LCS 유사도 임계값
LCS_THRESH = 0.6

def score_text_alignment(reference_segments: list[dict], user_whisper_json: dict) -> list[dict]:
    """
    기준 음성 세그먼트와 유저 음성의 STT를 3중 필터링으로 비교해 각 단어별 pass/fail 상태 반환
    3중 필터링: 1) 단어 일치 2) 시간 구간 겹침 3) 시작 시점 근접성
    """
    results = []
    

    user_segments = parse_whisper_cpp_result(user_whisper_json)             # whiser json에서 세그먼트 파싱
    print("▶ score_text_alignment parsed user_segments:", user_segments)
    user_normalized = normalize_segments_to_zero(user_segments)
    ref_normalized = normalize_segments_to_zero(reference_segments)         # 기준 세그먼트를 0초 정규화
    
    
    # 3중 필터링 파라미터
    tol = 0.25   # 윈도우 크기 허용 오차
    start_tol = 0.5 # 시작점 허용 오차

    for ref in ref_normalized:                                                              # 정규화된 각 기준 세그먼트에 대해 순회 시작
        matched = False
        window_start, window_end = ref['start_time'] - tol, ref['end_time'] + tol           # 기준 세그먼트 시간에 허용 오차를 더한 윈도우 범위 계산
        
        # 후보군 필터링 : 시간 구간이 겹치는 유저 세그먼트만 추출
        candidates = [
            user for user in user_normalized
            if not (user['end_time'] < window_start or user['start_time'] > window_end)
        ]

        if candidates:
            # 최대 겹침 후보 계산
            best_user = max(
                candidates,
                key=lambda user: min(window_end, user['end_time']) - max(window_start, user['start_time'])
            )
            # 유사도 계산
            if word_match_with_normalization(ref['word'], best_user['word']):
                lcs_sim = 1.0
            else:
                lcs_sim = compute_char_lcs_ratio(ref['word'], best_user['word'])
            
            # 임계값 미만이면 불일치 처리
            if lcs_sim < LCS_THRESH:
                results.append({
                    "word": ref['word'],
                    "status": "fail",
                    "lcs_similarity": float(lcs_sim)
                })
                continue

            # 5단계 : 시작점 허용 오차 및 겹침률 체크
            start_ok = abs(best_user['start_time'] - ref['start_time']) <= start_tol
            overlap = min(window_end, best_user['end_time']) - max(window_start, best_user['start_time'])
            ref_dur = ref['end_time'] - ref['start_time']
            ratio = overlap / ref_dur if ref_dur else 0.0
            if start_ok and ratio >= 0.70:
                status = "pass"
            else:
                status = "fail"

            results.append({
                "word": ref['word'], 
                "status": status, 
                "lcs_similarity": float(lcs_sim),
                "confidence": best_user.get('confidence', 0.0)
                })
        else:
            results.append({
                "word": ref['word'], 
                "status": "fail", 
                "lcs_similarity": 0.0,
                "confidence" : 0.0
                })
    return results

def compare_texts(reference_segments: list[dict], user_whisper_json: dict) -> list[dict]:
    """기준 음성 세그먼트와 유저 STT 결과를 비교하여 텍스트 유사도 평가 결과 반환"""
    return score_text_alignment(reference_segments, user_whisper_json)

# 문자 단위 LCS 유사도 계산 함수
def compute_char_lcs_ratio(a: str, b: str) -> float:
    """
    두 문자열 a, b의 LCS(Longest Common Subsequence) 길이를 계산한 뒤,
    max(len(a), len(b))로 나누어 유사도 비율(0.0~1.0)을 반환합니다.
    예: a='hello', b='hallo' → LCS='hllo' → 4/5 = 0.8
    """
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n] / max(m, n) if max(m, n) > 0 else 0.0