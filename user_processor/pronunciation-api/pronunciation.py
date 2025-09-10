from fastapi import FastAPI, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from datetime import datetime
from zoneinfo import ZoneInfo
import time
from urllib.parse import urlparse
import uuid
import boto3
import tempfile
import json
import numpy as np
import shutil
import requests
import re
from pathlib import Path
from run_whisper_cpp import speech_to_text
from text_similarity import compare_texts, parse_whisper_cpp_result, normalize_segments_to_zero, parse_time, normalize_and_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from mfcc_similarity import extract_mfcc_from_audio, extract_mfcc_segment, compare_mfcc_segments

import threading
import asyncio

app = FastAPI(
    title = "Voice Analysis API",
    description = "유저 음성 분석 및 평가 자동화 API")

# SQS 설정 추가
SQS_QUEUE_URL = 'https://sqs.ap-northeast-2.amazonaws.com/975049946580/audio-analysis-queue'
sqs_client = boto3.client('sqs', region_name='ap-northeast-2')

# 작업 상태 저장 딕셔너리
job_status = {}

@app.get("/")
async def root():
    return {"message" : "Voice Analysis API", "status" : "running"}

@app.get("/status")
async def server_status():
    return {
        "status" : "running",
        "timestamp" : datetime.now().isoformat(),
        "message" : "Voice Analysis API 정상 작동 중"
    }

@app.post("/analyze-voice")
async def analyze_voice(
    background_tasks: BackgroundTasks,
    request_data: str = Form(...)
):
    # 고유한 job_id 생성
    job_id = str(uuid.uuid4())

    # 시작 시간 기록
    start_time = time.time()
    start_datetime_kst = datetime.now(ZoneInfo("Asia/Seoul"))
    print(f"[{job_id}] 🚀 요청 시작: {start_datetime_kst.strftime('%Y-%m-%d %H:%M:%S KST')}")
    
    data = json.loads(request_data)
    s3_audio_url = data["s3_audio_url"]
    webhook_url = data["webhook_url"]
    script_data = data["script"]

    # 기본 정보 로그
    print(f"[{job_id}] 📁 S3 URL: {s3_audio_url}")

    try:
        # 작업 상태 초기화
        job_status[job_id] = {
            "status": "processing",
            "s3_audio_url": s3_audio_url,
            "started_at": datetime.now().isoformat()
        }

        audio_path = download_from_s3(s3_audio_url, job_id)

        # 백그라운드에서 처리
        background_tasks.add_task(analyze_pronunciation_pipeline, job_id, audio_path, script_data, s3_audio_url, webhook_url, start_time)

        return {
            "job_id": job_id,
            "status": "processing",
            "message": "음성 분석이 시작되었습니다."
        }

    except ValueError as e:
        job_status[job_id] = {
            "status": "failed",
            "s3_audio_url": s3_audio_url,
            "started_at": datetime.now().isoformat(),
            "error": str(e)
        }
        return JSONResponse(
            status_code=400,
            content={"error": f"잘못된 요청: {str(e)}"}
        )

    except Exception as e:
        job_status[job_id] = {
            "status": "failed",
            "s3_audio_url": s3_audio_url,
            "started_at": datetime.now().isoformat(),
            "error": str(e)
        }
        return JSONResponse(
            status_code=500,
            content={"error": f"서버 오류: {str(e)}"}
        )

def parse_s3_url(s3_url: str) -> tuple[str, str]:
    if s3_url.startswith('s3://'):
        parts = s3_url.split('/', 3)
        if len(parts) < 4:
            raise ValueError("올바른 s3://bucket/key 형식이 아닙니다.")
        return parts[2], parts[3]
    elif 'amazonaws.com' in s3_url:
        parsed = urlparse(s3_url)
        netloc_parts = parsed.netloc.split('.')
        if len(netloc_parts) < 3 or not parsed.path:
            raise ValueError("올바르지 않은 S3 URL 형식입니다.")
        bucket = netloc_parts[0]
        key = parsed.path.lstrip('/')
        return bucket, key
    else:
        raise ValueError("지원되지 않은 S3 URL 형식입니다.")

def download_from_s3(s3_audio_url: str, job_id: str) -> str:
    s3_client = boto3.client('s3')
    audio_bucket, audio_key = parse_s3_url(s3_audio_url)

    job_folder = Path(f"/tmp/{job_id}")
    job_folder.mkdir(exist_ok=True)

    audio_path = job_folder / "user_audio.wav"
    s3_client.download_file(audio_bucket, audio_key, str(audio_path))

    # shared_data 폴더에도 백업 저장
    import os
    shared_dir = os.path.expanduser('~/shared_data/user')
    os.makedirs(shared_dir, exist_ok=True)
    shared_file_path = f"{shared_dir}/{job_id}.wav"
    shutil.copy2(str(audio_path), shared_file_path)
    return str(audio_path)
    print(f"[{job_id}] 📁 유저 음성 백업: {shared_file_path}")

def calculate_time_overlap(ref_start, ref_end, user_start, user_end):
    """두 시간 구간의 겹침 정도를 계산"""
    overlap_start = max(ref_start, user_start)
    overlap_end = min(ref_end, user_end)
    
    if overlap_start >= overlap_end:
        return 0.0, 0.0  # 겹침 없음
    
    overlap_duration = overlap_end - overlap_start
    ref_duration = ref_end - ref_start
    overlap_ratio = overlap_duration / ref_duration if ref_duration > 0 else 0.0
    
    return overlap_duration, overlap_ratio

def log_time_overlap_analysis(job_id, ref_word, ref_start, ref_end, user_word, user_start, user_end, overlap_ratio, threshold=0.4):
    """시간 겹침 분석 로그 출력"""
    ref_duration = ref_end - ref_start
    user_duration = user_end - user_start
    start_diff = user_start - ref_start
    end_diff = user_end - ref_end
    
    if overlap_ratio >= threshold:
        status = "✅"
        result = "매칭"
    else:
        status = "❌" if overlap_ratio < threshold else "⚠️"
        result = "매칭 실패" if overlap_ratio < threshold else "매칭 (시간 불일치)"
    
    print(f"[{job_id}]   {status} \"{ref_word}\" {result}:")
    print(f"[{job_id}]     └─ 기준: {ref_start:.3f}s~{ref_end:.3f}s vs 유저: {user_start:.3f}s~{user_end:.3f}s")
    print(f"[{job_id}]     └─ 겹침: {max(ref_start, user_start):.3f}s~{min(ref_end, user_end):.3f}s ({overlap_ratio*ref_duration:.3f}s) / 기준길이: {ref_duration:.3f}s = {overlap_ratio*100:.1f}% 겹침")
    print(f"[{job_id}]     └─ 시작점 차이: {start_diff:+.3f}s, 종료점 차이: {end_diff:+.3f}s")
    
    if overlap_ratio < threshold:
        print(f"[{job_id}]     └─ 실패 원인: 겹침률 {overlap_ratio*100:.1f}% < 임계값 {threshold*100:.0f}%")
    
    return overlap_ratio >= threshold

def extract_word_timestamps(stt_result):
    """STT 결과에서 단어별 시간 정보 추출"""
    word_timestamps = []
    
    if not stt_result.get('transcription'):
        return word_timestamps
    
    transcription = stt_result['transcription'][0]
    
    if 'words' in transcription and isinstance(transcription['words'], list) and transcription['words']:
        word_timestamps = []
        for word in transcription['words']:
            if 'word' in word and 'start' in word and 'end' in word:
                word_timestamps.append({
                    'word': word['word'],
                    'start_time': word['start'],
                    'end_time': word['end']
                    })
        return word_timestamps

    # tokens 리스트 가져오기
    tokens = transcription.get('tokens', [])

    # tokens가 없으면 빈 리스트 반환
    if not tokens:
        return word_timestamps
    
    # 단어 병합을 위한 변수 초기화
    current_text = ""       # 병합중인 단어 텍스트
    word_start = None       # 현재 단어의 시작 시간
    prev_end = None         # 직전 토큰의 종료 시간

    # 각 토큰을 순회하며 단어 병합 및 타임스탬프 추출
    for token in tokens:
        # 제어 토큰 스킵
        text = token.get('text', '')
        if text.startswith('[') and text.endswith(']'):
            continue
        # 순수 구두점 토큰 제외
        if re.fullmatch(r'[\W_]+', text):
            continue
        # 토큰의 시작, 종료 시간을 초 단위로 파싱
        start = parse_time(token['timestamps']['from'])
        end = parse_time(token['timestamps']['to'])

        # subword 경계: 앞에 공백이 있으면 새 단어 시작
        if text.startswith(' '):
            # 이전 단어가 있으면 먼저 저장
            if current_text:
                word_timestamps.append({
                    'word': current_text,
                    'start_time': word_start,
                    'end_time': prev_end
                })
            current_text = text.strip()
            word_start = start
        else:
            # subword 이어 붙이기
            current_text += text.strip()

        #prev_end 갱신
        prev_end = end

    # 마지막 단어 저장
    if current_text:
        word_timestamps.append({
            'word': current_text,
            'start_time': word_start,
            'end_time': prev_end
        })
    return word_timestamps

async def analyze_pronunciation_pipeline(job_id: str, audio_path: str, script_data: dict, s3_audio_url: str, webhook_url: str, start_time: float):
    """
    시간 정보와 겹침 분석이 포함된 개선된 발음 분석 파이프라인
    """
    
    step_times = {}  # 각 단계별 처리 시간 기록
    
    try:
        # 유저 음성 파일 길이 확인
        try:
            import librosa
            audio_duration = librosa.get_duration(path=str(audio_path))
            print(f"[{job_id}] 📊 유저 음성 길이: {audio_duration:.3f}초")
        except Exception as e:
            print(f"[{job_id}] ⚠️ 오디오 길이 확인 실패: {e}")
            audio_duration = 0.0

        # 기준 스크립트 정보 로그
        reference_words = [word_data['word'] for word_data in script_data['words']]
        reference_text = ' '.join(reference_words)
        
        # 기준 스크립트 총 길이 계산
        if script_data['words'] and script_data['words'][-1].get('end_time', None) is not None:
            first_start = script_data['words'][0].get('start_time', 0)
            last_end = script_data['words'][-1]['end_time']
            ref_total_duration = last_end - first_start  # 상대적 길이 계산
        
        print(f"[{job_id}] 📝 기준 문장: \"{reference_text}\"")
        print(f"[{job_id}] 🎯 기준 단어 수: {len(reference_words)}개, 총 길이: {ref_total_duration:.3f}초")

        # Step 1: STT 변환
        stt_start = time.time()
        print(f"[{job_id}] 🎤 STT 변환 중...")
        
        output_dir = Path(audio_path).parent
        model_path = Path("./whisper.cpp/models/ggml-medium.en.bin")
        
        user_stt_result = speech_to_text(audio_path=Path(audio_path), output_dir=output_dir, model_path=model_path)
        
        stt_end = time.time()
        step_times['stt'] = stt_end - stt_start
        
        # STT 결과 추출
        user_text = ""
        if user_stt_result.get('transcription'):
            user_text = user_stt_result['transcription'][0].get('text', '').strip()
        
        print(f"[{job_id}] 🎤 STT 인식 결과: \"{user_text}\"")
        # 유저 단어 시간 정보 추출
        reference_segments = normalize_segments_to_zero(script_data['words'])
        if user_text.startswith('['):
            print(f"[{job_id}] ⚠️ 음성 신호가 감지되지 않음 ({user_text})")
            user_word_timestamps = []
        else:
            # STT 토큰에서 직접 word-level 타임 스탬프 추출
            raw_timestamps = extract_word_timestamps(user_stt_result)
            # 0초 기준으로 정규화
            user_word_timestamps = raw_timestamps

        user_words = [item['word'] for item in user_word_timestamps]
        print(f"[{job_id}] 📊 인식된 단어 수: {len(user_words)}개 (기준: {len(reference_words)}개)")
        if len(user_words) != len(reference_words):
            diff = len(reference_words) - len(user_words)
            if diff > 0:
                print(f"[{job_id}] ⚠️ 단어 수 불일치 감지: {diff}개 단어 누락")
            else:
                print(f"[{job_id}] ⚠️ 단어 수 불일치 감지: {abs(diff)}개 단어 초과")

        # 유저 단어별 시간 정보 로그
        print(f"[{job_id}] ⏰ 유저 단어별 시간 정보:")
        for item in user_word_timestamps:
            start = item.get('start_time')
            end = item.get('end_time')
            # None인 경우 건너뛰기
            if start is None or end is None:
                continue
            duration = end - start
            print(f"[{job_id}]   \"{item['word']}\": {start:.3f}s ~ {end:.3f}s (길이: {duration:.3f}s)")

        # 기준 단어별 시간 정보 로그
        print(f"[{job_id}] 📋 기준 단어별 시간 정보:")
        for word_data in reference_segments:
            word = word_data['word']
            start = word_data.get('start_time', 0.0)
            end = word_data.get('end_time', 0.0)
            duration = end - start
            print(f"[{job_id}]   \"{word}\": {start:.3f}s ~ {end:.3f}s (길이: {duration:.3f}s)")

        # Step 2: 시간 겹침 분석
        overlap_start = time.time()
        
        time_matching_results = analyze_time_overlap(job_id, reference_segments, user_word_timestamps)
        
        overlap_end = time.time()
        step_times['overlap'] = overlap_end - overlap_start

        # Step 3: 텍스트 비교
        text_start = time.time()
        
        text_comparison_result = compare_texts(reference_segments, user_stt_result)
        
        text_end = time.time()
        step_times['text'] = text_end - text_start
        
        # 텍스트 매칭 요약 로그
        log_text_matching_summary(job_id, text_comparison_result, time_matching_results, [word_data['word'] for word_data in reference_segments])

        # Step 4: MFCC 분석
        mfcc_start = time.time()
        
        user_mfcc, user_frame_times = extract_mfcc_from_audio(audio_path)
        mfcc_comparison_result = compare_mfcc_segments(
            cached_segments=reference_segments, 
            user_mfcc=user_mfcc, 
            user_frame_times=user_frame_times, 
            job_id=job_id
        )
        
        mfcc_end = time.time()
        step_times['mfcc'] = mfcc_end - mfcc_start
        
        # MFCC 결과 상세 로그
        log_mfcc_analysis_with_time(job_id, reference_segments, mfcc_comparison_result, time_matching_results, user_word_timestamps)

        # Step 5: 종합 결과 생성
        final_results = generate_comprehensive_results(
            job_id, [word_data['word'] for word_data in reference_segments], text_comparison_result, 
            mfcc_comparison_result, time_matching_results
        )
        
        # 유저 STT 결과 추가
        user_stt_data = {
            "text": user_text,
            "word_timestamps": user_word_timestamps
        }
        final_results["user_stt"] = user_stt_data
        
        # Step 6: 웹훅 전송
        webhook_start = time.time()
        print(f"[{job_id}] 📤 웹훅 전송 중...")
        
        webhook_response = requests.post(webhook_url, json={
            "job_id": job_id,
            "status": "completed",
            "result": final_results
        }, timeout=10)
        
        webhook_end = time.time()
        step_times['webhook'] = webhook_end - webhook_start
        
        total_end = time.time()
        total_duration = total_end - start_time
        
        print(f"[{job_id}] ⏱️ 처리 시간 분석:")
        print(f"[{job_id}]   STT 변환: {step_times['stt']:.2f}초")
        print(f"[{job_id}]   시간 겹침 분석: {step_times['overlap']:.2f}초")
        print(f"[{job_id}]   텍스트 비교: {step_times['text']:.2f}초")
        print(f"[{job_id}]   MFCC 분석: {step_times['mfcc']:.2f}초")
        print(f"[{job_id}]   웹훅 전송: {step_times['webhook']:.2f}초")
        print(f"[{job_id}] 🏁 전체 처리 시간: {total_duration:.2f}초 (상태: {webhook_response.status_code})")

    except Exception as e:
        print(f"[{job_id}] ❌ 파이프라인 에러 발생: {str(e)}")
        
        # 실패 웹훅 전송
        try:
            webhook_start = time.time()
            requests.post(webhook_url, json={
                "job_id": job_id,
                "status": "failed",
                "error": str(e)
            }, timeout=10)
            webhook_end = time.time()
            total_end = time.time()
            print(f"[{job_id}] ⏱️ 실패 웹훅 전송 시간: {webhook_end - webhook_start:.2f}초")
            print(f"[{job_id}] 🏁 전체 처리 시간: {total_end - start_time:.2f}초")
        except Exception as webhook_error:
            print(f"[{job_id}] ❌ 웹훅 호출 실패: {webhook_error}")

    finally:
        try:
            job_folder = Path(audio_path).parent
            shutil.rmtree(job_folder)
            print(f"[{job_id}] 🧹 임시 파일 정리 완료")
        except Exception as cleanup_error:
            print(f"[{job_id}] ⚠️ 임시 파일 정리 실패: {cleanup_error}")
            
def analyze_time_overlap(job_id, reference_words, user_word_timestamps, threshold=0.4):
    """시간 겹침 분석 수행 및 상세 로그 출력"""
    print(f"[{job_id}] 🔍 단어별 시간 겹침 분석:")
    
    time_matching_results = []
    
    # 유저 단어를 딕셔너리로 변환 (빠른 검색을 위해)
    #user_words_dict = {item['word'].lower(): item for item in user_word_timestamps}
    user_words_dict = {
        normalize_and_tokenize(item['word'])[0]: item for item in user_word_timestamps
    }
    
    for ref_word_data in reference_words:
        ref_word = ref_word_data['word']
        ref_start = ref_word_data.get('start_time', 0.0)
        ref_end = ref_word_data.get('end_time', 0.0)
        
        # 해당 단어가 유저 STT에 있는지 확인
        ref_key = normalize_and_tokenize(ref_word)[0]
        user_match = user_words_dict.get(ref_key)
        
        if user_match:
            user_start = user_match['start_time']
            user_end = user_match['end_time']
            
            overlap_duration, overlap_ratio = calculate_time_overlap(ref_start, ref_end, user_start, user_end)
            is_time_match = log_time_overlap_analysis(job_id, ref_word, ref_start, ref_end, ref_word, user_start, user_end, overlap_ratio, threshold)
            
            time_matching_results.append({
                'word': ref_word,
                'time_match': is_time_match,
                'overlap_ratio': overlap_ratio,
                'ref_start': ref_start,
                'ref_end': ref_end,
                'user_start': user_start,
                'user_end': user_end
            })
        else:
            print(f"[{job_id}]   ❌ \"{ref_word}\" 매칭 실패:")
            print(f"[{job_id}]     └─ 기준: {ref_start:.3f}s~{ref_end:.3f}s vs 유저: 매칭 후보 없음")
            print(f"[{job_id}]     └─ 실패 원인: STT에서 인식되지 않음")
            
            time_matching_results.append({
                'word': ref_word,
                'time_match': False,
                'overlap_ratio': 0.0,
                'ref_start': ref_start,
                'ref_end': ref_end,
                'user_start': None,
                'user_end': None
            })
    
    return time_matching_results

def log_text_matching_summary(job_id, text_comparison_result, time_matching_results, reference_words):
    """텍스트 매칭 요약 로그 출력"""
    print(f"[{job_id}] 📝 텍스트 매칭 요약:")
    
    complete_success = 0  # 텍스트 + 시간 모두 성공
    text_only_success = 0  # 텍스트만 성공
    time_insufficient = 0  # 시간 겹침 부족
    stt_failure = 0  # STT 인식 실패
    
    for i, word in enumerate(reference_words):
        text_status = text_comparison_result[i]["status"] if i < len(text_comparison_result) else "fail"
        time_result = time_matching_results[i] if i < len(time_matching_results) else None
        
        if text_status == "pass" and time_result and time_result['time_match']:
            complete_success += 1
        elif text_status == "pass" and time_result and not time_result['time_match'] and time_result['user_start'] is not None:
            text_only_success += 1
        elif text_status == "fail" and time_result and time_result['user_start'] is not None:
            time_insufficient += 1
        else:
            stt_failure += 1
    
    print(f"[{job_id}]   ✅ 완전 성공: {complete_success}개 (텍스트 + 시간 매칭)")
    if text_only_success > 0:
        print(f"[{job_id}]   ⚠️ 텍스트만 일치: {text_only_success}개 (시간 불일치)")
    if time_insufficient > 0:
        print(f"[{job_id}]   ❌ 시간 겹침 부족: {time_insufficient}개")
    if stt_failure > 0:
        print(f"[{job_id}]   ❌ STT 인식 실패: {stt_failure}개")
    
    text_success_rate = (complete_success + text_only_success) / len(reference_words) * 100
    time_success_rate = complete_success / len(reference_words) * 100
    
    print(f"[{job_id}] 📈 텍스트 매칭 성공률: {complete_success + text_only_success}/{len(reference_words)} ({text_success_rate:.1f}%)")
    print(f"[{job_id}] ⏰ 시간 매칭 성공률: {complete_success}/{len(reference_words)} ({time_success_rate:.1f}%)")

def log_mfcc_analysis_with_time(job_id, reference_words, mfcc_comparison_result, time_matching_results, user_word_timestamps):
    """MFCC 분석 결과를 시간 정보와 함께 로그 출력"""
    print(f"[{job_id}] 🎵 MFCC 유사도 분석:")
    
    time_matched_scores = []
    all_scores = []
    
    for i, word_data in enumerate(reference_words):
        word = word_data['word']
        ref_start = word_data.get('start_time', 0.0)
        ref_end = word_data.get('end_time', 0.0)
        
        mfcc_similarity = 0.0
        if i < len(mfcc_comparison_result):
            mfcc_similarity = mfcc_comparison_result[i].get("similarity", 0.0)
        
        time_result = time_matching_results[i] if i < len(time_matching_results) else None
        
        if time_result and time_result['user_start'] is not None:
            user_start = time_result['user_start']
            user_end = time_result['user_end']
            
            if time_result['time_match']:
                status = "🎯"
                quality = get_mfcc_quality_description(mfcc_similarity)
                time_matched_scores.append(mfcc_similarity)
                print(f"[{job_id}]   {status} \"{word}\" (시간매칭 성공): {mfcc_similarity:.3f} ({quality})")
                print(f"[{job_id}]     └─ 분석구간: 기준 {ref_start:.3f}s~{ref_end:.3f}s vs 유저 {user_start:.3f}s~{user_end:.3f}s")
            else:
                status = "⚠️"
                quality = get_mfcc_quality_description(mfcc_similarity)
                print(f"[{job_id}]   {status} \"{word}\" (시간매칭 실패): {mfcc_similarity:.3f} ({quality})")
                print(f"[{job_id}]     └─ 분석구간: 기준 {ref_start:.3f}s~{ref_end:.3f}s vs 유저 {user_start:.3f}s~{user_end:.3f}s")
                print(f"[{job_id}]     └─ 낮은 이유: 시간 불일치로 인한 잘못된 구간 비교")
            
            all_scores.append(mfcc_similarity)
        else:
            print(f"[{job_id}]   ❌ \"{word}\": N/A (STT 인식 실패)")
    
    # 평균 계산
    if all_scores:
        overall_avg = sum(all_scores) / len(all_scores)
        print(f"[{job_id}] 📊 평균 MFCC 유사도: {overall_avg:.3f}", end="")
        
        if time_matched_scores:
            time_matched_avg = sum(time_matched_scores) / len(time_matched_scores)
            print(f" (시간매칭 성공한 단어만: {time_matched_avg:.3f})")
        else:
            print()
    else:
        print(f"[{job_id}] 📊 평균 MFCC 유사도: N/A (분석 가능한 단어 없음)")

def get_mfcc_quality_description(similarity):
    """MFCC 유사도 점수에 따른 품질 설명"""
    if similarity >= 0.8:
        return "우수"
    elif similarity >= 0.6:
        return "양호"
    elif similarity >= 0.4:
        return "보통"
    elif similarity >= 0.2:
        return "낮음"
    else:
        return "매우 낮음"

def generate_comprehensive_results(job_id, reference_words, text_comparison_result, mfcc_comparison_result, time_matching_results):
    """종합 결과 생성 및 실패 원인 분석"""
    
    # 단어별 통합 결과 생성
    word_analysis = []
    text_pass_count = 0
    time_pass_count = 0
    
    # 실패 원인 분류
    stt_failures = []
    time_failures = []
    mfcc_low_quality = []
    
    for i, word in enumerate(reference_words):
        # 텍스트 결과
        text_status = text_comparison_result[i]["status"] if i < len(text_comparison_result) else "fail"
        
        # 시간 매칭 결과
        time_result = time_matching_results[i] if i < len(time_matching_results) else None
        time_match = time_result['time_match'] if time_result else False
        overlap_ratio = time_result['overlap_ratio'] if time_result else 0.0
        
        # MFCC 결과
        mfcc_similarity = 0.0
        if i < len(mfcc_comparison_result):
            # 환산된 점수가 있으면 사용, 없으면 원래 유사도 사용
            if "adjusted_score" in mfcc_comparison_result[i]:
                mfcc_similarity = mfcc_comparison_result[i].get("adjusted_score", 0.0)
            else:
                mfcc_similarity = mfcc_comparison_result[i].get("similarity", 0.0)
        
        # 종합 점수 계산 (텍스트 70% + MFCC 30%)
        confidence = text_comparison_result[i].get("confidence", 0.0)
        text_score = confidence if text_status == "pass" else 0.0
        #text_score = 1.0 if text_status == "pass" else 0.0
        print(f"❌{confidence * 0.7}")
        word_score = (text_score * 0.7) + (mfcc_similarity * 0.3)
        
        word_analysis.append({
            "word": word,
            "text_status": text_status,
            "time_match": time_match,
            "overlap_ratio": round(overlap_ratio, 3),
            "mfcc_similarity": round(mfcc_similarity, 3),
            "word_score": round(word_score, 3)
        })
        
        # 통과 개수 계산
        if text_status == "pass":
            text_pass_count += 1
        if time_match:
            time_pass_count += 1
        
        # 실패 원인 분류
        if time_result and time_result['user_start'] is None:
            stt_failures.append(word)
        elif not time_match and time_result and time_result['user_start'] is not None:
            time_failures.append({
                'word': word,
                'overlap_ratio': overlap_ratio
            })
        
        if mfcc_similarity < 0.4:  # 낮은 MFCC 품질
            mfcc_low_quality.append({
                'word': word,
                'similarity': mfcc_similarity
            })
    
    # 전체 요약 계산
    text_accuracy = text_pass_count / len(reference_words) if reference_words else 0.0
    time_accuracy = time_pass_count / len(reference_words) if reference_words else 0.0
    
    # MFCC 평균 계산
    mfcc_total = sum(result["mfcc_similarity"] for result in word_analysis)
    mfcc_average = mfcc_total / len(word_analysis) if word_analysis else 0.0
    
    # 전체 점수 계산
    score_total = sum(result["word_score"] for result in word_analysis)
    overall_score = score_total / len(word_analysis) if word_analysis else 0.0
    
    # 실패 원인 분석 로그
    print(f"[{job_id}] 🔍 실패 원인 상세 분석:")
    
    if stt_failures:
        quoted_words = [f'"{w}"' for w in stt_failures]
        print(f"[{job_id}]   📝 STT 인식 실패: {len(stt_failures)}개 단어 ({', '.join(quoted_words)})")

        # If every word failed, assume blank audio
        if len(stt_failures) == len(reference_words):
            print(f"[{job_id}]     └─ 원인: 음성 신호가 감지되지 않음 (모두 인식 실패)")
            print(f"[{job_id}]     └─ 제안: 다시 녹음하거나 마이크 및 볼륨 설정을 확인하세요")
        else:
            # Otherwise, check for short words
            short_words = [w for w in stt_failures if len(w) <= 2]
            if short_words:
                print(f"[{job_id}]     └─ 원인: 짧은 단어 인식 실패")
                print(f"[{job_id}]     └─ 제안: 짧은 단어를 더 명확하게 발음")
            else:
                print(f"[{job_id}]     └─ 제안: 발음을 더 명확하게 하거나 속도를 조절")
    
    if time_failures:
        print(f"[{job_id}]   ⏰ 시간 불일치: {len(time_failures)}개 단어")
        for failure in time_failures:
            word = failure['word']
            ratio = failure['overlap_ratio']
            print(f"[{job_id}]     └─ \"{word}\": {ratio*100:.1f}% 겹침 (임계값 40% 미달)")
        print(f"[{job_id}]     └─ 제안: 기준 음성과 비슷한 속도로 발음")
    
    if mfcc_low_quality:
        high_quality_words = [item for item in mfcc_low_quality if item['similarity'] >= 0.2]
        very_low_words = [item for item in mfcc_low_quality if item['similarity'] < 0.2]
        
        if high_quality_words:
            print(f"[{job_id}]   🎵 MFCC 품질 개선 필요: {len(high_quality_words)}개 단어")
            print(f"[{job_id}]     └─ 제안: 발음 정확도 향상 필요")
        
        if very_low_words:
            print(f"[{job_id}]   🎵 MFCC 품질 매우 낮음: {len(very_low_words)}개 단어")
            print(f"[{job_id}]     └─ 제안: 해당 단어들의 발음을 다시 연습")
    
    if not stt_failures and not time_failures and not mfcc_low_quality:
        print(f"[{job_id}]   ✅ 전반적으로 양호한 발음 품질")
        print(f"[{job_id}]     └─ 제안: 현재 수준 유지")
    
    # 최종 결과 로그
    print(f"[{job_id}] 📋 최종 결과:")
    print(f"[{job_id}]   🏆 전체 점수: {overall_score:.3f} ({overall_score*100:.1f}%)")
    print(f"[{job_id}]   📝 텍스트 정확도: {text_accuracy:.3f} ({text_pass_count}/{len(reference_words)})")
    print(f"[{job_id}]   ⏰ 시간 정확도: {time_accuracy:.3f} ({time_pass_count}/{len(reference_words)})")
    print(f"[{job_id}]   🎵 MFCC 평균: {mfcc_average:.3f}")
    
    return {
        "overall_score": round(overall_score, 3),
        "word_analysis": word_analysis,
        "summary": {
            "text_accuracy": round(text_accuracy, 3),
            "time_accuracy": round(time_accuracy, 3),
            "mfcc_average": round(mfcc_average, 3),
            "total_words": len(reference_words),
            "passed_words": text_pass_count,
            "time_matched_words": time_pass_count
        },
        "failure_analysis": {
            "stt_failures": stt_failures,
            "time_failures": [f["word"] for f in time_failures],
            "mfcc_low_quality": [f["word"] for f in mfcc_low_quality]
        }
    }

# SQS 관련 함수들
def sqs_message_processor():
    """SQS에서 메시지를 받아서 처리하는 함수"""
    while True:
        try:
            response = sqs_client.receive_message(
                QueueUrl=SQS_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20
            )

            messages = response.get('Messages', [])
            for message in messages:
                print(f"📨 SQS 메시지 수신: {message['MessageId']}")

                # 메시지 처리
                process_sqs_message(message)

                # 메시지 삭제
                sqs_client.delete_message(
                    QueueUrl=SQS_QUEUE_URL,
                    ReceiptHandle=message['ReceiptHandle']
                )
                print(f"✅ SQS 메시지 처리 완료: {message['MessageId']}")

        except Exception as e:
            print(f"❌ SQS 처리 오류: {e}")
            time.sleep(5)

def process_sqs_message(message):
    """SQS 메시지를 파싱하여 분석 실행"""
    try:
        # 메시지 파싱
        body = json.loads(message['Body'])

        job_id = body['job_id']
        s3_audio_url = body['s3_audio_url']
        webhook_url = body['webhook_url']
        video_id = body.get('video_id', 'unknown')

        print(f"🎯 SQS 분석 시작 - Job: {job_id}")

        # 스크립트 데이터 준비
        #script_data = get_script_data_by_video_id(video_id)

        # 기존 분석 파이프라인 실행
        start_time = time.time()
        audio_path = download_from_s3(s3_audio_url, job_id)

        # 비동기 함수를 동기적으로 실행
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            analyze_pronunciation_pipeline(
                job_id, audio_path, script_data, s3_audio_url, webhook_url, start_time
            )
        )
        loop.close()

    except Exception as e:
        print(f"❌ SQS 메시지 처리 실패: {e}")
        # 실패 시에도 웹훅 전송
        try:
            requests.post(webhook_url, json={
                "job_id": body.get('job_id', 'unknown'),
                "status": "failed",
                "error": str(e)
            }, timeout=10)
        except Exception as webhook_error:
            print(f"❌ 실패 웹훅 전송 실패: {webhook_error}")

# def get_script_data_by_video_id(video_id: str) -> dict:
#     """video_id로 스크립트 데이터 조회"""
#     # TODO: 실제 데이터로 교체 필요
#     return {
#         "words": [
#             {"word": "hello", "start": 0.0, "end": 0.5, "mfcc": None},
#             {"word": "world", "start": 0.6, "end": 1.0, "mfcc": None},
#             {"word": "test", "start": 1.1, "end": 1.5, "mfcc": None}
#         ]
#     }

@app.on_event("startup")
async def startup_event():
    """FastAPI 서버 시작 시 SQS 처리 스레드 시작"""
    print("🚀 Voice Analysis API 서버 시작 - SQS 처리 스레드 활성화")

    sqs_thread = threading.Thread(target=sqs_message_processor, daemon=True)
    sqs_thread.start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pronunciation:app", host="0.0.0.0", port=8001, reload=False)