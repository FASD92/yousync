"""
User Processor 메인 파이프라인 (main.py)

전체 음성 분석 및 평가 파이프라인:
1. 음성 전처리 (audio_cleaner.py)
2. STT 처리 (stt_processor.py)  
3. MFA 정렬 (mfa_aligner.py)
4. 종합 비교 분석 (textgrid_comparator.py)
5. 최종 점수 산출 (scorer.py)

사용법:
    python main.py --input input_audio/user_voice.mp4
    python main.py --input input_audio/user_voice.mp4 --video_id jZOywn1qArI
"""

import argparse
import sys
import json
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime

# 로컬 모듈들 import
from audio_cleaner import clean_user_audio
from stt_processor import transcribe_user_audio, get_optimized_segments_for_pitch
from mfa_aligner import get_reference_data_from_local_storage, create_reference_lab_file, prepare_mfa_corpus, run_mfa_align_docker
from textgrid_comparator import comprehensive_analysis
from scorer import calculate_final_score
from voice_to_pitch import create_user_pitch_json

def ensure_shared_data_structure():
    """shared_data 디렉토리 구조가 없으면 자동 생성"""
    shared_data_path = Path("../shared_data")
    
    if not shared_data_path.exists():
        print("📁 shared_data 디렉토리가 없습니다. 자동 생성 중...")
        from setup_directories import create_shared_data_structure
        create_shared_data_structure()

def clean_previous_results():
    """이전 실행 결과 자동 정리"""
    print("🧹 이전 실행 결과 정리 중...")
    
    shared_data_path = Path("../shared_data")
    
    try:
        # 정리할 디렉토리들 (내용만 삭제, 디렉토리는 유지)
        cleanup_dirs = [
            "mfa_corpus",
            "pitch_data/user",
            "user/lab",
            "user/processed"
        ]
        
        # mfa_output에서 사용자 파일만 삭제
        mfa_output_path = shared_data_path / "mfa_output"
        if mfa_output_path.exists():
            user_textgrid = mfa_output_path / "user.TextGrid"
            if user_textgrid.exists():
                user_textgrid.unlink()
        
        for dir_name in cleanup_dirs:
            dir_path = shared_data_path / dir_name
            if dir_path.exists():
                # 디렉토리 내용만 삭제
                for item in dir_path.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
        
        # 정리할 파일들
        cleanup_files = [
            "comparison_result.json",
            "final_result.json"
        ]
        
        for file_name in cleanup_files:
            file_path = shared_data_path / file_name
            if file_path.exists():
                file_path.unlink()
        
        print("✅ 이전 실행 결과 정리 완료")
        
    except Exception as e:
        print(f"⚠️ 정리 중 오류 (무시하고 계속): {e}")

def process_user_audio(input_audio_path: str, video_id: str = None, 
                      remove_background: bool = True) -> dict:
    """
    사용자 음성 전체 처리 파이프라인
    
    Args:
        input_audio_path: 입력 음성 파일 경로
        video_id: 특정 기준 비디오 ID (None이면 자동 선택)
        remove_background: 배경 제거 여부
    
    Returns:
        전체 처리 결과 딕셔너리
    """
    print("=" * 60)
    print("🚀 User Processor 파이프라인 시작")
    print("=" * 60)
    
    start_time = datetime.now()
    shared_data_path = Path("../shared_data")
    results = {}
    
    # shared_data 디렉토리 구조 확인 및 생성
    ensure_shared_data_structure()
    
    # 이전 실행 결과 자동 정리
    #clean_previous_results()
    
    try:
        # 1. 음성 전처리
        print("\n🧹 1단계: 음성 전처리")
        print("-" * 40)
        
        input_path = Path(input_audio_path)
        output_path = shared_data_path / "user" / "processed" / f"cleaned_{input_path.stem}.wav"
        
        cleaned_audio = clean_user_audio(str(input_path), str(output_path), remove_background)
        results['cleaned_audio'] = cleaned_audio
        
        # 2. STT 처리
        print("\n🎯 2단계: STT 처리")
        print("-" * 40)
        
        stt_result = transcribe_user_audio(cleaned_audio)
        results['stt_result'] = stt_result
        
        # 3. MFA 정렬
        print("\n🔗 3단계: MFA 정렬")
        print("-" * 40)
        
        # 기준 데이터 로드 (로컬 shared_data에서)
        reference_data = get_reference_data_from_local_storage(video_id)
        
        # 기준 .lab 파일 생성
        reference_lab_path = shared_data_path / "reference" / "lab" / f"{reference_data['video_id']}.lab"
        create_reference_lab_file(reference_data['reference_text'], reference_lab_path)
        
        # MFA 정렬 실행
        corpus_dir = prepare_mfa_corpus(
            reference_data['audio_file'],
            str(reference_lab_path),
            cleaned_audio,
            stt_result['lab_file']
        )
        
        output_dir = run_mfa_align_docker(corpus_dir)
        
        # 결과 TextGrid 파일 확인
        reference_textgrid = Path(output_dir) / "reference.TextGrid"
        user_textgrid = Path(output_dir) / "user.TextGrid"
        
        mfa_result = {
            "corpus_dir": corpus_dir,
            "output_dir": output_dir,
            "reference_textgrid": str(reference_textgrid) if reference_textgrid.exists() else None,
            "user_textgrid": str(user_textgrid) if user_textgrid.exists() else None
        }
        
        print("🎉 MFA 정렬 완료!")
        if mfa_result["reference_textgrid"]:
            print(f"📊 기준 TextGrid: {mfa_result['reference_textgrid']}")
        if mfa_result["user_textgrid"]:
            print(f"📊 사용자 TextGrid: {mfa_result['user_textgrid']}")
        
        results['mfa_result'] = mfa_result
        results['reference_data'] = reference_data
        
        # 4. 종합 비교 분석
        print("\n📊 4단계: 종합 비교 분석")
        print("-" * 40)
        
        # 세그먼트 정보 생성 - STT 결과에서 실제 시간 구간 사용
        segments = get_optimized_segments_for_pitch(stt_result)
        
        # 세그먼트가 없으면 전체 구간을 하나의 세그먼트로 처리 (fallback)
        if not segments:
            print("⚠️ STT 세그먼트가 없어 전체 구간을 사용합니다.")
            segments = [{
                "start": 0.0,
                "end": 10.0,  # 임시값
                "text": reference_data["reference_text"]
            }]
        else:
            print(f"✅ STT에서 {len(segments)}개 세그먼트 추출 완료")
        
        comparison_result = comprehensive_analysis(
            mfa_result['reference_textgrid'],
            mfa_result['user_textgrid'],
            reference_data['audio_file'],
            cleaned_audio,
            segments
        )
        results['comparison_result'] = comparison_result
        
        # 5. 최종 점수 산출
        print("\n📊 5단계: 최종 점수 산출")
        print("-" * 40)
        
        final_scores = calculate_final_score(comparison_result)
        results['final_scores'] = final_scores
        
        # 6. 결과 정리 및 저장
        print("\n💾 6단계: 결과 저장")
        print("-" * 40)
        
        final_result = compile_final_result(results)
        
        # 처리 시간 계산
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        final_result['processing_info'] = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "processing_time_seconds": round(processing_time, 2)
        }
        
        # 최종 결과 저장
        result_path = shared_data_path / "final_result.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 최종 결과 저장: {result_path}")
        
        # 결과 요약 출력
        print_summary(final_result)
        
        return final_result
        
    except Exception as e:
        print(f"❌ 파이프라인 실행 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def compile_final_result(results: dict) -> dict:
    """
    모든 단계의 결과를 종합하여 최종 결과 생성
    
    Args:
        results: 각 단계별 처리 결과
    
    Returns:
        종합된 최종 결과
    """
    return {
        "user_info": {
            "input_file": str(results.get('cleaned_audio', '')),
            "recognized_text": results.get('stt_result', {}).get('text', ''),
            "language": results.get('stt_result', {}).get('language', '')
        },
        "reference_info": {
            "video_id": results.get('reference_data', {}).get('video_id', ''),
            "reference_text": results.get('reference_data', {}).get('reference_text', ''),
            "audio_file": results.get('reference_data', {}).get('audio_file', '')
        },
        "analysis_results": {
            "pronunciation_score": results.get('comparison_result', {}).get('overall_scores', {}).get('pronunciation_score', 0),
            "timing_score": results.get('comparison_result', {}).get('overall_scores', {}).get('timing_score', 0),
            "pitch_score": results.get('comparison_result', {}).get('overall_scores', {}).get('pitch_score', 0),
            "final_score": results.get('final_scores', {}).get('overall_score', 0)
        },
        "detailed_analysis": {
            "textgrid_analysis": results.get('comparison_result', {}).get('textgrid_analysis', {}),
            "pitch_analysis": results.get('comparison_result', {}).get('pitch_analysis', {}),
            "scoring_breakdown": results.get('final_scores', {})
        },
        "file_paths": {
            "cleaned_audio": results.get('cleaned_audio', ''),
            "user_lab": results.get('stt_result', {}).get('lab_file', ''),
            "reference_textgrid": results.get('mfa_result', {}).get('reference_textgrid', ''),
            "user_textgrid": results.get('mfa_result', {}).get('user_textgrid', ''),
            "comparison_result": str(Path("../shared_data") / "comparison_result.json"),
            "final_result": str(Path("../shared_data") / "final_result.json")
        }
    }

def print_summary(result: dict):
    """
    결과 요약 출력
    
    Args:
        result: 최종 결과 딕셔너리
    """
    print("\n" + "=" * 60)
    print("🎉 User Processor 완료!")
    print("=" * 60)
    
    analysis = result.get('analysis_results', {})
    
    print(f"📝 인식된 텍스트: {result.get('user_info', {}).get('recognized_text', '')[:100]}...")
    print(f"🎬 기준 비디오: {result.get('reference_info', {}).get('video_id', '')}")
    print()
    print("📊 점수 결과:")
    print(f"   🗣️  발음 점수:  {analysis.get('pronunciation_score', 0):6.1f}/100")
    print(f"   ⏰ 타이밍 점수: {analysis.get('timing_score', 0):6.1f}/100")
    print(f"   🎵 피치 점수:  {analysis.get('pitch_score', 0):6.1f}/100")
    print(f"   🏆 최종 점수:  {analysis.get('final_score', 0):6.1f}/100")
    print()
    print(f"⏱️  처리 시간: {result.get('processing_info', {}).get('processing_time_seconds', 0):.1f}초")
    print(f"📁 결과 파일: {result.get('file_paths', {}).get('final_result', '')}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='User Processor - 음성 분석 및 평가 파이프라인')
    parser.add_argument('--input', '-i', required=True, help='입력 음성 파일 경로')
    parser.add_argument('--video_id', '-v', help='기준 비디오 ID (선택사항)')
    parser.add_argument('--no_background_removal', action='store_true', help='배경 제거 건너뛰기')
    
    args = parser.parse_args()
    
    # 입력 파일 확인
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)
    
    print("🎤 User Processor 초기화 완료")
    print(f"📁 공유 데이터 경로: {Path('../shared_data').absolute()}")
    
    try:
        result = process_user_audio(
            str(input_path),
            args.video_id,
            not args.no_background_removal
        )
        
        print("\n🎉 모든 처리가 완료되었습니다!")
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 처리 중 오류 발생: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

def process_user_audio_with_s3_reference(s3_textgrid_url: str,
                                        s3_pitch_url: str,
                                        s3_user_audio_url: str,
                                        remove_background: bool = True) -> dict:
    """
    S3 기준 데이터를 사용한 사용자 음성 처리 파이프라인
    
    Args:
        s3_textgrid_url: 기준 TextGrid S3 URL
        s3_pitch_url: 기준 피치 JSON S3 URL  
        s3_user_audio_url: 사용자 음성 S3 URL
        remove_background: 배경 제거 여부
    
    Returns:
        전체 처리 결과 딕셔너리
    """
    print("=" * 60)
    print("🎯 S3 기준 데이터 기반 음성 분석 파이프라인 시작")
    print("=" * 60)
    
    start_time = datetime.now()
    results = {}
    
    # shared_data 디렉토리 구조 확인
    ensure_shared_data_structure()
    
    # 이전 결과 정리
    #clean_previous_results()
    
    shared_data_path = Path("../shared_data")
    
    try:
        # 1. 사용자 음성 준비 및 전처리
        print("🧹 1단계: 사용자 음성 준비 및 전처리")
        print("-" * 40)
        
        # S3에서 사용자 음성 다운로드
        print(f"📥 S3에서 사용자 음성 다운로드: {s3_user_audio_url}")
        from s3_downloader import download_user_audio_from_s3
        temp_audio_path = f"/tmp/user_audio_{start_time.strftime('%Y%m%d_%H%M%S')}.mp4"
        download_user_audio_from_s3(s3_user_audio_url, temp_audio_path)
        input_path = Path(temp_audio_path)
        
        output_path = shared_data_path / "user" / "processed" / f"cleaned_{input_path.stem}.wav"
        
        cleaned_audio = clean_user_audio(str(input_path), str(output_path), remove_background)
        results['cleaned_audio'] = cleaned_audio
        results['cleaned_audio'] = cleaned_audio
        
        # 2. STT 처리
        print("\n🎯 2단계: STT 처리")
        print("-" * 40)
        
        stt_result = transcribe_user_audio(cleaned_audio)
        results['stt_result'] = stt_result
        
        # 3. S3에서 기준 데이터 다운로드
        print("\n🌐 3단계: S3 기준 데이터 다운로드")
        print("-" * 40)
        
        from s3_downloader import download_reference_data_from_s3
        reference_data = download_reference_data_from_s3(s3_textgrid_url, s3_pitch_url)
        results['reference_data'] = reference_data
        
        # 4. User 전용 MFA 정렬
        print("\n🔗 4단계: User MFA 정렬")
        print("-" * 40)
        
        from mfa_aligner import run_mfa_align_user_only
        user_textgrid_path = run_mfa_align_user_only(
            cleaned_audio,
            stt_result['lab_file']
        )
        
        mfa_result = {
            "reference_textgrid": reference_data['textgrid_file'],
            "user_textgrid": user_textgrid_path,
            "source": "s3"
        }
        results['mfa_result'] = mfa_result
        
        # 5. 종합 비교 분석
        print("\n📊 5단계: 종합 비교 분석")
        print("-" * 40)
        
        # 세그먼트 정보 생성 - STT 결과에서 실제 시간 구간 사용
        segments = get_optimized_segments_for_pitch(stt_result)
        
        # 세그먼트가 없으면 전체 구간을 하나의 세그먼트로 처리 (fallback)
        if not segments:
            print("⚠️ STT 세그먼트가 없어 전체 구간을 사용합니다.")
            segments = [{
                "start": 0.0,
                "end": 10.0,  # 임시값
                "text": "Full audio segment"
            }]
        else:
            print(f"✅ STT에서 {len(segments)}개 세그먼트 추출 완료")
        
        comparison_result = comprehensive_analysis(
            mfa_result['reference_textgrid'],
            mfa_result['user_textgrid'],
            None,  # 기준 오디오 파일 불필요 (S3 방식)
            cleaned_audio,
            segments
        )
        results['comparison_result'] = comparison_result
        
        # 6. 최종 점수 산출
        print("\n📊 6단계: 최종 점수 산출")
        print("-" * 40)
        
        final_scores = calculate_final_score(comparison_result)
        results['final_scores'] = final_scores
        
        # 7. 결과 정리 및 저장
        print("\n💾 7단계: 결과 저장")
        print("-" * 40)
        
        final_result = compile_final_result_s3(results)
        
        # 처리 시간 계산
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        final_result['processing_info'] = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "processing_time_seconds": round(processing_time, 2)
        }
        
        # 최종 결과 저장
        result_path = shared_data_path / "final_result_s3.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 S3 기반 분석 완료! (처리 시간: {processing_time:.1f}초)")
        print_summary_s3(final_result)
        
        return {
            "success": True,
            "processing_time": processing_time,
            "analysis_results": final_result
        }
        
    except Exception as e:
        print(f"\n❌ S3 기반 파이프라인 실행 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "processing_time": (datetime.now() - start_time).total_seconds()
        }
    
    finally:
        # 임시 데이터 정리
        try:
            from s3_downloader import cleanup_temp_reference_data
            #cleanup_temp_reference_data()
        except Exception as e:
            print(f"⚠️ 임시 데이터 정리 중 오류: {e}")

def compile_final_result_s3(results: dict) -> dict:
    """
    S3 기반 최종 결과 컴파일
    """
    return {
        "analysis_type": "s3_reference",
        "reference_source": {
            "textgrid_url": results.get('reference_data', {}).get('s3_textgrid_url'),
            "pitch_url": results.get('reference_data', {}).get('s3_pitch_url')
        },
        "scores": {
            "pronunciation_score": results.get('comparison_result', {}).get('overall_scores', {}).get('pronunciation_score', 0),
            "timing_score": results.get('comparison_result', {}).get('overall_scores', {}).get('timing_score', 0),
            "pitch_score": results.get('comparison_result', {}).get('overall_scores', {}).get('pitch_score', 0),
            "final_score": results.get('final_scores', {}).get('overall_score', 0)
        },
        "detailed_analysis": {
            "textgrid_analysis": results.get('comparison_result', {}).get('textgrid_analysis', {}),
            "pitch_analysis": results.get('comparison_result', {}).get('pitch_analysis', {}),
            "scoring_breakdown": results.get('final_scores', {})
        },
        "file_paths": {
            "cleaned_audio": results.get('cleaned_audio', ''),
            "user_lab": results.get('stt_result', {}).get('lab_file', ''),
            "reference_textgrid": results.get('reference_data', {}).get('textgrid_file', ''),
            "user_textgrid": results.get('mfa_result', {}).get('user_textgrid', ''),
            "final_result": str(Path("../shared_data") / "final_result_s3.json")
        }
    }

def print_summary_s3(result: dict):
    """
    S3 기반 결과 요약 출력
    """
    print("\n" + "=" * 50)
    print("📊 S3 기반 분석 결과 요약")
    print("=" * 50)
    
    scores = result.get('scores', {})
    print(f"🎯 최종 점수: {scores.get('final_score', 0):.1f}점")
    print(f"📢 발음 점수: {scores.get('pronunciation_score', 0):.1f}점")
    print(f"⏰ 타이밍 점수: {scores.get('timing_score', 0):.1f}점")
    print(f"🎵 피치 점수: {scores.get('pitch_score', 0):.1f}점")
    
    ref_source = result.get('reference_source', {})
    print(f"\n📁 기준 데이터 소스:")
    print(f"   TextGrid: {ref_source.get('textgrid_url', 'N/A')}")
    print(f"   Pitch: {ref_source.get('pitch_url', 'N/A')}")
