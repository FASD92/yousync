"""
MFA 정렬 모듈 (mfa_aligner.py)

기능:
- 기준 음성과 사용자 음성을 각각 MFA align 수행
- Docker를 통한 MFA 실행
- TextGrid 파일 생성 및 처리

입력: 
- 기준 음성 WAV + LAB 파일
- 사용자 음성 WAV + LAB 파일

출력: 각각의 TextGrid 파일
"""

import subprocess
import shutil
import json
import sys
import os
from pathlib import Path

def get_reference_data_from_local_storage(video_id: str = None) -> dict:
    """
    로컬 shared_data에서 기준 데이터 가져오기
    
    Args:
        video_id: 특정 비디오 ID (None이면 사용 가능한 첫 번째)
    
    Returns:
        기준 데이터 정보
    """
    # shared_data 경로에서 데이터 찾기
    shared_data_path = Path("../shared_data")
    
    if video_id:
        # 특정 비디오 ID 찾기 - .lab 파일에서
        token_files = list((shared_data_path / "reference" / "lab").glob(f"{video_id}*.lab"))
        if token_files:
            token_file = token_files[0]  # 첫 번째 매칭 파일 사용
        else:
            raise FileNotFoundError(f"비디오 ID {video_id}에 해당하는 토큰 파일을 찾을 수 없습니다.")
        
        actual_video_id = video_id
        # 기준 오디오 파일 (있다면)
        audio_file = shared_data_path / "reference" / "audio" / f"{video_id}.mp3"
    else:
        # 사용 가능한 첫 번째 파일 찾기
        token_files = list((shared_data_path / "reference" / "lab").glob("*.lab"))
        if not token_files:
            raise FileNotFoundError("shared_data에서 토큰 파일을 찾을 수 없습니다.")
        
        token_file = token_files[0]
        # 파일명에서 비디오 ID 추출
        filename = token_file.stem
        if '_' in filename:
            actual_video_id = filename.split('_')[0]  # jZOywn1qArI_Liam_Neeson_Taken_token -> jZOywn1qArI
        else:
            actual_video_id = filename  # jZOywn1qArI
        audio_file = shared_data_path / "reference" / "audio" / f"{actual_video_id}.mp3"
    
    # 토큰 데이터 로드 (.lab 파일은 텍스트 파일)
    if token_file.exists():
        with open(token_file, 'r', encoding='utf-8') as f:
            reference_text = f.read().strip()
    else:
        raise FileNotFoundError(f"토큰 파일을 찾을 수 없습니다: {token_file}")
    
    print(f"✅ 로컬 기준 데이터 로드 완료: {actual_video_id}")
    print(f"📄 토큰 파일: {token_file}")
    print(f"📝 기준 텍스트: {reference_text[:100]}...")
    
    return {
        "video_id": actual_video_id,
        "audio_file": str(audio_file) if audio_file.exists() else None,
        "token_file": str(token_file),
        "reference_text": reference_text
    }

def create_reference_lab_file(reference_text: str, output_path: str) -> str:
    """
    기준 텍스트로 .lab 파일 생성
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(reference_text.strip())
    
    print(f"📄 기준 .lab 파일 생성: {output_path}")
    return output_path

def prepare_mfa_corpus(reference_wav: str, reference_lab: str, 
                      user_wav: str, user_lab: str) -> str:
    """
    MFA align을 위한 corpus 디렉토리 준비
    """
    print("📁 MFA corpus 디렉토리 준비 중...")
    
    shared_data_path = Path("../shared_data")
    corpus_path = shared_data_path / "mfa_corpus"
    
    # corpus 디렉토리 초기화
    #if corpus_path.exists():
    #    shutil.rmtree(corpus_path)
    #corpus_path.mkdir(exist_ok=True)
    
    try:
        # 파일들을 corpus 디렉토리로 복사
        shutil.copy2(reference_wav, corpus_path / "reference.wav")
        shutil.copy2(reference_lab, corpus_path / "reference.lab")
        shutil.copy2(user_wav, corpus_path / "user.wav")
        shutil.copy2(user_lab, corpus_path / "user.lab")
        
        print(f"✅ Corpus 준비 완료: {corpus_path}")
        print(f"   - reference.wav, reference.lab")
        print(f"   - user.wav, user.lab")
        
        return str(corpus_path)
        
    except Exception as e:
        print(f"❌ Corpus 준비 실패: {str(e)}")
        raise

def optimize_docker_performance():
    """Docker 성능 최적화 (이미지 미리 로드 및 확인)"""
    
    print("🐳 Docker 성능 최적화 중...")
    
    # Docker 데몬 실행 확인
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=10)
        print("✅ Docker 데몬 실행 중")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        print("⚠️ Docker 데몬이 실행되지 않았습니다. Docker Desktop을 시작해주세요.")
        return False
    
    # MFA 이미지가 로컬에 있는지 확인
    result = subprocess.run(
        ["docker", "images", "mmcauliffe/montreal-forced-aligner", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True, text=True
    )
    
    if "mmcauliffe/montreal-forced-aligner:latest" not in result.stdout:
        print("📥 MFA Docker 이미지 다운로드 중... (최초 1회만)")
        try:
            subprocess.run(["docker", "pull", "mmcauliffe/montreal-forced-aligner:latest"], 
                         check=True, timeout=300)
            print("✅ MFA Docker 이미지 준비 완료")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print("❌ Docker 이미지 다운로드 실패")
            return False
    else:
        print("✅ MFA Docker 이미지 캐시됨 (빠른 실행 가능)")
    
    return True

def run_mfa_align_docker(corpus_dir: str) -> str:
    """
    Docker를 통해 MFA align 실행
    """
    print("🚀 MFA align 실행 중...")
    
    # Docker 성능 최적화
    if not optimize_docker_performance():
        raise RuntimeError("Docker 최적화 실패")
    
    shared_data_path = Path("../shared_data")
    output_path = shared_data_path / "mfa_output"
    output_path.mkdir(exist_ok=True)
    
    try:
        # shared_data의 MFA 모델 경로로 변경
        mfa_models_path = Path("../shared_data/mfa")
        
        # Docker 명령어 구성
        cmd = [
            "docker", "exec", "mfa-persistent",
            "mfa", "align", 
            "/data", 
            "/models/english_us_arpa.dict",
            "/models/english_us_arpa",
            "/output"
        ]
        
        print(f"🚀 실행 명령어: docker exec mfa-persistent mfa align")
        
        # MFA 실행
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300
        )
        
        if result.returncode == 0:
            print("✅ MFA align 성공!")
            print(f"📁 출력 디렉토리: {output_path}")
            return str(output_path)
        else:
            print(f"❌ MFA align 실패:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd)
            
    except subprocess.TimeoutExpired:
        print("❌ MFA align 타임아웃 (5분 초과)")
        raise
    except Exception as e:
        print(f"❌ MFA align 오류: {str(e)}")
        raise

def run_mfa_align_user_only(user_wav_path: str, user_lab_path: str) -> str:
    """
    User 음성만 MFA 정렬 (기존 마운트 활용)
    """
    print("🚀 User 전용 MFA align 실행 중... (기존 마운트 활용)")
    
    # Docker 성능 최적화
    if not optimize_docker_performance():
        raise RuntimeError("Docker 최적화 실패")
    
    shared_data_path = Path("../shared_data")
    
    try:
        # 기존 /data 디렉토리 정리 후 user 파일만 복사
        import subprocess
        import shutil
        
        # 호스트의 mfa_corpus 디렉토리 정리
        mfa_corpus_dir = shared_data_path / "mfa_corpus"
        
        # 기존 파일들 정리
        #for file in mfa_corpus_dir.glob("*"):
        #    file.unlink()
        
        # user 파일들을 mfa_corpus로 복사
        user_wav_dest = mfa_corpus_dir / "user.wav"
        user_lab_dest = mfa_corpus_dir / "user.lab"
        
        shutil.copy2(user_wav_path, user_wav_dest)
        shutil.copy2(user_lab_path, user_lab_dest)
        
        print(f"📁 User 파일들을 기존 마운트로 복사 완료")
        print(f"   - {user_wav_dest}")
        print(f"   - {user_lab_dest}")
        
        # MFA 출력 디렉토리
        output_path = shared_data_path / "mfa_output"
        output_path.mkdir(exist_ok=True)
        
        # 기존 user.TextGrid 제거
        user_textgrid = output_path / "user.TextGrid"
        if user_textgrid.exists():
            user_textgrid.unlink()
        
        # MFA 실행 (기존 마운트 사용)
        cmd = [
            "docker", "exec", "mfa-persistent",
            "mfa", "align", 
            "/data", 
            "/models/english_us_arpa.dict",
            "/models/english_us_arpa",
            "/output",
            "--single_speaker",
            "--use_mp",
            "--clear"
        ]
        
        print(f"🚀 실행 명령어: docker exec mfa-persistent mfa align (기존 마운트)")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            if user_textgrid.exists():
                print("✅ User MFA align 성공!")
                print(f"📊 User TextGrid: {user_textgrid}")
                return str(user_textgrid)
            else:
                raise FileNotFoundError("User TextGrid 파일이 생성되지 않았습니다.")
        else:
            print(f"❌ User MFA align 실패:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd)
            
    except subprocess.TimeoutExpired:
        print("❌ User MFA align 타임아웃 (5분 초과)")
        raise
    except Exception as e:
        print(f"❌ User MFA align 오류: {str(e)}")
        raise