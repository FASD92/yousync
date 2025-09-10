"""
whisper.cpp를 CLI를 호출해 STT 결과를 JSON으로 반환!

whisper.cpp는 파이썬 라이브러리가 아니라 C++로 작성된 독립 실행 바이너리.

그래서 파이썬에서 다른 프로그램(바이너리 등)을 외부에서 실행시킬 수 있게 해주는
subprocess라는 표준 라이브러리를 이용해야 함
"""

import subprocess
import json
from pathlib import Path

def speech_to_text(audio_path: Path, output_dir: Path, model_path: Path) -> dict:
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_path = Path(output_dir) / f"{Path(audio_path).stem}.json"

    cmd = [
        "./whisper.cpp/build/bin/whisper-cli",
        "-m", model_path,
        "-f", audio_path,
        "-l", "en",
        "-t", "2",
        "--no-prints",
        "-nf",
        "-of", output_path.with_suffix(""),
        "--output-json-full"    # JSON 출력
        #"-oj" # JSON 출력
    ]

    try:
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Whisper.cpp 실행 실패: {e}")
        raise

    with open(output_path, "r", encoding="utf-8") as f:
        """
        json.load()는 JSON -> dict or list 같은 파이썬 객체로 변환
        """
        result = json.load(f)
        
    return result