#!/usr/bin/env python3
"""
shared_data 디렉토리 구조 자동 생성 스크립트

사용법:
    python3 setup_directories.py
"""

from pathlib import Path

def create_shared_data_structure():
    """shared_data 디렉토리 구조 생성"""
    
    base_path = Path("../shared_data")
    
    # 필요한 디렉토리들
    directories = [
        "user/input",
        "user/processed", 
        "user/lab",
        "reference/audio",
        "reference/lab",
        "reference/tokens",
        "mfa_corpus",
        "mfa_output", 
        "pitch_data/reference",
        "pitch_data/user"
    ]
    
    print("📁 shared_data 디렉토리 구조 생성 중...")
    
    for dir_path in directories:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ {full_path}")
    
    # README 파일 생성
    readme_content = """# Shared Data Directory

이 디렉토리는 User Processor 실행 시 자동으로 생성되는 데이터들을 저장합니다.

## 디렉토리 구조

```
shared_data/
├── user/                   # 사용자 관련 데이터
│   ├── input/             # 입력 음성 파일들
│   ├── processed/         # 전처리된 음성 파일들
│   └── lab/              # STT 결과 파일들
├── reference/             # 기준 데이터
│   ├── audio/            # 기준 음성 파일들
│   ├── lab/              # 기준 텍스트 파일들
│   └── tokens/           # 토큰 데이터
├── mfa_corpus/           # MFA 입력 데이터
├── mfa_output/           # MFA 출력 (TextGrid)
├── pitch_data/           # 피치 분석 데이터
│   ├── reference/        # 기준 피치 데이터
│   └── user/            # 사용자 피치 데이터
├── comparison_result.json # 비교 분석 결과
└── final_result.json     # 최종 결과
```

## 주의사항

- 이 디렉토리의 파일들은 Git에 포함되지 않습니다
- 실행할 때마다 새로 생성되는 임시 데이터입니다
- 사용자 음성 파일은 `user/input/`에 넣어주세요
"""
    
    readme_path = base_path / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"📖 {readme_path}")
    print("\n🎉 shared_data 디렉토리 구조 생성 완료!")
    print("💡 이제 user/input/ 디렉토리에 음성 파일을 넣고 분석을 시작하세요.")

if __name__ == "__main__":
    create_shared_data_structure()
