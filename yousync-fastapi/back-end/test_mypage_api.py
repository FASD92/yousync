#!/usr/bin/env python3
"""
마이페이지 API 테스트 스크립트
"""

import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def test_api_endpoint(endpoint: str, method: str = "GET", headers: Dict[str, str] = None, data: Dict[str, Any] = None):
    """API 엔드포인트 테스트"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        
        print(f"\n{'='*50}")
        print(f"Testing: {method} {endpoint}")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
            except:
                print(f"Response: {response.text}")
        else:
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Server might not be running on {BASE_URL}")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """메인 테스트 함수"""
    print("🚀 마이페이지 API 테스트 시작")
    
    # 인증이 필요한 API들이므로 실제 토큰 없이는 401 에러가 예상됨
    # 하지만 엔드포인트가 존재하는지는 확인 가능
    
    # 1. 서버 상태 확인
    test_api_endpoint("/")
    
    # 2. 마이페이지 API 엔드포인트들 테스트 (인증 없이)
    mypage_endpoints = [
        "/mypage/bookmarks/",
        "/mypage/my-dubbed-tokens",
        "/mypage/overview",
        "/mypage/tokens/1/analysis-status",
    ]
    
    for endpoint in mypage_endpoints:
        test_api_endpoint(endpoint)
    
    print(f"\n{'='*50}")
    print("✅ 테스트 완료!")
    print("📝 참고: 인증이 필요한 API들은 401 Unauthorized 에러가 정상입니다.")
    print("🔑 실제 사용 시에는 Google OAuth 토큰이 필요합니다.")

if __name__ == "__main__":
    main()
