import requests

# 테스트용 파라미터
data = {
    "s3_audio_url": "s3://jungle-youtube-downloader-20250627/user_taken.wav",
    "video_id": "jZOywn1qArI",
    "webhook_url": "https://httpbin.org/post"
}

headers = {
    "ngrok-skip-browser-warning": "true",
    "Content-Type": "application/x-www-form-urlencoded"
}

# 실제 API 호출
try:
    response = requests.post(
        "https://escargot-immune-jaguar.ngrok-free.app/analyze-voice",
        data=data,
        headers=headers,
        timeout=30
    )

    print("✅ 상태 코드:", response.status_code)
    print("📨 응답 내용:", response.json())

except requests.exceptions.RequestException as e:
    print("❌ 요청 실패:", e)
