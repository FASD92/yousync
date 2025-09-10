#!/bin/bash

# YouSync FastAPI 오디오 분석 서비스용 netdata 경량화 스크립트
echo "🎵 YouSync 서비스용 netdata 최적화를 시작합니다..."

# 현재 메모리 사용량 확인
echo "📊 최적화 전 netdata 메모리 사용량:"
ps aux | grep netdata | grep -v grep | awk '{print "PID: " $2 ", RSS: " $6/1024 " MB"}'

# 백업 생성
sudo cp /etc/netdata/netdata.conf /etc/netdata/netdata.conf.backup.$(date +%Y%m%d_%H%M%S)
echo "✅ 기존 설정 백업 완료"

# YouSync 오디오 분석 서비스에 최적화된 설정
sudo tee /etc/netdata/netdata.conf > /dev/null << 'EOF'
[global]
    # 메모리 사용량 대폭 감소 (오디오 분석용 RAM 확보)
    history = 180                    # 3분간만 데이터 보관 (기본: 3600초)
    update every = 5                 # 5초마다 업데이트 (실시간성 < 자원절약)
    
    # 디스크 사용 최소화
    dbengine disk space = 0          # 디스크 DB 완전 비활성화
    dbengine multihost disk space = 0
    
    # 보안 및 접근 제한
    bind to = 127.0.0.1              # localhost만 접근 허용
    default port = 19999
    
    # 로그 최소화
    debug log = none
    error log = syslog
    access log = none

[plugins]
    # 오디오 분석 서비스에 불필요한 플러그인 모두 비활성화
    python.d = no                    # Python 플러그인 비활성화
    go.d = no                        # Go 플러그인 비활성화
    node.d = no                      # Node.js 플러그인 비활성화
    charts.d = no                    # 차트 플러그인 비활성화
    apps = no                        # 애플리케이션 모니터링 비활성화
    cgroups = no                     # 컨테이너 모니터링 비활성화
    
    # 핵심 시스템 정보만 유지 (FastAPI + 오디오 분석 모니터링용)
    proc = yes                       # CPU, 메모리, 디스크 기본 정보
    diskspace = yes                  # 디스크 사용량 (S3 업로드 모니터링)
    
[plugin:proc]
    # 오디오 분석 서비스 모니터링에 필요한 최소 정보만
    /proc/stat = yes                 # CPU 사용률
    /proc/meminfo = yes              # 메모리 사용률 (오디오 분석 중요)
    /proc/diskstats = yes            # 디스크 I/O (S3 업로드 모니터링)
    /proc/net/dev = yes              # 네트워크 (API 요청 모니터링)
    
    # 불필요한 세부 정보 모두 비활성화
    /proc/interrupts = no
    /proc/softirqs = no
    /proc/vmstat = no
    /proc/loadavg = yes              # 로드 평균은 유지
    /proc/pressure = no              # PSI 정보 비활성화
    
[plugin:diskspace]
    # S3 업로드용 임시 파일 디스크 모니터링만
    exclude space metrics on paths = /dev /proc /sys /var/run /run /var/cache
    
# 웹 인터페이스 최소화
[web]
    web files owner = netdata
    web files group = netdata
    disconnect idle clients after seconds = 60
    enable gzip compression = yes
EOF

echo "✅ YouSync 서비스 최적화 설정 적용 완료"

# netdata 재시작
echo "🔄 netdata 재시작 중..."
sudo systemctl restart netdata

# 잠시 대기 후 결과 확인
sleep 3

echo "📊 최적화 후 netdata 메모리 사용량:"
ps aux | grep netdata | grep -v grep | awk '{print "PID: " $2 ", RSS: " $6/1024 " MB"}'

echo ""
echo "🎉 YouSync 오디오 분석 서비스용 netdata 최적화 완료!"
echo ""
echo "📈 최적화 내용:"
echo "  - 메모리 히스토리: 3600초 → 180초 (RAM 사용량 대폭 감소)"
echo "  - 업데이트 주기: 1초 → 5초 (CPU 사용량 감소)"
echo "  - 불필요한 플러그인 모두 비활성화"
echo "  - 오디오 분석에 필요한 시스템 모니터링만 유지"
echo ""
echo "💡 모니터링 접근: http://localhost:19999 (서버 내부에서만)"
echo "⚠️  복원 필요시: sudo cp /etc/netdata/netdata.conf.backup.* /etc/netdata/netdata.conf"
