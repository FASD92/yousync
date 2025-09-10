# Yousync-FE 포스트세션 가이드

## 📋 프로젝트 개요

**Yousync**는 YouTube 영상을 활용한 더빙 서비스입니다. 사용자가 YouTube 영상의 대사를 따라 더빙하고, 음성 분석을 통해 피드백을 받을 수 있는 웹 애플리케이션입니다.

### 🎯 핵심 기능
- YouTube 영상 기반 더빙 연습
- 일반 더빙 (Solo Dubbing)
- 듀엣 더�bing (Duet Dubbing) - 두 명이 함께 더빙
- 음성 분석 및 피드백
- 사용자 인증 (Google OAuth)
- 마이페이지 및 북마크 기능

## 🛠 기술 스택

### Frontend
- **Framework**: Next.js 15.3.4 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS 4.1.11
- **State Management**: Zustand 5.0.6
- **Data Fetching**: TanStack React Query 5.81.5
- **Authentication**: Google OAuth (@react-oauth/google)
- **Charts**: Chart.js, ApexCharts, React-ApexCharts
- **Animation**: Framer Motion, GSAP
- **Audio Processing**: AudioBuffer-to-WAV, PitchFinder

### Development Tools
- **Package Manager**: npm
- **Linting**: ESLint
- **Build Tool**: Turbopack (개발 모드)

## 📁 프로젝트 구조

```
src/
├── app/                    # Next.js App Router 페이지
│   ├── home/              # 홈 페이지
│   ├── login/             # 로그인 페이지
│   ├── signup/            # 회원가입 페이지
│   ├── movie/             # 영화 선택 페이지
│   ├── dubbing/           # 일반 더빙 페이지
│   ├── duetdubbing/       # 듀엣 더빙 페이지
│   ├── result/            # 결과 페이지
│   ├── mypage/            # 마이페이지
│   ├── api/               # API 라우트
│   └── providers/         # Context Providers
├── components/            # 재사용 가능한 컴포넌트
│   ├── ui/               # 공통 UI 컴포넌트
│   ├── dubbing/          # 더빙 관련 컴포넌트
│   ├── movie/            # 영화 관련 컴포넌트
│   ├── result/           # 결과 관련 컴포넌트
│   ├── mypage/           # 마이페이지 컴포넌트
│   └── modal/            # 모달 컴포넌트
├── hooks/                # Custom Hooks
├── services/             # API 서비스
├── store/                # Zustand 스토어
├── types/                # TypeScript 타입 정의
├── utils/                # 유틸리티 함수
└── styles/               # 스타일 파일
```

## 🔑 핵심 컴포넌트 분석

### 1. 더빙 시스템 (DubbingContainer)
**위치**: `src/components/dubbing/DubbingContainer.tsx`

**주요 기능**:
- 일반 더빙과 듀엣 더빙을 하나의 컴포넌트로 통합
- `isDuet` 플래그로 모드 전환
- 화자 구분 로직 (`isMyLine` 함수)
- 자동 재생 및 흐름 제어

**Props**:
```typescript
interface DubbingContainerProps {
  isDuet?: boolean;
  selectedActor?: string;
}
```

### 2. 스크립트 표시 (ScriptDisplay)
**위치**: `src/components/dubbing/ScriptDisplay.tsx`

**주요 기능**:
- 대사 스크립트 표시
- 화자별 마이크 버튼 활성화/비활성화
- 듀엣 모드에서 화자 구분 스타일링

### 3. 사이드바 (Sidebar)
**위치**: `src/components/dubbing/Sidebar.tsx`

**주요 기능**:
- 스크립트 목록 표시
- 화자별 스타일 구분
- 진행 상황 표시

## 🎣 Custom Hooks

### 1. useDubbingState
**위치**: `src/hooks/useDubbingState.ts`
- 더빙 상태 관리 (재생, 일시정지, 녹음 등)
- 스크립트 데이터 관리
- 자동 재생 로직

### 2. useDubbingRecorder
**위치**: `src/hooks/useDubbingRecorder.ts`
- 음성 녹음 기능
- 오디오 스트림 관리
- 녹음 파일 처리

### 3. useVoiceRecorder
**위치**: `src/hooks/useVoiceRecorder.ts`
- 음성 녹음 및 분석
- 피치 분석
- 오디오 데이터 처리

### 4. useVideos
**위치**: `src/hooks/useVideos.ts`
- 비디오 데이터 관리
- 북마크 기능
- 비디오 검색

## 🗄 상태 관리 (Zustand Stores)

### 1. useAudioStore
**위치**: `src/store/useAudioStore.ts`
- 오디오 재생 상태 관리
- 볼륨 제어
- 재생 위치 관리

### 2. useTokenStore
**위치**: `src/store/useTokenStore.ts`
- 인증 토큰 관리
- 로그인 상태 관리

### 3. useResultStore
**위치**: `src/store/useResultStore.ts`
- 더빙 결과 데이터 관리
- 분석 결과 저장

## 🌐 API 서비스

### 1. API 클라이언트
**위치**: `src/services/api.ts`
- Axios 기반 HTTP 클라이언트
- 인터셉터를 통한 토큰 관리
- 에러 처리

**주요 엔드포인트**:
- `/auth/` - 인증 관련
- `/videos/` - 비디오 관련
- `/dubbing/` - 더빙 관련
- `/users/` - 사용자 관련

### 2. 인증 서비스
**위치**: `src/services/auth.ts`
- Google OAuth 처리
- 토큰 관리
- 사용자 정보 관리

## 🎨 스타일링

### Tailwind CSS 설정
**위치**: `tailwind.config.ts`
- 커스텀 색상 팔레트
- 반응형 디자인
- 다크 테마 지원

### 글로벌 스타일
**위치**: `src/styles/globals.css`
- 기본 스타일 설정
- 폰트 설정 (Geist)
- 애니메이션 정의

## 🔧 환경 설정

### 환경 변수 (.env)
```env
NEXT_PUBLIC_API_BASE_URL=https://yousync.link/api
NEXT_PUBLIC_GOOGLE_CLIENT_ID=539927629178-t1f6h2tutqgp8d9pvvpmn3it8hq4mbmd.apps.googleusercontent.com
NEXT_PUBLIC_YT_API_KEY=AIzaSyA2S3q7qc5mxIM9l0CIyWBagFwASXyy2QE
```

### Next.js 설정 (next.config.ts)
- 이미지 도메인 허용 (YouTube, Naver 등)
- Webpack 설정 (deprecation warning 억제)
- 성능 최적화 설정

## 📱 주요 페이지 플로우

### 1. 랜딩 페이지 → 홈
- `/` → `/home`
- 메인 시작 버튼을 통한 진입

### 2. 영화 선택 → 더빙
- `/movie` → `/dubbing` (일반 더빙)
- `/movie` → `/duetdubbing` (듀엣 더빙)

### 3. 더빙 → 결과
- `/dubbing` → `/result`
- 더빙 완료 후 분석 결과 표시

### 4. 마이페이지
- `/mypage` - 사용자 정보 및 더빙 기록

## 🚀 빌드 및 배포

### 개발 서버 실행
```bash
npm run dev  # Turbopack 사용
```

### 프로덕션 빌드
```bash
npm run build
npm run start
```

### 주요 스크립트
- `dev`: 개발 서버 (Turbopack + deprecation warning 억제)
- `build`: 프로덕션 빌드
- `start`: 프로덕션 서버
- `lint`: ESLint 실행

## 🔍 포스트세션 예상 질문

### 기술적 질문
1. **"Next.js App Router를 선택한 이유는?"**
   - 최신 Next.js 기능 활용
   - 서버 컴포넌트와 클라이언트 컴포넌트 분리
   - 향상된 라우팅 시스템

2. **"Zustand를 상태 관리로 선택한 이유는?"**
   - Redux보다 간단한 설정
   - TypeScript 친화적
   - 작은 번들 사이즈

3. **"TanStack React Query를 사용한 이유는?"**
   - 서버 상태 관리 최적화
   - 캐싱 및 동기화
   - 에러 처리 및 로딩 상태 관리

### 아키텍처 질문
1. **"일반 더빙과 듀엣 더빙을 어떻게 통합했나?"**
   - `isDuet` 플래그를 통한 조건부 렌더링
   - `isMyLine` 함수로 화자 구분
   - 공통 컴포넌트 재사용

2. **"음성 처리는 어떻게 구현했나?"**
   - Web Audio API 활용
   - AudioBuffer-to-WAV로 파일 변환
   - PitchFinder로 피치 분석

3. **"실시간 음성 분석은 어떻게 처리했나?"**
   - MediaRecorder API 사용
   - 실시간 스트림 처리
   - 백엔드 API와 연동

### 성능 최적화 질문
1. **"성능 최적화를 위해 어떤 방법을 사용했나?"**
   - React.memo를 통한 불필요한 리렌더링 방지
   - 이미지 최적화 (Next.js Image 컴포넌트)
   - 코드 스플리팅 (동적 import)

2. **"번들 사이즈 최적화는?"**
   - Tree shaking
   - 불필요한 라이브러리 제거
   - 동적 import 활용

### 사용자 경험 질문
1. **"접근성은 어떻게 고려했나?"**
   - 키보드 네비게이션 지원
   - ARIA 라벨 적용
   - 색상 대비 고려

2. **"반응형 디자인은?"**
   - Tailwind CSS의 반응형 클래스 활용
   - 모바일 우선 디자인
   - 터치 인터페이스 최적화

### 보안 질문
1. **"인증은 어떻게 처리했나?"**
   - Google OAuth 2.0 사용
   - JWT 토큰 관리
   - 토큰 자동 갱신

2. **"API 보안은?"**
   - HTTPS 통신
   - 토큰 기반 인증
   - 환경 변수를 통한 민감 정보 관리

## 🐛 알려진 이슈 및 개선사항

### 현재 이슈
1. **Deprecation Warning**
   - Next.js의 url.parse() 관련 경고
   - Webpack 설정으로 억제 중

2. **성능 최적화 필요**
   - 대용량 오디오 파일 처리
   - 메모리 사용량 최적화

### 향후 개선사항
1. **테스트 코드 추가**
   - 단위 테스트 (Jest)
   - 통합 테스트 (Cypress)

2. **PWA 지원**
   - 서비스 워커 추가
   - 오프라인 기능

3. **성능 모니터링**
   - 웹 바이탈 측정
   - 에러 트래킹

## 📚 참고 자료

### 공식 문서
- [Next.js Documentation](https://nextjs.org/docs)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [Zustand](https://github.com/pmndrs/zustand)
- [TanStack React Query](https://tanstack.com/query/latest)

### 프로젝트 관련
- `INTEGRATION_SUMMARY.md` - 더빙 컴포넌트 통합 작업 요약
- `README.md` - 기본 프로젝트 정보

---

**💡 팁**: 포스트세션에서는 기술적 선택의 이유, 트레이드오프, 그리고 실제 구현 과정에서의 어려움과 해결 방법에 대해 구체적으로 설명할 준비를 하세요.
