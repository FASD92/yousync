//이해완료
"use client";

import YouTube from "react-youtube";
import { useRef, useEffect, useImperativeHandle, forwardRef } from "react";
import { useState } from "react";


interface VideoPlayerProps {
  videoId: string;
  onTimeUpdate?: (currentTime: number) => void;
  startTime?: number;
  endTime?: number;
  disableAutoPause?: boolean;
  onEndTimeReached?: () => void;
  onPause?: () => void;
  onPlay?: () => void;
  overlayType?: 'full' | 'header'; // 오버레이 타입
  overlayHeight?: number; // header 오버레이 높이(px)
  // 스크립트 목록 버튼 관련 props
  onOpenSidebar?: () => void;
}

export interface VideoPlayerRef {
  seekTo: (time: number) => void;
  playVideo: () => void;
  pauseVideo: () => void;
  getCurrentTime: () => number;
  getIsPlaying: () => boolean;
}

// YouTube 플레이어 내부 타입 정의
interface YouTubePlayer {
  seekTo: (time: number) => void;
  playVideo: () => void;
  pauseVideo: () => void;
  getCurrentTime: () => number;
  getPlayerState: () => number;
}

const VideoPlayer = forwardRef<VideoPlayerRef, VideoPlayerProps>(
  ({ videoId, onTimeUpdate, startTime = 0, endTime, disableAutoPause = false, onEndTimeReached, onPause, onPlay, overlayType = 'header', overlayHeight = 48, onOpenSidebar }, ref) => {
    const playerRef = useRef<YouTubePlayer | null>(null);
    const intervalRef = useRef<NodeJS.Timeout | null>(null);
    const initialStartTimeRef = useRef(startTime);
    const isPlayingRef = useRef<boolean>(false);
    const [isMobile, setIsMobile] = useState(false);

    // 모바일 디바이스 감지
    useEffect(() => {
      const checkMobile = () => {
        const userAgent = navigator.userAgent || navigator.vendor;
        const isMobileDevice = /android|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(userAgent.toLowerCase());
        const isSmallScreen = window.innerWidth <= 768;
        const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
        setIsMobile(isMobileDevice || (isSmallScreen && isTouchDevice));
      };
      
      checkMobile();
      window.addEventListener('resize', checkMobile);
      return () => window.removeEventListener('resize', checkMobile);
    }, []);

    // 외부에서 호출할 수 있는 메서드들
    useImperativeHandle(ref, () => ({
      seekTo: (time: number) => {
        console.log('[VideoPlayer] seekTo 호출:', time, 'playerRef.current:', !!playerRef.current);
        if (playerRef.current) {
          try {
            // YouTube 플레이어가 준비되었는지 확인
            if (typeof playerRef.current.seekTo === 'function') {
              playerRef.current.seekTo(time);
              console.log('[VideoPlayer] seekTo 성공');
            } else {
              console.warn('[VideoPlayer] seekTo 메서드가 준비되지 않았습니다.');
            }
          } catch (error) {
            console.error('[VideoPlayer] seekTo 에러:', error);
          }
        } else {
          console.error('[VideoPlayer] seekTo 실패: playerRef.current가 null');
        }
      },
      playVideo: () => {
        console.log('[VideoPlayer] playVideo 호출, playerRef.current:', !!playerRef.current);
        if (playerRef.current) {
          try {
            // YouTube 플레이어가 준비되었는지 더 안전하게 확인
            if (typeof playerRef.current.playVideo === 'function') {
              // getPlayerState 메서드가 있는지 확인 후 사용
              const player = playerRef.current as any;
              if (typeof player.getPlayerState === 'function') {
                const playerState = player.getPlayerState();
                console.log('[VideoPlayer] playVideo - 현재 플레이어 상태:', playerState);
                
                // 플레이어가 정지 상태이거나 일시정지 상태일 때만 재생
                if (playerState === 2 || playerState === 5 || playerState === -1) { // 2: 일시정지, 5: 정지, -1: 시작안됨
                  playerRef.current.playVideo();
                  isPlayingRef.current = true;
                  console.log('[VideoPlayer] playVideo 성공');
                  // 강제로 onPlay 콜백 호출
                  if (onPlay) {
                    setTimeout(() => {
                      console.log('[VideoPlayer] onPlay 콜백 호출');
                      onPlay();
                    }, 100);
                  }
                } else if (playerState === 1) {
                  console.log('[VideoPlayer] 이미 재생 중입니다');
                  isPlayingRef.current = true;
                } else {
                  // 상태에 관계없이 재생 시도
                  playerRef.current.playVideo();
                  isPlayingRef.current = true;
                  console.log('[VideoPlayer] 강제 playVideo 실행');
                }
              } else {
                // getPlayerState가 없으면 그냥 재생 시도
                playerRef.current.playVideo();
                isPlayingRef.current = true;
                console.log('[VideoPlayer] getPlayerState 없음, 직접 playVideo 실행');
                if (onPlay) {
                  setTimeout(() => {
                    console.log('[VideoPlayer] onPlay 콜백 호출');
                    onPlay();
                  }, 100);
                }
              }
            } else {
              console.warn('[VideoPlayer] YouTube 플레이어 메서드가 준비되지 않았습니다.');
            }
          } catch (error) {
            console.error('[VideoPlayer] playVideo 에러:', error);
          }
        } else {
          console.error('[VideoPlayer] playVideo 실패: playerRef.current가 null');
        }
      },
      pauseVideo: () => {
        console.log('[VideoPlayer] pauseVideo 호출, playerRef.current:', !!playerRef.current);
        if (playerRef.current) {
          try {
            // YouTube 플레이어가 준비되었는지 더 안전하게 확인
            if (typeof playerRef.current.pauseVideo === 'function') {
              // getPlayerState 메서드가 있는지 확인 후 사용
              const player = playerRef.current as any;
              if (typeof player.getPlayerState === 'function') {
                const playerState = player.getPlayerState();
                console.log('[VideoPlayer] pauseVideo - 현재 플레이어 상태:', playerState);
                
                // 플레이어가 재생 중이거나 버퍼링 중일 때만 정지
                if (playerState === 1 || playerState === 3) { // 1: 재생중, 3: 버퍼링
                  playerRef.current.pauseVideo();
                  isPlayingRef.current = false;
                  console.log('[VideoPlayer] pauseVideo 성공');
                  // 강제로 onPause 콜백 호출
                  if (onPause) {
                    setTimeout(() => {
                      console.log('[VideoPlayer] onPause 콜백 호출');
                      onPause();
                    }, 100);
                  }
                } else {
                  console.log('[VideoPlayer] 플레이어가 재생 상태가 아니므로 정지하지 않음');
                  isPlayingRef.current = false;
                }
              } else {
                // getPlayerState가 없으면 그냥 정지 시도
                playerRef.current.pauseVideo();
                isPlayingRef.current = false;
                console.log('[VideoPlayer] getPlayerState 없음, 직접 pauseVideo 실행');
                if (onPause) {
                  setTimeout(() => {
                    console.log('[VideoPlayer] onPause 콜백 호출');
                    onPause();
                  }, 100);
                }
              }
            } else {
              console.warn('[VideoPlayer] pauseVideo 메서드가 준비되지 않았습니다.');
            }
          } catch (error) {
            console.error('[VideoPlayer] pauseVideo 에러:', error);
          }
        } else {
          console.error('[VideoPlayer] pauseVideo 실패: playerRef.current가 null');
        }
      },
      getCurrentTime: () => {
        if (playerRef.current) {
          try {
            // YouTube 플레이어가 준비되었는지 확인
            if (typeof playerRef.current.getCurrentTime === 'function') {
              const time = playerRef.current.getCurrentTime();
              console.log('[VideoPlayer] getCurrentTime:', time);
              return time;
            } else {
              console.warn('[VideoPlayer] getCurrentTime 메서드가 준비되지 않았습니다.');
              return 0;
            }
          } catch (error) {
            console.error('[VideoPlayer] getCurrentTime 에러:', error);
            return 0;
          }
        }
        console.error('[VideoPlayer] getCurrentTime 실패: playerRef.current가 null');
        return 0;
      },
      getIsPlaying: () => {
        console.log('[VideoPlayer] getIsPlaying:', isPlayingRef.current);
        return isPlayingRef.current;
      }
    }));

    // 컴포넌트 언마운트 시 인터벌 정리
    useEffect(() => {
      return () => {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
      };
    }, []);

    // 오버레이 상태
    const [showOverlay, setShowOverlay] = useState(true);
    const [playerReady, setPlayerReady] = useState(false);
    // 4초 후 오버레이 해제
    useEffect(() => {
      const timer = setTimeout(() => setShowOverlay(false), 4000);
      return () => clearTimeout(timer);
    }, []);
    // 자동 재생 비활성화 - 사용자가 직접 재생 버튼을 눌러야 재생됨
    useEffect(() => {
      // 자동 재생 비활성화 - 아무 동작 하지 않음
    }, [showOverlay, playerReady]);

    const onReady = (event: { target: any }) => {
      console.log('[VideoPlayer] onReady 호출됨');
      playerRef.current = event.target as YouTubePlayer;
      setPlayerReady(true);
      
      // 플레이어 메서드들이 사용 가능한지 확인
      console.log('[VideoPlayer] 플레이어 메서드 확인:', {
        seekTo: typeof event.target.seekTo,
        playVideo: typeof event.target.playVideo,
        pauseVideo: typeof event.target.pauseVideo,
        getCurrentTime: typeof event.target.getCurrentTime
      });
      
      // 시작 시간이 설정되어 있으면 해당 시간으로 이동
      if (startTime > 0) {
        console.log('[VideoPlayer] 시작 시간으로 이동:', startTime);
        try {
          event.target.seekTo(startTime);
        } catch (error) {
          console.error('[VideoPlayer] 시작 시간 이동 에러:', error);
        }
      }
      
      // 영상을 잠시 재생했다가 정지하여 첫 프레임이 보이도록 함
      console.log('[VideoPlayer] 첫 프레임 로드를 위한 재생/정지');
      try {
        event.target.playVideo();
        
        // 약간의 지연 후에 정지 (첫 프레임이 로드될 시간을 줌)
        setTimeout(() => {
          if (playerRef.current) {
            console.log('[VideoPlayer] 첫 프레임 로드 완료, 정지');
            try {
              // YouTube 플레이어가 준비되었는지 더 안전하게 확인
              if (typeof playerRef.current.pauseVideo === 'function') {
                // getPlayerState 메서드가 있는지 확인 후 사용
                const player = playerRef.current as any;
                if (typeof player.getPlayerState === 'function') {
                  const playerState = player.getPlayerState();
                  console.log('[VideoPlayer] 현재 플레이어 상태:', playerState);
                  
                  // 플레이어가 재생 중이거나 버퍼링 중일 때만 정지
                  if (playerState === 1 || playerState === 3) { // 1: 재생중, 3: 버퍼링
                    playerRef.current.pauseVideo();
                    console.log('[VideoPlayer] 첫 프레임 정지 완료');
                  } else {
                    console.log('[VideoPlayer] 플레이어가 재생 상태가 아니므로 정지하지 않음');
                  }
                } else {
                  // getPlayerState가 없으면 그냥 정지 시도
                  playerRef.current.pauseVideo();
                  console.log('[VideoPlayer] getPlayerState 없음, 직접 pauseVideo 실행');
                }
              } else {
                console.warn('[VideoPlayer] pauseVideo 메서드가 준비되지 않았습니다.');
              }
            } catch (error) {
              console.error('[VideoPlayer] 첫 프레임 로드 후 정지 에러:', error);
            }
          } else {
            console.warn('[VideoPlayer] playerRef.current가 null입니다.');
          }
        }, 800); // 지연 시간을 800ms로 더 증가
      } catch (error) {
        console.error('[VideoPlayer] 첫 프레임 로드 재생 에러:', error);
      }
    };

    const onStateChange = (event: { data: number }) => {
      console.log('[VideoPlayer] 상태 변경:', event.data);
      
      // 기존 인터벌 정리
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }

      // 재생 중일 때만 시간 업데이트
      if (event.data === 1) { // 1 = 재생 중
        console.log('[VideoPlayer] 재생 시작');
        isPlayingRef.current = true;
        if (typeof onPlay === 'function') {
          onPlay();
        }
        
        intervalRef.current = setInterval(() => {
          if (playerRef.current && onTimeUpdate) {
            const currentTime = playerRef.current.getCurrentTime();
            onTimeUpdate(currentTime);
            // endTime 체크 후 자동 정지
            if (endTime !== undefined && currentTime >= endTime) {
              console.log('[VideoPlayer] endTime 도달, 정지');
              playerRef.current.pauseVideo();
              isPlayingRef.current = false;
              if (onEndTimeReached) onEndTimeReached();
            }
          }
        }, 50);
      } else if (event.data === 2) { // 2 = 일시정지
        console.log('[VideoPlayer] 일시정지');
        isPlayingRef.current = false;
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
        if (typeof onPause === 'function') {
          onPause();
        }
      } else if (event.data === 0) { // 0 = 종료
        console.log('[VideoPlayer] 재생 종료');
        isPlayingRef.current = false;
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      } else if (event.data === 3) { // 3 = 버퍼링
        console.log('[VideoPlayer] 버퍼링 중');
      } else if (event.data === 5) { // 5 = 동영상 신호
        console.log('[VideoPlayer] 동영상 신호');
      }
    };



    return (
      <div className="bg-black overflow-hidden w-full relative">
        <div className={`relative w-full ${isMobile ? 'pt-[56.25%]' : 'pt-[40%]'}`}>
          <YouTube
            videoId={videoId}
            className="absolute top-0 left-0 w-full h-full"
            onReady={onReady}
            onStateChange={onStateChange}
            opts={{
              width: "100%",
              height: "100%",
              playerVars: {
                autoplay: 0,
                controls: 0, // 모든 디바이스에서 컨트롤 숨김
                modestbranding: 1,
                rel: 0,
                showinfo: 0,
                iv_load_policy: 3,
                disablekb: 1, // 모든 디바이스에서 키보드 컨트롤 비활성화
                fs: 0, // 모든 디바이스에서 전체화면 비활성화
                start: initialStartTimeRef.current,
                playsinline: 1, // iOS에서 인라인 재생 허용
                enablejsapi: 1, // JavaScript API 활성화
                origin: typeof window !== 'undefined' ? window.location.origin : undefined,
              },
            }}
          />
          {/* 상단 오버레이 (텍스트/공유/타이틀 등 가림) */}
          <div
            className="absolute top-0 left-0 w-full z-20 pointer-events-none"
            style={{ height: 80, background: 'linear-gradient(to bottom, rgba(0,0,0,1) 0%, rgba(0,0,0,0.7) 70%, rgba(0,0,0,0) 100%)' }}
          />
        </div>
        
        {/* 전체 스크립트 보기 버튼 - 오른쪽 상단에 위치 */}
        {onOpenSidebar && (
          <button
            onClick={onOpenSidebar}
            className={`absolute top-2 right-2 flex items-center justify-center ${
              isMobile ? 'w-10 h-10' : 'w-8 h-8'
            } bg-gray-800/80 text-gray-300 rounded-md hover:bg-gray-700 hover:text-white transition-all duration-200 z-30`}
            title="전체 스크립트 보기"
          >
            <svg 
              width={isMobile ? "24" : "20"} 
              height={isMobile ? "24" : "20"} 
              viewBox="0 2 28 28" 
              fill="none"
            >
              <g stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="5,8 8,11 13,6" />
                <line x1="16" y1="8" x2="23" y2="8" />
                <polyline points="5,16 8,19 13,14" />
                <line x1="16" y1="16" x2="23" y2="16" />
                <polyline points="5,24 8,27 13,22" />
                <line x1="16" y1="24" x2="23" y2="24" />
              </g>
            </svg>
          </button>
        )}
      </div>
    );
  }
);

VideoPlayer.displayName = 'VideoPlayer';

export default VideoPlayer; 