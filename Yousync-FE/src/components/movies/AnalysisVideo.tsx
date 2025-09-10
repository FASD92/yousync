//이해완료
"use client";

import YouTube from "react-youtube";
import { useRef, useEffect, useImperativeHandle, forwardRef } from "react";
import { useState } from "react";

interface AnalysisVideoProps {
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
}

export interface AnalysisVideoRef {
  seekTo: (time: number) => void;
  playVideo: () => void;
  pauseVideo: () => void;
  getCurrentTime: () => number;
}

const AnalysisVideo = forwardRef<AnalysisVideoRef, AnalysisVideoProps>(
  ({ videoId, onTimeUpdate, startTime = 0, endTime, disableAutoPause = false, onEndTimeReached, onPause, onPlay, overlayType = 'header', overlayHeight = 48 }, ref) => {
    const playerRef = useRef<{ seekTo: (time: number) => void; playVideo: () => void; pauseVideo: () => void; getCurrentTime: () => number } | null>(null);
    const intervalRef = useRef<NodeJS.Timeout | null>(null);
    const initialStartTimeRef = useRef(startTime);

    // 외부에서 호출할 수 있는 메서드들
    useImperativeHandle(ref, () => ({
      seekTo: (time: number) => {
        if (playerRef.current) {
          playerRef.current.seekTo(time);
          console.log('영상 시간 이동:', time);
        }
      },
      playVideo: () => {
        if (playerRef.current) {
          playerRef.current.playVideo();
          console.log('영상 재생 시작');
        }
      },
      pauseVideo: () => {
        if (playerRef.current) {
          playerRef.current.pauseVideo();
          console.log('영상 일시정지');
        }
      },
      getCurrentTime: () => {
        if (playerRef.current) {
          return playerRef.current.getCurrentTime();
        }
        return 0;
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
    // 오버레이가 사라지고, 플레이어가 준비된 경우 자동재생
    useEffect(() => {
      if (!showOverlay && playerReady && playerRef.current) {
        playerRef.current.playVideo();
      }
    }, [showOverlay, playerReady]);

    // 임시 로딩 상태 (실제 분석 API 연동 시 교체)
    const isLoading = true;

    const onReady = (event: { target: { seekTo: (time: number) => void; playVideo: () => void; pauseVideo: () => void; getCurrentTime: () => number } }) => {
      playerRef.current = event.target;
      setPlayerReady(true);
      // 시작 시간이 설정되어 있으면 해당 시간부터 재생
      if (startTime > 0) {
        event.target.seekTo(startTime);
        console.log('플레이어 시작 시간 설정:', startTime);
      }
      console.log('AnalysisVideo onReady - disableAutoPause:', disableAutoPause);
    };

    const onStateChange = (event: { data: number }) => {
      // 기존 인터벌 정리
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      // 재생 중일 때만 시간 업데이트
      if (event.data === 1) { // 1 = 재생 중
        console.log('영상 재생 시작 - 시간 업데이트 인터벌 시작');
        if (typeof onPlay === 'function') onPlay();
        intervalRef.current = setInterval(() => {
          if (playerRef.current && onTimeUpdate) {
            const currentTime = playerRef.current.getCurrentTime();
            onTimeUpdate(currentTime);
            // endTime 체크 후 자동 정지 (주석 복구)
            if (endTime !== undefined && currentTime >= endTime) {
              playerRef.current.pauseVideo();
              if (onEndTimeReached) onEndTimeReached();
              console.log('endTime 도달 - 영상 자동 정지', { currentTime, endTime });
            }
          }
        }, 100); // 100ms마다 시간 체크
      } else if (event.data === 2) { // 2 = 일시정지
        console.log('영상 일시정지 - 시간 업데이트 인터벌 정지');
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
        if (typeof onPause === 'function') onPause();
      } else if (event.data === 0) { // 0 = 영상 끝남
        // 영상이 끝나면 처음부터 다시 재생
        if (playerRef.current) {
          playerRef.current.seekTo(0);
          playerRef.current.playVideo();
        }
      }
    };

    return (
      <div className="bg-black rounded-xl overflow-hidden">
        <div className="relative w-full pt-[56.25%]">
          {/* 로딩 오버레이 */}
          {isLoading && (
            <div className="absolute inset-0 z-40 flex flex-col items-center justify-center pointer-events-none">
              {/* 흐림 효과 */}
              <div className="absolute inset-0 bg-black/30 backdrop-blur-[2px] z-[-1]" />
              {/* 중앙 원형 스피너 */}
              <div className="flex flex-col items-center justify-center">
                <svg className="animate-spin h-16 w-16 text-emerald-400 mb-6" viewBox="0 0 50 50">
                  <circle className="opacity-20" cx="25" cy="25" r="20" stroke="currentColor" strokeWidth="6" fill="none" />
                  <circle className="opacity-80" cx="25" cy="25" r="20" stroke="currentColor" strokeWidth="6" fill="none" strokeDasharray="31.4 94.2" strokeLinecap="round" />
                </svg>
                <span className="text-emerald-300 text-lg font-bold drop-shadow-lg animate-pulse">영상 대사 추출중</span>
              </div>
            </div>
          )}
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
                controls: 0,
                modestbranding: 1,
                rel: 0,
                showinfo: 0,
                iv_load_policy: 3,
                disablekb: 1,
                fs: 0,
                start: initialStartTimeRef.current, // 시작 시간 설정
                mute: 1, // 항상 음소거
              },
            }}
          />
          {/* 영상 클릭 방지 오버레이 */}
          <div
            className="absolute top-0 left-0 w-full h-full z-30"
            style={{ cursor: 'not-allowed', background: 'transparent' }}
            onClick={e => e.preventDefault()}
            onMouseDown={e => e.preventDefault()}
            onDoubleClick={e => e.preventDefault()}
            onContextMenu={e => e.preventDefault()}
          />
        </div>
      </div>
    );
  }
);

AnalysisVideo.displayName = 'AnalysisVideo';

export default AnalysisVideo; 