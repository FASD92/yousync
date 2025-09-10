"use client";
import React, { useRef, useEffect } from "react";
import YouTube from "react-youtube";

interface PreviewVideoProps {
  videoId: string;
  startTime?: number;
  endTime?: number;
  className?: string;
}

const PreviewVideo: React.FC<PreviewVideoProps> = ({ videoId, startTime = 0, endTime, className }) => {
  const playerRef = useRef<any>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // 영상 준비되면 해당 시점으로 이동 후 일시정지
  const onReady = (event: any) => {
    playerRef.current = event.target;
    if (startTime > 0) {
      event.target.seekTo(startTime);
    }
    event.target.pauseVideo();
  };

  // 구간 반복: endTime에 도달하면 startTime으로 이동
  const onStateChange = (event: any) => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (event.data === 1 && playerRef.current) {
      // 재생 중일 때 구간 체크
      intervalRef.current = setInterval(() => {
        if (playerRef.current && endTime !== undefined) {
          const current = playerRef.current.getCurrentTime();
          if (current >= endTime) {
            playerRef.current.seekTo(startTime);
            playerRef.current.pauseVideo();
          }
        }
      }, 100);
      // 항상 일시정지
      playerRef.current.pauseVideo();
    }
    // 일시정지 시 인터벌 해제
    if (event.data === 2 && intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  // 언마운트 시 인터벌 해제
  useEffect(() => {
    console.log('[PreviewVideo] videoId:', videoId, 'startTime:', startTime, 'endTime:', endTime);
  }, [videoId, startTime, endTime]);

  return (
    <div className={className || "rounded-xl overflow-hidden shadow-lg bg-black w-[560px] h-[315px]"}>
      <YouTube
        videoId={videoId}
        className="w-[560px] h-[315px]"
        onReady={onReady}
        onStateChange={onStateChange}
        opts={{
          width: 560,
          height: 315,
          playerVars: {
            autoplay: 0,
            controls: 0,
            modestbranding: 1,
            rel: 0,
            showinfo: 0,
            iv_load_policy: 3,
            disablekb: 1,
            fs: 0,
            start: startTime,
            mute: 1,
          },
        }}
      />
    </div>
  );
};

export default PreviewVideo; 