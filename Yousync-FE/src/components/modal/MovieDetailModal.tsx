"use client";

import React, { MouseEvent, useRef, useEffect, useState } from "react";
import dynamic from 'next/dynamic';
import { FaMicrophone, FaUser, FaTag, FaClock } from "react-icons/fa";
import { BookmarkIcon } from '@heroicons/react/24/solid';
import { useRouter } from "next/navigation";
import type { TokenDetailResponse } from "@/types/pitch";
import { tokenApi } from "@/services/api";
import { useBookmark } from '@/hooks/useBookmark';

const YouTube = dynamic(() => import('react-youtube'), { ssr: false });

interface VideoModalProps {
  youtubeId: string;
  isOpen: boolean;
  onClose: () => void;
  tokenData: TokenDetailResponse;
}

export default function MovieDetailModal({
  youtubeId,
  isOpen,
  onClose,
  tokenData,
}: VideoModalProps) {
  const router = useRouter();
  const playerRef = useRef<any>(null);
  const [isDubbingLoading, setIsDubbingLoading] = useState(false);
  
  // 북마크 관련 상태 및 훅
  const { isLoading: bookmarkLoading, addBookmark, removeBookmark, getBookmarks, isLoggedIn, isBookmarked } = useBookmark();
  const [bookmarked, setBookmarked] = useState(false);
  
  // 마우스 호버 관련 상태 추가
  const [isHovering, setIsHovering] = useState(false);
  const [shouldPlay, setShouldPlay] = useState(false);
  const hoverTimerRef = useRef<NodeJS.Timeout | null>(null);
  const videoContainerRef = useRef<HTMLDivElement>(null);

  // tokenData가 있을 때만 접근
  const startTime = Number(tokenData?.start_time) || 0;
  const endTime = Number(tokenData?.end_time) || undefined;

  // 모달이 열릴 때 해당 영상의 북마크 상태 확인
  useEffect(() => {
    if (isOpen && tokenData && isLoggedIn()) {
      // 북마크 목록이 없으면 먼저 로드
      getBookmarks().then(() => {
        setBookmarked(isBookmarked(Number(tokenData.id)));
      }).catch(() => {
        // 에러 시 캐시된 데이터로 확인
        setBookmarked(isBookmarked(Number(tokenData.id)));
      });
    } else {
      setBookmarked(false);
    }
  }, [isOpen, tokenData, isLoggedIn]);

  // 북마크 클릭 핸들러
  const handleBookmarkClick = async (e: React.MouseEvent) => {
    e.stopPropagation();
    
    if (!isLoggedIn()) {
      alert('북마크 기능은 로그인 후 이용 가능합니다.');
      return;
    }
    
    if (!tokenData) return;
    
    try {
      if (!bookmarked) {
        await addBookmark(Number(tokenData.id));
        setBookmarked(true);
      } else {
        const success = await removeBookmark(Number(tokenData.id));
        if (success) {
          setBookmarked(false);
        }
      }
    } catch (error) {
      console.error('북마크 처리 중 오류 발생:', error);
    }
  };

  // 반복 재생: 영상이 끝나면 startTime으로 이동 후 재생
  const handlePlayerReady = (event: any) => {
    try {
      // event와 event.target이 존재하는지 확인
      if (!event || !event.target) {
        console.warn('Invalid player event received');
        return;
      }
      
      // playerRef에 저장
      playerRef.current = event.target;
      
      // 안전하게 래핑하여 메서드 호출
      setTimeout(() => {
        try {
          // startTime이 있으면 해당 시간으로 이동
          if (startTime > 0 && playerRef.current && typeof playerRef.current.seekTo === 'function') {
            playerRef.current.seekTo(startTime);
          }
          
          // 자동 재생하지 않고 대기
          if (playerRef.current && typeof playerRef.current.pauseVideo === 'function') {
            playerRef.current.pauseVideo();
          }
        } catch (e) {
          console.warn('Failed in handlePlayerReady:', e);
        }
      }, 0);
    } catch (error) {
      console.error('Error in handlePlayerReady:', error);
    }
  };

  const handlePlayerEnd = () => {
    if (!shouldPlay) return;
    
    try {
      const player = playerRef.current;
      // player가 null이 아니고 필요한 메서드가 있는지 확인
      if (player && typeof player.seekTo === 'function' && typeof player.playVideo === 'function') {
        // 시간 이동 및 재생 시도 - 안전하게 래핑
        setTimeout(() => {
          try {
            if (playerRef.current) {
              playerRef.current.seekTo(startTime);
              playerRef.current.playVideo();
            }
          } catch (e) {
            console.warn('Failed to handle player end:', e);
          }
        }, 0);
      }
    } catch (error) {
      console.error('Error in handlePlayerEnd:', error);
    }
  };

  // 마우스가 영상 위에 1.5초 이상 머물면 재생
  useEffect(() => {
    let isMounted = true; // 컴포넌트 마운트 상태 추적

    if (isHovering) {
      hoverTimerRef.current = setTimeout(() => {
        if (!isMounted) return; // 컴포넌트가 언마운트되었으면 중단
        setShouldPlay(true);
        
        try {
          const player = playerRef.current;
          // player가 null이 아니고 필요한 메서드가 있는지 확인
          if (player && typeof player.seekTo === 'function' && typeof player.playVideo === 'function') {
            // 시간 이동 시도 - 안전하게 래핑
            setTimeout(() => {
              try {
                if (playerRef.current) {
                  playerRef.current.seekTo(startTime);
                }
              } catch (e) {
                console.warn('Failed to seek video:', e);
              }
            }, 0);
            
            // 재생 시도 - 안전하게 래핑
            setTimeout(() => {
              try {
                if (playerRef.current) {
                  playerRef.current.playVideo();
                }
              } catch (e) {
                console.warn('Failed to play video:', e);
              }
            }, 0);
          }
        } catch (error) {
          console.error('Error playing video:', error);
        }
      }, 1500); // 1.5초 후 재생
    } else {
      if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
        hoverTimerRef.current = null;
      }
      setShouldPlay(false);
      
      try {
        const player = playerRef.current;
        // player가 null이 아니고 필요한 메서드가 있는지 확인
        if (player && typeof player.pauseVideo === 'function') {
          // 일시 정지 시도 - 안전하게 래핑
          setTimeout(() => {
            try {
              if (playerRef.current) {
                playerRef.current.pauseVideo();
              }
            } catch (e) {
              console.warn('Failed to pause video:', e);
            }
          }, 0);
          
          // 시간 이동 시도 (seekTo 메서드가 있는지 확인)
          if (typeof player.seekTo === 'function') {
            setTimeout(() => {
              try {
                if (playerRef.current) {
                  playerRef.current.seekTo(startTime);
                }
              } catch (e) {
                console.warn('Failed to seek video:', e);
              }
            }, 0);
          }
        }
      } catch (error) {
        console.error('Error pausing video:', error);
      }
    }

    // 클린업 함수
    return () => {
      isMounted = false; // 컴포넌트 언마운트 표시
      if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
        hoverTimerRef.current = null;
      }
    };
  }, [isHovering, startTime]);

  // endTime이 지정된 경우, 100ms마다 체크해서 endTime 도달 시 반복
  useEffect(() => {
    if (!endTime || !shouldPlay) return;
    
    let isMounted = true; // 컴포넌트 마운트 상태 추적
    
    const interval = setInterval(() => {
      if (!isMounted) {
        clearInterval(interval);
        return;
      }
      
      try {
        const player = playerRef.current;
        // player가 null이거나 필요한 메서드가 없으면 중단
        if (!player || typeof player.getCurrentTime !== 'function') {
          return;
        }
        
        // 현재 시간 가져오기
        const current = player.getCurrentTime();
        
        // endTime 도달 시 처리
        if (typeof current === 'number' && current >= endTime) {
          // player가 필요한 메서드를 가지고 있는지 확인
          if (typeof player.seekTo === 'function' && typeof player.playVideo === 'function') {
            // 안전하게 래핑
            setTimeout(() => {
              try {
                if (playerRef.current) {
                  playerRef.current.seekTo(startTime);
                  playerRef.current.playVideo();
                }
              } catch (e) {
                console.warn('Failed to handle end time reached:', e);
              }
            }, 0);
          }
        }
      } catch (error) {
        console.error('Error in interval:', error);
      }
    }, 100);
    
    // 클린업 함수
    return () => {
      isMounted = false; // 컴포넌트 언마운트 표시
      clearInterval(interval);
    };
  }, [endTime, startTime, shouldPlay, isOpen]);

  if (!isOpen || !youtubeId || !tokenData) return null;

  const handleDubbingClick = async () => {
    setIsDubbingLoading(true);
    try {
      await tokenApi.incrementView(tokenData.id);
    } catch (error) {
      console.error("Failed to increment view count:", error);
    } finally {
      router.push(`/dubbing/${tokenData.id}?modalId=${youtubeId}`);
    }
  };

  // 마우스 이벤트 핸들러
  const handleMouseEnter = () => {
    setIsHovering(true);
  };

  const handleMouseLeave = () => {
    setIsHovering(false);
  };
  const isSmallScreen = typeof window !== "undefined" && window.innerWidth < 640;

  return (
    <div className="fixed inset-0 z-51 flex items-center justify-center bg-black/40 backdrop-blur-sm">
      <div className="absolute inset-0" onClick={onClose} />
      <div
        className="relative z-50 w-full max-w-[90vw] sm:max-w-2xl md:max-w-3xl lg:max-w-4xl max-h-[90vh] overflow-y-auto rounded-lg bg-[#181818] p-4 sm:p-6 shadow-2xl"
        onClick={(e: MouseEvent<HTMLDivElement>) => e.stopPropagation()}
      >
        {/* 닫기 버튼은 주석 처리됨. 바깥(배경) 클릭 시 모달이 닫힙니다. */}
        {/* <button
          className="absolute top-4 right-4 text-2xl text-white hover:text-gray-300"
          onClick={onClose}
        >
          &times;
        </button> */}
        {/* 메인 영상 */}
        <div 
          ref={videoContainerRef}
          className="mx-auto aspect-video w-full overflow-hidden rounded-xl relative shadow-lg"
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseLeave}
        >
          <YouTube
            key={youtubeId}
            videoId={youtubeId}
            className="h-full w-full"
            onReady={handlePlayerReady}
            onEnd={handlePlayerEnd}
            opts={{
              width: "100%",
              height: "100%",
              playerVars: {
                controls: 0, // 재생바 숨김
                autoplay: 0, // 자동 재생 비활성화
                start: startTime,
                showinfo: 0, // 제목 숨김
                modestbranding: 1, // 유튜브 로고 최소화
                rel: 0, // 관련 영상 숨김
                iv_load_policy: 3, // 인포카드 숨김
                fs: 0, // 전체화면 버튼 숨김
                disablekb: 1, // 키보드 제어 비활성화
              }
            }}
          />
          {/* 상단 오버레이 */}
          <div
            className="absolute top-0 left-0 w-full z-20 pointer-events-none"
            style={{ height: 64, background: '#000', opacity: 1 }}
          />
          
          {/* 재생 상태 표시 오버레이 */}
          {!shouldPlay && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
              <div className="text-white text-center">
                {/* 작은 화면일 때 */}
                {/* <p className="text-lg block sm:hidden">화면을 터치하면 재생됩니다</p> */}
                {/* 큰 화면일 때 */}
                {/* <p className="text-lg hidden sm:block">마우스를 1.5초간 올려두면 재생됩니다</p> */}
              {isSmallScreen ? "터치하면 재생됩니다" : "마우스를 1.5초간 올려두면 재생됩니다"}
              </div>
            </div>
          )}
        </div>
        <div className="flex flex-col sm:flex-row items-center justify-between gap-6 mt-6 bg-[#20232a] rounded-2xl shadow-lg p-6 border border-[#23272f]">
          <div className="flex-1 flex flex-col gap-2">
            <div
              className="flex items-center gap-2 text-2xl text-emerald-400 font-bold cursor-pointer"
              onClick={() => router.push(`/actor/${encodeURIComponent(tokenData.actor_name.replace(/\s/g, ''))}`)}
            >
              <FaUser />
              {tokenData.actor_name}
            </div>
            <div className="flex items-center gap-2 text-base text-gray-300">
              <FaTag />
              {tokenData.category}
            </div>
            <div className="flex items-center gap-2 text-base text-gray-400">
              <FaClock />
              재생 시간: {(Number(tokenData.end_time) - Number(tokenData.start_time)).toFixed(2)}초
            </div>
          </div>
          <div className="flex items-center gap-4">
            {/* 북마크 버튼 */}
            <button
              onClick={handleBookmarkClick}
              disabled={bookmarkLoading}
              className={`flex items-center gap-2 px-4 py-3 rounded-full text-white font-bold shadow-lg transition-all duration-200 focus:outline-none ${
                bookmarkLoading 
                  ? 'bg-gray-500 cursor-not-allowed' 
                  : bookmarked 
                    ? 'bg-yellow-500 hover:bg-yellow-600' 
                    : 'bg-gray-600 hover:bg-gray-700'
              }`}
              title={isLoggedIn() ? (bookmarked ? "북마크 삭제" : "북마크 추가") : "로그인 필요"}
            >
              <BookmarkIcon className="w-5 h-5" />
              {bookmarked ? '북마크됨' : '북마크'}
            </button>
            
            {/* 더빙하기 버튼 */}
            <button
              onClick={handleDubbingClick}
              disabled={isDubbingLoading}
              className={`flex items-center gap-3 px-8 py-3 rounded-full text-white text-lg font-bold shadow-lg transition-all duration-200 focus:outline-none ${isDubbingLoading ? 'bg-emerald-500 animate-pulse cursor-not-allowed' : 'bg-emerald-500 hover:bg-emerald-600'}`}
            >
              <FaMicrophone className="text-2xl" />
              더빙하기
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}