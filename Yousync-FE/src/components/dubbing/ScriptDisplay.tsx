"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { ChevronLeftIcon, ChevronRightIcon } from "@heroicons/react/24/solid";
import { VideoPlayerRef } from "./VideoPlayer";
import PronunciationTimingGuide from "./PronunciationTimingGuide";
import "@/styles/analysis-animations.css";
import { useRouter } from "next/navigation";
import { useUser } from "@/hooks/useUser";

interface ScriptDisplayProps {
  captions: Array<{
    id: number;
    script: string;
    translation: string;
    start_time: number;
    end_time: number;
    actor?: {
      name: string;
      id: number;
    };
  }>;
  currentScriptIndex: number;
  onScriptChange: (index: number) => void;
  currentVideoTime?: number;
  playbackRange?: {
    startTime: number;
    endTime?: number;
  };
  videoPlayerRef?: React.RefObject<any>;
  currentWords?: Array<any>;
  recording?: boolean;
  recordingCompleted?: boolean;
  isAnalyzing?: boolean;
  onStopLooping?: () => void;
  showAnalysisResult?: boolean;
  analysisResult?: any;
  isVideoPlaying?: boolean;
  onPlay?: () => void;
  onPause?: () => void;
  onMicClick?: () => void;
  isLooping?: boolean;
  onLoopToggle?: () => void;
  isDuet?: boolean;
  isMyLine?: boolean;
  showCompletedButtons?: boolean;
  onOpenDubbingListenModal?: () => void;
  onShowResults?: () => void;
  id?: string | number;
}

const ScriptDisplay = ({
  captions,
  currentScriptIndex,
  onScriptChange,
  currentVideoTime,
  playbackRange,
  videoPlayerRef,
  currentWords = [],
  recording = false,
  recordingCompleted = false,
  isAnalyzing = false,
  showAnalysisResult = false,
  analysisResult,
  isVideoPlaying = false,
  onPlay,
  onPause,
  onMicClick,
  isLooping = false,
  onLoopToggle,
  isDuet = false,
  isMyLine = true,
  showCompletedButtons = false,
  onOpenDubbingListenModal,
  onShowResults,
  id,
}: ScriptDisplayProps) => {
  const {isLoggedIn, user} = useUser();
  const router = useRouter();
  // 시간 포맷 함수
  function formatTime(sec?: number) {
    if (typeof sec !== 'number' || isNaN(sec)) return '--:--.--';
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${String(m).padStart(2, '0')}:${s.toFixed(2).padStart(5, '0')}`;
  }

  // HTML 엔티티 디코딩 함수
  const decodeHtmlEntities = (text: string) => {
    // 브라우저 환경에서만 실행
    if (typeof document !== 'undefined') {
      const textarea = document.createElement('textarea');
      textarea.innerHTML = text;
      return textarea.value;
    }
    // 서버 환경에서는 기본적인 HTML 엔티티 디코딩
    return text
      .replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&quot;/g, '"')
      .replace(/&#39;/g, "'")
      .replace(/&nbsp;/g, ' ');
  };

  // 스크립트 렌더링 함수
  const renderScriptWithWords = () => {
    if (!currentWords || currentWords.length === 0) {
      return (
        <div className="text-white font-bold text-center leading-tight tracking-wide" style={{ fontSize: 'clamp(24px, 3.5vw, 48px)' }}>
          <span className="bg-gradient-to-br from-white to-gray-300 bg-clip-text text-transparent">
            {decodeHtmlEntities(captions[currentScriptIndex]?.script || "")}
          </span>
        </div>
      );
    }

    return (
      <div className="text-white font-bold text-center leading-tight tracking-wide" style={{ fontSize: 'clamp(14px, 2vw, 32px)' }}>
        {currentWords.map((word, index) => {
          // 시간 허용 오차 (0.1초) - 네트워크 지연이나 처리 지연 고려
          const TIME_TOLERANCE = 0.1;
          
          const isCurrentWord = currentVideoTime && 
            currentVideoTime >= (word.start_time - TIME_TOLERANCE) && 
            currentVideoTime <= (word.end_time + TIME_TOLERANCE);
          
          // 이전 단어와의 시간 간격 계산 (첫 번째 단어는 스크립트 시작 시간과 비교)
          const prevWord = index > 0 ? currentWords[index - 1] : null;
          const scriptStartTime = currentWords[0]?.start_time || 0;
          
          let timeGap = 0;
          let gapStartTime = 0;
          
          if (index === 0) {
            // 첫 번째 단어: 스크립트 시작 지점부터 첫 번째 단어까지의 간격
            const scriptStartTime = captions[currentScriptIndex]?.start_time || 0;
            timeGap = word.start_time - scriptStartTime;
            gapStartTime = scriptStartTime;
          } else if (prevWord) {
            // 나머지 단어: 이전 단어와의 간격
            timeGap = word.start_time - prevWord.end_time;
            gapStartTime = prevWord.end_time;
          }
          
          const hasLongGap = timeGap >= 2; // 2초 이상 간격
          
          // 카운트다운 계산 (간격 시간 동안)
          const isInGap = currentVideoTime && hasLongGap &&
            currentVideoTime >= gapStartTime && 
            currentVideoTime < word.start_time;
          const remainingTime = isInGap 
            ? Math.max(0, Math.ceil(word.start_time - currentVideoTime))
            : -1; // -1로 설정해서 조건에서 제외
          
          // 전체 스크립트 진행률 계산 (연속적)
          //const scriptStartTime = currentWords[0]?.start_time || 0;
          const scriptEndTime = currentWords[currentWords.length - 1]?.end_time || 0;
          const scriptDuration = scriptEndTime - scriptStartTime;
          
          // 현재 단어의 상대적 위치 계산
          const wordStartOffset = (word.start_time - scriptStartTime) / scriptDuration;
          const wordEndOffset = (word.end_time - scriptStartTime) / scriptDuration;
          const wordWidth = wordEndOffset - wordStartOffset;
          
          // 전체 진행률 (0~1)
          const overallProgress = currentVideoTime && scriptDuration > 0
            ? Math.min(1, Math.max(0, (currentVideoTime - scriptStartTime) / scriptDuration))
            : 0;
          
          // 현재 단어 내에서의 진행률 - 더 정교한 계산
          let wordProgress = 0;
          if (currentVideoTime) {
            if (currentVideoTime >= (word.start_time - TIME_TOLERANCE) && currentVideoTime <= (word.end_time + TIME_TOLERANCE)) {
              const wordDuration = word.end_time - word.start_time;
              if (wordDuration > 0) {
                // 더 정확한 시간 계산
                const adjustedCurrentTime = Math.max(word.start_time, Math.min(word.end_time, currentVideoTime));
                wordProgress = (adjustedCurrentTime - word.start_time) / wordDuration;
                
                // 더 부드러운 진행률 계산 (easing 함수 적용)
                wordProgress = Math.min(1, Math.max(0, wordProgress));
                
                // 시작과 끝 부분을 더 부드럽게 처리
                if (wordProgress < 0.1) {
                  wordProgress = wordProgress * 5; // 시작 부분 가속
                } else if (wordProgress > 0.9) {
                  wordProgress = 0.9 + (wordProgress - 0.9) * 2; // 끝 부분 감속
                }
                
                wordProgress = Math.min(1, Math.max(0, wordProgress));
              } else {
                wordProgress = 1;
              }
            } else if (currentVideoTime > word.end_time) {
              wordProgress = 1;
            }
          }
          


          // 내 대사인지 상대방 대사인지에 따라 색상 결정
          const getWordColors = () => {
            if (isMyLine) {
              // 내 대사 - 초록색 계열
              return {
                gradient: 'from-emerald-200 to-green-300',
                shadow: '0 0 20px rgba(16, 185, 129, 0.6), 0 0 40px rgba(16, 185, 129, 0.3)',
                fillColor: '#22c55e' // green-500
              };
            } else {
              // 상대방 대사 - 파랑/보라색 계열
              return {
                gradient: 'from-blue-200 to-indigo-300',
                shadow: '0 0 20px rgba(59, 130, 246, 0.6), 0 0 40px rgba(99, 102, 241, 0.3)',
                fillColor: '#6366f1' // indigo-500
              };
            }
          };

          const colors = getWordColors();
          
          return (
            <span key={index} style={{ display: 'inline-block', marginRight: '0.45em' }} className="relative">
              {/* 카운트다운 표시 */}
              {hasLongGap && isInGap && remainingTime >= 0 && (
                <div 
                  className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-black/70 text-white text-xs px-2 py-1 rounded-full animate-pulse"
                  style={{ 
                    fontSize: 'clamp(10px, 1vw, 14px)',
                    zIndex: 10
                  }}
                >
                  {remainingTime}
                </div>
              )}
              
              <span
                className={`relative transition-all duration-75 ${
                  wordProgress > 0
                    ? `font-extrabold inline-block` 
                    : 'bg-gradient-to-br from-white to-gray-300 bg-clip-text text-transparent'
                }`}
                style={{
                  position: 'relative',
                  display: 'inline-block'
                }}
              >
                {wordProgress > 0 ? (
                  <span className="relative inline-block">
                    {/* 배경 글씨 (회색) */}
                    <span className="text-gray-500">
                      {decodeHtmlEntities(word.word)}
                    </span>
                    
                    {/* 색깔 글씨 (진행률에 따라 차오름) - 더 정교한 그라데이션 */}
                    <span
                      className={`absolute inset-0 ${isMyLine ? 'text-green-400' : 'text-blue-400'}`}
                      style={{
                        WebkitMask: `linear-gradient(90deg, 
                          black 0%, 
                          black ${Math.max(0, wordProgress * 100 - 2)}%, 
                          rgba(0,0,0,0.8) ${wordProgress * 100}%, 
                          rgba(0,0,0,0.3) ${Math.min(100, wordProgress * 100 + 3)}%, 
                          transparent ${Math.min(100, wordProgress * 100 + 8)}%, 
                          transparent 100%)`,
                        mask: `linear-gradient(90deg, 
                          black 0%, 
                          black ${Math.max(0, wordProgress * 100 - 2)}%, 
                          rgba(0,0,0,0.8) ${wordProgress * 100}%, 
                          rgba(0,0,0,0.3) ${Math.min(100, wordProgress * 100 + 3)}%, 
                          transparent ${Math.min(100, wordProgress * 100 + 8)}%, 
                          transparent 100%)`,
                        transition: 'none',
                        filter: `brightness(${1 + wordProgress * 0.2})` // 진행되면서 약간 밝아짐
                      }}
                    >
                      {decodeHtmlEntities(word.word)}
                    </span>
                    
                    {/* 글로우 효과 (진행 중인 부분에만) */}
                    {wordProgress > 0 && wordProgress < 1 && (
                      <span
                        className={`absolute inset-0 ${isMyLine ? 'text-green-300' : 'text-blue-300'}`}
                        style={{
                          WebkitMask: `linear-gradient(90deg, 
                            transparent 0%, 
                            transparent ${Math.max(0, wordProgress * 100 - 5)}%, 
                            rgba(0,0,0,0.4) ${Math.max(0, wordProgress * 100 - 2)}%, 
                            rgba(0,0,0,0.8) ${wordProgress * 100}%, 
                            rgba(0,0,0,0.4) ${Math.min(100, wordProgress * 100 + 2)}%, 
                            transparent ${Math.min(100, wordProgress * 100 + 5)}%, 
                            transparent 100%)`,
                          mask: `linear-gradient(90deg, 
                            transparent 0%, 
                            transparent ${Math.max(0, wordProgress * 100 - 5)}%, 
                            rgba(0,0,0,0.4) ${Math.max(0, wordProgress * 100 - 2)}%, 
                            rgba(0,0,0,0.8) ${wordProgress * 100}%, 
                            rgba(0,0,0,0.4) ${Math.min(100, wordProgress * 100 + 2)}%, 
                            transparent ${Math.min(100, wordProgress * 100 + 5)}%, 
                            transparent 100%)`,
                          transition: 'none',
                          filter: 'blur(0.5px)',
                          opacity: 0.6
                        }}
                      >
                        {decodeHtmlEntities(word.word)}
                      </span>
                    )}
                  </span>
                ) : (
                  <span className="text-white">
                    {decodeHtmlEntities(word.word)}
                  </span>
                )}
              </span>
            </span>
          );
        })}
      </div>
    );
  };

  return (
    <div className={`bg-gray-900/80 backdrop-blur-sm rounded-lg sm:rounded-xl w-full flex flex-col relative border ${isDuet && !isMyLine ? 'border-blue-800' : 'border-gray-800'} shadow-lg`}>
      <div className={`bg-gradient-to-br ${isDuet &&  !isMyLine ? 'from-[#0f1a2a] to-[#1e2b3b]' : 'from-[#0f172a] to-[#1e293b]'} rounded-lg sm:rounded-xl p-2 sm:p-[0.6vw] shadow-xl text-white border ${isDuet && !isMyLine ? 'border-blue-700/50' : 'border-gray-700/50'} flex flex-col justify-between min-h-[10vh] sm:min-h-[12vh]`}>
        
        {/* 상단 통합 영역: 배우 정보(좌) + 버튼들(중) + 시간 정보(우) */}
        <div className="flex items-center justify-between w-full py-1 sm:py-[0.3vh]">
          {/* 왼쪽: 배우 정보 - 크기 2배 증가 및 반응형 */}
          <div className="flex items-center flex-1">
            {isDuet && (
              <span className={`px-2 sm:px-4 py-1 sm:py-2 rounded-lg sm:rounded-xl text-sm sm:text-base md:text-lg font-bold shadow-lg ${
                isMyLine 
                  ? 'bg-gradient-to-r from-green-500 to-emerald-600 text-white border border-green-400/30' 
                  : 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white border border-blue-400/30'
              }`}>
                {isMyLine ? (user?.name || '나') : (captions[currentScriptIndex]?.actor?.name || '상대배우')}
              </span>
            )}
          </div>

          {/* 중앙: 버튼들 - 3D 효과 추가 */}
          <div className="flex items-center justify-center gap-1 sm:gap-[0.8vw] flex-1 flex-wrap min-h-0">
            {/* 재생/정지 버튼 */}
            <button
              onClick={() => {
                console.log('[ScriptDisplay] 재생/정지 버튼 클릭');
                console.log('[ScriptDisplay] videoPlayerRef?.current:', !!videoPlayerRef?.current);
                
                if (!videoPlayerRef?.current) {
                  console.error('[ScriptDisplay] videoPlayerRef.current가 null입니다');
                  return;
                }
                
                if (isVideoPlaying) {
                  console.log('[ScriptDisplay] 정지 요청');
                  videoPlayerRef.current.pauseVideo();
                } else {
                  console.log('[ScriptDisplay] 재생 요청');
                  const currentScript = captions[currentScriptIndex];
                  const startTime = currentScript?.start_time || 0;
                  const endTime = currentScript?.end_time || 0;
                  
                  const currentTime = videoPlayerRef.current.getCurrentTime();
                  console.log('[ScriptDisplay] 현재 시간:', currentTime, '시작 시간:', startTime);
                  
                  if (currentTime < startTime || currentTime >= endTime) {
                    console.log('[ScriptDisplay] 시간 이동:', startTime);
                    videoPlayerRef.current.seekTo(startTime);
                    setTimeout(() => {
                      if (videoPlayerRef?.current) {
                        videoPlayerRef.current.playVideo();
                      }
                    }, 200);
                  } else {
                    videoPlayerRef.current.playVideo();
                  }
                }
              }}
              className={`w-8 h-8 sm:w-[3.5vw] sm:h-[3.5vw] sm:min-w-[35px] sm:min-h-[35px] sm:max-w-[55px] sm:max-h-[55px] ${
                recording 
                  ? 'bg-gray-700/70 backdrop-blur-sm cursor-not-allowed' 
                  : isVideoPlaying
                    ? 'bg-gradient-to-br from-red-700/70 to-rose-800/70 backdrop-blur-sm hover:from-red-600/80 hover:to-rose-700/80'
                    : 'bg-gradient-to-br from-emerald-700/70 to-green-800/70 backdrop-blur-sm hover:from-emerald-600/80 hover:to-green-700/80'
              } rounded-2xl sm:rounded-3xl flex items-center justify-center transition-all duration-200 border-2 border-white/30 shadow-[0_4px_15px_rgba(0,0,0,0.3),inset_0_1px_0_rgba(255,255,255,0.2),inset_0_-1px_0_rgba(0,0,0,0.2)] transform hover:scale-110 hover:shadow-[0_6px_20px_rgba(0,0,0,0.4),inset_0_2px_0_rgba(255,255,255,0.3),inset_0_-2px_0_rgba(0,0,0,0.3)] active:scale-95 active:shadow-[0_2px_8px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(0,0,0,0.2),inset_0_-1px_0_rgba(255,255,255,0.1)]`}
              disabled={recording}
            >
              {isVideoPlaying || recording ? (
                <svg className="w-3 h-3 sm:w-[1.2vw] sm:h-[1.2vw] sm:min-w-[14px] sm:min-h-[14px] sm:max-w-[20px] sm:max-h-[20px] drop-shadow-sm" fill="currentColor" viewBox="0 0 24 24">
                  <rect x="6" y="4" width="4" height="16" rx="2" />
                  <rect x="14" y="4" width="4" height="16" rx="2" />
                </svg>
              ) : (
                <svg className="w-4 h-4 sm:w-[1.6vw] sm:h-[1.6vw] sm:min-w-[18px] sm:min-h-[18px] sm:max-w-[24px] sm:max-h-[24px] drop-shadow-sm" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5.14v14l11-7-11-7z" />
                </svg>
              )}
            </button>

            {/* 마이크 버튼 */}
            <button
              onClick={onMicClick}
              disabled={recording || recordingCompleted || (isDuet && !isMyLine)}
              className={`w-8 h-8 sm:w-[3.5vw] sm:h-[3.5vw] sm:min-w-[35px] sm:min-h-[35px] sm:max-w-[55px] sm:max-h-[55px] rounded-2xl sm:rounded-3xl flex items-center justify-center transition-all duration-200 border-2 border-white/30 shadow-[0_4px_15px_rgba(0,0,0,0.3),inset_0_1px_0_rgba(255,255,255,0.2),inset_0_-1px_0_rgba(0,0,0,0.2)] transform hover:scale-110 hover:shadow-[0_6px_20px_rgba(0,0,0,0.4),inset_0_2px_0_rgba(255,255,255,0.3),inset_0_-2px_0_rgba(0,0,0,0.3)] active:scale-95 active:shadow-[0_2px_8px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(0,0,0,0.2),inset_0_-1px_0_rgba(255,255,255,0.1)] ${
                recording 
                  ? 'bg-gradient-to-br from-red-700/70 to-rose-800/70 backdrop-blur-sm animate-pulse-mic' 
                  : recordingCompleted
                    ? 'bg-gradient-to-br from-green-700/70 to-emerald-800/70 backdrop-blur-sm'
                    : (isDuet && !isMyLine)
                      ? 'bg-gray-700/50 backdrop-blur-sm cursor-not-allowed opacity-50'
                      : 'bg-gradient-to-br from-gray-700/70 to-gray-800/70 backdrop-blur-sm hover:from-gray-600/80 hover:to-gray-700/80'
              }`}
            >
              <svg 
                className="w-4 h-4 sm:w-[1.6vw] sm:h-[1.6vw] sm:min-w-[18px] sm:min-h-[18px] sm:max-w-[24px] sm:max-h-[24px] relative z-10 drop-shadow-sm" 
                fill="currentColor" 
                viewBox="0 0 24 24"
              >
                <path d="M12 14c1.66 0 2.99-1.34 2.99-3L15 5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.48 6-3.3 6-6.72h-1.7z"/>
              </svg>
            </button>

            {/* 구간반복 버튼 */}
            <button
              onClick={onLoopToggle}
              disabled={recording || recordingCompleted || !videoPlayerRef?.current}
              className={`w-8 h-8 sm:w-[3.5vw] sm:h-[3.5vw] sm:min-w-[35px] sm:min-h-[35px] sm:max-w-[55px] sm:max-h-[55px] rounded-2xl sm:rounded-3xl flex items-center justify-center transition-all duration-200 border-2 border-white/30 shadow-[0_4px_15px_rgba(0,0,0,0.3),inset_0_1px_0_rgba(255,255,255,0.2),inset_0_-1px_0_rgba(0,0,0,0.2)] transform hover:scale-110 hover:shadow-[0_6px_20px_rgba(0,0,0,0.4),inset_0_2px_0_rgba(255,255,255,0.3),inset_0_-2px_0_rgba(0,0,0,0.3)] active:scale-95 active:shadow-[0_2px_8px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(0,0,0,0.2),inset_0_-1px_0_rgba(255,255,255,0.1)] ${
                isLooping 
                  ? 'bg-gradient-to-br from-amber-700/70 to-orange-800/70 backdrop-blur-sm' 
                  : 'bg-gradient-to-br from-gray-700/70 to-gray-800/70 backdrop-blur-sm hover:from-gray-600/80 hover:to-gray-700/80'
              }`}
            >
              <svg viewBox="0 0 24 24" fill="none" className={`w-4 h-4 sm:w-[1.6vw] sm:h-[1.6vw] sm:min-w-[18px] sm:min-h-[18px] sm:max-w-[24px] sm:max-h-[24px] drop-shadow-sm ${isLooping ? 'animate-spin' : ''}`} stroke="currentColor" strokeWidth="2">
                <path d="M4 12c0-4.4 3.6-8 8-8 2.2 0 4.2 0.9 5.7 2.3" strokeLinecap="round"/>
                <path d="M20 4v4h-4" strokeLinecap="round"/>
                <path d="M20 12c0 4.4-3.6 8-8 8-2.2 0-4.2-0.9-5.7-2.3" strokeLinecap="round"/>
                <path d="M4 20v-4h4" strokeLinecap="round"/>
              </svg>
            </button>

            {/* 완료 버튼들 */}
            {showCompletedButtons && (
              <>
                {/* 더빙본 들어보기 버튼 */}
                <button
                  onClick={onOpenDubbingListenModal}
                  className="px-2 py-1 sm:px-[0.8vw] sm:py-[0.5vh] h-8 sm:h-[2.5vw] rounded-lg sm:rounded-xl flex items-center justify-center gap-1 transition-all duration-200 border-2 border-purple-400/30 shadow-[0_4px_15px_rgba(0,0,0,0.3),inset_0_1px_0_rgba(255,255,255,0.2),inset_0_-1px_0_rgba(0,0,0,0.2)] transform hover:scale-110 hover:shadow-[0_6px_20px_rgba(0,0,0,0.4),inset_0_2px_0_rgba(255,255,255,0.3),inset_0_-2px_0_rgba(0,0,0,0.3)] active:scale-95 active:shadow-[0_2px_8px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(0,0,0,0.2),inset_0_-1px_0_rgba(255,255,255,0.1)] bg-gradient-to-br from-purple-700/70 to-indigo-800/70 backdrop-blur-sm hover:from-purple-600/80 hover:to-indigo-700/80"
                >
                  <span className="text-xs sm:text-[0.8vw] font-medium whitespace-nowrap drop-shadow-sm">더빙영상</span>
                </button>

                {/* 결과보기 버튼 */}
                <button
                  onClick={onShowResults}
                  className="px-2 py-1 sm:px-[0.8vw] sm:py-[0.5vh] h-8 sm:h-[2.5vw] rounded-lg sm:rounded-xl flex items-center justify-center gap-1 transition-all duration-200 border-2 border-emerald-400/30 shadow-[0_4px_15px_rgba(0,0,0,0.3),inset_0_1px_0_rgba(255,255,255,0.2),inset_0_-1px_0_rgba(0,0,0,0.2)] transform hover:scale-110 hover:shadow-[0_6px_20px_rgba(0,0,0,0.4),inset_0_2px_0_rgba(255,255,255,0.3),inset_0_-2px_0_rgba(0,0,0,0.3)] active:scale-95 active:shadow-[0_2px_8px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(0,0,0,0.2),inset_0_-1px_0_rgba(255,255,255,0.1)] bg-gradient-to-br from-emerald-700/70 to-teal-800/70 backdrop-blur-sm hover:from-emerald-600/80 hover:to-teal-700/80"
                >
                  <span className="text-xs sm:text-[0.8vw] font-medium whitespace-nowrap drop-shadow-sm">결과보기</span>
                </button>
              </>
            )}
          </div>

          {/* 오른쪽: 시간 정보 */}
          <div className="flex items-center justify-end flex-1" style={{ fontSize: 'clamp(8px, 1vw, 16px)' }}>
            {playbackRange && (
              <span className="text-gray-300 font-medium">
                {formatTime(playbackRange.startTime - captions[0]?.start_time || 0)} ~ {playbackRange.endTime ? formatTime(playbackRange.endTime - captions[0]?.start_time || 0) : '끝'}
              </span>
            )}
          </div>
        </div>



        {/* 스크립트 본문 영역 */}
        <div className="flex flex-col items-center flex-1 justify-center gap-1 sm:gap-[0.5vh] mt-1 sm:mt-[0.5vh]">
          {/* 스크립트 본문 + 내비게이션 */}
          <div className="flex items-center gap-2 sm:gap-[1.2vw] w-full h-full">
            {/* 왼쪽 네비게이션 버튼 */}
            <button
              onClick={() => {
                if (currentScriptIndex > 0) {
                  onScriptChange(currentScriptIndex - 1);
                }
              }}
              disabled={currentScriptIndex === 0 || recording || recordingCompleted}
              className={`p-1 sm:p-[0.6vw] rounded-lg sm:rounded-xl transition-all duration-200 border-2 shadow-[0_4px_15px_rgba(0,0,0,0.3),inset_0_1px_0_rgba(255,255,255,0.2),inset_0_-1px_0_rgba(0,0,0,0.2)] transform hover:scale-110 hover:shadow-[0_6px_20px_rgba(0,0,0,0.4),inset_0_2px_0_rgba(255,255,255,0.3),inset_0_-2px_0_rgba(0,0,0,0.3)] active:scale-95 active:shadow-[0_2px_8px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(0,0,0,0.2),inset_0_-1px_0_rgba(255,255,255,0.1)] ${
                currentScriptIndex === 0 
                  ? 'bg-gray-800/50 text-gray-500 cursor-not-allowed border-gray-600/30' 
                  : isDuet && !isMyLine
                    ? 'bg-indigo-900/50 backdrop-blur-sm text-blue-400 hover:bg-indigo-800/70 hover:text-blue-300 border-blue-700/30'
                    : 'bg-emerald-900/50 backdrop-blur-sm text-green-400 hover:bg-emerald-800/70 hover:text-green-300 border-emerald-700/30'
              }`}
            >
              <ChevronLeftIcon className="w-4 h-4 sm:w-[1.2vw] sm:h-[1.2vw] sm:min-w-[14px] sm:min-h-[14px] sm:max-w-[20px] sm:max-h-[20px] drop-shadow-sm" />
            </button>

            {/* 중앙 스크립트 박스 */}
            <div 
              className={`bg-gray-800/80 rounded-lg sm:rounded-xl p-2 sm:p-[0.5vw] flex-1 shadow-[inset_0_2px_8px_rgba(0,0,0,0.3),inset_0_-1px_0_rgba(255,255,255,0.1)] border border-gray-700/50 flex items-center justify-center relative overflow-visible ${
                showAnalysisResult && analysisResult 
                  ? 'min-h-[6vh] sm:min-h-[8vh]' // 분석 결과 표시 시 최소 높이만 설정
                  : 'min-h-[4vh] sm:min-h-[6vh]'  // 일반 텍스트 시 최소 높이만 설정
              }`}
              style={{
                background: isAnalyzing 
                  ? 'linear-gradient(45deg, rgba(59, 130, 246, 0.1), rgba(147, 51, 234, 0.1))' 
                  : undefined
              }}
            >
              {isAnalyzing ? (
                <div className="text-center">
                  <div className="animate-spin w-6 h-6 sm:w-8 sm:h-8 border-3 sm:border-4 border-blue-400 border-t-transparent rounded-full mx-auto mb-2"></div>
                  <div className="text-blue-400 font-medium text-sm sm:text-base">음성 분석 중...</div>
                </div>
              ) : showAnalysisResult && analysisResult ? (
                <PronunciationTimingGuide
                  captions={captions}
                  currentScriptIndex={currentScriptIndex}
                  currentVideoTime={currentVideoTime || 0}
                  currentWords={currentWords}
                  showAnalysisResult={showAnalysisResult}
                  analysisResult={analysisResult}
                  recording={recording}
                  id={id}
                />
              ) : (
                <div className="text-white font-bold text-center leading-tight relative" style={{ fontSize: 'clamp(14px, 2.5vw, 32px)' }}>
                  {renderScriptWithWords()}
                </div>
              )}
            </div>

            {/* 오른쪽 네비게이션 버튼 */}
            <button
              onClick={() => {
                if (currentScriptIndex < captions.length - 1) {
                  onScriptChange(currentScriptIndex + 1);
                }
              }}
              disabled={currentScriptIndex === captions.length - 1 || recording || recordingCompleted}
              className={`p-1 sm:p-[0.6vw] rounded-lg sm:rounded-xl transition-all duration-200 border-2 shadow-[0_4px_15px_rgba(0,0,0,0.3),inset_0_1px_0_rgba(255,255,255,0.2),inset_0_-1px_0_rgba(0,0,0,0.2)] transform hover:scale-110 hover:shadow-[0_6px_20px_rgba(0,0,0,0.4),inset_0_2px_0_rgba(255,255,255,0.3),inset_0_-2px_0_rgba(0,0,0,0.3)] active:scale-95 active:shadow-[0_2px_8px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(0,0,0,0.2),inset_0_-1px_0_rgba(255,255,255,0.1)] ${
                currentScriptIndex === captions.length - 1 
                  ? 'bg-gray-800/50 text-gray-500 cursor-not-allowed border-gray-600/30' 
                  : isDuet && !isMyLine
                    ? 'bg-indigo-900/50 backdrop-blur-sm text-blue-400 hover:bg-indigo-800/70 hover:text-blue-300 border-blue-700/30'
                    : 'bg-emerald-900/50 backdrop-blur-sm text-green-400 hover:bg-emerald-800/70 hover:text-green-300 border-emerald-700/30'
              }`}
            >
              <ChevronRightIcon className="w-4 h-4 sm:w-[1.2vw] sm:h-[1.2vw] sm:min-w-[14px] sm:min-h-[14px] sm:max-w-[20px] sm:max-h-[20px] drop-shadow-sm" />
            </button>
          </div>


          {/* 완료 버튼들 - 제거됨, 상단으로 이동 */}

        </div>
      </div>
    </div>
  );
};

export default ScriptDisplay;
