"use client";

import { motion } from "framer-motion";
import { useEffect, useState, useRef } from "react";
import Loader from './Loader';
import { useUser } from '@/hooks/useUser';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  captions: { 
    script: string;
    actor?: {
      name: string;
      id: number;
    };
  }[];
  currentScriptIndex: number;
  onScriptSelect: (index: number) => void;
  actorName?: string;
  movieTitle?: string;
  analyzedCount?: number;
  totalCount?: number;
  recording?: boolean;
  onStopLooping?: () => void;
  recordedScripts?: boolean[];
  latestResultByScript?: Record<string, any>; // 추가
  recordingCompleted?: boolean; // 추가
  // 듀엣 모드 관련 props
  isDuet?: boolean;
  isMyLine?: (index: number) => boolean;
}

function normalizeScript(str: string) {
  if (!str || typeof str !== 'string') return '';
  return str.toLowerCase().replace(/[^a-z0-9]/g, '');
}

export default function Sidebar({
  isOpen,
  onClose,
  captions,
  currentScriptIndex,
  onScriptSelect,
  recording = false,
  onStopLooping,
  recordedScripts = [],
  latestResultByScript = {},
  recordingCompleted = false,
  isDuet = false,
  isMyLine,
}: SidebarProps) {
  const { user } = useUser();
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [detailIndex, setDetailIndex] = useState<number | null>(null); // 클릭 시 세부 정보 표시용

  // 자동스크롤을 위한 ref
  const sidebarRef = useRef<HTMLDivElement | null>(null);
  const listRef = useRef<HTMLUListElement | null>(null);

  useEffect(() => {
    console.log("[Sidebar] captions:", captions);
  }, [captions]);

  useEffect(() => {
    // currentScriptIndex가 변경될 때마다 자동스크롤 실행
    scrollToSelectedItem();
  }, [currentScriptIndex, captions])

  const scrollToSelectedItem = () => {
    if(!listRef.current || !sidebarRef.current) return;

    // 현재 선택된 문장의 DOM요소 찾기
    const selectedElement = listRef.current.querySelector(`[data-index="${currentScriptIndex}"]`) as HTMLElement;
    if(!selectedElement) return;

    // 사이드바의 높이와 스크롤 위치 계산
    const sidebarHeight = sidebarRef.current.clientHeight;
    const listTop = listRef.current.offsetTop;
    const itemTop = selectedElement.offsetTop;
    const itemHeight = selectedElement.clientHeight;

    // 현재 스크롤 위치
    const currentScrollTop = sidebarRef.current.scrollTop;
   
    // 아이템이 보이는 영역 계산
    const itemVisibleTop = itemTop - currentScrollTop;
    const itemVisibleBottom = itemVisibleTop + itemHeight;

    // 목표 위치 (6번째 위치)
    const targetVisibleTop = 5 * itemHeight;

    // 아이템이 목표 위치보다 아래에 있으면 스크롤 조정
    if(itemVisibleTop > targetVisibleTop) {
      const scrollOffset = itemVisibleTop - targetVisibleTop;
      
      // 부드러운 스크롤 애니메이션
      sidebarRef.current.scrollTo({
        top: currentScrollTop + scrollOffset,
        behavior: 'smooth'
      });
    }
  }

  // captions에서 정보 추출 (존재하지 않으면 '-')
  const actorName = (captions[0] && (captions[0] as any).actor && (captions[0] as any).actor.name) ? (captions[0] as any).actor.name : '-';
  const movieTitle = (captions[0] && (captions[0] as any).movie_name) ? (captions[0] as any).movie_name : '-';
  const totalCount = captions.length;
  const analyzedCount = captions.filter(c => {
    const scriptKey = normalizeScript(c.script);
    return !!latestResultByScript[scriptKey];
  }).length;

  // 시간 포맷 함수
  function formatTime(sec?: number) {
    if (typeof sec !== 'number' || isNaN(sec)) return '--:--.--';
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${String(m).padStart(2, '0')}:${s.toFixed(2).padStart(5, '0')}`;
  }

  return (
    <motion.div
      ref={sidebarRef}
      initial={{ x: 400 }}
      animate={{ x: isOpen ? 0 : 400 }}
      transition={{ type: "tween", duration: 0.3 }}
      className="fixed top-16 sm:top-20 lg:top-24 right-0 bottom-0 w-[20vw]  z-40 shadow-2xl bg-gradient-to-br from-gray-900/95 via-gray-800/95 to-gray-900/95 backdrop-blur-sm border-l border-gray-700 text-gray-100 overflow-y-auto custom-scrollbar"
      style={{ boxShadow: 'rgba(0,0,0,0.3) -5px 0 20px' }}
    >
      <div className="flex justify-between items-center px-2 sm:px-3 py-2 border-b border-gray-700 bg-gray-900/90 sticky top-0 z-10">
        <span className="font-bold text-lg sm:text-xl tracking-tight text-emerald-400 select-none">스크립트 목록</span>
        <button
          onClick={onClose}
          className="ml-1 p-1 rounded-full text-gray-400 hover:text-emerald-400 hover:bg-gray-800 transition text-xl sm:text-2xl focus:outline-none focus:ring-2 focus:ring-emerald-400"
          aria-label="사이드바 닫기"
        >
          <span className="text-2xl sm:text-3xl">×</span>
        </button>
      </div>
      {/* 상단 정보 박스 */}
      <div className="px-2 sm:px-3 py-2 border-b border-gray-800 bg-gray-900/90 flex flex-col gap-1">
        <div className="flex items-center gap-1 text-base text-gray-300">
          <span className="font-semibold text-emerald-400">배우명</span>
          <span className="truncate">{actorName}</span>
        </div>
        <div className="flex items-center gap-1 text-base text-gray-300">
          <span className="font-semibold text-emerald-400">영화명</span>
          <span className="truncate">{movieTitle}</span>
        </div>

        {/* 문장 진행률 게이지 */}
        <div className="mt-1">
          <div className="flex justify-between text-sm text-gray-400 mb-1">
            <span>문장 진행률</span>
            <span>{currentScriptIndex + 1} / {totalCount}</span>
          </div>
          <div className="w-full h-2 bg-gray-700 rounded">
            <div
              className="h-2 bg-emerald-400 rounded"
              style={{ width: `${((currentScriptIndex + 1) / totalCount) * 100}%` }}
            />
          </div>
        </div>
        {/* 분석 완료 게이지 */}
        <div className="mt-1">
          <div className="flex justify-between text-sm text-gray-400 mb-1">
            <span>분석 완료</span>
            <span>{analyzedCount} / {totalCount}</span>
          </div>
          <div className="w-full h-2 bg-gray-700 rounded">
            <div
              className="h-2 bg-blue-400 rounded"
              style={{ width: `${(analyzedCount / totalCount) * 100}%` }}
            />
          </div>
        </div>
      </div>
      {/* 범례 */}
      <div className="flex items-center justify-center space-x-2 text-sm text-gray-400 mt-1 px-1">
        <div className="flex items-center space-x-1">
          <div className="w-3 h-3 bg-gray-600 rounded"></div>
          <span>미완료</span>
        </div>
        <div className="flex items-center space-x-1">
          <div className="w-3 h-3 bg-green-400 rounded"></div>
          <span>분석중</span>
        </div>
        <div className="flex items-center space-x-1">
          <div className="w-3 h-3 bg-blue-400 rounded"></div>
          <span>분석완료</span>
        </div>
      </div>

      <ul ref={listRef} className="px-2 py-2 pb-32">
        {captions.map((caption, index) => {
          const scriptKey = normalizeScript(caption.script);
          const isAnalyzed = !!latestResultByScript[scriptKey];
          const isSelected = currentScriptIndex === index;
          const isCurrentMyLine = isDuet && isMyLine ? isMyLine(index) : true;
          
          return (
            <li
              key={index}
              data-index={index}
              onClick={() => {
                if (recording) return;
                if (onStopLooping) onStopLooping();
                onScriptSelect(index);
              }}
              className={`cursor-pointer px-2 py-3 transition-all duration-150 select-none relative rounded-md mb-2 text-base
                ${isSelected
                  ? `border ${isCurrentMyLine ? 'border-emerald-400' : 'border-blue-400'} scale-[1.02] bg-transparent text-white shadow-md z-10`
                  : isAnalyzed
                  ? `border ${isCurrentMyLine ? 'border-emerald-400/50' : 'border-blue-400/50'} bg-transparent text-white`
                  : `hover:bg-gray-800/50 hover:text-${isCurrentMyLine ? 'emerald' : 'blue'}-300 text-gray-200`}
                ${isSelected ? "transition-transform" : ""}
                ${isDuet && !isCurrentMyLine ? 'bg-blue-900/10' : ''}
              `}
              style={{ wordBreak: 'break-word', zIndex: isSelected ? 10 : 1 }}
            >
              {/* Loader 오버레이 - 현재 선택된 문장에서만 표시 */}
              {isSelected && !recording && recordingCompleted && (
                <div className="absolute inset-0 bg-gray-900/30 backdrop-blur-[1px] flex items-center justify-center z-20 rounded pointer-events-none">
                  <Loader />
                </div>
              )}
              <div className="flex items-start">
                {/* 아이콘 + 번호 */}
                <span className="flex items-center mr-2 mt-0.5 select-none" style={{ zIndex: 2 }}>
                  {isSelected ? (
                    // 빙빙 도는 아이콘 (SVG)
                    <svg className={`w-4 h-4 mr-1 text-${isCurrentMyLine ? 'emerald' : 'blue'}-300 animate-spin`} viewBox="0 0 20 20" fill="none" aria-label="재생 중">
                      <circle cx="10" cy="10" r="8" stroke="currentColor" strokeWidth="3" strokeDasharray="20 10" />
                    </svg>
                  ) : isAnalyzed ? (
                    // 체크 아이콘 (SVG) - 분석 완료
                    <svg className={`w-4 h-4 mr-1 text-${isCurrentMyLine ? 'emerald' : 'blue'}-400`} viewBox="0 0 20 20" fill="currentColor" aria-label="분석 완료">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  ) : (
                    // 플레이 아이콘 (SVG)
                    <svg className={`w-4 h-4 mr-1 text-gray-400 group-hover:text-${isCurrentMyLine ? 'emerald' : 'blue'}-300 transition-colors`} viewBox="0 0 20 20" fill="currentColor" aria-label="플레이">
                      <polygon points="6,4 16,10 6,16" />
                    </svg>
                  )}
                </span>
                <span className="flex-1 leading-relaxed text-base">
                  {isDuet && (
                    <span className={`inline-block px-2 py-0.5 rounded text-sm mr-2 ${isCurrentMyLine ? 'bg-green-900/30 text-green-300' : 'bg-blue-900/30 text-blue-300'}`}>
                      {isCurrentMyLine ? (user?.name || '나') : (caption.actor?.name || '상대배우')}
                    </span>
                  )}
                  {caption.script}
                </span>
              </div>
              {/* 타임라인 */}
              {typeof (caption as any).start_time === 'number' && typeof (caption as any).end_time === 'number' && (
                <div className={`text-sm mt-1 ml-4 font-mono transition-colors duration-150 ${isSelected ? 'text-white' : 'text-gray-400'}`}>
                  {formatTime((caption as any).start_time - (captions[0] as any)?.start_time || 0)} ~ {formatTime((caption as any).end_time - (captions[0] as any)?.start_time || 0)}
                </div>
              )}
              {index < captions.length - 1 && (
                <div className="absolute bottom-0 left-2 right-2 h-0.5"
                  style={{ background: isSelected ? '#34d399' : '#a3a3a3', opacity: isSelected ? 0.7 : 0.4, zIndex: 0 }}
                />
              )}
            </li>
          );
        })}
      </ul>
      {/* 세부 정보 모달/패널 (분석 완료 문장 클릭 시) */}
      {detailIndex !== null && latestResultByScript[normalizeScript(captions[detailIndex].script)] && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40" onClick={() => setDetailIndex(null)}>
          <div className="bg-gray-900 rounded-lg p-6 min-w-[320px] max-w-[90vw] shadow-2xl relative" onClick={e => e.stopPropagation()}>
            <button className="absolute top-2 right-2 text-gray-400 hover:text-white text-2xl" onClick={() => setDetailIndex(null)}>×</button>
            <div className="font-bold text-xl mb-3 text-emerald-300">문장 {detailIndex + 1} 분석 세부 정보</div>
            <div className="mb-3 text-gray-200 text-base">{captions[detailIndex].script}</div>
            <div className="space-y-2 text-base">
              {Object.entries(latestResultByScript[normalizeScript(captions[detailIndex].script)]).map(([k, v]) => (
                <div key={k}><span className="text-emerald-300">{k}</span>: {typeof v === 'object' ? JSON.stringify(v, null, 2) : String(v)}</div>
              ))}
            </div>
          </div>
        </div>
      )}
      <style jsx global>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #2d3748;
          border-radius: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
      `}</style>
    </motion.div>
  );
}