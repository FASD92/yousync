import React, { useState, useEffect } from "react";
import Loader from "../ui/Loader";
import WordDetailModal from "./wordDetailModal";
import demoDataRaw from "../../../data.json";

const demoData: Record<string, any>[] = demoDataRaw as Record<string, any>[];

interface PronunciationTimingGuideProps {
  captions: Array<{
    id: number;
    script: string;
    translation: string;
    start_time: number;
    end_time: number;
  }>;
  currentScriptIndex: number;
  currentVideoTime: number;
  currentWords: Array<{
    script_id: number;
    start_time: number;
    end_time: number;
    word: string;
    probability: number;
    id: number;
  }>;
  showAnalysisResult?: boolean;
  analysisResult?: any;
  recording?: boolean;
  id?: string | number; // 추가
}

export default function PronunciationTimingGuide({
  captions,
  currentScriptIndex,
  currentVideoTime,
  currentWords = [],
  showAnalysisResult = false,
  analysisResult = null,
  recording = false,
  id, // 추가
}: PronunciationTimingGuideProps) {
  // id가 35일 때만 demoData의 첫 번째 결과를 analysisResult로 사용하도록 분기 조건을 변경합니다.
  let displayAnalysisResult = analysisResult;
  let userSTT = undefined;
  if (id === 35 || id === '35') {
    // demoData는 배열이므로 currentScriptIndex에 따라 결과를 선택
    const demoIdx = Math.min(currentScriptIndex, demoData.length - 1);
    const demoObj = demoData[demoIdx];
    const firstKey = Object.keys(demoObj)[0];
    displayAnalysisResult = demoObj[firstKey];
    userSTT = demoObj[firstKey]?.user_stt;
  }

  const sentence = captions[currentScriptIndex];
  // 분석 결과가 없어도 컴포넌트는 렌더링하되, 내용만 조건부로 표시
  const words = displayAnalysisResult?.word_analysis || [];

  // 게이지 애니메이션을 위한 상태
  const [animatedScores, setAnimatedScores] = useState<Record<string, number>>({});
  
  // 부드러운 전환을 위한 상태
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [showContent, setShowContent] = useState(false);

  // 단어 클릭 시 WordDetailModal이 열리도록 useState로 모달 상태와 선택 단어 관리
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedWord, setSelectedWord] = useState<any>(null);

  // 분석 결과가 표시될 때 게이지 애니메이션 시작
  useEffect(() => {
    if (displayAnalysisResult?.word_analysis) {
      const targetScores: Record<string, number> = {};
      displayAnalysisResult.word_analysis.forEach((word: any) => {
        targetScores[word.word] = word.word_score;
      });

      // 즉시 최종 점수로 설정 (애니메이션 제거)
      setAnimatedScores(targetScores);
    } else {
      setAnimatedScores({});
    }
  }, [displayAnalysisResult]);

  // 분석 결과가 표시될 때 id 콘솔 출력
  useEffect(() => {
    if (displayAnalysisResult?.word_analysis && id !== undefined) {
      console.log('[PronunciationTimingGuide] 분석 결과 도착, id:', id);
    }
  }, [displayAnalysisResult, id]);

  // 부드러운 전환 효과
  useEffect(() => {
    const hasAnalysisResult = displayAnalysisResult?.word_analysis && displayAnalysisResult.word_analysis.length > 0;
    
    if (hasAnalysisResult && !showContent) {
      // 분석 결과가 도착했을 때 부드럽게 전환
      setIsTransitioning(true);
      setTimeout(() => {
        setShowContent(true);
        setIsTransitioning(false);
      }, 150); // 150ms 지연으로 부드러운 전환
    } else if (!hasAnalysisResult && showContent) {
      // 분석 결과가 없어졌을 때 즉시 로딩창으로
      setShowContent(false);
      setIsTransitioning(false);
    }
  }, [displayAnalysisResult, showContent]);

  // 녹음이 시작될 때 애니메이션/색상 상태 초기화
  useEffect(() => {
    if (recording) {
      setAnimatedScores({});
      setShowContent(false);
      setIsTransitioning(false);
    }
  }, [recording]);

  useEffect(() => {
    if (id !== undefined) {
      console.log('[PronunciationTimingGuide] 받은 id:', id);
    }
  }, [id]);

  // RGB 그라데이션 색상 계산
  const getGradientColor = (score: number) => {
    let r, g, b;
    if (score <= 0.5) {
      const t = score * 2;
      r = 255;
      g = Math.round(255 * t);
      b = 0;
    } else {
      const t = (score - 0.5) * 2;
      r = Math.round(255 * (1 - t));
      g = 255;
      b = 0;
    }
    return `rgb(${r}, ${g}, ${b})`;
  };

  // HTML 엔티티를 디코딩하는 함수 (SSR 호환)
  const decodeHtmlEntities = (text: string) => {
    if (typeof window === 'undefined') {
      // 서버 사이드에서는 기본적인 HTML 엔티티만 처리
      return text
        .replace(/&amp;/g, '&')
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/&quot;/g, '"')
        .replace(/&#39;/g, "'")
        .replace(/&nbsp;/g, ' ');
    }
    
    // 클라이언트 사이드에서는 textarea를 사용
    const textarea = document.createElement('textarea');
    textarea.innerHTML = text;
    return textarea.value;
  };

  // 현재 시간을 분:초 형식으로 변환
  const getMinutesAndSeconds = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return {
      minutes: minutes.toString().padStart(2, '0'),
      seconds: seconds.toString().padStart(2, '0'),
    };
  };

  const timeBoxClass = "inline-block font-mono text-lg bg-gray-800 rounded px-1 w-[36px] text-center align-middle";
  const current = getMinutesAndSeconds(currentVideoTime);
  
  // 스크립트의 마지막 end_time에서 종료시간 계산
  const lastScriptEndTime = captions.length > 0 ? captions[captions.length - 1].end_time : 0;
  const total = getMinutesAndSeconds(lastScriptEndTime);

  return (
    <div className="relative w-full h-full flex items-center justify-center">
      {/* 배우 뱃지 - 듀엣더빙이 아닐 때는 표시하지 않음 */}
      {process.env.NEXT_PUBLIC_MODE === 'duet' && (
        <div className="absolute top-1/2 -translate-y-1/2 left-3 flex items-center gap-2 px-3 py-1 rounded-full text-xl font-semibold bg-emerald-600 text-white">
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
          </svg>
          내 대사
        </div>
      )}
      {/* 자막 텍스트 - 단어 게이지를 여러 줄로 분할 */}
      <div className="text-center w-full">
        <div className="text-sm sm:text-base md:text-lg lg:text-xl font-bold leading-tight text-emerald-100">
          {showContent && displayAnalysisResult?.word_analysis && displayAnalysisResult.word_analysis.length > 0 ? (
            // 분석 결과가 있을 때만 표시
            <div className={`transition-all duration-300 ease-out ${isTransitioning ? 'opacity-0 scale-95' : 'opacity-100 scale-100'}`}> 
              <div className="flex flex-wrap justify-center items-center gap-2">
                {words.map((word: any, idx: number) => {
                  const animatedScore = animatedScores[word.word] || 0;
                  return (
                    <div
                      key={word.word + idx}
                      className="flex flex-col items-center cursor-pointer transition-transform duration-200 hover:scale-110 hover:bg-emerald-900/20 rounded-lg p-1"
                      onClick={() => {
                        setSelectedWord(word);
                        setIsModalOpen(true);
                      }}
                    >
                      <span className="text-emerald-100 mb-1">{decodeHtmlEntities(word.word)}</span>
                      <div className="w-12 h-1 sm:w-16 sm:h-1.5 md:w-20 md:h-2 bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className="h-full transition-all duration-300 ease-out"
                          style={{
                            width: `${Math.round(animatedScore * 100)}%`,
                            backgroundColor: getGradientColor(animatedScore)
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            // 분석 결과가 없을 때 로딩창 표시
            <div className={`flex flex-col items-center justify-center space-y-3 w-full transition-all duration-300 ease-out ${isTransitioning ? 'opacity-0 scale-95' : 'opacity-100 scale-100'}`}>
              <Loader />
              <span className="text-gray-400 text-sm text-center">분석 결과를 기다리는 중...</span>
            </div>
          )}
        </div>
      </div>
      <WordDetailModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        word={selectedWord}
        userSTT={userSTT}
      />
    </div>
  );
} 