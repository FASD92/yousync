"use client";
import React, { useState, useRef, useEffect } from "react";
import { Radar } from "react-chartjs-2";
import {Chart,RadialLinearScale,PointElement,LineElement,Filler,Tooltip,Legend} from "chart.js";
import { useDubbingState } from "@/hooks/useDubbingState";
import ScoreCards from "./ScoreCards";
import SentenceAnalysis from "./SentenceAnalysis";
import { COLORS } from "../../../tailwind.config";
import ResultViewBtn from "./ResultViewBtn";
Chart.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

function calcAvg(arr: number[]) {
  if (!arr.length) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function getTotalScores(finalResults: any[]) {
  console.log('getTotalScores 호출됨, 입력값:', finalResults);
  console.log('입력값 타입:', Array.isArray(finalResults) ? 'Array' : typeof finalResults);
  
  // 배열이 아닌 경우 배열로 변환
  let resultsArray = finalResults;
  if (!Array.isArray(finalResults) && typeof finalResults === 'object') {
    console.log('객체를 배열로 변환');
    resultsArray = Object.values(finalResults);
    console.log('변환된 배열:', resultsArray);
  }
  
  const safeResults = Array.isArray(resultsArray) ? resultsArray.filter(Boolean) : [];
  console.log('필터링된 결과 개수:', safeResults.length);
  
  if (safeResults.length === 0) {
    console.log('유효한 결과 없음, 기본값 반환');
    return { syncRate: 0, pronunciation: 0, intonation: 0, timing: 0 };
  }
  
  // 각 결과의 구조 확인
  safeResults.forEach((result, index) => {
    console.log(`결과[${index}] 키:`, Object.keys(result));
    console.log(`결과[${index}] word_analysis 존재 여부:`, !!result.word_analysis);
    if (result.word_analysis) {
      console.log(`결과[${index}] word_analysis 길이:`, result.word_analysis.length);
    }
  });
  
  const allWords = safeResults.flatMap(s =>
    Array.isArray(s.word_analysis) ? s.word_analysis : []
  );
  
  console.log('모든 단어 분석 개수:', allWords.length);
  
  if (allWords.length === 0) {
    console.log('단어 분석 없음, 기본값 반환');
    return { syncRate: 0, pronunciation: 0, intonation: 0, timing: 0 };
  }
  
  // 발음: word_score 평균
  const pronunciation = calcAvg(allWords.map(w => w.word_score || 0));
  // 억양: mfcc_similarity 평균
  const intonation = calcAvg(allWords.map(w => w.mfcc_similarity || 0));
  // 발화타임: text_status === 'pass' 비율
  const timing =
    allWords.length > 0
      ? allWords.filter(w => w.text_status === 'pass').length / allWords.length
      : 0;
  // 싱크율 %: 세 점수의 평균
  const syncRate = (pronunciation + intonation + timing) / 3;
  
  console.log('계산된 점수:', { syncRate, pronunciation, intonation, timing });
  
  return { syncRate, pronunciation, intonation, timing };
}

function getRank(score: number) {
  if (score >= 0.95) return "S";
  if (score >= 0.85) return "A";
  if (score >= 0.7) return "B";
  return "C";
}

function getRankColor(rank: string) {
  if (rank === "S") return COLORS.point;
  if (rank === "A") return COLORS.yellow;
  if (rank === "B") return "#6cd4ff";
  return COLORS.fail;
}

export interface TestResultAnalysisSectionProps {
  finalResults?: Record<string, any> | any[];
  latestResultByScript?: Record<string, any>;
  hasAnalysisResults?: boolean;
  showResults?: boolean;
  showCompleted?: boolean;
  onViewResults?: () => void;
}

const ResultComponent: React.FC<TestResultAnalysisSectionProps> = ({ 
  finalResults: propFinalResults,
  latestResultByScript: propLatestResultByScript,
  hasAnalysisResults = false,
  showResults = false,
  showCompleted = false,
  onViewResults
}) => {
  const [openIdx, setOpenIdx] = useState<number | null>(null);

  // 결과 컨테이너 ref
  const resultRef = useRef<HTMLDivElement | null>(null);
  const prevShowResults = useRef(showResults);

  // showResults가 false -> true로 바뀔 때만 스크롤
  useEffect(() => {
    if (!prevShowResults.current && showResults && resultRef.current) {
      requestAnimationFrame(() => {
        resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
        if (resultRef.current) {
          window.scrollTo({ top: resultRef.current.offsetTop, behavior: "smooth" });
        }
      });
    }
    prevShowResults.current = showResults;
  }, [showResults]);

  // props로 받은 데이터가 있으면 사용, 없으면 훅에서 가져오기
  const dubbingState = useDubbingState();
  const latestResultByScriptObj = propLatestResultByScript || dubbingState.latestResultByScript;
  
  // finalResults가 이미 배열인 경우 그대로 사용
  const finalResults = Array.isArray(propFinalResults) 
    ? propFinalResults 
    : latestResultByScriptObj && typeof latestResultByScriptObj === "object"
      ? Object.values(latestResultByScriptObj)
      : propFinalResults && typeof propFinalResults === "object"
        ? Object.values(propFinalResults)
        : dubbingState.finalResults
          ? Array.isArray(dubbingState.finalResults)
            ? dubbingState.finalResults
            : Object.values(dubbingState.finalResults)
          : [];
          
  // null/undefined 필터링
  const filteredResults = finalResults.filter(Boolean);

  // 디버깅 로그 추가
  useEffect(() => {
    console.log('ResultComponent - propFinalResults:', propFinalResults);
    console.log('ResultComponent - propFinalResults 타입:', Array.isArray(propFinalResults) ? 'Array' : typeof propFinalResults);
    
    console.log('ResultComponent - finalResults(처리 후):', filteredResults);
    console.log('ResultComponent - finalResults 타입:', Array.isArray(filteredResults) ? 'Array' : typeof filteredResults);
    console.log('ResultComponent - finalResults 길이:', filteredResults.length);
    
    if (filteredResults.length > 0) {
      console.log('ResultComponent - 첫 번째 항목:', filteredResults[0]);
      console.log('ResultComponent - word_analysis 존재 여부:', !!filteredResults[0]?.word_analysis);
      
      // 모든 항목에 word_analysis가 있는지 확인
      const allHaveWordAnalysis = filteredResults.every(item => !!item?.word_analysis);
      console.log('ResultComponent - 모든 항목에 word_analysis가 있는지:', allHaveWordAnalysis);
      
      // word_analysis가 있는 항목 수
      const itemsWithWordAnalysis = filteredResults.filter(item => !!item?.word_analysis).length;
      console.log('ResultComponent - word_analysis가 있는 항목 수:', itemsWithWordAnalysis);
    }
  }, [propFinalResults, filteredResults]);

  const { syncRate, pronunciation, intonation, timing } = getTotalScores(filteredResults);
  const rank = getRank(syncRate);

  const radarData = {
    labels: ["싱크율 %", "발음", "발화타임", "억양"],
    datasets: [
      {
        label: "Score",
        data: [
          Math.round(syncRate * 100),
          Math.round(pronunciation * 100),
          Math.round(timing * 100),
          Math.round(intonation * 100)
        ],
        backgroundColor: "rgba(34,255,136,0.18)", // 더 연한 내부 색(반투명)
        borderColor: "rgba(34,255,136,0.2)", // 연한 선
        borderWidth: 1, // 매우 가는 선
        pointBackgroundColor: "rgba(0,0,0,0)", // 점 완전 투명
        pointRadius: 1, // 아주 작은 점
        pointHoverRadius: 2,
        fill: true
      }
    ]
  };
  const radarOptions = {
    scales: {
      r: {
        angleLines: { color: "#242626" },
        grid: { color: "#232626" },
        pointLabels: { color: COLORS.text, font: { size: 16 } },
        suggestedMin: 0,
        suggestedMax: 100,
        ticks: { display: false },
      }
    },
    plugins: { legend: { display: false } },
    responsive: true,
    maintainAspectRatio: false
  };

  return (
    <>
      {(showResults || showCompleted) && (
        <div ref={resultRef} className="min-h-screen flex flex-col items-center justify-center px-4 py-12 overflow-hidden bg-gradient-to-b from-gray-900 via-gray-900 to-gray-950">
          {filteredResults.length > 0 ? (
            <>
              <div className="w-full max-w-4xl mx-auto mb-12">
                <h2 className="text-3xl md:text-4xl font-bold text-center mb-8 text-transparent bg-clip-text bg-gradient-to-r from-indigo-300 via-purple-300 to-pink-300">
                  더빙 분석 결과
                </h2>
                <div className="h-1 w-32 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 mx-auto rounded-full mb-12"></div>
              </div>
              
              <ScoreCards
                syncRate={syncRate}
                pronunciation={pronunciation}
                intonation={intonation}
                timing={timing}
              />
              
              <div className="flex flex-col md:flex-row items-center justify-center mb-16 w-full max-w-4xl mx-auto">
                <div className="flex-1 flex justify-center items-center mb-8 md:mb-0">
                  <div
                    className="text-center"
                    style={{
                      fontSize: 120,
                      fontWeight: 900,
                      lineHeight: 1,
                      color: getRankColor(rank),
                      textShadow: `0 0 30px ${getRankColor(rank)}66, 0 8px 45px #000a`,
                      minWidth: 150
                    }}
                  >
                    {rank}
                    <div className="text-base font-medium text-gray-400 mt-2">등급</div>
                  </div>
                </div>
                
                <div className="flex-1 flex justify-center">
                  <div className="w-64 h-64 bg-gray-900/80 backdrop-blur-sm rounded-2xl border border-gray-800/50 p-4 flex items-center justify-center shadow-xl"
                    style={{
                      boxShadow: '0 10px 30px -5px rgba(0, 0, 0, 0.3), 0 0 20px rgba(79, 70, 229, 0.15) inset'
                    }}
                  >
                    <Radar data={radarData} options={radarOptions} />
                  </div>
                </div>
              </div>
              
              <SentenceAnalysis finalResults={filteredResults} />
            </>
          ) : (
            <div className="text-center max-w-md mx-auto p-8 bg-gray-900/80 backdrop-blur-sm rounded-2xl border border-gray-800/50 shadow-xl">
              <div className="text-6xl mb-6">🎬</div>
              <h2 className="text-2xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-indigo-300 via-purple-300 to-pink-300">분석 결과가 없습니다</h2>
              <p className="text-gray-400 mb-6">
                아직 이 토큰에 대한 분석 결과가 없습니다. 먼저 더빙을 진행해주세요.
              </p>
            </div>
          )}
          
          {/* 다시 더빙하기 버튼 추가 */}
          <div className="w-full flex justify-center px-2 md:px-0 mt-8 gap-4">
            <button
              onClick={() => {
                // URL에서 token_id 파라미터 추출
                const urlParams = new URLSearchParams(window.location.search);
                const tokenId = urlParams.get('token_id');
                if (tokenId) {
                  // 더빙 페이지로 이동 (경로 형식: /dubbing/[id])
                  window.location.href = `/dubbing/${tokenId}`;
                }
              }}
              className="w-full max-w-xs px-8 py-4 bg-gradient-to-r from-blue-500 to-green-500 text-white font-semibold rounded-lg shadow-lg hover:from-blue-600 hover:to-green-600 transition-all duration-300 transform hover:scale-105"
            >
              🎙️ 다시 더빙하기
            </button>
            
            {/* 뒤로가기 버튼 추가 */}
            <button
              onClick={() => {
                // 마이페이지로 이동
                window.location.href = '/mypage';
              }}
              className="w-full max-w-xs px-8 py-4 bg-gradient-to-r from-gray-500 to-gray-700 text-white font-semibold rounded-lg shadow-lg hover:from-gray-600 hover:to-gray-800 transition-all duration-300 transform hover:scale-105"
            >
              ↩️ 뒤로가기
            </button>
          </div>
        </div>
      )}
      <ResultViewBtn
        hasAnalysisResults={hasAnalysisResults}
        showResults={showResults}
        showCompleted={showCompleted}
        onViewResults={onViewResults}
      />
    </>
  );
};

export default ResultComponent; 