import React from "react";
import ResultComponent from "./ResultComponent";
import ScoreCards from "./ScoreCards";
import SentenceAnalysis from "./SentenceAnalysis";

// 필요한 props 타입을 ResultComponent에서 재사용
import type { TestResultAnalysisSectionProps } from "./ResultComponent";

const ResultContainer: React.FC<TestResultAnalysisSectionProps> = (props) => {
  // 추후 ScoreCards, SentenceAnalysis 등 개별적으로도 쓸 수 있도록 구조화
  return (
    <div className="w-full">
      {/* 전체 결과(그래프, 카드, 분석 등) */}
      <ResultComponent {...props} />
      {/* 필요시 개별 컴포넌트도 아래처럼 추가 가능 */}
      {/* <ScoreCards ... /> */}
      {/* <SentenceAnalysis ... /> */}
    </div>
  );
};

export default ResultContainer; 