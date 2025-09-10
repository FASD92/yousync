"use client";
import React from "react";

interface ResultViewBtnProps {
  hasAnalysisResults: boolean;
  showResults: boolean;
  showCompleted: boolean;
  onViewResults?: () => void;
}

const ResultViewBtn: React.FC<ResultViewBtnProps> = ({
  hasAnalysisResults,
  showResults,
  showCompleted,
  onViewResults
}) => {
  const isBtnVisible = hasAnalysisResults && !showResults && !showCompleted;

  if (!isBtnVisible) return null;

  return (
    <div className="w-full flex justify-center px-2 md:px-0 mt-8">
      <button
        onClick={onViewResults}
        className="w-full max-w-xs px-8 py-4 bg-gradient-to-r from-green-500 to-blue-500 text-white font-semibold rounded-lg shadow-lg hover:from-green-600 hover:to-blue-600 transition-all duration-300 transform hover:scale-105"
      >
        📊 분석 결과 조회
      </button>
    </div>
  );
};

export default ResultViewBtn; 