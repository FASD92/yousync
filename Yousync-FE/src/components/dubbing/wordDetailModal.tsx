import React from "react";
import ReactDOM from "react-dom";

interface WordDetailModalProps {
  isOpen: boolean;
  onClose: () => void;
  word: {
    word: string;
    word_score: number;
    pronunciation_score?: number;
    fluency_score?: number;
    timing_score?: number;
  } | null;
  userSTT?: {
    text: string;
    word_timestamps: Array<{
      word: string;
      start: number;
      end: number;
    }>;
  };
}

export default function WordDetailModal({ isOpen, onClose, word, userSTT }: WordDetailModalProps) {
  if (!isOpen || !word) return null;

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

  const getScoreLabel = (score: number) => {
    if (score >= 0.9) return "매우 좋음";
    if (score >= 0.8) return "좋음";
    if (score >= 0.7) return "보통";
    if (score >= 0.6) return "개선 필요";
    return "많이 개선 필요";
  };

  const modalContent = (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-gray-900 rounded-2xl p-6 max-w-md w-full mx-4 shadow-2xl border border-gray-700 flex flex-col items-center justify-center">
        {/* 헤더 */}
        <div className="flex items-center justify-between mb-6 w-full">
          <h2 className="text-2xl font-bold text-white text-center w-full">단어 상세 분석</h2>
          <button
            onClick={onClose}
            className="absolute top-6 right-6 text-gray-400 hover:text-white transition-colors duration-200"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* 단어 */}
        <div className="text-center mb-6 w-full">
          <div className="text-4xl font-bold text-emerald-300 mb-2">
            "{word.word.replace(/,$/, "")}"
          </div>
        
        </div>

        {/* 전체 점수 - 중앙 강조 */}
        <div className="mb-8 w-full flex flex-col items-center justify-center">
          <div className="flex items-center justify-center mb-2 w-full">
            <span className="text-white font-medium text-lg mr-2">전체 유사도</span>
            <span className="text-emerald-300 font-bold text-2xl">
              {Math.round(word.word_score * 100)}%
            </span>
          </div>
          <div className="w-3/4 h-5 bg-gray-700 rounded-full overflow-hidden mb-2">
            <div
              className="h-full transition-all duration-500 ease-out"
              style={{
                width: `${Math.round(word.word_score * 100)}%`,
                backgroundColor: getGradientColor(word.word_score)
              }}
            />
          </div>
          <div className="text-center mt-2">
            <span className="text-lg text-gray-400 font-semibold">
              {getScoreLabel(word.word_score)}
            </span>
          </div>
        </div>

        {/* 세부 점수들 */}
        {(word.pronunciation_score !== undefined || word.fluency_score !== undefined || word.timing_score !== undefined) && (
          <div className="space-y-4 w-full">
            <h3 className="text-lg font-semibold text-white mb-3 text-center w-full">세부 분석</h3>
            
            {word.pronunciation_score !== undefined && (
              <div className="w-full">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-gray-300 text-sm">전체 유사도</span>
                  <span className="text-emerald-300 font-medium">
                    {Math.round(word.pronunciation_score * 100)}%
                  </span>
                </div>
                <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full transition-all duration-500 ease-out"
                    style={{
                      width: `${Math.round(word.pronunciation_score * 100)}%`,
                      backgroundColor: getGradientColor(word.pronunciation_score)
                    }}
                  />
                </div>
              </div>
            )}

            {word.fluency_score !== undefined && (
              <div className="w-full">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-gray-300 text-sm">유창성</span>
                  <span className="text-emerald-300 font-medium">
                    {Math.round(word.fluency_score * 100)}%
                  </span>
                </div>
                <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full transition-all duration-500 ease-out"
                    style={{
                      width: `${Math.round(word.fluency_score * 100)}%`,
                      backgroundColor: getGradientColor(word.fluency_score)
                    }}
                  />
                </div>
              </div>
            )}

            {word.timing_score !== undefined && (
              <div className="w-full">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-gray-300 text-sm">타이밍</span>
                  <span className="text-emerald-300 font-medium">
                    {Math.round(word.timing_score * 100)}%
                  </span>
                </div>
                <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full transition-all duration-500 ease-out"
                    style={{
                      width: `${Math.round(word.timing_score * 100)}%`,
                      backgroundColor: getGradientColor(word.timing_score)
                    }}
                  />
                </div>
              </div>
            )}
          </div>
        )}

        {/* 닫기 버튼 */}
        <div className="mt-8 text-center w-full">
          <button
            onClick={onClose}
            className="px-6 py-2 bg-emerald-600 hover:bg-emerald-700 text-white font-medium rounded-lg transition-colors duration-200"
          >
            닫기
          </button>
        </div>
      </div>
    </div>
  );

  if (typeof window !== "undefined") {
    return ReactDOM.createPortal(modalContent, document.body);
  }
  return null;
} 