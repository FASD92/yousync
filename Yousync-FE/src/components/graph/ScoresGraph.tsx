import React from 'react';

export const ScoresGraph = () => {
  // 예시 값 (0~100)
  const gaugeValue = 75;
  // 게이지 각도 계산 (반원: -90~+90도)
  const angle = (gaugeValue / 100) * 180 - 90;

  return (
    <div className="flex items-center gap-4 px-4 py-2 bg-gray-800 rounded-md w-fit text-white text-sm">
      {/* 계기판 게이지 */}
      <div className="flex flex-col items-center w-[80px]">
        <svg width="80" height="48" viewBox="0 0 80 48">
          {/* 배경 반원 */}
          <path
            d="M 8 40 A 32 32 0 0 1 72 40"
            fill="none"
            stroke="#374151"
            strokeWidth="8"
          />
          {/* 진행 반원 */}
          <path
            d="M 8 40 A 32 32 0 0 1 72 40"
            fill="none"
            stroke="#3b82f6"
            strokeWidth="8"
            strokeDasharray="100"
            strokeDashoffset={100 - (gaugeValue * 100) / 100}
            style={{
              strokeDasharray: 100,
              strokeDashoffset: 100 - (gaugeValue * 100) / 100,
            }}
          />
          {/* 바늘 */}
          <g>
            <line
              x1="40"
              y1="40"
              x2={40 + 28 * Math.cos((angle * Math.PI) / 180)}
              y2={40 + 28 * Math.sin((angle * Math.PI) / 180)}
              stroke="#facc15"
              strokeWidth="3"
              strokeLinecap="round"
            />
            {/* 중앙 원 */}
            <circle cx="40" cy="40" r="4" fill="#facc15" />
          </g>
          {/* 중앙 숫자 */}
          <text
            x="40"
            y="30"
            textAnchor="middle"
            fontSize="16"
            fill="#fff"
            fontWeight="bold"
          >
            
          </text>
        </svg>
        <span className="text-[10px] text-gray-400 mt-1">실시간 피치</span>
      </div>

      {/* 피치정확도 */}
      <div className="flex flex-col w-[100px]">
        <span className="text-[11px] text-gray-300">피치 정확도</span>
        <div className="w-full bg-gray-600 h-2 rounded">
          <div className="h-full bg-blue-500 rounded" style={{ width: '75%' }}></div>
        </div>
        <span className="text-[10px] text-right text-gray-400 mt-1">75%</span>
      </div>

      {/* 발음정확도 */}
      <div className="flex flex-col w-[100px]">
        <span className="text-[11px] text-gray-300">발음 정확도</span>
        <div className="w-full bg-gray-600 h-2 rounded">
          <div className="h-full bg-green-500 rounded" style={{ width: '90%' }}></div>
        </div>
        <span className="text-[10px] text-right text-gray-400 mt-1">90%</span>
      </div>
    </div>
  )
}