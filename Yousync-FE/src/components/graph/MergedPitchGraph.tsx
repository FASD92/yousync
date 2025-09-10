import React, { useMemo } from "react";
import { Caption } from "@/types/caption";

interface PitchPoint {
  time: number;
  hz: number | null;
}

interface MergedPitchGraphProps {
  myPitchData: PitchPoint[];
  serverPitchData: PitchPoint[];
  captionState: { currentIdx: number; captions: Caption[] };
}

export default function MergedPitchGraph({ myPitchData = [], serverPitchData = [], captionState }: MergedPitchGraphProps) {
  const { currentIdx, captions } = captionState;
  const currentCaption = captions[currentIdx];

  // 현재 스크립트 구간에 해당하는 데이터만 필터링
  const filterByCurrent = (data: PitchPoint[] | undefined | null) => {
    if (!Array.isArray(data) || !currentCaption) {
      // 최소 2개 점이 있어야 path가 그려지므로, 0값 dummy 데이터 반환
      return [
        { time: 0, hz: 0 },
        { time: 1, hz: 0 }
      ];
    }
    return data.filter(
      (p) =>
        p.hz !== null &&
        p.time >= currentCaption.start_time &&
        p.time <= currentCaption.end_time
    );
  };

  // MyPitchGraph, ServerPitchGraph 스타일의 데이터 변환
  const myData = useMemo(() => filterByCurrent(myPitchData).map((p, i, arr) => ({
    x: p.time - (arr[0]?.time ?? 0),
    y: p.hz ?? 0
  })), [myPitchData, currentCaption]);
  const serverData = useMemo(() => filterByCurrent(serverPitchData).map((p, i, arr) => ({
    x: p.time - (arr[0]?.time ?? 0),
    y: p.hz ?? 0
  })), [serverPitchData, currentCaption]);

  // y값 스케일링 (두 곡선 모두 포함, MyPitchGraph/ServerPitchGraph 방식)
  const allY = [...myData, ...serverData].map((d) => d.y);
  const minY = Math.min(...allY, 80); // 80Hz 이하로 안내려감
  const maxY = Math.max(...allY, 1000); // 1000Hz 이상으로 안올라감
  const getY = (y: number) => {
    if (!y || isNaN(y) || !isFinite(y)) return 20;
    if (maxY === minY) return 20;
    const scaledY = 200 - ((y - minY) / (maxY - minY)) * 200;
    if (isNaN(scaledY) || !isFinite(scaledY)) return 20;
    return Math.max(0, Math.min(200, scaledY)); // 0~200(px) SVG 높이 기준
  };

  // path 생성 함수 (캔버스 느낌의 라인)
  const makePath = (data: {x: number, y: number}[]) => {
    if (!data.length) return '';
    return `M 0,${getY(data[0].y)} ` +
      data.map((point, idx) => `L ${(idx / (data.length - 1)) * 600},${getY(point.y)}`).join(' ');
  };

  // 데이터가 없으면 안내 메시지
  if (!myData.length && !serverData.length) {
    return (
      <div className="w-full h-full flex items-center justify-center text-gray-400 text-sm">
        피치 데이터 없음
      </div>
    );
  }

  return (
    <svg
      width={600}
      height={200}
      style={{
        width: '100%',
        height: '200px',
        border: '1px solid #444',
        borderRadius: 4,
        background: '#111',
        display: 'block',
      }}
      viewBox="0 0 600 200"
      preserveAspectRatio="none"
    >
      {/* 배경 그리드/눈금선 */}
      <g>
        {[0, 0.25, 0.5, 0.75, 1].map((t, i) => (
          <line
            key={i}
            x1={0}
            x2={600}
            y1={200 * t}
            y2={200 * t}
            stroke="#222"
            strokeWidth={i === 0.5 ? 2 : 1}
          />
        ))}
      </g>
      {/* 원본 곡선 (파랑) */}
      {serverData.length > 1 && (
        <path
          d={makePath(serverData)}
          stroke="deepskyblue"
          strokeWidth="2.5"
          fill="none"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      )}
      {/* 내 곡선 (토마토) */}
      {myData.length > 1 && (
        <path
          d={makePath(myData)}
          stroke="tomato"
          strokeWidth="2.5"
          fill="none"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      )}
    </svg>
  );
} 