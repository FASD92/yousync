import { useEffect, useState, useMemo } from "react";
import axios from "axios";
import { Caption } from "@/types/caption";

interface CaptionState {
  currentIdx: number;
  captions: Caption[];
}

interface ServerPitch {
  time: number;
  hz: number | null;
}

interface TokenDetailResponse {
  pitch: ServerPitch[];
}

interface ServerPitchGraphProps {
  captionState: CaptionState;
  token_id: number | string | undefined | null;
  serverPitchData?: ServerPitch[];
}

export default function ServerPitchGraph({
  captionState = { currentIdx: 0, captions: [] },
  token_id,
  serverPitchData,
}: ServerPitchGraphProps) {
  const { currentIdx, captions } = captionState;
  const [pitch, setPitch] = useState<ServerPitch[]>([]);
  const [error, setError] = useState<string | null>(null);

  // token_id를 문자열로 변환하여 항상 API에서 쓸 수 있게
  const token_id_str =
    typeof token_id === "string" ? token_id : String(token_id);

  useEffect(() => {
    // serverPitchData가 있으면 그것을 사용, 없으면 API에서 가져오기
    if (serverPitchData && serverPitchData.length > 0) {
      setPitch(serverPitchData);
      return;
    }
    
    // token_id가 undefined, null, 빈문자열 등 falsy면 요청하지 않음
    if (!token_id) return;
    const fetchData = async () => {
      try {
        setError(null);
        const res = await axios.get<TokenDetailResponse>(
          `${process.env.NEXT_PUBLIC_API_BASE_URL}/tokens/${token_id_str}`
        );
        setPitch(res.data.pitch || []);
      } catch (err: unknown) {
        setError("서버 피치 데이터 불러오기 에러");
        setPitch([]);
        if (err && typeof err === 'object' && 'response' in err) {
          const axiosError = err as { response: { status: number; data: unknown } };
          console.error("서버 에러 상태코드:", axiosError.response.status, axiosError.response.data);
        } else {
          console.error("네트워크 또는 기타 에러:", err);
        }
      }
    };
    fetchData();
  }, [token_id_str, token_id, serverPitchData]);

  const filteredData = useMemo(() => {
    if (!captions.length || pitch.length === 0) return [];
    const currentCaption = captions[currentIdx];
    if (!currentCaption) return [];
    return pitch
      .filter(
        (p) =>
          p.hz !== null &&
          p.time >= currentCaption.start_time &&
          p.time <= currentCaption.end_time
      )
      .map((p) => ({
        x: p.time - currentCaption.start_time,
        y: p.hz !== null && p.hz !== undefined ? p.hz : 0,
      }));
  }, [pitch, captions, currentIdx]);

  // y값 스케일링 (최소~최대 정규화)
  const yValues = filteredData.map(d => d.y);
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);
  const getY = (y: number) => {
    // 유효하지 않은 값 체크
    if (!y || isNaN(y) || !isFinite(y)) return 20;
    
    if (maxY === minY) return 20; // flat line
    
    const scaledY = 40 - ((y - minY) / (maxY - minY)) * 40;
    
    // 결과값이 유효한지 체크
    if (isNaN(scaledY) || !isFinite(scaledY)) return 20;
    
    return Math.max(0, Math.min(40, scaledY)); // 0-40 범위로 제한
  };

  if (error) {
    return (
      <div className="w-full h-16 flex items-center justify-center text-red-400 text-sm font-semibold">
        {error}
      </div>
    );
  }

  // 유효한 데이터만 필터링
  const validFilteredData = filteredData.filter(point => 
    point && 
    typeof point.y === 'number' && 
    !isNaN(point.y) && 
    isFinite(point.y)
  );

  // 데이터가 없으면 빈 그래프 표시
  if (validFilteredData.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center text-gray-400 text-sm">
        피치 데이터 없음
      </div>
    );
  }

  return (
    <div className="w-full h-full relative">
      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 40" preserveAspectRatio="none">
        <defs>
          <linearGradient id="pitchGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#10B981" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#10B981" stopOpacity="0.2" />
          </linearGradient>
        </defs>
        <path
          d={`M 0,${getY(validFilteredData[0].y)} ${validFilteredData.map((point, index) => 
            `L ${(index / (validFilteredData.length - 1)) * 100},${getY(point.y)}`
          ).join(' ')}`}
          stroke="#10B981"
          strokeWidth="2"
          fill="none"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d={`M 0,${getY(validFilteredData[0].y)} ${validFilteredData.map((point, index) => 
            `L ${(index / (validFilteredData.length - 1)) * 100},${getY(point.y)}`
          ).join(' ')} L 100,40 L 0,40 Z`}
          fill="url(#pitchGradient)"
        />
      </svg>
    </div>
  );
}