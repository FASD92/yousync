import React, { useState } from "react";

// 커스텀 훅
type StatusType = "idle" | "pending" | "processing" | "success" | "failed";

interface YoutubeProcessResult {
  jobId: string | null;
  status: StatusType;
  progress: number; // 0~100
  data: any;        // 처리 결과 데이터
  error: string | null;
  start: (url: string) => Promise<void>;
  reset: () => void;
}

export function useYoutubeProcess(apiBaseUrl: string): YoutubeProcessResult {
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<StatusType>("idle");
  const [progress, setProgress] = useState(0);
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const progressInterval = React.useRef<NodeJS.Timeout | null>(null);
  const statusInterval = React.useRef<NodeJS.Timeout | null>(null);

  // 유튜브 처리 시작
  const start = React.useCallback(async (url: string) => {
    setStatus("pending");
    setProgress(0);
    setError(null);
    setData(null);
    setJobId(null);
    try {
      const res = await fetch(`${apiBaseUrl}/youtube/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      if (!res.ok) throw new Error("요청 실패");
      const { job_id } = await res.json();
      setJobId(job_id);
      setStatus("processing");
    } catch (e: any) {
      setError(e?.message || "처리 시작 실패");
      setStatus("failed");
    }
  }, [apiBaseUrl]);

  // 상태/진행률 폴링
  React.useEffect(() => {
    if (!jobId) return;

    setStatus("processing");
    // 진행률 폴링
    progressInterval.current = setInterval(async () => {
      try {
        const res = await fetch(`${apiBaseUrl}/youtube/process-progress/${jobId}`);
        if (!res.ok) return;
        const json = await res.json();
        setProgress(json.progress ?? 0);
      } catch { }
    }, 1000);

    // 상태 폴링
    statusInterval.current = setInterval(async () => {
      try {
        const res = await fetch(`${apiBaseUrl}/youtube/process-status/${jobId}`);
        if (!res.ok) return;
        const json = await res.json();
        if (json.status === "success") {
          setStatus("success");
          setData(json.data);
        }
        if (json.status === "failed") {
          setStatus("failed");
          setError(json.error || "처리 실패");
        }
      } catch { }
    }, 2000);

    // 클린업
    return () => {
      if (progressInterval.current) clearInterval(progressInterval.current);
      if (statusInterval.current) clearInterval(statusInterval.current);
    };
  }, [jobId, apiBaseUrl]);

  // 완료/실패시 폴링 정지
  React.useEffect(() => {
    if (status === "success" || status === "failed") {
      if (progressInterval.current) clearInterval(progressInterval.current);
      if (statusInterval.current) clearInterval(statusInterval.current);
    }
  }, [status]);

  // 초기화 함수
  const reset = () => {
    setJobId(null);
    setStatus("idle");
    setProgress(0);
    setData(null);
    setError(null);
  };

  return { jobId, status, progress, data, error, start, reset };
}
