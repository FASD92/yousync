"use client";

// import ServerPitchGraph from "@/components/graph/ServerPitchGraph";
// import { MyPitchGraph } from '@/components/graph/MyPitchGraph';
import type { Caption } from "@/types/dubbing";
import { VideoPlayerRef } from "./VideoPlayer";
import { useDubbingRecorder } from '@/hooks/useDubbingRecorder';
import { useEffect, useRef, useState, forwardRef, useImperativeHandle } from 'react';
import { useAudioStore } from '@/store/useAudioStore';
import { ScriptItem } from "@/types/pitch";
import { useJobIdsStore } from '@/store/useJobIdsStore';
import LiquidGauge from '@/components/result/LiquidGauge';

interface PitchComparisonProps {
  currentScriptIndex: number;
  captions: Caption[];
  tokenId: string;
  serverPitchData: Array<{ time: number; hz: number | null }>;
  videoPlayerRef?: React.RefObject<VideoPlayerRef | null>;
  onNextScript?: (nextIndex: number) => void;
  onPlay?: () => void;
  onPause?: () => void;
  isVideoPlaying: boolean;
  scripts?: ScriptItem[];
  onUploadComplete?: (success: boolean, jobIds?: string[]) => void;
  onRecordingChange?: (recording: boolean) => void;
  handleRecordingComplete?: () => void;
  showAnalysisResult?: boolean;
  recordingCompleted?: boolean;
  onRecordingPlaybackChange?: (isPlaying: boolean) => void;
  onOpenSidebar?: () => void;
  onShowResults?: () => void;
  onOpenDubbingListenModal?: () => void;
  latestResultByScript?: { [key: string]: { overall_score: number } };
}

const PitchComparison = forwardRef<{ handleExternalStop: () => void; stopLooping?: () => void; handleMicClick: () => void }, PitchComparisonProps>(function PitchComparison({ 
  currentScriptIndex, 
  captions, 
  tokenId, 
  serverPitchData,
  videoPlayerRef,
  onNextScript,
  onPlay,
  onPause,
  isVideoPlaying,
  scripts,
  onUploadComplete,
  onRecordingChange,
  handleRecordingComplete,
  showAnalysisResult = false,
  recordingCompleted = false,
  onRecordingPlaybackChange,
  onOpenSidebar,
  onShowResults,
  onOpenDubbingListenModal,
  latestResultByScript,
}: PitchComparisonProps, ref) {

  const {
    recording,
    recordedScripts,
    uploading,
    startScriptRecording,
    stopScriptRecording,
    getAllBlobs,
  } = useDubbingRecorder({
    captions,
    tokenId,
    scripts,
    onUploadComplete: (success: boolean, jobIds: string[]) => {
      if (success) {
        if (Array.isArray(jobIds)) {
          jobIds.forEach((jobId) => {
          });
        } else {
          console.warn(`[⚠️ 경고] jobIds가 배열이 아님: ${typeof jobIds}`);
        }
      } else {
        console.error(`[❌ 업로드 실패] 문장 ${currentScriptIndex + 1}번 업로드 실패`);
      }
      
      // 상위 컴포넌트로 콜백 전달
      onUploadComplete?.(success, jobIds);
    },
  });

  // zustand 전역 상태 사용
  const setMultiJobIds = useJobIdsStore((state) => state.setMultiJobIds);

  const [volume, setVolume] = useState(0);
  const rafRef = useRef<number | null>(null);
  const [isVideoEnded, setIsVideoEnded] = useState(false);
  const [isLooping, setIsLooping] = useState(false);
  const loopIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  
  // 녹음된 오디오 재생 관련 상태
  const [isPlayingRecording, setIsPlayingRecording] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // 녹음된 오디오 재생 함수
  const playRecording = () => {
    const blobs = getAllBlobs();
    const currentBlob = blobs[currentScriptIndex];
    
    if (!currentBlob) {
      console.warn('[WARN] 현재 문장의 녹음 파일이 없습니다.');
      return;
    }

    // 기존 오디오 정지
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }

    // 새 오디오 생성 및 재생
    const audioUrl = URL.createObjectURL(currentBlob);
    const audio = new Audio(audioUrl);
    audioRef.current = audio;
    
    audio.onended = () => {
      setIsPlayingRecording(false);
      onRecordingPlaybackChange?.(false);
      URL.revokeObjectURL(audioUrl);
    };
    
    audio.onerror = () => {
      setIsPlayingRecording(false);
      onRecordingPlaybackChange?.(false);
      URL.revokeObjectURL(audioUrl);
      console.error('[ERROR] 오디오 재생 실패');
    };

    audio.play().then(() => {
      setIsPlayingRecording(true);
      onRecordingPlaybackChange?.(true);
    }).catch((error) => {
      console.error('[ERROR] 오디오 재생 시작 실패:', error);
      setIsPlayingRecording(false);
      onRecordingPlaybackChange?.(false);
      URL.revokeObjectURL(audioUrl);
    });
  };

  // 오디오 정지 함수
  const stopRecordingPlayback = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    setIsPlayingRecording(false);
    onRecordingPlaybackChange?.(false);
  };

  // 컴포넌트 언마운트 시 오디오 정리
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!recording) {
      setVolume(0);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      return;
    }
    const analyser = useAudioStore.getState().analyser;
    if (!analyser) return;
    const dataArray = new Uint8Array(analyser.fftSize);
    const update = () => {
      analyser.getByteTimeDomainData(dataArray);
      const avg = dataArray.reduce((a, b) => a + Math.abs(b - 128), 0) / dataArray.length;
      setVolume(Math.min(1, (avg / 128) * 2));
      rafRef.current = requestAnimationFrame(update);
    };
    update();
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [recording]);

  useEffect(() => {
    if (typeof onRecordingChange === 'function') {
      onRecordingChange(recording);
    }
  }, [recording, onRecordingChange]);

  const handleMicClick = () => {
    if (videoPlayerRef?.current && captions[currentScriptIndex]) {
      const currentScript = captions[currentScriptIndex];
      
      videoPlayerRef.current.seekTo(currentScript.start_time);
      videoPlayerRef.current.playVideo();
      
      // 영상이 실제로 재생되기 시작할 때까지 대기
      const checkVideoPlaying = () => {
        if (!videoPlayerRef?.current) return;
        
        const currentTime = videoPlayerRef.current.getCurrentTime();
        const targetTime = currentScript.start_time;
        
        // 영상이 목표 시간에 도달했는지 확인 (0.1초 허용 오차)
        if (Math.abs(currentTime - targetTime) < 0.1) {
          startScriptRecording(currentScriptIndex);
          
          if (typeof onNextScript === 'function') {
            onNextScript(currentScriptIndex);
          }
        } else {
          // 아직 재생되지 않았으면 다시 체크
          setTimeout(checkVideoPlaying, 50);
        }
      };
      
      // 100ms 후부터 체크 시작 (브라우저 렉 고려)
      setTimeout(checkVideoPlaying, 100);
    }
  };

  // useImperativeHandle remains for external stop
  useImperativeHandle(ref, () => ({
    handleExternalStop: () => {
      console.log('[DEBUG][handleExternalStop] 외부에서 녹음 중지 요청');
      stopScriptRecording(currentScriptIndex);
    },
    stopLooping,
    handleMicClick, // 마이크 버튼 클릭 함수 추가
  }));

  // 녹음 상태 변경 감지하여 완료 시 handleRecordingComplete 호출
  useEffect(() => {
    // 녹음이 true에서 false로 변경되었을 때만 (녹음 완료)
    if (prevRecordingRef.current === true && recording === false && handleRecordingComplete) {
      console.log('[DEBUG][PitchComparison] 녹음 완료 감지, handleRecordingComplete 호출');
      handleRecordingComplete();
    }
  }, [recording, handleRecordingComplete]);

  // allRecorded, uploadAllRecordings 관련 useEffect 완전 제거

  const handleNextScript = () => {
    if (!captions || captions.length === 0) return;
    const nextIndex = Math.min(currentScriptIndex + 1, captions.length - 1);
    if (nextIndex !== currentScriptIndex && onNextScript) {
      onNextScript(nextIndex);
      // 영상도 다음 문장 시작으로 이동 및 재생
      if (videoPlayerRef?.current) {
        videoPlayerRef.current.seekTo(captions[nextIndex].start_time);
        videoPlayerRef.current.playVideo();
      }
    }
  };

  const handlePrevScript = () => {
    if (!captions || captions.length === 0) return;
    const prevIndex = Math.max(currentScriptIndex - 1, 0);
    if (prevIndex !== currentScriptIndex && onNextScript) {
      onNextScript(prevIndex);
      // 영상도 이전 문장 시작으로 이동 및 재생
      if (videoPlayerRef?.current) {
        videoPlayerRef.current.seekTo(captions[prevIndex].start_time);
        videoPlayerRef.current.playVideo();
      }
    }
  };

  const getCurrentScriptPlaybackRange = () => {
    if (!captions || captions.length === 0) {
      return { startTime: 0, endTime: undefined };
    }
    const currentScript = captions[currentScriptIndex];
    return {
      startTime: currentScript?.start_time || 0,
      endTime: currentScript?.end_time || undefined,
    };
  };

  // recording 값이 바뀔 때만 로그 출력
  const prevRecordingRef = useRef(recording);
  useEffect(() => {
    if (prevRecordingRef.current !== recording) {
      prevRecordingRef.current = recording;
    }
  }, [recording]);

  // 영상 상태 추적
  useEffect(() => {
    if (!videoPlayerRef?.current) return;
    let interval: ReturnType<typeof setInterval> | null = null;
    const checkState = () => {
      if (!videoPlayerRef.current) return;
      const currentTime = videoPlayerRef.current.getCurrentTime();
      const endTime = captions[currentScriptIndex]?.end_time;
      // 영상이 끝났는지 체크
      if (endTime !== undefined && currentTime >= endTime - 0.1) {
        setIsVideoEnded(true);
      } else {
        setIsVideoEnded(false);
      }
    };
    interval = setInterval(() => {
      if (!videoPlayerRef.current) return;
      // YouTube API의 getPlayerState가 있으면 더 정확하게 체크 가능
      if (videoPlayerRef.current.getCurrentTime) {
        const currentTime = videoPlayerRef.current.getCurrentTime();
        const endTime = captions[currentScriptIndex]?.end_time;
        if (endTime !== undefined && currentTime >= endTime - 0.1) {
          setIsVideoEnded(true);
        } else {
          setIsVideoEnded(false);
        }
      }
    }, 200);
    return () => { if (interval) clearInterval(interval); };
  }, [videoPlayerRef, currentScriptIndex, captions]);

  // 영상 play/pause 이벤트 핸들러 필요시 추가
  const handlePlay = () => {
    if (onPlay) onPlay();
  };
  const handlePause = () => {
    if (onPause) onPause();
  };

  const handleLoopToggle = () => {
    if (isLooping) {
      setIsLooping(false);
      if (loopIntervalRef.current) {
        clearInterval(loopIntervalRef.current);
        loopIntervalRef.current = null;
      }
    } else {
      setIsLooping(true);
      // 구간 반복 감시 시작
      loopIntervalRef.current = setInterval(() => {
        if (!videoPlayerRef?.current) return;
        const currentTime = videoPlayerRef.current.getCurrentTime();
        const startTime = captions[currentScriptIndex]?.start_time || 0;
        const endTime = captions[currentScriptIndex]?.end_time;
        if (endTime !== undefined && currentTime >= endTime - 0.1) {
          videoPlayerRef.current.seekTo(startTime);
          videoPlayerRef.current.playVideo();
        }
      }, 200);
    }
  };

  useEffect(() => {
    // 문장 인덱스가 바뀌면 반복 해제
    setIsLooping(false);
    if (loopIntervalRef.current) {
      clearInterval(loopIntervalRef.current);
      loopIntervalRef.current = null;
    }
  }, [currentScriptIndex]);

  useEffect(() => {
    // 컴포넌트 언마운트 시 반복 해제
    return () => {
      if (loopIntervalRef.current) {
        clearInterval(loopIntervalRef.current);
      }
    };
  }, []);

  const stopLooping = () => {
    setIsLooping(false);
    if (loopIntervalRef.current) {
      clearInterval(loopIntervalRef.current);
      loopIntervalRef.current = null;
    }
  };

  // 게이지에 분석 점수 반영 (진행률 × 평균 점수)
  const total = captions.length;
  const analyzedScores = Object.entries(latestResultByScript ?? {})
    .map(([_, v]) => v?.overall_score ?? 0)
    .filter((v) => typeof v === 'number' && !isNaN(v));
  const analyzedCount = analyzedScores.length;
  const avgScore = analyzedCount > 0 ? analyzedScores.reduce((a, b) => a + b, 0) / analyzedCount : 0;
  const percent = total > 0 ? Math.round((analyzedCount / total) * avgScore * 100) : 0;
  


  return (
    <div className="bg-gradient-to-br from-gray-900 to-gray-950 rounded-xl p-2 h-auto relative border border-gray-800 shadow-lg">
      <div className="flex justify-between items-center mb-1">
        <h3 className="text-xs font-semibold text-white">Pitch Comparison</h3>
      </div>
      <div className="space-y-1">
        <div className="w-full h-16 flex justify-start items-center">
          <LiquidGauge value={percent} size={50} />
        </div>
      </div>
    </div>
  );
});

export default PitchComparison; 