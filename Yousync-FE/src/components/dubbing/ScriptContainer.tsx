import { useEffect, useRef, useState } from "react";
import { Caption } from "@/types/caption";
import { delayPlay } from "@/utils/delayPlay";
import {motion, AnimatePresence} from "framer-motion";
import React from "react";
import { Timer } from "@/components/Timer";

interface Props {
  caption: Caption;
  currentIdx: number;
  playerRef: React.RefObject<YT.Player | null>;
  startRecording: () => void;
  stopRecording: (idx: number) => void;
}

const ScriptContainerComponent = ({
  caption,
  currentIdx,
  playerRef,
  startRecording,
  stopRecording,
}: Props) => {
  const [highlightEnabled, setHighlightEnabled] = useState(true);
  const [gaugeProgress, setGaugeProgress] = useState(0);
  const [showTimer, setShowTimer] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [prevIdx, setPrevIdx] = useState<number | null>(null);
  const [countDown, setCountDown] = useState<number | null>(null);
  const [tmp, setTmp] = useState<number | null> (null);

  // 반드시 필요
  const [displayedCaption, setDisplayedCaption] = useState<Caption>(caption);

  // 1. 타이머 끼워넣기 -> 첫대사 버그 빼고 완료
  // 2. 녹음 끼워넣기
  // 3. 두번째 대사에 초록색 칠해지게
  useEffect(() => {
    const duration = caption.end_time - caption.start_time;
    let delayTimeout: NodeJS.Timeout | undefined; // 여기서 선언!
    let recordingTimeout: NodeJS.Timeout | undefined; // 여기서 선언!
    // duration 후에 구간 시작으로 이동 + 3초 멈춤 + 다시 재생
    const timeOut = setTimeout(() => {
      playerRef.current?.seekTo(caption.start_time, true);
      // playerRef.current?.pauseVideo(); // 자동 정지 비활성화
  
      // 3초 후에 다시 재생
      setShowTimer(true);
      

      const delayTimeout = setTimeout(() => {
        playerRef.current?.playVideo();
        setShowTimer(false);
        startRecording();
        console.log(">>녹음 시작")

      }, 3000);
      
      // cleanup에서 delayTimeout도 정리
      recordingTimeout = setTimeout(async () => {
        stopRecord();

        
      }, duration * 1000)

      const stopRecord = async () => {
        await stopRecording(currentIdx);
        console.log(">>녹음 종료")

      }
    }, duration * 1000 - 100);
  
    // cleanup
    return () => {
      clearTimeout(timeOut);
      if(delayTimeout) clearTimeout(delayTimeout);
    };
  }, [currentIdx]);
  return (
    <>
   {displayedCaption.script}
   {showTimer && <Timer />}
    </>
  );
};

export const ScriptContainer = React.memo(
  ScriptContainerComponent,
  (prevProps, nextProps) => prevProps.currentIdx === nextProps.currentIdx
);