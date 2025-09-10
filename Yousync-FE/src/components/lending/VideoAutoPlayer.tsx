import React, { useRef, useEffect, useState } from "react";
import { gsap } from "gsap";

const videos = [
  "/lending/mp4/kingsman_002.mp4",
  "/lending/mp4/taken_002.mp4",
  "/lending/mp4/taken_003.mp4",
];

const captions = [
  '"Do you know what that means?"',
  '"i dont know what you want "',
  '"I will find you, and I will kill you"',
];

interface VideoAutoPlayerProps {
  onComplete?: () => void;
}

export default function VideoAutoPlayer({ onComplete }: VideoAutoPlayerProps) {
  const [index, setIndex] = useState(0);
  const [fade, setFade] = useState(true);
  const [showFinalMessage, setShowFinalMessage] = useState(false);
  const [showPlayBtn, setShowPlayBtn] = useState(true);   // 버튼 보여줄지 여부
  const [videoReady, setVideoReady] = useState(false);    // 영상 시작여부

  const playBtnRef = useRef<HTMLButtonElement | null>(null);
  const playIconRef = useRef<SVGPolylineElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // GSAP: SVG 그려지는 애니메이션
  useEffect(() => {
    if (showPlayBtn && playIconRef.current) {
      gsap.set(playIconRef.current, { strokeDasharray: 180, strokeDashoffset: 180, opacity: 1 });
      gsap.to(playIconRef.current, {
        strokeDashoffset: 0,
        duration: 1.1,
        ease: "power1.inOut"
      });
    }
  }, [showPlayBtn]);

  // 클릭 시 애니메이션(커지면서 사라짐)+영상 준비
  const handlePlayClick = () => {
    if (playBtnRef.current) {
      gsap.to(playBtnRef.current, {
        scale: 2.2,
        opacity: 0,
        duration: 2.5,
        ease: "power2.in",
        onComplete: () => {
          setShowPlayBtn(false);
          setTimeout(() => setVideoReady(true), 200); // 자연스러운 딜레이 후 영상 시작
        }
      });
    }
  };

  useEffect(() => {
    if (videoReady && videoRef.current) {
      // autoplay + 소리 ON
      videoRef.current.muted = false;
      videoRef.current.play();
    }
  }, [videoReady, index]);

  useEffect(() => {
    if (showFinalMessage) {
      // 스크롤을 막지 않도록 수정
      // document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "auto";
    }
  }, [showFinalMessage]);

  const handleEnded = () => {
    if (index < videos.length - 1) {
      setFade(false);
      setTimeout(() => {
        setIndex((prev) => prev + 1);
        setFade(true);
      }, 300);
    } else {
      setShowFinalMessage(true);
      // 영상 완료 후 3초 뒤에 onComplete 콜백 호출
      setTimeout(() => {
        onComplete?.();
      }, 3000);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#001210] from-55% to-[#122a17] to-100% relative">
      {/* ─── 인트로 재생버튼 GSAP 애니메이션 (클릭 필요) ─── */}
      {showPlayBtn && (
        <div className="w-screen h-screen fixed left-0 top-0 z-30 flex items-center justify-center bg-black/98">
          <button
            className="w-40 h-40 bg-gradient-radial from-[#101c13] from-74% to-[#022f18] to-100% rounded-full flex items-center justify-center shadow-[0_8px_32px_#1dff8165,0_1px_4px_#001a13] drop-shadow-[0_0_24px_#2dff8c30] scale-100 opacity-100 transition-all duration-[0.35s] md:w-24 md:h-24"
            ref={playBtnRef}
            tabIndex={0}
            aria-label="영상 재생"
            onClick={handlePlayClick}
            style={{
              border: "none",
              background: "transparent",
              cursor: "pointer",
              outline: "none",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <svg width="120" height="120" viewBox="0 0 120 120">
              <circle
                cx="60" cy="60" r="58"
                stroke="#2dff8c" strokeWidth="4"
                fill="rgba(0,0,0,0.12)"
              />
              <polyline
                ref={playIconRef}
                points="50,45 50,75 78,60 50,45"
                fill="none"
                stroke="#39ff14"
                strokeWidth="7"
                strokeLinejoin="round"
                strokeLinecap="round"
                style={{
                  filter: "drop-shadow(0 0 14px #39ff14cc)"
                }}
              />
            </svg>
          </button>
        </div>
      )}

      {/* ─── 영상/자막 시퀀스 (영상+자막, 소리 O) ─── */}
      {videoReady && !showFinalMessage && index < videos.length && (
        <>
          <video
            ref={videoRef}
            key={videos[index]}
            src={videos[index]}
            autoPlay
            onEnded={handleEnded}
            className="w-screen h-screen object-cover fixed top-0 left-0 z-[1] transition-opacity duration-300"
            style={{
              opacity: fade ? 1 : 0,
            }}
          />
          <div className="fixed bottom-[10%] w-screen z-[2] flex flex-col items-center justify-center pointer-events-none">
            <span className="text-white text-3xl drop-shadow-[0_2px_8px_#000] pointer-events-auto block text-center">
              {captions[index]}
            </span>
          </div>
        </>
      )}

      {/* ─── 영상 끝 메시지 ─── */}
      {showFinalMessage && (
        <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-auto z-20 flex flex-col items-center justify-center pointer-events-none">
          <span
            className="text-[#39ff14] text-[2.5rem] font-bold drop-shadow-[0_0_8px_#39ff14,0_0_24px_#39ff14,0_0_48px_#39ff14] bg-black/70 px-12 py-8 rounded-[2rem] animate-[growTextBig_2s_cubic-bezier(0.23,1.01,0.32,1)_forwards] block whitespace-nowrap text-center m-0"
          >
            Now it&apos;s your turn
          </span>
        </div>
      )}

      <style>{`
        @keyframes growTextBig {
          0% {
            transform: scale(0.3);
            font-size: 2.5rem;
            opacity: 0.2;
          }
          100% {
            transform: scale(1.2);
            font-size: 7vw;
            opacity: 1;
          }
        }
      `}</style>
    </div>
  );
}
