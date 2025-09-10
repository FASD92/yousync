import React, { useRef, useEffect, useState } from "react";
import { gsap } from "gsap";

const videos = [
  "/kingsman_002.mp4",
  "/taken_002.mp4",
  "/taken_003.mp4",
];

const captions = [
  "“I don't know who you are”",
  "“I'll be back”",
  "“Manners maketh man”",
  "“This is not that kind of movie”",
  "“Good luck”",
  "“I will find you, and I will kill you”",
];

interface LanderSequenceProps {
  onEnd?: () => void;
}

export default function LanderSequence({ onEnd }: LanderSequenceProps) {
  // step: 0=대기(재생버튼), 1=버튼애니, 3=영상, 4=Now
  const [step, setStep] = useState(0);
  const [index, setIndex] = useState(0);
  const [fade, setFade] = useState(true);

  const playBtnRef = useRef(null);
  const playIconRef = useRef(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // 재생 버튼 클릭 핸들러
  const handlePlayClick = () => {
    setStep(3); // 영상 재생 단계로 이동
  };

  // 1. 재생버튼 SVG 애니메이션
  useEffect(() => {
    if (step === 0 && playIconRef.current) {
      gsap.set(playIconRef.current, { strokeDasharray: 180, strokeDashoffset: 180, opacity: 1 });
      gsap.to(playIconRef.current, {
        strokeDashoffset: 0,
        duration: 1.1,
        ease: "power1.inOut"
      });
    }
  }, [step]);


  // 3. 영상+스크립트 재생
  useEffect(() => {
    if (step === 3 && videoRef.current) {
      videoRef.current.muted = false;
      videoRef.current.play();
    }
  }, [step, index]);

  // 영상 종료 시 다음 영상 or Now 메시지
  const handleEnded = () => {
    if (index < videos.length - 1) {
      setFade(false);
      setTimeout(() => {
        setIndex((prev) => prev + 1);
        setFade(true);
      }, 300);
    } else {
      setStep(4); // 마지막 영상 끝나면 Now 메시지로
    }
  };

  // 4. Now it's your turn 메시지 → 애니메이션 끝나면 onEnd 콜백 호출
  useEffect(() => {
    if (step === 4) {
      document.body.style.overflow = "hidden";
      const timer = setTimeout(() => {
        if (onEnd) onEnd();
      }, 2200); // 메시지 애니메이션 후 바로 onEnd
      return () => clearTimeout(timer);
    } else {
      document.body.style.overflow = "auto";
    }
  }, [step, onEnd]);

  return (
    <>
      {/* 1. 재생버튼 */}
      {step === 0 && (
        <div
          ref={playBtnRef}
          onClick={handlePlayClick}
          style={{
            position: "fixed",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            zIndex: 10,
            width: 120,
            height: 120,
            background: "rgba(0,0,0,0.7)",
            borderRadius: "50%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
            boxShadow: "0 0 24px #39ff14, 0 0 48px #39ff14",
          }}
        >
          <svg width="60" height="60" viewBox="0 0 60 60">
            <circle cx="30" cy="30" r="28" stroke="#39ff14" strokeWidth="4" fill="none" />
            <polygon points="25,20 45,30 25,40" fill="#39ff14" ref={playIconRef} />
          </svg>
        </div>
      )}
      {/* 3. 영상+스크립트 재생 */}
      {step === 3 && index < videos.length && (
        <>
          <video
            ref={videoRef}
            key={videos[index]}
            src={videos[index]}
            autoPlay
            onEnded={handleEnded}
            controls
            style={{
              width: "100vw",
              height: "100vh",
              objectFit: "cover",
              opacity: fade ? 1 : 0,
              transition: "opacity 0.3s",
              position: "fixed",
              top: 0,
              left: 0,
              zIndex: 1,
            }}
          />
        </>
      )}
      {step === 4 && (
        <div
          style={{
            position: "fixed",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            width: "auto",
            zIndex: 20,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            pointerEvents: "none",
          }}
        >
          <span
            style={{
              color: "#39ff14",
              fontSize: "2.5rem",
              fontWeight: "bold",
              textShadow:
                "0 0 8px #39ff14, 0 0 24px #39ff14, 0 0 48px #39ff14",
              background: "rgba(0,0,0,0.7)",
              padding: "2rem 3rem",
              borderRadius: "2rem",
              animation:
                "growTextBig 2s cubic-bezier(0.23,1.01,0.32,1) forwards",
              display: "block",
              whiteSpace: "nowrap",
              textAlign: "center",
              margin: 0,
            }}
          >
            Now it&apos;s your turn
          </span>
        </div>
      )}
      <style>{`
        .theme-bg {
          min-height: 100vh;
          background: linear-gradient(140deg,#001210 55%,#122a17 100%);
          position:relative;
        }
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
    </>
  );
} 