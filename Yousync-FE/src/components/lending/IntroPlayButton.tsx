import React, { useRef, useEffect } from 'react';
import { gsap } from "gsap";

interface IntroPlayButtonProps {
  onPlay?: () => void;
}

export default function IntroPlayButton({ onPlay }: IntroPlayButtonProps) {
  const playBtnRef = useRef(null);
  const playIconRef = useRef(null);

  // 재생버튼 SVG 애니메이션
  useEffect(() => {
    if (playIconRef.current) {
      gsap.set(playIconRef.current, { strokeDasharray: 180, strokeDashoffset: 180, opacity: 1 });
      gsap.to(playIconRef.current, {
        strokeDashoffset: 0,
        duration: 1.1,
        ease: "power1.inOut"
      });
    }
  }, []);

  const handlePlayClick = () => {
    if (onPlay) onPlay();
  };

  return (
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
  );
}
