import React, { useRef, useEffect } from "react";
import { gsap } from "gsap";

interface IntroPlayButtonProps {
  onPlay: () => void;
}

export default function IntroPlayButton({ onPlay }: IntroPlayButtonProps) {
  const playBtnRef = useRef<HTMLDivElement | null>(null);
  const playIconRef = useRef<SVGPolygonElement | null>(null);

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

  const handleClick = () => {
    if (playBtnRef.current) {
      gsap.to(playBtnRef.current, {
        scale: 2.2,
        opacity: 0,
        duration: 0.9,
        ease: "power2.in",
        onComplete: onPlay
      });
    }
  };

  return (
    <div
      ref={playBtnRef}
      onClick={handleClick}
      style={{
        position: "fixed",
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
        zIndex: 100,
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