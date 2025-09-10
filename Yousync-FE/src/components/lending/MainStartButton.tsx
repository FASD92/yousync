"use client";
import React, { useRef, useEffect } from "react";
import { gsap } from "gsap";
import { useRouter } from "next/navigation";

interface MainStartButtonProps {
  onPlay?: () => void;
}

const basePosterSet = [
  // 마블 영화들
  "https://image.tmdb.org/t/p/w500/1g0dhYtJI95apcsWiqjWrH1JBcO.jpg", // Spider-Man: No Way Home
  "https://image.tmdb.org/t/p/w500/74xTEgt7R36Fpooo50r9T25onhq.jpg", // Spider-Man: Into the Spider-Verse
  "https://image.tmdb.org/t/p/w500/7WsyChQLEftFiDOVTGkv3hFpyyt.jpg", // Avengers: Infinity War
  "https://image.tmdb.org/t/p/w500/or06FN3Dka5tukK1e9sl16pB3iy.jpg", // Avengers: Endgame
  "https://image.tmdb.org/t/p/w500/9Gtg2DzBhmYamXBS1hKAhiwbBKS.jpg", // Doctor Strange
  "https://image.tmdb.org/t/p/w500/qAKvO3hF7yLPqKGT3XFw9fZuiy2.jpg", // Captain America: Civil War
  "https://image.tmdb.org/t/p/w500/5kAG070BFpMGtSyuOvetMbeVI1d.jpg", // Iron Man
  "https://image.tmdb.org/t/p/w500/F9Hz1LR6nVjBjcifBEcNI9eHSDt.jpg", // Guardians of the Galaxy
  "https://image.tmdb.org/t/p/w500/y4MBh0EjBlMuOzv9axM4qJlmhzz.jpg", // Guardians of the Galaxy Vol. 2
  "https://image.tmdb.org/t/p/w500/rzRwTcFvttcN1ZpX2xv4j3tSdJu.jpg", // Thor: Ragnarok
  "https://image.tmdb.org/t/p/w500/cezWGskPY5x7GaglTTRN4Fugfb8.jpg", // Black Panther
  "https://image.tmdb.org/t/p/w500/BjtyMtVKEYlYqJvbB6sBVnT8Ot4.jpg", // Captain Marvel
  
  // 해리포터 시리즈
  "https://image.tmdb.org/t/p/w500/wuMc08IPKEatf9rnMNXvIDxqP4W.jpg", // Harry Potter and the Philosopher's Stone
  "https://image.tmdb.org/t/p/w500/sdEOH0992YZ0QSxgXNIGLq1ToUi.jpg", // Harry Potter and the Chamber of Secrets
  "https://image.tmdb.org/t/p/w500/aWxwnYoe8p2d2fcxOqtvAtJ72Rw.jpg", // Harry Potter and the Prisoner of Azkaban
  "https://image.tmdb.org/t/p/w500/fECBtHlr0RB3foNHDiCBXeg9Bv9.jpg", // Harry Potter and the Goblet of Fire
  "https://image.tmdb.org/t/p/w500/5aQsGUWqjmsJzjkjkuHoyLQTbQO.jpg", // Harry Potter and the Order of the Phoenix
  "https://image.tmdb.org/t/p/w500/z7uo9zmQdQwU5ZJHFpv2Upl30i1.jpg", // Harry Potter and the Half-Blood Prince
  
  // 트랜스포머 시리즈
  "https://image.tmdb.org/t/p/w500/4KRnAXXyKSWbG0Dh1QudOkb2ievv.jpg", // Transformers
  "https://image.tmdb.org/t/p/w500/pLBb0whOzVDtJvyD4DPeQyQNOqp.jpg", // Transformers: Revenge of the Fallen
  "https://image.tmdb.org/t/p/w500/28YlCLrFhONteYSs9hKjD1Km0Cj.jpg", // Transformers: Dark of the Moon
  "https://image.tmdb.org/t/p/w500/jyzrfx2WaeY60kYZpPYepSjGz4S.jpg", // Transformers: Age of Extinction
  "https://image.tmdb.org/t/p/w500/s5HQf2Gb3lIO2cRcFwNL9sn1o1o.jpg", // Transformers: The Last Knight
  "https://image.tmdb.org/t/p/w500/AqLcLsGGTzAjm3pCCq0CZCQrp6m.jpg", // Bumblebee
  
  // DC 영화들
  "https://image.tmdb.org/t/p/w500/9yBVqNruk6Ykrwc32qrK2TIE5xw.jpg", // The Dark Knight
  "https://image.tmdb.org/t/p/w500/8UlWHLMpgZm9bx6QYh0NFoq67TZ.jpg", // The Dark Knight Rises
  "https://image.tmdb.org/t/p/w500/gfJGlDaHuWimErCr5Ql0I8x9QSy.jpg", // Wonder Woman
  "https://image.tmdb.org/t/p/w500/eifGNCSDuxJeS1loAXil5bIGgvC.jpg", // Batman v Superman
  "https://image.tmdb.org/t/p/w500/xLxgVxFWvb9hhUyCDDXxRPPnFck.jpg", // Justice League
  "https://image.tmdb.org/t/p/w500/udDclJoHjfjb8Ekgsd4FDteOkCU.jpg", // Joker
  
  // 액션/판타지 영화들
  "https://image.tmdb.org/t/p/w500/6CoRTJTmijhBLJTUNoVSUNxZMEI.jpg", // John Wick
  "https://image.tmdb.org/t/p/w500/hEpWvX6Bp79eLxY1kX5ZZJcme5U.jpg", // John Wick: Chapter 2
  "https://image.tmdb.org/t/p/w500/ziEuG1essDuWuC5lpWUaw1uXY2O.jpg", // John Wick: Chapter 3
  "https://image.tmdb.org/t/p/w500/rCzpDGLbOoPwLjy3OAm5NUPOTrC.jpg", // The Lord of the Rings
  "https://image.tmdb.org/t/p/w500/5VTN0pR8gcqV3EPUHHfMGnJYN9L.jpg", // The Lord of the Rings: The Two Towers
  "https://image.tmdb.org/t/p/w500/uexxR7Kw1qYbZk0RYaF9Rx5ykbj.jpg", // The Lord of the Rings: The Return of the King
  
  // 애니메이션 영화들
  "https://image.tmdb.org/t/p/w500/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg", // The Incredibles
  "https://image.tmdb.org/t/p/w500/9lFKBtaVIhP7E2Pk0IY1CwTKTMZ.jpg", // The Incredibles 2
  "https://image.tmdb.org/t/p/w500/3bgtUfKQKNi3nJsAB5URpP2wdRt.jpg", // Frozen
  "https://image.tmdb.org/t/p/w500/mINJaa34MtknCTTJ5T2u2cj1Agi.jpg", // Frozen II
  "https://image.tmdb.org/t/p/w500/sKCr78MXSLixwmZ8DyJLrpMsd15.jpg", // Toy Story 4
  "https://image.tmdb.org/t/p/w500/w9kR8qbmQ01HwnvK4alvnQ2ca0L.jpg", // Finding Nemo
  
  // 액션 스릴러
  "https://image.tmdb.org/t/p/w500/lmZFxXgJE3vgrciwuDib0N8CfQo.jpg", // Mission: Impossible
  "https://image.tmdb.org/t/p/w500/4q2NNj4S5dG2RLF9CpXsej7yXl.jpg", // Mission: Impossible - Fallout
  "https://image.tmdb.org/t/p/w500/AkJQpZp9WoNdj7pLYSj1L0RcMMN.jpg", // Fast & Furious 6
  "https://image.tmdb.org/t/p/w500/hek3koDUyRQk7FIhPXsa6mT2Zc3.jpg", // Fast Five
  "https://image.tmdb.org/t/p/w500/dW4BIVfz8XplqUBWTfFTpflyUoN.jpg", // Fast & Furious 7
  "https://image.tmdb.org/t/p/w500/e1mjopzAS2KNsvpbpahQ1a6SkSn.jpg", // Mad Max: Fury Road
];

// 20번 복제하여 진짜 무한루프 (총 960개 포스터)
const posters = Array(20).fill(basePosterSet).flat();

export default function DoubleGridBlurSlider({ onPlay }: MainStartButtonProps) {
  const blurRef = useRef(null);
  const clearRef = useRef(null);
  const router = useRouter();

  useEffect(() => {
    // CSS 애니메이션으로 변경하므로 GSAP 코드 제거
  }, []);

  const renderPosterFrame = (src: string, i: number) => (
    <div className="frame" key={i}>
      <img src={src} alt="" />
    </div>
  );

  const handlePlay = () => {
    if (onPlay) {
      onPlay();
    } else {
      router.push("/home");
    }
  };

  return (
    <div className="doublegrid-root">
      {/* 뒤쪽 흐릿한 블러 슬라이드 */}
      <div className="doublegrid-blur-layer">
        <div className="doublegrid-blur-wrap">
          <div className="doublegrid-grid blur" ref={blurRef}>
            {posters.map(renderPosterFrame)}
          </div>
        </div>
      </div>
      {/* 위쪽(앞) 선명(약간 흐릿) 슬라이드 */}
      <div className="doublegrid-clear-layer">
        <div className="doublegrid-clear-wrap">
          <div className="doublegrid-grid clear" ref={clearRef}>
            {posters.map(renderPosterFrame)}
          </div>
        </div>
      </div>
      {/* 텍스트/버튼 (항상 선명) */}
      <div className="doublegrid-hero-text">
        <div className="brand-logo">
          <span className="you">You</span>
          <span className="sync">Sync</span>
        </div>
        <h1>
          Step Into the Scene.<br />
          <span className="highlight">Create Your Own Classic.</span>
        </h1>
        <p>
          YouSync와 함께하는 더빙의 새로운 경험<br />
          좋아하는 영화 장면에 당신만의 감정을 담아보세요
        </p>
        <div className="cta-section">
          <button className="main-cta" onClick={handlePlay}>
            <span className="button-text">지금 시작하기</span>
          </button>
          <div className="features">
            <div className="feature">
            </div>
            
          </div>
        </div>
      </div>
      <style>{`
        @keyframes scrollUpSmooth {
          from { transform: translateY(0%); }
          to { transform: translateY(-95%); }
        }
        
        .doublegrid-root {
          background: linear-gradient(135deg, #0a0f0d 0%, #1a2420 50%, #0f1814 100%);
          min-height: 100vh;
          position: relative;
          overflow: hidden;
        }
        .doublegrid-blur-layer {
          position: fixed;
          inset: 0;
          z-index: 0;
          pointer-events: none;
        }
        .doublegrid-blur-wrap {
          position: absolute;
          top: 50%;
          left: 68%;
          transform: translate(-50%, -50%) rotate(-10deg);
          width: 130vw;
          height: 130vh;
          overflow: hidden;
        }
        .doublegrid-grid.blur {
          filter: blur(32px) brightness(0.35) saturate(0.8);
          animation: scrollUpSmooth 400s linear infinite;
        }
        .doublegrid-clear-layer {
          position: fixed;
          inset: 0;
          z-index: 1;
          pointer-events: none;
        }
        .doublegrid-clear-wrap {
          position: absolute;
          top: 50%;
          left: 68%;
          transform: translate(-50%, -50%) rotate(-10deg);
          width: 130vw;
          height: 130vh;
          overflow: hidden;
        }
        .doublegrid-grid {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 2000%;
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          grid-auto-rows: min-content;
          gap: 1rem;
          padding: 0.5rem;
        }
        .doublegrid-grid.clear {
          filter: brightness(0.4) blur(5px) saturate(0.95);
          animation: scrollUpSmooth 300s linear infinite;
        }
        .frame {
          aspect-ratio: 2/3;
          width: 100%;
          border-radius: 0.6rem;
          overflow: hidden;
          background: #131e18;
          box-shadow: 0 4px 12px rgba(18, 255, 174, 0.1);
        }
        .frame img {
          width: 100%;
          height: 100%;
          object-fit: cover;
          display: block;
          filter: brightness(0.8);
        }
        .doublegrid-hero-text {
          position: relative;
          z-index: 3;
          max-width: 750px;
          margin-left: 7vw;
          margin-top: 8vh;
          padding: 2rem 0;
        }
        .brand-logo {
          margin-bottom: 2rem;
          font-size: 2.5rem;
          font-weight: 900;
          letter-spacing: -0.02em;
        }
        .brand-logo .you {
          color: #ffffff;
          text-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
        }
        .brand-logo .sync {
          background: linear-gradient(135deg, #00ffc3 0%, #41ffb1 50%, #0af 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          text-shadow: 0 0 30px rgba(0, 255, 195, 0.4);
        }
        .doublegrid-hero-text h1 {
          font-size: clamp(2.5rem, 5.5vw, 4.8rem);
          font-weight: 800;
          line-height: 1.2;
          color: #ffffff;
          margin-bottom: 1.5rem;
          text-shadow: 0 2px 20px rgba(0, 0, 0, 0.5);
        }
        .doublegrid-hero-text h1 .highlight {
          background: linear-gradient(135deg, #00ffc3 0%, #41ffb1 50%, #0af 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          position: relative;
        }
        .doublegrid-hero-text h1 .highlight::after {
          content: '';
          position: absolute;
          bottom: -4px;
          left: 0;
          right: 0;
          height: 3px;
          background: linear-gradient(90deg, #00ffc3 0%, #0af 100%);
          border-radius: 2px;
          opacity: 0.6;
        }
        .doublegrid-hero-text p {
          margin-bottom: 3rem;
          font-size: clamp(1.1rem, 2vw, 1.3rem);
          line-height: 1.8;
          color: #b8e6d3;
          text-shadow: 0 2px 10px rgba(9, 31, 23, 0.8);
          font-weight: 400;
        }
        .cta-section {
          display: flex;
          flex-direction: column;
          gap: 2rem;
        }
        .main-cta {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          font-size: 1.2rem;
          padding: 1.3rem 3rem;
          border: none;
          border-radius: 60px;
          background: linear-gradient(135deg, #00ffc3 0%, #0af 100%);
          color: #ffffff;
          cursor: pointer;
          font-weight: 600;
          box-shadow: 
            0 4px 20px rgba(0, 255, 195, 0.25),
            0 1px 3px rgba(0, 0, 0, 0.12);
          transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1);
          position: relative;
          overflow: hidden;
          text-shadow: none;
          align-self: flex-start;
          letter-spacing: 0.02em;
        }
        .main-cta::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, transparent 50%);
          opacity: 0;
          transition: opacity 0.4s ease;
        }
        .main-cta:hover::before {
          opacity: 1;
        }
        .main-cta:hover {
          transform: translateY(-1px);
          box-shadow: 
            0 8px 30px rgba(0, 255, 195, 0.35),
            0 2px 8px rgba(0, 0, 0, 0.15);
        }
        .main-cta:active {
          transform: translateY(0);
          transition: all 0.15s ease;
        }
        .button-text {
          position: relative;
          z-index: 1;
        }
        .features {
          display: flex;
          gap: 2.5rem;
          flex-wrap: wrap;
          margin-top: 0.5rem;
        }
        .feature {
          display: flex;
          align-items: center;
          gap: 0.7rem;
          color: #a8d4be;
          font-size: 0.95rem;
          font-weight: 500;
          opacity: 0.9;
        }
        .feature-icon {
          font-size: 1.1rem;
          display: flex;
          align-items: center;
          justify-content: center;
          width: 24px;
          height: 24px;
        }
        @media (max-width: 950px) {
          .doublegrid-blur-wrap, .doublegrid-clear-wrap {
            left: 50%;
            width: 98vw;
            height: 98vw;
          }
          .doublegrid-hero-text {
            margin-left: 5vw;
            margin-right: 5vw;
            margin-top: 6vh;
          }
          .brand-logo {
            font-size: 2rem;
          }
          .features {
            gap: 1.5rem;
          }
          .main-cta {
            font-size: 1.1rem;
            padding: 1rem 2rem;
          }
        }
        @media (max-width: 640px) {
          .doublegrid-hero-text {
            margin-left: 4vw;
            margin-right: 4vw;
          }
          .features {
            flex-direction: column;
            gap: 1rem;
          }
          .cta-section {
            gap: 1.5rem;
          }
        }
      `}</style>
    </div>
  );
}