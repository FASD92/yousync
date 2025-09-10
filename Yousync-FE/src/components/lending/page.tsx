"use client";

import { useState, Suspense } from "react";
import IntroPlayButton from "@/components/lending/IntroPlayButton";
import LanderSequence from "@/components/graph/LanderSequence"; // 영상/스크립트 재생
import MainStartButton from "@/components/lending/MainStartButton"; // "재생" 버튼
import HomeClient from "@/app/home/HomeClient";

export default function HomePage() {
  const [step, setStep] = useState<"intro" | "video" | "main">("intro");

  return (
    <>
      <Suspense fallback={<div>Loading...</div>}>
        <HomeClient />
      </Suspense>
    </>
  );
}
