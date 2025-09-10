'use client';

import MainStartButton from "@/components/lending/MainStartButton";
import { useRouter } from "next/navigation";
import React from "react";

export default function LandingPage() {
  const router = useRouter();
  
  const handlePlay = () => {
    router.push("/home");
  };
  
  return (
    <div className="bg-neutral-950 text-white min-h-screen">
      <MainStartButton onPlay={handlePlay} />
    </div>
  );
} 