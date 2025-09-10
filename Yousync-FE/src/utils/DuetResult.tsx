"use client";

import { useMemo } from "react";
import { useResultStore } from "@/store/useResultStore";

interface DuetResultProps {
  scriptCount: number;
  children: (isAllAnalyzed: boolean) => React.ReactNode;
}

export default function DuetResult({ scriptCount, children }: DuetResultProps) {
  const { finalResults } = useResultStore();

  const isAllAnalyzed = useMemo(() => {
    return finalResults.length === scriptCount;
  }, [finalResults.length, scriptCount]);

  return <>{children(isAllAnalyzed)}</>;
} 