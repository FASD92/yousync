'use client';

import { Suspense } from 'react';
import ResultPageContent from '@/components/result/ResultPageContent';

export default function ResultPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-neutral-950 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-4" />
          <p>로딩 중...</p>
        </div>
      </div>
    }>
      <ResultPageContent />
    </Suspense>
  );
}
