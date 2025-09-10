//이해완료
"use client";

import { useRouter, useSearchParams } from "next/navigation";

interface DubbingHeaderProps {
  title: string;
  category: string;
  actorName: string;
}

export default function DubbingHeader({ title, category, actorName }: DubbingHeaderProps) {
  const router = useRouter();
  const searchParams = useSearchParams();

  const handleBackClick = () => {
    // 쿼리 파라미터에서 modalId 확인
    const modalId = searchParams.get('modalId');
    if (modalId) {
      // modalId가 있으면 sessionStorage에 저장하고 홈페이지로 이동
      sessionStorage.setItem('modalId', modalId);
      router.replace('/');
    } else {
      router.back();
    }
  };

  return (
    <header className="bg-black/50 backdrop-blur-sm border-b border-gray-800">
      <div className="w-full mx-auto px-2 sm:px-4 lg:px-6 py-2 sm:py-3">
        <div className="flex items-center justify-between">
          <div className="flex-1 min-w-0 mr-2 sm:mr-4">
            <h1 className="text-base sm:text-lg lg:text-xl font-bold truncate">{title}</h1>
            <p className="text-gray-400 text-xs sm:text-sm truncate">
              <span className="hidden sm:inline">{category} - </span>
              {actorName}
            </p>
          </div>
          <button
            onClick={handleBackClick}
            className="flex-shrink-0 px-2 py-1 sm:px-3 sm:py-1.5 lg:px-4 lg:py-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors text-xs sm:text-sm lg:text-base font-medium"
          >
            <span className="hidden sm:inline">뒤로가기</span>
            <span className="sm:hidden">←</span>
          </button>
        </div>
      </div>
    </header>
  );
} 