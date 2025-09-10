'use client';

import { useRouter, usePathname } from 'next/navigation';
import { useState } from 'react';

export default function FloatingRequestButton() {
  const router = useRouter();
  const pathname = usePathname();
  const [isHovered, setIsHovered] = useState(false);

  // 메인페이지(/home)에서만 버튼을 표시
  if (pathname !== '/home') {
    return null;
  }

  const handleClick = () => {
    router.push('/uploadrequest');
  };

  return (
    <button
      onClick={handleClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className="fixed bottom-6 right-6 z-50 w-14 h-14 bg-gradient-to-r from-emerald-400 via-green-500 to-emerald-400 rounded-full shadow-lg hover:shadow-xl transition-all duration-300 flex items-center justify-center group hover:scale-110"
      title="영화 요청하기"
    >
      <svg 
        width="24" 
        height="24" 
        fill="none" 
        viewBox="0 0 24 24"
        className="text-black"
      >
        <rect x="10" y="4" width="4" height="16" rx="2" fill="currentColor" />
        <rect x="4" y="10" width="16" height="4" rx="2" fill="currentColor" />
      </svg>
      
      {/* 호버 시 나타나는 툴팁 */}
      {isHovered && (
        <div className="absolute bottom-16 right-0 bg-neutral-800 text-white px-3 py-2 rounded-lg text-sm whitespace-nowrap shadow-lg">
          영화 요청하기
          <div className="absolute top-full right-4 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-neutral-800"></div>
        </div>
      )}
    </button>
  );
}
