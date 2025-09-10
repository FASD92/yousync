'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useState } from 'react';

export default function QueryProvider({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 5 * 60 * 1000, // 5분으로 증가 (더 오래 캐시 유지)
            gcTime: 10 * 60 * 1000, // 10분 (캐시 유지 시간)
            refetchOnWindowFocus: false,
            refetchOnMount: false, // 마운트 시 자동 리페치 비활성화
            refetchOnReconnect: true,
            retry: 1, // 재시도 횟수 제한
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
} 