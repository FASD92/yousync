import { useState, useEffect } from 'react';
import { backendApi } from '@/services/api';
import { MyPageOverview } from '@/types/MypageType';

interface UseMyPageOverviewResult {
  data: MyPageOverview | null;
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

export const useMyPageOverview = (): UseMyPageOverviewResult => {
  const [data, setData] = useState<MyPageOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchOverview = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await backendApi.get<MyPageOverview>('/mypage/overview');
      setData(response);
    } catch (err) {
      console.error('마이페이지 통합 정보 조회 실패:', err);
      setError('마이페이지 정보를 불러오는데 실패했습니다.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchOverview();
  }, []);

  return {
    data,
    loading,
    error,
    refetch: fetchOverview,
  };
}; 