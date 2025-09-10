import { useState, useEffect } from 'react';
import { mypageService, BookmarkListOut, MyDubbedTokenResponse } from '@/services/mypage';
import { authService, UserInfo } from '@/services/auth';

export interface MyPageData {
  user: UserInfo | null;
  bookmarks: BookmarkListOut[];
  dubbedTokens: MyDubbedTokenResponse[];
  stats: {
    totalBookmarks: number;
    totalDubbedTokens: number;
    averageScore: number;
    totalPracticeCount: number;
  };
}

export const useMyPage = () => {
  const [data, setData] = useState<MyPageData>({
    user: null,
    bookmarks: [],
    dubbedTokens: [],
    stats: {
      totalBookmarks: 0,
      totalDubbedTokens: 0,
      averageScore: 0,
      totalPracticeCount: 0,
    }
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMyPageData = async () => {
    try {
      setLoading(true);
      setError(null);

      // 사용자 정보 가져오기
      const userInfo = await authService.getCurrentUser();
      
      // 북마크 목록 가져오기
      const bookmarks = await mypageService.getBookmarks(10, 0);
      
      // 내가 더빙한 토큰 목록 가져오기
      const dubbedTokens = await mypageService.getMyDubbedTokens(10, 0);

      // 통계 계산
      const totalPracticeCount = dubbedTokens.reduce((sum, token) => sum + token.completed_scripts, 0);
      const averageScore = dubbedTokens.length > 0 ? 
        Math.round(dubbedTokens.reduce((sum, token) => sum + (token.completed_scripts / token.total_scripts * 100), 0) / dubbedTokens.length) : 0;

      setData({
        user: userInfo,
        bookmarks,
        dubbedTokens,
        stats: {
          totalBookmarks: bookmarks.length,
          totalDubbedTokens: dubbedTokens.length,
          averageScore,
          totalPracticeCount,
        }
      });
    } catch (err) {
      console.error('마이페이지 데이터 로딩 실패:', err);
      setError('데이터를 불러오는데 실패했습니다.');
    } finally {
      setLoading(false);
    }
  };

  const addBookmark = async (tokenId: number) => {
    try {
      await mypageService.createBookmark(tokenId);
      // 북마크 목록 새로고침
      const bookmarks = await mypageService.getBookmarks(10, 0);
      setData(prev => ({
        ...prev,
        bookmarks,
        stats: {
          ...prev.stats,
          totalBookmarks: bookmarks.length
        }
      }));
    } catch (err) {
      console.error('북마크 추가 실패:', err);
      throw err;
    }
  };

  const removeBookmark = async (tokenId: number) => {
    try {
      await mypageService.deleteBookmark(tokenId);
      // 북마크 목록 새로고침
      const bookmarks = await mypageService.getBookmarks(10, 0);
      setData(prev => ({
        ...prev,
        bookmarks,
        stats: {
          ...prev.stats,
          totalBookmarks: bookmarks.length
        }
      }));
    } catch (err) {
      console.error('북마크 삭제 실패:', err);
      throw err;
    }
  };

  useEffect(() => {
    fetchMyPageData();
  }, []);

  return {
    data,
    loading,
    error,
    refetch: fetchMyPageData,
    addBookmark,
    removeBookmark,
  };
};
