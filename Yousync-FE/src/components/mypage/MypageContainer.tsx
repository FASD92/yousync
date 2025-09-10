"use client";

import React, { useState, useEffect } from 'react';
import { useMyPageOverview } from '@/hooks/useMyPageOverview';
import { API_ENDPOINTS } from '@/lib/constants';
import UserProfile from './UserProfile';
import PageHeader from './PageHeader';
import { extractYoutubeVideoId, getYoutubeThumbnail } from '@/utils/extractYoutubeVideoId';
import { useUser } from '@/hooks/useUser';
import { useRouter } from 'next/navigation';
import axios from 'axios';
import { toast } from 'react-hot-toast';
import { useVideos } from '@/hooks/useVideos';
import MovieDetailModal from '@/components/modal/MovieDetailModal';
import type { TokenDetailResponse } from '@/types/pitch';
import MypageListenModal from './MypageListenModal';

const MypageContainer: React.FC = () => {
  const { data, loading, error, refetch } = useMyPageOverview();
  const [bookmarkPage, setBookmarkPage] = useState(1);
  const [dubbedPage, setDubbedPage] = useState(1);
  const [isRemovingBookmark, setIsRemovingBookmark] = useState(false);
  const [approvedTokens, setApprovedTokens] = useState<any[]>([]);
  const { data: tokens = [] } = useVideos();
  const [selectedToken, setSelectedToken] = useState<TokenDetailResponse | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const itemsPerPage = 6; // 페이지당 표시할 항목 수
  const [dubbingModalOpen, setDubbingModalOpen] = useState(false);
  const [dubbingTokenId, setDubbingTokenId] = useState<number | null>(null);
  const [dubbingYoutubeId, setDubbingYoutubeId] = useState<string | null>(null);

  // 북마크 정렬 및 페이지네이션 계산
  const sortedBookmarks = data ? [...data.recent_bookmarks].sort((a, b) => b.id - a.id) : [];
  const totalBookmarkPages = data ? Math.ceil(sortedBookmarks.length / itemsPerPage) : 0;
  const currentBookmarks = data ? sortedBookmarks.slice(
    (bookmarkPage - 1) * itemsPerPage,
    bookmarkPage * itemsPerPage
  ) : [];

  // 더빙 토큰 정렬 및 페이지네이션 계산
  const sortedDubbedTokens = data ? [...data.recent_dubbed_tokens].sort((a, b) => b.token_id - a.token_id) : [];
  const totalDubbedPages = data ? Math.ceil(sortedDubbedTokens.length / itemsPerPage) : 0;
  const currentDubbedTokens = data ? sortedDubbedTokens.slice(
    (dubbedPage - 1) * itemsPerPage,
    dubbedPage * itemsPerPage
  ) : [];

  // 승인된 영상 불러오기
  useEffect(() => {
    const fetchApprovedTokens = async () => {
      const accessToken = localStorage.getItem('access_token');
      if (!accessToken) return;
      try {
        const res = await axios.get(
          `${process.env.NEXT_PUBLIC_API_BASE_URL}/request/mine`,
          {
            headers: {
              Authorization: `Bearer ${accessToken}`,
            },
          }
        );
        // status가 '승인됨'인 것만 필터링
        if (Array.isArray(res.data)) {
          setApprovedTokens(res.data.filter((item: any) => item.status === '승인됨'));
        } else {
          setApprovedTokens([]);
        }
      } catch (error) {
        console.error('승인된 영상 불러오기 실패:', error);
      }
    };
    fetchApprovedTokens();
  }, []);



  // 페이지 번호 버튼 생성 함수
  const renderPaginationButtons = (totalPages: number, currentPage: number, setPage: (page: number) => void) => {
    const buttons = [];
    for (let i = 1; i <= totalPages; i++) {
      buttons.push(
        <button
          key={i}
          onClick={() => setPage(i)}
          className={`w-8 h-8 rounded-full flex items-center justify-center ${
            currentPage === i
              ? 'bg-blue-600 text-white'
              : 'bg-neutral-800 text-gray-300 hover:bg-neutral-700'
          }`}
        >
          {i}
        </button>
      );
    }
    return buttons;
  };

  const handleRemoveBookmark = async (tokenId: number) => {
    setIsRemovingBookmark(true);
    try {
      const response = await fetch(`${API_ENDPOINTS.BASE_URL}/mypage/bookmarks/${tokenId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        }
      });

      if (response.status === 401) {
        localStorage.removeItem('access_token');
        window.location.href = '/login';
        return;
      }

      if (!response.ok) {
        throw new Error('북마크 삭제에 실패했습니다.');
      }

      refetch();
    } catch (error) {
      console.error('북마크 삭제 실패:', error);
      alert('북마크 삭제에 실패했습니다.');
    } finally {
      setIsRemovingBookmark(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-neutral-950 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-4" />
          <p>마이페이지 정보를 불러오는 중...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-neutral-950 text-white flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-400 mb-4">{error}</p>
          <button
            onClick={refetch}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition"
          >
            다시 시도
          </button>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="min-h-screen bg-neutral-950 text-white flex items-center justify-center">
        <div className="text-center">
          <p className="mb-4">데이터를 불러올 수 없습니다.</p>
          <button
            onClick={refetch}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition"
          >
            다시 시도
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-neutral-950 text-white">
      <PageHeader title="마이페이지" subtitle="" />

      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="space-y-8">
          {/* 사용자 프로필 */}
          <UserProfile
            user={data.user_info}
            stats={{
              totalDubbedTokens: data.total_dubbed_tokens,
              averageScore: data.average_completion_rate,
              totalPracticeCount: data.total_practice_count,
            }}
          />

          {/* 북마크 섹션 */}
          <div className="bg-neutral-900 rounded-2xl p-8 border border-neutral-800">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold">
                내 북마크{' '}
                <span className="text-blue-400 text-lg ml-2">
                  ({data.recent_bookmarks.length}개)
                </span>
              </h2>
            </div>

            {data.recent_bookmarks.length === 0 ? (
              <div className="text-center py-12 text-gray-400">
                <div className="text-4xl mb-4">📌</div>
                <p>아직 북마크한 토큰이 없습니다.</p>
                <p className="text-sm mt-2">관심있는 토큰을 북마크해보세요!</p>
              </div>
            ) : (
              <>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-3 gap-6">
                  {currentBookmarks.map((bookmark) => (
                    <div
                      key={bookmark.id}
                      className="group bg-neutral-800 rounded-xl overflow-hidden border border-neutral-700 hover:border-neutral-600 transition-all duration-300 cursor-pointer"
                      onClick={() => {
                        // 북마크 클릭 시 모달창 열기
                        if (!bookmark.token.youtube_url) {
                          toast.error('YouTube URL 정보가 없습니다.');
                          return;
                        }
                        const youtubeId = extractYoutubeVideoId(bookmark.token.youtube_url);
                        if (!youtubeId) {
                          toast.error('유효하지 않은 YouTube URL입니다.');
                          return;
                        }
                        const foundToken = tokens.find(t => t.youtubeId === youtubeId);
                        if (foundToken) {
                          setSelectedToken(foundToken);
                          setModalOpen(true);
                        } else {
                          toast.error('해당 영상 정보를 찾을 수 없습니다.');
                        }
                      }}
                    >
                      {/* 썸네일 */}
                      <div className="relative aspect-[16/9] w-full overflow-hidden">
                        <img
                          src={
                            bookmark.token.youtube_url 
                              ? getYoutubeThumbnail(bookmark.token.youtube_url)
                              : bookmark.token.thumbnail_url ||
                                'https://images.unsplash.com/photo-1519125323398-675f0ddb6308'
                          }
                          alt={bookmark.token.token_name}
                          className="absolute inset-0 w-full h-full object-cover"
                          onError={(e) => {
                            const target = e.target as HTMLImageElement;
                            target.src =
                              'https://images.unsplash.com/photo-1519125323398-675f0ddb6308';
                          }}
                        />
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRemoveBookmark(bookmark.token.id);
                          }}
                          className="absolute top-2 right-2 bg-black/70 hover:bg-red-600 text-white p-1.5 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                          disabled={isRemovingBookmark}
                        >
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-4 w-4"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M6 18L18 6M6 6l12 12"
                            />
                          </svg>
                        </button>
                      </div>

                      {/* 정보 */}
                      <div className="p-4">
                        <h3 className="font-medium text-white mb-1 truncate">
                          {bookmark.token.token_name}
                        </h3>
                        <p className="text-gray-400 text-sm truncate">{bookmark.token.actor_name}</p>
                      </div>
                    </div>
                  ))}
                </div>
                
                {/* 페이지네이션 버튼 */}
                {totalBookmarkPages > 1 && (
                  <div className="flex justify-center mt-8 gap-2">
                    {renderPaginationButtons(totalBookmarkPages, bookmarkPage, setBookmarkPage)}
                  </div>
                )}
              </>
            )}
          </div>

          {/* 최근 더빙한 토큰 섹션 */}
          <div className="bg-neutral-900 rounded-2xl p-8 border border-neutral-800">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold">
                내가 더빙한 영상{' '}
                <span className="text-green-400 text-lg ml-2">
                  ({data.recent_dubbed_tokens.length}개)
                </span>
              </h2>
            </div>

            {data.recent_dubbed_tokens.length === 0 ? (
              <div className="text-center py-12 text-gray-400">
                <div className="text-4xl mb-4">🎬</div>
                <p>아직 더빙한 토큰이 없습니다.</p>
                <p className="text-sm mt-2">토큰을 더빙해보세요!</p>
              </div>
            ) : (
              <>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-3 gap-6">
                  {currentDubbedTokens.map((token) => (
                    <div
                      key={token.token_id}
                      className="group bg-neutral-800 rounded-xl overflow-hidden border border-neutral-700 hover:border-neutral-600 transition-all duration-300 cursor-pointer"
                      onClick={() => {
                        // 토큰 클릭 시 결과 페이지로 이동
                        // window.location.href = `/result?token_id=${token.token_id}`;
                        setDubbingTokenId(token.token_id);
                        setDubbingYoutubeId(token.youtube_url ?? null);
                        setDubbingModalOpen(true);
                      }}
                    >
                      <div className="relative aspect-[16/9] w-full overflow-hidden flex items-center justify-center bg-black">
                        <img
                          src={
                            token.youtube_url && extractYoutubeVideoId(token.youtube_url)
                              ? `https://img.youtube.com/vi/${extractYoutubeVideoId(token.youtube_url)}/mqdefault.jpg`
                              : "https://images.unsplash.com/photo-1519125323398-675f0ddb6308"
                          }
                          alt={token.token_name}
                          className="object-contain w-full h-full"
                          onError={(e) => {
                            const target = e.target as HTMLImageElement;
                            target.src = "https://images.unsplash.com/photo-1519125323398-675f0ddb6308";
                          }}
                        />
                      </div>
                      <div className="p-4">
                        <h3 className="font-medium text-white mb-1 truncate">{token.token_name}</h3>
                        <p className="text-gray-400 text-sm truncate">{token.actor_name}</p>
                      </div>
                    </div>
                  ))}
                </div>
                
                {/* 페이지네이션 버튼 */}
                {totalDubbedPages > 1 && (
                  <div className="flex justify-center mt-8 gap-2">
                    {renderPaginationButtons(totalDubbedPages, dubbedPage, setDubbedPage)}
                  </div>
                )}
              </>
            )}
          </div>

          {/* 승인된 영상 섹션 */}
          <div className="bg-neutral-900 rounded-2xl p-8 border border-neutral-800 mt-10">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold">
                승인된 영상{' '}
                <span className="text-green-400 text-lg ml-2">
                  ({approvedTokens.length}개)
                </span>
              </h2>
            </div>
            {approvedTokens.length === 0 ? (
              <div className="text-center py-12 text-gray-400">
                <p>아직 승인된 영상이 없습니다.</p>
                <p className="text-sm mt-2">심사 후 승인된 영상이 여기에 표시됩니다.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-3 gap-6">
                {approvedTokens.map((token) => (
                  <div
                    key={token.token_id}
                    className="group bg-neutral-800 rounded-xl overflow-hidden border border-neutral-700 hover:border-green-600 transition-all duration-300 cursor-pointer"
                    onClick={() => {
                      const youtubeId = extractYoutubeVideoId(token.url);
                      const found = tokens.find(t => t.youtubeId === youtubeId);
                      if (found) {
                        setSelectedToken(found);
                        setModalOpen(true);
                      }
                    }}
                  >
                    <div className="relative aspect-[16/9] w-full overflow-hidden">
                      <img
                        src={
                          extractYoutubeVideoId(token.url)
                            ? `https://img.youtube.com/vi/${extractYoutubeVideoId(token.url)}/mqdefault.jpg`
                            : "https://images.unsplash.com/photo-1519125323398-675f0ddb6308"
                        }
                        alt={token.token_name}
                        className="w-full h-full object-cover object-center"
                        onError={(e) => {
                          const target = e.target as HTMLImageElement;
                          target.src = "https://images.unsplash.com/photo-1519125323398-675f0ddb6308";
                        }}
                      />
                    </div>
                    {/* <div className="p-4">
                      <h3 className="font-medium text-white mb-1 truncate">{token.token_name}</h3>
                      <p className="text-gray-400 text-sm truncate">{token.actor_name}</p>
                    </div> */}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
      {selectedToken && (
        <MovieDetailModal
          youtubeId={extractYoutubeVideoId(selectedToken.youtube_url) || ''}
          isOpen={modalOpen}
          onClose={() => setModalOpen(false)}
          tokenData={selectedToken}
        />
      )}
      {dubbingModalOpen && dubbingTokenId && (
        <MypageListenModal 
          open={dubbingModalOpen}
          onClose={() => setDubbingModalOpen(false)}
          tokenId={String(dubbingTokenId)}
          modalId={extractYoutubeVideoId(dubbingYoutubeId ?? "") ?? undefined}
        />
      )}
    </div>
  );
};

export default MypageContainer;