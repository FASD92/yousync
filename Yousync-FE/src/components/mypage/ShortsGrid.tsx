"use client";

import { BookmarkListOut } from '@/services/mypage';

interface ShortsGridProps {
  bookmarks: BookmarkListOut[];
  loading?: boolean;
  onRemoveBookmark?: (tokenId: number) => void;
}

export default function ShortsGrid({ bookmarks, loading, onRemoveBookmark }: ShortsGridProps) {
  if (loading) {
    return (
      <div className="lg:col-span-2 bg-neutral-900 rounded-2xl p-8 border border-neutral-800">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold">내 북마크</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[...Array(4)].map((_, index) => (
            <div key={index} className="bg-neutral-800 rounded-xl overflow-hidden animate-pulse">
              <div className="aspect-[16/9] bg-neutral-700"></div>
              <div className="p-4">
                <div className="h-4 bg-neutral-700 rounded mb-2"></div>
                <div className="h-3 bg-neutral-700 rounded w-2/3"></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="lg:col-span-2 bg-neutral-900 rounded-2xl p-8 border border-neutral-800">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold">내 북마크 <span className="text-blue-400 text-lg ml-2">({bookmarks.length}개)</span></h2>
      </div>
      
      {bookmarks.length === 0 ? (
        <div className="text-center py-12 text-gray-400">
          <div className="text-4xl mb-4">📌</div>
          <p>아직 북마크한 토큰이 없습니다.</p>
          <p className="text-sm mt-2">관심있는 토큰을 북마크해보세요!</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {bookmarks.map((bookmark) => (
            <div
              key={bookmark.id}
              className="group bg-neutral-800 rounded-xl overflow-hidden border border-neutral-700 hover:border-neutral-600 transition-all duration-300 cursor-pointer"
            >
              {/* 썸네일 */}
              <div className="relative aspect-[16/9] w-full overflow-hidden">
                <img
                  src={bookmark.token.thumbnail_url || "https://images.unsplash.com/photo-1519125323398-675f0ddb6308"}
                  alt={bookmark.token.token_name}
                  className="absolute inset-0 w-full h-full object-cover"
                  onError={(e) => {
                    const target = e.target as HTMLImageElement;
                    target.src = "https://images.unsplash.com/photo-1519125323398-675f0ddb6308";
                  }}
                />
                <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                  <button className="bg-white/20 backdrop-blur-sm rounded-full p-3 text-white">
                    ▶
                  </button>
                </div>
                {onRemoveBookmark && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onRemoveBookmark(bookmark.token_id);
                    }}
                    className="absolute top-2 right-2 bg-red-500/80 hover:bg-red-500 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                )}
                <div className="absolute top-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
                  {bookmark.token.category}
                </div>
              </div>
              <div className="p-4">
                <h3 className="font-semibold mb-1 truncate">{bookmark.token.token_name}</h3>
                <p className="text-gray-400 text-sm mb-1">{bookmark.token.actor_name}</p>
                <p className="text-gray-500 text-xs">
                  {new Date(bookmark.created_at).toLocaleDateString('ko-KR')}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
} 