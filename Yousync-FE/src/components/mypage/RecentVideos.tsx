"use client";

import { MyDubbedTokenResponse } from '@/services/mypage';

interface RecentVideosProps {
  dubbedTokens: MyDubbedTokenResponse[];
  loading?: boolean;
}

export default function RecentVideos({ dubbedTokens, loading }: RecentVideosProps) {
  if (loading) {
    return (
      <section className="mb-12">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold">내가 더빙한 영상</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[...Array(4)].map((_, index) => (
            <div key={index} className="bg-neutral-900 rounded-xl overflow-hidden animate-pulse">
              <div className="aspect-video bg-neutral-700"></div>
              <div className="p-4">
                <div className="h-4 bg-neutral-700 rounded mb-2"></div>
                <div className="h-3 bg-neutral-700 rounded w-2/3 mb-2"></div>
                <div className="h-3 bg-neutral-700 rounded w-1/2"></div>
              </div>
            </div>
          ))}
        </div>
      </section>
    );
  }

  return (
    <section className="mb-12">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold">내가 더빙한 영상</h2>
      </div>
      
      {dubbedTokens.length === 0 ? (
        <div className="text-center py-12 text-gray-400">
          <div className="text-4xl mb-4">🎬</div>
          <p>아직 더빙한 토큰이 없습니다.</p>
          <p className="text-sm mt-2">첫 더빙을 시작해보세요!</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {dubbedTokens.map((token) => {
            const completionRate = Math.round((token.completed_scripts / token.total_scripts) * 100);
            const scoreColor = completionRate >= 80 ? 'bg-green-500' : 
                              completionRate >= 60 ? 'bg-yellow-500' : 'bg-red-500';
            
            return (
              <div
                key={token.token_id}
                className="group bg-neutral-900 rounded-xl overflow-hidden border border-neutral-800 hover:border-neutral-700 transition-all duration-300 cursor-pointer"
              >
                <div className="relative aspect-video w-full overflow-hidden rounded-xl">
                  <img
                    src="https://images.unsplash.com/photo-1519125323398-675f0ddb6308"
                    alt={token.token_name}
                    className="absolute inset-0 w-full h-full object-cover"
                  />
                  <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                    <button className="bg-white/20 backdrop-blur-sm rounded-full p-3 text-white">
                      ▶
                    </button>
                  </div>
                  <div className={`absolute top-2 right-2 ${scoreColor} text-white text-xs px-2 py-1 rounded font-bold`}>
                    {completionRate}%
                  </div>
                  <div className="absolute top-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
                    {token.category}
                  </div>
                </div>
                <div className="p-4">
                  <h3 className="font-semibold mb-1 truncate">{token.token_name}</h3>
                  <p className="text-gray-400 text-sm mb-2">{token.actor_name}</p>
                  <div className="flex justify-between items-center text-xs text-gray-500">
                    <span>{token.completed_scripts}/{token.total_scripts} 완료</span>
                    <span>{new Date(token.last_dubbed_at).toLocaleDateString('ko-KR')}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
} 