'use client';

import { useState, useEffect, useCallback } from "react";
import MovieDetailModal from "@/components/modal/MovieDetailModal";
import { useVideos, usePopularVideos, useLatestVideos, useRomanticVideos, useDuetScenes, useActionVideos, useComedyVideos, useAnimationVideos, useFantasyVideos, useDramaVideos, useSyncCollection } from "@/hooks/useVideos";
import Movie from "@/components/movie/Movie";
import type { TokenDetailResponse } from "@/types/pitch";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";

export default function HomeClient() {
  const [selectedVideoId, setSelectedVideoId] = useState<string | null>(null);
  const [hoverTimeout, setHoverTimeout] = useState<NodeJS.Timeout | null>(null);
  const { data: tokens = [], isLoading, error } = useVideos();
  const { data: popularTokens = [] } = usePopularVideos();
  const { data: latestTokens = [] } = useLatestVideos();
  const { data: romanticTokens = [] } = useRomanticVideos();
  // 추가한 장르별 hook
  const {data: actionTokens = []} = useActionVideos();
  const { data: comedyTokens = [] } = useComedyVideos();
  const { data: animationTokens = [] } = useAnimationVideos();
  const { data: fantasyTokens = [] } = useFantasyVideos();
  const { data: dramaTokens = [] } = useDramaVideos();
  const { data: syncCollectionTokens = [] } = useSyncCollection();
  const { data: duetScenes = [], isLoading: duetScenesLoading, error: duetScenesError } = useDuetScenes();
  const router = useRouter();
  const searchParams = useSearchParams();

  const openModal = useCallback((youtubeId: string) => {
    if (hoverTimeout) clearTimeout(hoverTimeout);
    setSelectedVideoId(youtubeId);
  }, [hoverTimeout]);

  const closeModal = () => {
    const timeout = setTimeout(() => {
      setSelectedVideoId(null);
    }, 200);
    setHoverTimeout(timeout);
  };

  useEffect(() => {
    window.scrollTo(0, 0);
    if (tokens.length === 0) return;
    if (searchParams.get('modalId')) {
      const modalId = searchParams.get('modalId') as string;
      setTimeout(() => {
        openModal(modalId);
        router.replace('/home');
      }, 500);
    }
  }, [searchParams, router, tokens.length, openModal]);

  const videos = tokens.map(({ id, youtubeId, actor_name }) => ({
    videoId: id,
    youtubeId,
    actor_name,
  }));

  const selectedTokenData: TokenDetailResponse | undefined =
    selectedVideoId && tokens.length
      ? tokens.find((v) => v.youtubeId === selectedVideoId)
      : undefined;

  return (
    <div className="bg-neutral-950 text-white font-sans min-h-full flex flex-col">
      {/* 모바일 레이아웃 (400px 이하) */}
      <div className="block sm:hidden px-2 py-2">
        {isLoading && tokens.length === 0 && (
          <div className="flex items-center justify-center min-h-[200px]">
            <div className="flex flex-col items-center space-y-4">
              <div className="animate-spin w-6 h-6 border-2 border-green-400 border-t-transparent rounded-full" />
              <span className="text-white text-sm">영화 목록을 불러오는 중...</span>
            </div>
          </div>

        )}
        {error && (
          <div className="flex items-center justify-center min-h-[200px]">
            <div className="text-red-500 text-sm">에러 발생: {error.toString()}</div>
          </div>
        )}
       {!isLoading && !error && (
        <Movie
          tokens={tokens}
          popularTokens={popularTokens}
          latestTokens={latestTokens}
          romanticTokens={romanticTokens}
          actionTokens={actionTokens}
          comedyTokens={comedyTokens}
          animationTokens={animationTokens}
          fantasyTokens={fantasyTokens}
          dramaTokens={dramaTokens}
          syncCollectionTokens={syncCollectionTokens}
          isLoading={isLoading}
          error={error}
          duetScenes={duetScenes}
          duetScenesLoading={duetScenesLoading}
          duetScenesError={duetScenesError}
          onOpenModal={openModal}
        />
      )}
      </div>

      {/* 데스크톱 레이아웃 (400px 초과) */}
      <div className="hidden sm:block px-6 py-4">
        {isLoading && tokens.length === 0 && (
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="flex flex-col items-center space-y-4">
              <div className="animate-spin w-8 h-8 border-3 border-green-400 border-t-transparent rounded-full" />
              <span className="text-white text-lg">영화 목록을 불러오는 중...</span>
            </div>
          </div>
        )}
        {error && (
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-red-500 text-lg">에러 발생: {error.toString()}</div>
          </div>
        )}
         {!isLoading && !error && (
        <Movie
          tokens={tokens}
          popularTokens={popularTokens}
          latestTokens={latestTokens}
          romanticTokens={romanticTokens}
          actionTokens={actionTokens}
          comedyTokens={comedyTokens}
          animationTokens={animationTokens}
          fantasyTokens={fantasyTokens}
          dramaTokens={dramaTokens}
          syncCollectionTokens={syncCollectionTokens}
          isLoading={isLoading}
          error={error}
          duetScenes={duetScenes}
          duetScenesLoading={duetScenesLoading}
          duetScenesError={duetScenesError}
          onOpenModal={openModal}
        />
      )}
      </div>


      {selectedVideoId && selectedTokenData && (
        <MovieDetailModal
          youtubeId={selectedVideoId}
          isOpen={!!selectedVideoId}
          onClose={closeModal}
          tokenData={selectedTokenData}
        />
      )}
    </div>
  );
} 