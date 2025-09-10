"use client"
import React, { useState } from 'react'
import { extractYoutubeVideoId } from '@/utils/extractYoutubeVideoId';
import MovieDetailModal from '@/components/modal/MovieDetailModal';
import type { TokenDetailResponse } from '@/types/pitch';
import { ActorMovie } from '@/types/actor';
import { LeftSection } from './LeftSection';
import { RightVideosSection } from './RightVideosSection';

interface ActorContainerProps {
  movies: ActorMovie[];
  actorName: string;
  isLoading: boolean;
}

export const ActorContainer: React.FC<ActorContainerProps> = ({ 
  movies, 
  actorName, 
  isLoading 
}) => {
  const [selectedMovie, setSelectedMovie] = useState<ActorMovie | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // 영상 카드 클릭 핸들러
  const handleMovieClick = (movie: ActorMovie) => {
    setSelectedMovie(movie);
    setIsModalOpen(true);
  };

  // 모달 닫기 핸들러
  const handleCloseModal = () => {
    setIsModalOpen(false);
    setSelectedMovie(null);
  };

  // ActorMovie를 TokenDetailResponse로 변환하는 함수
  const convertToTokenDetailResponse = (movie: ActorMovie): TokenDetailResponse => {
    const youtubeId = extractYoutubeVideoId(movie.youtube_url);
    return {
      ...movie,
      pitch: [],
      bgvoice_url: movie.s3_bgvoice_url,
      scripts: [],
      youtubeId: youtubeId || '',
    };
  };

  return (
    <div className="h-screen overflow-hidden bg-neutral-950">
      <div className="flex flex-col lg:flex-row h-full pt-20">
        {/* 왼쪽 섹션 */}
        <LeftSection movies={movies} actorName={actorName} />
        {/* 오른쪽 영화 리스트 섹션 */}
        <RightVideosSection 
          movies={movies} 
          isLoading={isLoading} 
          onMovieClick={handleMovieClick} 
        />
      </div>

      {/* MovieDetailModal */}
      {selectedMovie && (
        <MovieDetailModal
          youtubeId={extractYoutubeVideoId(selectedMovie.youtube_url) || ''}
          isOpen={isModalOpen}
          onClose={handleCloseModal}
          tokenData={convertToTokenDetailResponse(selectedMovie)}
        />
      )}
    </div>
  );
}
