import React, { useEffect, useRef, useState } from 'react'
import { getYoutubeThumbnail } from '@/utils/extractYoutubeVideoId'
import { ActorMovie } from '@/types/actor'
import { MovieDetail } from './MovieDetail'
import {motion, AnimatePresence} from "framer-motion";

interface RightVideosSectionProps {
  movies: ActorMovie[];
  isLoading: boolean;
  onMovieClick: (movie: ActorMovie) => void;
}

export const RightVideosSection: React.FC<RightVideosSectionProps> = ({ 
  movies, 
  isLoading, 
  onMovieClick 
}) => {
  return (
    <div className="lg:w-1/2 lg:h-full overflow-y-auto">
        <div className="p-4 lg:p-8">
            {isLoading && <div className="text-center text-gray-400">로딩중...</div>}
            <div className="space-y-4 lg:space-y-6">
                {movies.map(movie => (
                    <div key={movie.id} className="flex flex-col gap-2">
                        <div
                            className="bg-gray-900 border-2 border-gray-800 rounded-xl overflow-hidden shadow-lg hover:border-green-500 transition-all duration-300 flex cursor-pointer transform hover:scale-105"
                            onClick={() => onMovieClick(movie)}
                        >
                            <img
                                src={getYoutubeThumbnail(movie.youtube_url)}
                                alt={movie.token_name}
                                className="w-32 lg:w-40 h-24 lg:h-28 object-cover flex-shrink-0"
                            />
                            <div className="p-3 lg:p-4 flex-1 flex flex-col justify-center">
                                <div className="text-sm lg:text-lg font-bold text-white mb-1">{movie.token_name}</div>
                                <div className="text-xs lg:text-sm text-gray-400 mb-2">{movie.category}</div>
                                <div className="text-xs text-gray-500">조회수 {movie.view_count}</div>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
            {(!isLoading && movies.length === 0) && (
                <div className="text-center text-gray-400 mt-12">등록된 영상이 없습니다.</div>
            )}
        </div>
    </div>
  )
}
