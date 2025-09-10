import React from 'react'
import { ActorInfo } from './ActorInfo'
import { getYoutubeThumbnail } from '@/utils/extractYoutubeVideoId'
import { ActorMovie } from '@/types/actor'

interface LeftSectionProps {
  movies: ActorMovie[];
  actorName: string;
}

export const LeftSection: React.FC<LeftSectionProps> = ({ movies, actorName }) => {
  return (
    <div className="lg:w-1/2 flex flex-col">
         
        {/* 상단 배너 섹션 */}
        {movies.length > 0 && (
            <div className="relative h-1/2">
                {/* 유튜브 썸네일 배경 */}
                <img
                    src={getYoutubeThumbnail(movies[0].youtube_url)}
                    alt="배너 배경"
                    className="w-full h-full object-cover object-center absolute inset-0"
                    style={{ filter: 'brightness(1)' }}
                />
                {/* 어두운 오버레이 */}
                <div className="absolute inset-0 bg-black/60"></div>
                <div className="relative z-10 h-full flex items-center justify-start">
                    <div className="max-w-2xl px-8 lg:px-12 text-left">
                        <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-4 drop-shadow-lg">
                            {actorName.split(' ').map((word, idx) => (
                                <span key={idx}>
                                    {word}
                                    <br />
                                </span>
                            ))}
                        </h1>
                    </div>
                </div>
                {/* 하단 그라데이션 */}
                <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-black/70 to-transparent"></div>
            </div>
        )}
        
        {/* 하단 배우 정보 섹션 */}
        <ActorInfo movies={movies} actorName={actorName} />
    </div>
  )
}
