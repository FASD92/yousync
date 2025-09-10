import { ActorMovie } from '@/types/actor'
import { extractYoutubeVideoId } from '@/utils/extractYoutubeVideoId';
import React, { useEffect } from 'react'
interface VideoDetailProps {
    movie: ActorMovie;
}
export const MovieDetail: React.FC<VideoDetailProps> = ({movie}) => {
    useEffect(() => {
        console.log(">> 흠", extractYoutubeVideoId(movie.youtube_url));
    }, [movie])

    // const handleDubbingClick = () => {
    //     router.replace(`/dubbing/${tokenData.id}?modalId=${youtubeId}`);
    //   };

    // 필요한것 :
    // tokenData.id, actor_id
  return (
    <div
    className="text-white bg-gray-900 border-2 border-gray-800 rounded-xl shadow-lg hover:border-green-500 transition-all duration-300 flex flex-col justify-between p-4"
    >
        <div className="flex-1">
            <div className="text-lg font-bold mb-2">{movie.token_name}</div>
            <div className="text-sm text-gray-400">{movie.category}</div>
            <div className="text-sm text-gray-400">총시간 {movie.end_time - movie.start_time}초</div>
        </div>
        <div className="flex justify-end">
            <button className="px-4 py-2 bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500 hover:from-green-600 hover:via-emerald-600 hover:to-teal-600 text-white rounded-lg font-medium transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105">
                더빙하기
            </button>
        </div>
    </div>
  )
}
export default MovieDetail;