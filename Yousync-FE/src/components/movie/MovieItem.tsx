import { PlayIcon, SpeakerWaveIcon } from "@heroicons/react/24/outline";
import type { MovieItemProps } from "@/types/video";

export default function MovieItem({
  video,
  isPlayable,
  isShorts,
  playingVideo,
  onPlay,
  onOpenModal,
  onStop,
}: MovieItemProps) {

  return (
    <div
      className={`relative group/video bg-gray-900 border-2 border-gray-800 rounded-3xl overflow-hidden cursor-pointer flex-shrink-0 transition-all duration-300 hover:scale-105 hover:border-green-500 hover:shadow-2xl ${
        isShorts ? "aspect-[9/16]" : "aspect-video"
      }`}
      style={{
        minWidth: isShorts ? "180px" : "280px",
        maxWidth: isShorts ? "180px" : "280px",
      }}
      onClick={() => {
        if (isPlayable) {
          if (onPlay) onPlay(video.videoId);
        } else {
          if (onOpenModal) onOpenModal(video.videoId);
        }
      }}
    >
      {/* 유튜브 썸네일만 표시 */}
      {video.youtubeId ? (
        <img
          src={`https://img.youtube.com/vi/${video.youtubeId}/mqdefault.jpg`}
          alt={video.actor_name}
          className="w-full h-full object-cover"
          draggable={false}
          onError={(e) => {
            // 에러 시 기본 배경으로 fallback
            e.currentTarget.style.display = 'none';
          }}
        />
      ) : null}

      {/* 아래 카드 설명 영역(더빙/버튼 등)는 그대로 남겨둡니다 */}
      {!isShorts && (
        <div className="p-6">
          <h3 className="font-bold mb-2 text-white group-hover/video:text-green-400 transition-colors text-lg">
            {video.actor_name} 더빙
          </h3>
          <p className="text-sm text-gray-500 font-medium">
            {isPlayable
              ? "클릭해서 바로 들어보세요!"
              : "AI와 함께하는 재미있는 더빙 연습!"}
          </p>
          {isPlayable && (
            <div className="mt-3">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  if (playingVideo && playingVideo === video.youtubeId) {
                    if (onStop) onStop();
                  } else {
                    if (onPlay) onPlay(video.videoId);
                  }
                }}
                className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-full text-sm font-bold hover:bg-green-700 transition-colors"
              >
                {playingVideo === video.youtubeId ? (
                  <>
                    <SpeakerWaveIcon className="w-4 h-4" />
                    재생 중...
                  </>
                ) : (
                  <>
                    <SpeakerWaveIcon className="w-4 h-4" />
                    들어보기
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}