// src/components/movie/Movie.tsx
import { useState, useEffect } from "react";
import MovieDetailModal from "@/components/modal/MovieDetailModal";
import MovieList from "./MovieList";
import { NavBar } from "@/components/ui/NavBar";
import type { TokenDetailResponse } from "@/types/pitch";
import {motion, AnimatePresence} from "framer-motion";
import {
  PlayIcon,
  StarIcon,
  HeartIcon,
  SparklesIcon,
  UserGroupIcon,
  InformationCircleIcon,
  VideoCameraIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  FilmIcon,
  BoltIcon,
  FaceSmileIcon,
  BookOpenIcon,
} from "@heroicons/react/24/outline";
import { useRouter } from "next/navigation";
import DuetDetailModal from "@/components/modal/DuetDetailModal";

interface MovieProps {
  tokens: TokenDetailResponse[];
  popularTokens: TokenDetailResponse[];
  latestTokens: TokenDetailResponse[];
  romanticTokens: TokenDetailResponse[];
  actionTokens: TokenDetailResponse[];
  comedyTokens: TokenDetailResponse[];
  animationTokens: TokenDetailResponse[];
  fantasyTokens: TokenDetailResponse[];
  dramaTokens: TokenDetailResponse[];
  syncCollectionTokens: TokenDetailResponse[];
  isLoading: boolean;
  error: string | null;
  onOpenModal?: (youtubeId: string) => void;
  duetScenes?: any[];
  duetScenesLoading?: boolean;
  duetScenesError?: any;
}

export default function Movie({ 
  tokens, 
  popularTokens, 
  latestTokens, 
  romanticTokens, 
  actionTokens,
  comedyTokens,
  animationTokens,
  fantasyTokens,
  dramaTokens,
  syncCollectionTokens,
  isLoading, 
  error, 
  onOpenModal, 
  duetScenes = [], 
  duetScenesLoading = false, 
  duetScenesError = null 
}: MovieProps) {
  
  const [selectedTokenId, setSelectedTokenId] = useState<string | null>(null);
  const [hoverTimeout, setHoverTimeout] = useState<NodeJS.Timeout | null>(null);
  const [playingVideo, setPlayingVideo] = useState<string | null>(null);
  const router = useRouter();
  const [selectedDuet, setSelectedDuet] = useState<{ scene: any; pair: any } | null>(null);
  const closeDuetModal = () => setSelectedDuet(null);

  

  const videos = tokens.map(({ id, youtubeId, actor_name }) => ({
    videoId: String(id),
    youtubeId,
    actor_name,
  }));

  const selectedTokenData: TokenDetailResponse | undefined =
    selectedTokenId && tokens.length
      ? tokens.find((v) => String(v.id) === selectedTokenId)
      : undefined;

  const openModal = (tokenId: string) => {
    if (hoverTimeout) clearTimeout(hoverTimeout);
    setSelectedTokenId(tokenId);
    if (onOpenModal) {
      onOpenModal(tokenId);
    }
  };

  const closeModal = () => {
    const timeout = setTimeout(() => {
      setSelectedTokenId(null);
    }, 200);
    setHoverTimeout(timeout);
  };

  const playDubbing = (videoId: string) => {
    setPlayingVideo(videoId);
    const currentModalId = selectedTokenId;
    const query = currentModalId ? `?modalId=${currentModalId}` : '';
    router.push(`/detail/${videoId}${query}`);
  };

  const stopDubbing = () => {
    setPlayingVideo(null);
    console.log("Stopping dubbing");
  };

  const topCreators = [
    { id: 1, name: "김더빙", followers: "12.5K", image: "https://images.unsplash.com/photo-1494790108755-2616b612b786?w=150&h=150&fit=crop&crop=face" },
    { id: 2, name: "이연기", followers: "8.9K", image: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&h=150&fit=crop&crop=face" },
    { id: 3, name: "박목소리", followers: "15.2K", image: "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=150&h=150&fit=crop&crop=face" },
  ];

  const sections = [
    {
      id: "top-collections",
      title: "베스트 더빙",
      subtitle: "완벽한 더빙의 정석",
      icon: <StarIcon className="w-6 h-6 text-yellow-500" />,
      videos: popularTokens.map(({ id, youtubeId, actor_name }) => ({ videoId: String(id), youtubeId, actor_name })),
      isPlayable: false,
    },
    
    {
      id: "sync-collection",
      title: "싱크 컬렉션",
      subtitle: "짧고 강한 더빙 숏츠",
      icon: <VideoCameraIcon className="w-6 h-6 text-purple-500" />,
      videos: syncCollectionTokens.map(({ id, youtubeId, actor_name }) => ({ videoId: String(id), youtubeId, actor_name })),
      isPlayable: false,
      isShorts: true,
    },
    {
      id: "new-releases",
      title: "신상 더빙",
      subtitle: "방금 나온 신작들",
      icon: <SparklesIcon className="w-6 h-6 text-purple-500" />,
      videos: latestTokens.map(({ id, youtubeId, actor_name }) => ({ videoId: String(id), youtubeId, actor_name })),
      isPlayable: false,
    },
    {
      id: "action-dubbing",
      title: "액션 더빙",
      subtitle: "흥미진진한 액션 더빙",
      icon: <BoltIcon className="w-6 h-6 text-green-500" />,
      videos: actionTokens.map(({ id, youtubeId, actor_name }) => ({ videoId: String(id), youtubeId, actor_name })),
      isPlayable: false,
    },
    {
      id: "animation-dubbing",
      title: "애니메이션 더빙",
      subtitle: "재미있는 애니메이션 더빙",
      icon: <SparklesIcon className="w-6 h-6 text-yellow-500" />,
      videos: animationTokens.map(({ id, youtubeId, actor_name }) => ({ videoId: String(id), youtubeId, actor_name })),
      isPlayable: false,
    },
    {
      id: "comedy-dubbing",
      title: "코미디 더빙",
      subtitle: "재미있는 코미디 더빙",
      icon: <FaceSmileIcon className="w-6 h-6 text-blue-500" />,
      videos: comedyTokens.map(({ id, youtubeId, actor_name }) => ({ videoId: String(id), youtubeId, actor_name })),
      isPlayable: false,
    },

    {
      id: "trending-collabo",
      title: "로맨틱 더빙",
      subtitle: "설레는 더빙 연기",
      icon: <HeartIcon className="w-6 h-6 text-pink-500" />,
      videos: romanticTokens.map(({ id, youtubeId, actor_name }) => ({ videoId: String(id), youtubeId, actor_name })),
      isPlayable: false,
    },
    {
      id: "fantasy-dubbing",
      title: "판타지 더빙",
      subtitle: "판타지 연기",
      icon: <span className="w-7 h-7">🧙‍♂️</span>,
      videos: fantasyTokens.map(({ id, youtubeId, actor_name }) => ({ videoId: String(id), youtubeId, actor_name })),
      isPlayable: false,
    },
    {
      id: "drama-dubbing",
      title: "드라마 더빙",
      subtitle: "드라마 연기",
      icon: <BookOpenIcon className="w-6 h-6 text-yellow-500" />,
      videos: dramaTokens.map(({ id, youtubeId, actor_name }) => ({ videoId: String(id), youtubeId, actor_name })),
      isPlayable: false,
    },
  ];

  const featuredVideo = videos[0];
  const heroVideos = latestTokens.slice(0, 5);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [touchStart, setTouchStart] = useState<number | null>(null);
  const [touchEnd, setTouchEnd] = useState<number | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState(0);

  // 스와이프 감지를 위한 최소 거리
  const minSwipeDistance = 50;

  // 5초마다 자동으로 오른쪽(다음)으로 이동
  useEffect(() => {
    const interval = setInterval(() => {
      goToNext();
    }, 5000);
    return () => clearInterval(interval);
  }, [heroVideos.length, currentIndex]);

  const goToPrev = () => {
    setDirection(-1);
    setCurrentIndex((prev) => (prev === 0 ? heroVideos.length - 1 : prev - 1))
  };
  const goToNext = () => {
    setDirection(1);
    setCurrentIndex((prev) => (prev === heroVideos.length - 1 ? 0 : prev + 1))
  };
  const goToIndex = (idx: number) => {
    setDirection(idx > currentIndex ? 1 : -1);
    setCurrentIndex(idx)
  };

  // 스와이프 핸들러 함수들
  const onTouchStart = (e: React.TouchEvent) => {
    setTouchEnd(null);
    setTouchStart(e.targetTouches[0].clientX);
    setIsDragging(true);
    setDragOffset(0);
  };

  const onTouchMove = (e: React.TouchEvent) => {
    if (!touchStart) return;
    
    const currentTouch = e.targetTouches[0].clientX;
    const diff = currentTouch - touchStart;
    
    setTouchEnd(currentTouch);
    setDragOffset(diff);
  };

  const onTouchEnd = () => {
    if (!touchStart || !touchEnd) {
      setIsDragging(false);
      setDragOffset(0);
      return;
    }
    
    const distance = touchStart - touchEnd;
    const isLeftSwipe = distance > minSwipeDistance;
    const isRightSwipe = distance < -minSwipeDistance;

    if (isLeftSwipe) {
      goToNext();
    } else if (isRightSwipe) {
      goToPrev();
    }
    
    setIsDragging(false);
    setDragOffset(0);
  };
  //슬라이드가 왼쪽/오른쪽으로 이동하는지 구분
  const [direction, setDirection] = useState(0); // -1: 왼쪽, 1: 오른쪽

  // 임시 듀엣 더빙용 데이터 (실제 데이터로 교체 가능)
  const duetTokens = [
    { videoId: "201", youtubeId: "S6KnqDc-tis", actor_name: "AI 배우 김연기" },
    
  ];

  return (
    <div className="bg-black min-h-screen text-white font-sans overflow-x-hidden flex flex-col">
      
      {/* Main Content */}
      <div className="pt-24">
        {/* Hero Banner */}
        {heroVideos.length > 0 && (
          <div 
            className="relative h-[40vh] sm:h-[70vh] min-h-[300px] sm:min-h-[500px] mb-4 sm:mb-8"
            onTouchStart={onTouchStart}
            onTouchMove={onTouchMove}
            onTouchEnd={onTouchEnd}
          >
            {/* 유튜브 썸네일 배경 */}
            <div className="absolute inset-0 overflow-hidden">
              {/* 이전 이미지 (오른쪽으로 드래그할 때 보임) */}
              {isDragging && dragOffset > 0 && (
                <motion.img
                  src={`https://img.youtube.com/vi/${heroVideos[currentIndex === 0 ? heroVideos.length - 1 : currentIndex - 1].youtubeId}/maxresdefault.jpg`}
                  alt="이전 배너"
                  className="w-full h-full object-cover object-center absolute inset-0"
                  style={{ 
                    filter: 'brightness(1)',
                    transform: `translateX(${dragOffset - window.innerWidth}px)`
                  }}
                />
              )}
              
              {/* 현재 이미지 */}
              <AnimatePresence initial={false} custom={direction}>
                <motion.img
                  key={currentIndex}
                  src={`https://img.youtube.com/vi/${heroVideos[currentIndex].youtubeId}/maxresdefault.jpg`}
                  alt="배너 배경"
                  className="w-full h-full object-cover object-center absolute inset-0"
                  style={{ 
                    filter: 'brightness(1)',
                    transform: isDragging ? `translateX(${dragOffset}px)` : undefined
                  }}
                  initial={{ x: direction > 0 ? 300 : -300, opacity: 0 }}
                  animate={{ 
                    x: isDragging ? dragOffset : 0, 
                    opacity: 1 
                  }}
                  exit={{ x: direction > 0 ? -300 : 300, opacity: 0 }}
                  transition={{ 
                    type: isDragging ? "tween" : "spring", 
                    stiffness: 500, 
                    damping: 40,
                    duration: isDragging ? 0 : undefined
                  }}
                />
              </AnimatePresence>
              
              {/* 다음 이미지 (왼쪽으로 드래그할 때 보임) */}
              {isDragging && dragOffset < 0 && (
                <motion.img
                  src={`https://img.youtube.com/vi/${heroVideos[currentIndex === heroVideos.length - 1 ? 0 : currentIndex + 1].youtubeId}/maxresdefault.jpg`}
                  alt="다음 배너"
                  className="w-full h-full object-cover object-center absolute inset-0"
                  style={{ 
                    filter: 'brightness(1)',
                    transform: `translateX(${dragOffset + window.innerWidth}px)`
                  }}
                />
              )}
              {/* 기존 어두운 오버레이 */}
              <div className="absolute inset-0 bg-black/60"></div>
            </div>
            <div className="relative z-10 h-full flex items-center">
              {/* 좌측 화살표 */}
              <button
                onClick={goToPrev}
                className="absolute left-2 sm:left-4 top-1/2 -translate-y-1/2 z-20 bg-black/40 hover:bg-black/70 text-white rounded-full w-8 h-8 sm:w-10 sm:h-10 flex items-center justify-center text-lg sm:text-2xl"
                aria-label="이전 비디오"
              >
                <ChevronLeftIcon className="w-4 h-4 sm:w-6 sm:h-6" />
              </button>
              {/* 비디오 정보 */}
              <div className="max-w-7xl mx-auto px-2 w-full">
                <div className="max-w-2xl">
                  <h1 className="text-2xl sm:text-5xl md:text-7xl font-bold text-white mb-2 sm:mb-4 drop-shadow-lg">
                    {heroVideos[currentIndex].actor_name}
                  </h1>
                  <p className="text-sm sm:text-xl text-white/90 mb-4 sm:mb-8 max-w-lg">
                    YouSync와 함께하는 {heroVideos[currentIndex].actor_name}의 최신 더빙 영상입니다.
                    <br className="hidden sm:block" />
                    다양한 장르의 영화를 즐겨보세요!
                  </p>
                  <div className="flex items-center gap-2 sm:gap-4">
                    <button
                      onClick={() => {
                        openModal(String(heroVideos[currentIndex].id))}}
                      className="flex items-center gap-1 sm:gap-2 px-4 sm:px-8 py-2 sm:py-4 bg-transparent text-white rounded-full font-bold hover:bg-white/20 transition-all duration-200 transform hover:scale-105 shadow-lg border-2 border-white text-sm sm:text-base"
                    >
                      <PlayIcon className="w-4 h-4 sm:w-6 sm:h-6 text-white" />
                      <span className="text-white font-bold">재생하기</span>
                    </button>
                    <button
                      className="flex items-center gap-1 sm:gap-2 px-4 sm:px-8 py-2 sm:py-4 bg-transparent text-white rounded-full font-bold hover:bg-white/20 transition-all duration-200 transform hover:scale-105 border-2 border-white text-sm sm:text-base"
                      onClick={() => {router.push(`/actor/${encodeURIComponent(heroVideos[currentIndex].actor_name)}`)}}
                    >
                      <InformationCircleIcon className="w-4 h-4 sm:w-6 sm:h-6 text-white" />
                      <span className="text-white font-bold">배우 정보</span>
                    </button>
                  </div>
                </div>
              </div>
              {/* 우측 화살표 */}
              <button
                onClick={goToNext}
                className="absolute right-2 sm:right-4 top-1/2 -translate-y-1/2 z-20 bg-black/40 hover:bg-black/70 text-white rounded-full w-8 h-8 sm:w-10 sm:h-10 flex items-center justify-center text-lg sm:text-2xl"
                aria-label="다음 비디오"
              >
                <ChevronRightIcon className="w-4 h-4 sm:w-6 sm:h-6" />
              </button>
            </div>
            {/* 인디케이터 */}
            <div className="absolute bottom-4 sm:bottom-8 left-1/2 -translate-x-1/2 flex gap-1 sm:gap-2 z-20">
              {heroVideos.map((_, idx) => (
                <button
                  key={idx}
                  onClick={() => goToIndex(idx)}
                  className={`w-2 h-2 sm:w-3 sm:h-3 rounded-full transition-all duration-200 ${currentIndex === idx ? 'bg-white scale-125' : 'bg-gray-500 opacity-60'}`}
                  aria-label={`비디오 ${idx + 1}번`}
                />
              ))}
            </div>
            <div className="absolute bottom-0 left-0 right-0 h-16 sm:h-32 bg-gradient-to-t from-black/70 to-transparent"></div>
          </div>
        )}
        <div className="max-w-7xl mx-auto px-1 sm:px-2">
          
          {isLoading && (
            <div className="flex items-center justify-center py-10 sm:py-20">
              <div className="flex flex-col items-center space-y-4">
                <div className="animate-spin w-6 h-6 sm:w-8 sm:h-8 border-2 sm:border-3 border-green-400 border-t-transparent rounded-full" />
                <div className="text-gray-500 font-medium text-sm sm:text-base">영화를 불러오는 중...</div>
              </div>
            </div>
          )}
          {error && (
            <div className="flex items-center justify-center py-10 sm:py-20">
              <div className="text-red-400 font-medium text-sm sm:text-base">에러 발생: {error.toString()}</div>
            </div>
          )}
          {!isLoading && !error && (
            <div className="space-y-6 sm:space-y-12">
              {sections.map((section) => (
                <div key={section.id} className="relative group">
                  <div className="mb-3 sm:mb-6">
                    <div className="flex items-center gap-2 sm:gap-3 mb-1 sm:mb-2">
                      {section.icon}
                      <h2 className="text-xl sm:text-3xl font-bold bg-gradient-to-r from-green-400 via-emerald-400 to-teal-400 bg-clip-text text-transparent">
                        {section.title}
                      </h2>
                    </div>
                    <p className="text-gray-500 ml-6 sm:ml-9 font-medium text-sm sm:text-base">{section.subtitle}</p>
                  </div>
                  {/* MovieList만! */}
                  <MovieList
                    sectionId={section.id}
                    videos={section.videos}
                    isPlayable={section.isPlayable}
                    isShorts={section.isShorts}
                    playingVideo={playingVideo}
                    onPlay={playDubbing}
                    onOpenModal={openModal}
                    onStop={stopDubbing}
                  />
                  {/* 싱크 컬렉션 아래에만 듀엣 더빙 섹션 삽입 */}
                  {section.id === "sync-collection" && (
                    <div className="relative group mt-8 sm:mt-16">
                      <div className="mb-3 sm:mb-6">
                        <div className="flex items-center gap-2 sm:gap-3 mb-1 sm:mb-2">
                          <UserGroupIcon className="w-5 h-5 sm:w-6 sm:h-6 text-blue-400" />
                          <h2 className="text-xl sm:text-3xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                            배우와 듀엣 더빙
                          </h2>
                        </div>
                        <p className="text-gray-500 ml-6 sm:ml-9 font-medium text-sm sm:text-base">실제 배우와 함께 명장면을 연기해보세요!</p>
                      </div>
                      {duetScenesLoading ? (
                        <div className="text-gray-400 text-base sm:text-lg py-4 sm:py-8">듀엣 더빙 목록을 불러오는 중...</div>
                      ) : duetScenesError ? (
                        <div className="text-red-400 text-base sm:text-lg py-4 sm:py-8">듀엣 더빙 목록을 불러오지 못했습니다.</div>
                      ) : (
                        <MovieList
                          sectionId="duet-dubbing"
                          videos={duetScenes.map((scene: any) => ({
                            videoId: scene.youtube_url,
                            youtubeId: scene.duet_pair[0].youtube_url.split('v=')[1],
                            actor_name: `${scene.duet_pair[0].actor_name} & ${scene.duet_pair[1].actor_name}`
                          }))}
                          isPlayable={false}
                          onOpenModal={() => {}}
                          customRender={(video, index) => (
                            <div
                              key={`duet-${index}`}
                              className="relative bg-gray-900 border-2 border-gray-800 rounded-2xl sm:rounded-3xl overflow-hidden hover:border-blue-400 hover:shadow-2xl transition-all duration-300 cursor-pointer flex-shrink-0 transform hover:scale-105 aspect-video"
                              style={{ minWidth: "280px", maxWidth: "280px" }}
                              onClick={() => {
                                const scene = duetScenes[index];
                                setSelectedDuet({ scene, pair: scene.duet_pair[0] });
                              }}
                            >
                              {/* 듀엣 전용 오버레이 */}
                              <div className="absolute inset-0 z-10 flex flex-col justify-between pointer-events-none">
                                <div className="flex justify-between p-2 sm:p-3">
                                  <div className="flex items-center gap-1 bg-blue-500/80 text-white px-1.5 sm:px-2 py-0.5 sm:py-1 rounded-full text-xs font-bold shadow">
                                    {duetScenes[index].duet_pair[0].actor_name}
                                  </div>
                                  <div className="flex items-center gap-1 bg-green-500/80 text-white px-1.5 sm:px-2 py-0.5 sm:py-1 rounded-full text-xs font-bold shadow">
                                    {duetScenes[index].duet_pair[1].actor_name}
                                  </div>
                                </div>
                                <div className="flex justify-center mb-2 sm:mb-3">
                                  <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 text-white px-2 sm:px-3 py-0.5 sm:py-1 rounded-full text-xs font-bold shadow-lg">듀엣 더빙</span>
                                </div>
                              </div>
                              {/* 유튜브 썸네일 */}
                              <img
                                src={`https://img.youtube.com/vi/${duetScenes[index].duet_pair[0].youtube_url.split('v=')[1]}/mqdefault.jpg`}
                                alt={duetScenes[index].scene_title}
                                className="w-full h-full object-cover"
                                draggable={false}
                              />
                              {/* 카드 설명 영역 */}
                              <div className="p-3 sm:p-6 relative z-20 bg-gradient-to-t from-black/80 via-black/30 to-transparent">
                                <h3 className="font-bold mb-1 sm:mb-2 text-white text-sm sm:text-lg">
                                  {duetScenes[index].scene_title}
                                </h3>
                                <p className="text-xs sm:text-sm text-gray-300 font-medium mb-2 sm:mb-3">
                                  {duetScenes[index].duet_pair[0].actor_name} & {duetScenes[index].duet_pair[1].actor_name}
                                </p>
                                <button
                                  className="flex items-center gap-1 sm:gap-2 px-2 sm:px-4 py-1 sm:py-2 rounded-full text-xs sm:text-sm font-bold transition-colors shadow-lg bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 text-white hover:brightness-110"
                                  style={{ pointerEvents: 'auto' }}
                                >
                                  <PlayIcon className="w-3 h-3 sm:w-4 sm:h-4" />
                                  듀엣 시작
                                </button>
                              </div>
                            </div>
                          )}
                        />
                      )}
                    </div>
                  )}
                </div>
              ))}
              {/* 듀엣 더빙 전용 섹션 */}
              {/* This section is now rendered inside the sections.map loop */}
            </div>
          )}
        </div>
      </div>
      {/* 모달 렌더링 */}
      <MovieDetailModal
        youtubeId={selectedTokenData?.youtubeId || ''}
        isOpen={!!selectedTokenId}
        onClose={closeModal}
        tokenData={selectedTokenData as any}
      />
      {/* 듀엣 디테일 모달 */}
      {selectedDuet && (
        <DuetDetailModal
          youtubeId={selectedDuet.pair.youtube_url.split("v=")[1]}
          isOpen={!!selectedDuet}
          onClose={closeDuetModal}
          tokenData={selectedDuet.pair}
          duetPair={selectedDuet.scene.duet_pair.map((a: any) => ({ actor_name: a.actor_name, actor_id: a.id }))}
        />
      )}
    </div>
  );
}