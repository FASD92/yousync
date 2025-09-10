"use client"
import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  ChatBubbleLeftIcon, 
  HeartIcon, 
  ShareIcon, 
  BookmarkIcon,
  PlayIcon,
  PauseIcon
} from '@heroicons/react/24/outline'
import { HeartIcon as HeartSolidIcon } from '@heroicons/react/24/solid'

const VIDEO_WIDTH = 432;
const VIDEO_HEIGHT = 768;
const COMMENT_WIDTH = 320;

const SynccollectionPage = () => {
  const [isLiked, setIsLiked] = useState(false)
  const [isBookmarked, setIsBookmarked] = useState(false)
  const [showComments, setShowComments] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)

  const handleLike = () => {
    setIsLiked(!isLiked)
  }

  const handleBookmark = () => {
    setIsBookmarked(!isBookmarked)
  }

  const toggleComments = () => {
    setShowComments(!showComments)
  }

  const togglePlay = () => {
    setIsPlaying(!isPlaying)
  }

  return (
    <div className="relative min-h-screen bg-black flex items-center justify-center">
      {/* 영상+댓글창을 감싸는 flex row 컨테이너 */}
      <div className="flex items-center justify-center">
        {/* Shorts 비율의 비디오 카드 */}
        <div
          className="relative bg-gradient-to-b from-gray-900 to-black rounded-2xl shadow-2xl overflow-hidden flex flex-col justify-between"
          style={{ width: VIDEO_WIDTH, height: VIDEO_HEIGHT }}
        >
          {/* 비디오 플레이스홀더 (실제로는 YouTube iframe이 들어갈 예정) */}
          <div className="absolute inset-0 flex items-center justify-center z-0">
            <div className="text-center text-white select-none pointer-events-none">
              <div className="text-6xl mb-4">🎬</div>
              <div className="text-xl font-semibold">더빙 숏츠</div>
              <div className="text-gray-400 mt-2">YouTube Shorts 스타일</div>
            </div>
          </div>

          {/* 우측 액션 버튼들 */}
          <div className="absolute right-4 bottom-24 flex flex-col items-center space-y-6 z-10">
            {/* 프로필 이미지 */}
            <div className="w-12 h-12 rounded-full bg-gradient-to-r from-green-400 to-blue-500 flex items-center justify-center text-white font-bold text-lg">
              Y
            </div>

            {/* 좋아요 버튼 */}
            <motion.button
              whileTap={{ scale: 0.9 }}
              onClick={handleLike}
              className="flex flex-col items-center space-y-1"
            >
              {isLiked ? (
                <HeartSolidIcon className="w-8 h-8 text-red-500" />
              ) : (
                <HeartIcon className="w-8 h-8 text-white" />
              )}
              <span className="text-white text-xs font-medium">좋아요</span>
            </motion.button>

            {/* 댓글 버튼 */}
            <motion.button
              whileTap={{ scale: 0.9 }}
              onClick={toggleComments}
              className="flex flex-col items-center space-y-1"
            >
              <ChatBubbleLeftIcon className="w-8 h-8 text-white" />
              <span className="text-white text-xs font-medium">댓글</span>
            </motion.button>

            {/* 공유 버튼 */}
            <motion.button
              whileTap={{ scale: 0.9 }}
              className="flex flex-col items-center space-y-1"
            >
              <ShareIcon className="w-8 h-8 text-white" />
              <span className="text-white text-xs font-medium">공유</span>
            </motion.button>

            {/* 북마크 버튼 */}
            <motion.button
              whileTap={{ scale: 0.9 }}
              onClick={handleBookmark}
              className="flex flex-col items-center space-y-1"
            >
              {isBookmarked ? (
                <BookmarkIcon className="w-8 h-8 text-green-500" />
              ) : (
                <BookmarkIcon className="w-8 h-8 text-white" />
              )}
              <span className="text-white text-xs font-medium">저장</span>
            </motion.button>
          </div>

          {/* 하단 비디오 정보 */}
          <div className="absolute bottom-4 left-4 right-20 z-10">
            <div className="text-white">
              <h3 className="text-lg font-bold mb-2">@yousync_official</h3>
              <p className="text-sm mb-2">재미있는 더빙 숏츠입니다! 🎭</p>
            </div>
          </div>

          {/* 재생/일시정지 버튼 */}
          <motion.button
            whileTap={{ scale: 0.9 }}
            onClick={togglePlay}
            className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-black/50 rounded-full p-4 z-10"
          >
            {isPlaying ? (
              <PauseIcon className="w-8 h-8 text-white" />
            ) : (
              <PlayIcon className="w-8 h-8 text-white" />
            )}
          </motion.button>
        </div>

        {/* 댓글 패널 */}
        <AnimatePresence>
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={showComments ? { width: COMMENT_WIDTH, opacity: 1 } : { width: 0, opacity: 0 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="overflow-hidden ml-4 bg-[#181f1a] rounded-2xl shadow-2xl flex flex-col border border-gray-700"
            style={{ height: VIDEO_HEIGHT, minWidth: 0, maxWidth: COMMENT_WIDTH }}
          >
            {showComments && (
              <div className="h-full flex flex-col">
                {/* 댓글 헤더 */}
                <div className="flex items-center justify-between p-4 border-b border-gray-700">
                  <h3 className="text-lg font-semibold text-green-500">댓글</h3>
                  <motion.button
                    whileTap={{ scale: 0.9 }}
                    onClick={toggleComments}
                    className="text-gray-400 hover:text-green-400"
                  >
                    ✕
                  </motion.button>
                </div>

                {/* 댓글 목록 */}
                <div className="flex-1 overflow-y-auto p-4 space-y-4">
                  <div className="flex space-x-3">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-r from-green-400 to-blue-500 flex items-center justify-center text-white text-sm font-bold">
                      U
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <span className="font-semibold text-sm text-gray-200">user123</span>
                        <span className="text-xs text-gray-500">2시간 전</span>
                      </div>
                      <p className="text-sm text-gray-100">정말 재미있네요! 👍</p>
                    </div>
                  </div>

                  <div className="flex space-x-3">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-r from-purple-400 to-pink-500 flex items-center justify-center text-white text-sm font-bold">
                      A
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <span className="font-semibold text-sm text-gray-200">actor_fan</span>
                        <span className="text-xs text-gray-500">1시간 전</span>
                      </div>
                      <p className="text-sm text-gray-100">더빙 실력이 대단해요! 🎭</p>
                    </div>
                  </div>

                  <div className="flex space-x-3">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-r from-yellow-400 to-orange-500 flex items-center justify-center text-white text-sm font-bold">
                      C
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <span className="font-semibold text-sm text-gray-200">creative_user</span>
                        <span className="text-xs text-gray-500">30분 전</span>
                      </div>
                      <p className="text-sm text-gray-100">이런 콘텐츠 더 만들어주세요! 💕</p>
                    </div>
                  </div>
                </div>

                {/* 댓글 입력 */}
                <div className="p-4 border-t border-gray-700">
                  <div className="flex space-x-2">
                    <input
                      type="text"
                      placeholder="댓글을 입력하세요..."
                      className="flex-1 px-3 py-2 border border-gray-700 bg-[#232a23] rounded-full text-sm text-white focus:outline-none focus:border-green-400 placeholder:text-gray-400"
                    />
                    <motion.button
                      whileTap={{ scale: 0.9 }}
                      className="px-4 py-2 bg-green-500 text-white rounded-full text-sm font-medium hover:bg-green-400"
                    >
                      게시
                    </motion.button>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  )
}

export default SynccollectionPage