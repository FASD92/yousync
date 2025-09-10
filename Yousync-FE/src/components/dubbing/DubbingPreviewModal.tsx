// import React, { useEffect, useRef, useState } from "react";
// import VideoPlayer from "./VideoPlayer";

// interface DubbingPreviewModalProps {
//   open: boolean;
//   onClose: () => void;
//   videoId?: string; // YouTube 영상 ID
//   audioUrl?: string; // 백그라운드 오디오 URL
//   startTime?: number; // 문장 시작 시간
//   endTime?: number; // 문장 종료 시간
//   currentScript?: string; // 현재 문장 내용
//   userRecordingBlob?: Blob; // 사용자 녹음 파일
// }

// const DubbingPreviewModal: React.FC<DubbingPreviewModalProps> = ({ 
//   open, 
//   onClose, 
//   videoId = "dQw4w9WgXcQ",
//   audioUrl,
//   startTime = 0,
//   endTime,
//   currentScript,
//   userRecordingBlob
// }) => {
//   const audioRef = useRef<HTMLAudioElement | null>(null);
//   const userAudioRef = useRef<HTMLAudioElement | null>(null);
//   const [audioDuration, setAudioDuration] = useState<number>(0);
//   const [currentTime, setCurrentTime] = useState<number>(0);
//   const [isAudioLoaded, setIsAudioLoaded] = useState<boolean>(false);
//   const [isVideoPlaying, setIsVideoPlaying] = useState<boolean>(false);
//   const [isAudioFinished, setIsAudioFinished] = useState<boolean>(false);
//   const [hasUserRecording, setHasUserRecording] = useState<boolean>(false);
//   const [isUserRecordingPlaying, setIsUserRecordingPlaying] = useState<boolean>(false);

//   // 시간 포맷팅 함수
//   const formatTime = (seconds: number): string => {
//     const mins = Math.floor(seconds / 60);
//     const secs = Math.floor(seconds % 60);
//     return `${mins}:${secs.toString().padStart(2, '0')}`;
//   };

//   // 사용자 녹음만 재생/정지하는 함수
//   const toggleUserRecording = () => {
//     if (!userRecordingBlob || !userAudioRef.current) return;
    
//     if (isUserRecordingPlaying) {
//       // 정지
//       userAudioRef.current.pause();
//       userAudioRef.current.currentTime = 0;
//       setIsUserRecordingPlaying(false);
//     } else {
//       // 재생
//       userAudioRef.current.currentTime = 0;
//       userAudioRef.current.play().catch(error => {
//         console.log('사용자 녹음 재생 실패:', error);
//       });
//       setIsUserRecordingPlaying(true);
//     }
//   };

//   // 모달이 열릴 때 초기화
//   useEffect(() => {
//     if (open && audioUrl && audioRef.current) {
//       // 모달이 열리면 오디오를 준비하지만 재생하지 않음
//       audioRef.current.currentTime = startTime;
//       setCurrentTime(startTime);
//       setIsAudioFinished(false);
//       setIsVideoPlaying(false); // 모달 열릴 때 재생 상태 초기화
//     }
    
//     // 사용자 녹음 파일 확인
//     setHasUserRecording(!!userRecordingBlob);
//   }, [open, audioUrl, startTime, userRecordingBlob]);

//   // 모달이 닫힐 때 오디오 정지
//   useEffect(() => {
//     if (!open) {
//       if (audioRef.current) {
//         audioRef.current.pause();
//         audioRef.current.currentTime = startTime;
//         setCurrentTime(startTime);
//       }
//       if (userAudioRef.current) {
//         userAudioRef.current.pause();
//         userAudioRef.current = null;
//       }
//       setIsVideoPlaying(false);
//       setIsAudioFinished(false);
//     }
//   }, [open, startTime]);

//   // 영상 재생 상태에 따라 배경음악과 사용자 녹음 제어
//   useEffect(() => {
//     if (isVideoPlaying && !isAudioFinished) {
//       // 배경음악 재생
//       if (audioRef.current) {
//         audioRef.current.currentTime = startTime;
//         audioRef.current.play().catch(error => {
//           console.log('배경음악 재생 실패:', error);
//         });
//       }
      
//       // 사용자 녹음 재생
//       if (userRecordingBlob && userAudioRef.current) {
//         userAudioRef.current.currentTime = 0;
//         userAudioRef.current.play().catch(error => {
//           console.log('사용자 녹음 재생 실패:', error);
//         });
//       }
//     } else {
//       // 모든 오디오 정지
//       if (audioRef.current) {
//         audioRef.current.pause();
//       }
//       if (userAudioRef.current) {
//         userAudioRef.current.pause();
//       }
//     }
//   }, [isVideoPlaying, isAudioFinished, startTime, userRecordingBlob]);

//   // 오디오 로드 완료 시 길이 설정
//   const handleAudioLoadedMetadata = () => {
//     if (audioRef.current) {
//       setAudioDuration(audioRef.current.duration);
//       setIsAudioLoaded(true);
//       console.log('오디오 길이:', audioRef.current.duration);
//       console.log('재생 구간:', startTime, '~', endTime);
//     }
//   };

//   // 오디오 재생 중 시간 업데이트
//   const handleTimeUpdate = () => {
//     if (audioRef.current) {
//       const currentAudioTime = audioRef.current.currentTime;
//       setCurrentTime(currentAudioTime);
      
//       // endTime에 도달하면 오디오 정지
//       if (endTime && currentAudioTime >= endTime) {
//         audioRef.current.pause();
//         if (userAudioRef.current) {
//           userAudioRef.current.pause();
//         }
//         setIsAudioFinished(true);
//         console.log('배경음악 구간 재생 완료');
//       }
//     }
//   };

//   // 영상 재생/정지 핸들러
//   const handleVideoPlay = () => {
//     console.log('영상 재생 시작 - 배경음악과 사용자 녹음도 재생');
//     setIsVideoPlaying(true);
//     setIsAudioFinished(false);
//   };

//   const handleVideoPause = () => {
//     console.log('영상 정지 - 배경음악과 사용자 녹음도 정지');
//     setIsVideoPlaying(false);
//   };

//   if (!open) return null;
//   console.log("videoId", videoId);
//   console.log("audioUrl", audioUrl);
//   console.log("재생 구간:", startTime, "~", endTime);
//   console.log("사용자 녹음 있음:", hasUserRecording);

//   return (
//     <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80">
//       <div className="bg-white rounded-lg shadow-xl p-6 min-w-[600px] max-w-[80vw] max-h-[80vh] relative overflow-y-auto">
//         <button
//           className="absolute top-4 right-4 text-gray-500 hover:text-gray-800 text-2xl font-bold z-10"
//           onClick={onClose}
//           aria-label="닫기"
//         >
//           ×
//         </button>
//         <h2 className="text-xl font-bold mb-4 text-gray-900">더빙본 미리보기</h2>
        
//         {/* 현재 문장 정보 */}
//         {currentScript && (
//           <div className="mb-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
//             <h3 className="text-lg font-semibold mb-2 text-blue-800">현재 문장</h3>
//             <div className="space-y-2">
//               <div className="text-gray-700 font-medium">{currentScript}</div>
//               <div className="flex items-center gap-4 text-sm text-blue-600">
//                 <span className="bg-blue-100 px-2 py-1 rounded">
//                   시작: {formatTime(startTime)}
//                 </span>
//                 <span className="bg-blue-100 px-2 py-1 rounded">
//                   종료: {formatTime(endTime || 0)}
//                 </span>
//                 <span className="bg-blue-100 px-2 py-1 rounded">
//                   길이: {formatTime((endTime || 0) - startTime)}
//                 </span>
//                 {hasUserRecording && (
//                   <span className="bg-green-100 px-2 py-1 rounded text-green-700">
//                     사용자 녹음 있음
//                   </span>
//                 )}
//               </div>
//             </div>
//           </div>
//         )}
        
//         <div className="w-full">
//           <VideoPlayer 
//             videoId={videoId} 
//             muted={true} 
//             onPlay={handleVideoPlay}
//             onPause={handleVideoPause}
//           />
          
//           {/* 백그라운드 오디오 정보 */}
//           {audioUrl && (
//             <div className="mt-4 p-4 bg-gray-100 rounded-lg">
//               <h3 className="text-lg font-semibold mb-2 text-gray-800">오디오 정보</h3>
//               <div className="space-y-2 text-sm text-gray-600">
//                 <div className="break-all">배경음 URL: {audioUrl}</div>
//                 {isAudioLoaded ? (
//                   <div className="flex flex-wrap items-center gap-4">
//                     <span className="bg-blue-100 px-2 py-1 rounded">전체 길이: {formatTime(audioDuration)}</span>
//                     <span className="bg-purple-100 px-2 py-1 rounded">재생 구간: {formatTime(startTime)} ~ {formatTime(endTime || 0)}</span>
//                     <span className="bg-green-100 px-2 py-1 rounded">현재: {formatTime(currentTime)}</span>
//                     <span className="bg-yellow-100 px-2 py-1 rounded">진행률: {endTime && startTime ? Math.round(((currentTime - startTime) / (endTime - startTime)) * 100) : 0}%</span>
//                     <span className={`px-2 py-1 rounded ${isVideoPlaying ? 'bg-green-200' : 'bg-red-200'}`}>
//                       상태: {isVideoPlaying ? '재생 중' : '정지'}
//                     </span>
//                     {isAudioFinished && (
//                       <span className="bg-orange-200 px-2 py-1 rounded">구간 재생 완료</span>
//                     )}
//                   </div>
//                 ) : (
//                   <div className="text-blue-600">오디오 로딩 중...</div>
//                 )}
//               </div>
//             </div>
//           )}
          
//           <div className="mt-4 text-center text-gray-600">확인용</div>
          
//           {/* 백그라운드 오디오 재생 */}
//           {audioUrl && (
//             <audio
//               ref={audioRef}
//               src={audioUrl}
//               preload="auto"
//               className="hidden"
//               onLoadedMetadata={handleAudioLoadedMetadata}
//               onTimeUpdate={handleTimeUpdate}
//             />
//           )}
          
//           {/* 사용자 녹음 오디오 재생 */}
//           {userRecordingBlob && (
//             <audio
//               ref={userAudioRef}
//               src={URL.createObjectURL(userRecordingBlob)}
//               preload="auto"
//               className="hidden"
//             />
//           )}
//         </div>
//       </div>
//     </div>
//   );
// };

// export default DubbingPreviewModal; 