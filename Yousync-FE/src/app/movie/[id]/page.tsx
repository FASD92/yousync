'use client'
import React, { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import { extractYoutubeVideoId } from '@/utils/extractYoutubeVideoId';
import AnalysisVideo from '@/components/movies/AnalysisVideo';
import PreviewVideo from '@/components/movies/PreviewVideo';
import { getDuetTokenDetail } from '@/app/api/hello/getDuetTokenDetail';
import { getTokenDetail } from '@/app/api/hello/getTokenDetail';
export default function MovieAnalysisPage() {
  const params = useParams();
  const videoId = params?.id as string;
  const [url, setUrl] = useState(videoId ? `https://youtu.be/${videoId}` : '');
  const [submittedUrl, setSubmittedUrl] = useState('');
  const [captions, setCaptions] = useState<{ id: number; script: string; start_time: number; end_time: number; actor?: { name: string } }[]>([]);
  const [expandedCardId, setExpandedCardId] = useState<number | null>(null);

  useEffect(() => {
    if (videoId) {
      setUrl(`https://youtu.be/${videoId}`);
    }
  }, [videoId]);

  // 하드코딩 id=1로 데이터 fetch
  useEffect(() => {
    async function fetchData() {
      try {
        const { front_data } = await getTokenDetail('1');
        setCaptions(front_data.captions.map((c: any) => ({ id: c.id, script: c.script, start_time: c.start_time, end_time: c.end_time, actor: c.actor })));
      } catch (e) {
        setCaptions([]);
      }
    }
    fetchData();
  }, []);

  useEffect(() => {
    console.log('[MovieAnalysisPage] captions:', captions);
  }, [captions]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSubmittedUrl(url);
  };

  // 분석 시작 시 videoId 추출
  const submittedVideoId = extractYoutubeVideoId(submittedUrl);

  // 유튜브 썸네일 URL 생성
  const getThumbnailUrl = (videoId: string) => `https://img.youtube.com/vi/${videoId}/mqdefault.jpg`;

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-900 py-12">
      {!submittedVideoId && (
        <div className="bg-gray-800 rounded-xl shadow-lg p-8 w-full max-w-xl border border-gray-700">
          <h1 className="text-2xl font-bold text-white mb-6 text-center">영화/영상 URL 분석</h1>
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <input
              type="text"
              className="px-4 py-2 rounded border border-gray-600 bg-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-emerald-400"
              placeholder="YouTube 등 영상 URL을 입력하세요"
              value={url}
              onChange={e => setUrl(e.target.value)}
              required
            />
            <button
              type="submit"
              className="bg-emerald-500 hover:bg-emerald-600 text-white font-semibold py-2 rounded transition-all"
            >
              분석 시작
            </button>
          </form>
        </div>
      )}
      {submittedVideoId && (
        <div className="mt-10 flex flex-row justify-start items-start w-full">
          <div className="w-[640px] max-w-full ml-10">
            <AnalysisVideo videoId={submittedVideoId} />
          </div>
          {/* 오른쪽 커다란 패널 */}
          <div className="ml-16 flex flex-col items-center w-[700px]">
            <div className="w-full bg-gray-800/90 rounded-3xl shadow-2xl p-8 flex flex-col gap-4">
              {captions.length > 0 ? (
                captions.map((c, idx) => {
                  const isExpanded = expandedCardId === c.id;
                  return (
                    <div
                      key={c.id}
                      className={
                        `w-full bg-gradient-to-r from-emerald-700/30 to-gray-700/60 border border-emerald-400/30 ${isExpanded ? 'rounded-3xl' : 'rounded-full'} shadow flex flex-col transition-transform duration-200 hover:scale-80 hover:z-20 cursor-pointer mb-2`
                        + (isExpanded ? ' ring-2 ring-emerald-400/60 bg-gray-900/90' : '')
                      }
                      onClick={() => setExpandedCardId(isExpanded ? null : c.id)}
                    >
                      <div className="flex flex-row items-center px-12 py-2 gap-4">
                        <div className="relative group">
                          <img
                            src={getThumbnailUrl(videoId)}
                            alt="썸네일"
                            className="w-24 h-16 object-cover rounded-xl border border-gray-700 transition duration-200 group-hover:brightness-75 transform group-hover:scale-110 group-hover:z-10"
                          />
                          {/* 플레이 아이콘 오버레이 */}
                          <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                            <svg className="w-10 h-10 text-white drop-shadow-lg" fill="currentColor" viewBox="0 0 24 24">
                              <path d="M8 5v14l11-7z" />
                            </svg>
                          </div>
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="text-emerald-300 text-base font-bold truncate">{c.actor?.name || `화자 ${idx+1}`}</div>
                          <div className="text-xs text-gray-400 mt-0.5">{c.start_time} ~ {c.end_time}초</div>
                          <div className="text-white text-base font-semibold mt-1 break-words">{c.script}</div>
                        </div>
                      </div>
                      {/* 확장 영역 */}
                      {isExpanded && (
                        <div className="w-full bg-gray-900/95 rounded-b-3xl shadow-2xl mt-0 p-6 flex flex-col items-center animate-fadein">
                          <div className="w-full flex flex-col items-center">
                            <div className="w-[560px] h-[315px] flex items-center justify-center">
                              <PreviewVideo
                                videoId={videoId}
                                startTime={c.start_time}
                                endTime={c.end_time}
                              />
                            </div>
                            <div className="mt-4 w-full flex flex-row gap-4 items-center">
                              <div className="flex-1 flex flex-col gap-2 text-left">
                                <div className="text-emerald-300 text-base font-bold">화자: {c.actor?.name || `화자 ${idx+1}`}</div>
                                <div className="text-xs text-gray-400">구간: {c.start_time} ~ {c.end_time}초</div>
                                <div className="text-white text-base font-semibold">문장: {c.script}</div>
                              </div>
                              <button
                                className="bg-emerald-500 hover:bg-emerald-600 text-white font-bold rounded-xl px-6 py-3 transition-all shadow-lg"
                                type="button"
                              >
                                더빙하기
                              </button>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })
              ) : (
                <div className="text-gray-400">화자 구간 데이터를 불러오는 중...</div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 