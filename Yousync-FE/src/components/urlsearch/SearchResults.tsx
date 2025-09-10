import { useState, useCallback } from 'react';
import type { TokenDetailResponse } from '@/types/pitch';
import { FireIcon } from "@heroicons/react/24/outline";

interface SearchResultsProps {
  tokens: TokenDetailResponse[];
  onOpenModal: (token: TokenDetailResponse) => void;
}

export default function SearchResults({ tokens, onOpenModal }: SearchResultsProps) {
  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds < 10 ? '0' : ''}${remainingSeconds}`;
  };

  return (
    <div className="space-y-12">
      <div className="relative group">
        <div className="mb-6 text-center">
          <div className="flex items-center gap-3 mb-2 justify-center">
            <FireIcon className="w-6 h-6 text-orange-500" />
            <h2 className="text-3xl font-bold bg-gradient-to-r from-green-400 via-emerald-400 to-teal-400 bg-clip-text text-transparent">
              {tokens.length > 0 ? tokens[0].token_name : '검색 결과'}
            </h2>
          </div>
          <p className="text-gray-500 font-medium text-center">유튜브 URL 검색 결과입니다.</p>
        </div>

        <div className="flex flex-col items-center gap-0">
          {tokens.length > 0 && (
            <div
              className="bg-black rounded-lg overflow-hidden w-full max-w-lg flex flex-col"
            >
              <div className="relative w-full" style={{ paddingTop: '56.25%' }}>
                <iframe
                  className="absolute top-0 left-0 w-full h-full rounded-lg"
                  src={`https://www.youtube.com/embed/${tokens[0].youtubeId}?autoplay=1&mute=1&controls=0&loop=1&playlist=${tokens[0].youtubeId}`}
                  frameBorder="0"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                  title={tokens[0].token_name}
                ></iframe>
              </div>
              <div className="flex-grow flex flex-col justify-center">
              </div>
            </div>
          )}

          <div className="flex flex-col items-center gap-4">
            {tokens.map((token) => (
              <div
                key={token.id}
                className="bg-black rounded-lg shadow-lg p-4 cursor-pointer transform transition duration-300 hover:scale-105 w-full max-w-xs text-center"
                onClick={() => onOpenModal(token)}
              >
                <p className="text-xl font-semibold bg-gradient-to-r from-green-400 via-emerald-400 to-teal-400 bg-clip-text text-transparent">{token.actor_name}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
