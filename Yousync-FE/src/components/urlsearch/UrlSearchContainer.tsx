// src/components/urlsearch/UrlSearchContainer.tsx
"use client";

import { useState, useCallback } from 'react';
import { NavBar } from '@/components/ui/NavBar';
import ErrorMessage from './ErrorMessage';
import SearchResults from './SearchResults';
import MovieDetailModal from '@/components/modal/MovieDetailModal';
import type { TokenDetailResponse } from '@/types/pitch';

interface UrlSearchContainerProps {
  initialTokens: TokenDetailResponse[];
  errorMessage: string | null;
}

export default function UrlSearchContainer({ initialTokens, errorMessage }: UrlSearchContainerProps) {
  const [selectedToken, setSelectedToken] = useState<TokenDetailResponse | null>(null);

  const openModal = useCallback((token: TokenDetailResponse) => {
    setSelectedToken(token);
  }, []);

  const closeModal = () => {
    setSelectedToken(null);
  };

  const selectedTokenData = selectedToken;

  return (
    <div className="bg-black min-h-screen text-white font-sans overflow-x-hidden flex flex-col">
      <NavBar />
      <div className="pt-24 max-w-7xl mx-auto px-2 w-full">
        {errorMessage && <ErrorMessage message={errorMessage} />}

        {!errorMessage && initialTokens.length > 0 && (
          <SearchResults tokens={initialTokens} onOpenModal={openModal} />
        )}
      </div>
      {selectedToken && selectedTokenData && (
        <MovieDetailModal
          youtubeId={selectedToken.youtubeId}
          isOpen={!!selectedToken}
          onClose={closeModal}
          tokenData={selectedTokenData}
        />
      )}
    </div>
  );
}
