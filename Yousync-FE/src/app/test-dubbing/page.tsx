"use client";

import { useState } from "react";
import Link from "next/link";

export default function TestDubbingPage() {
  const [actor1, setActor1] = useState("1"); // 기본값 설정
  const [actor2, setActor2] = useState("2"); // 기본값 설정
  const [selectedActor, setSelectedActor] = useState("1"); // 기본값 설정
  const [movieId, setMovieId] = useState("1"); // 기본값 설정

  return (
    <div className="container mx-auto p-8">
      <h1 className="text-3xl font-bold mb-8">더빙 테스트 페이지</h1>
      
      <div className="bg-gray-800 p-6 rounded-lg mb-8">
        <h2 className="text-2xl font-semibold mb-4">일반 더빙</h2>
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">영화 ID:</label>
          <input
            type="text"
            value={movieId}
            onChange={(e) => setMovieId(e.target.value)}
            className="bg-gray-700 text-white px-4 py-2 rounded w-full"
          />
        </div>
        <Link
          href={`/dubbing/${movieId}`}
          className="inline-block bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded"
        >
          일반 더빙 시작하기
        </Link>
      </div>
      
      <div className="bg-gray-800 p-6 rounded-lg">
        <h2 className="text-2xl font-semibold mb-4">듀엣 더빙</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium mb-2">배우 1 ID:</label>
            <input
              type="text"
              value={actor1}
              onChange={(e) => setActor1(e.target.value)}
              className="bg-gray-700 text-white px-4 py-2 rounded w-full"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">배우 2 ID:</label>
            <input
              type="text"
              value={actor2}
              onChange={(e) => setActor2(e.target.value)}
              className="bg-gray-700 text-white px-4 py-2 rounded w-full"
            />
          </div>
        </div>
        
        <div className="mb-6">
          <label className="block text-sm font-medium mb-2">내가 연기할 배우:</label>
          <div className="flex space-x-4">
            <label className="flex items-center">
              <input
                type="radio"
                name="selectedActor"
                value={actor1}
                checked={selectedActor === actor1}
                onChange={() => setSelectedActor(actor1)}
                className="mr-2"
              />
              배우 1
            </label>
            <label className="flex items-center">
              <input
                type="radio"
                name="selectedActor"
                value={actor2}
                checked={selectedActor === actor2}
                onChange={() => setSelectedActor(actor2)}
                className="mr-2"
              />
              배우 2
            </label>
          </div>
        </div>
        
        <Link
          href={`/duetdubbing/${movieId}?actor1=${actor1}&actor2=${actor2}&selected=${selectedActor}`}
          className="inline-block bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-4 rounded"
        >
          듀엣 더빙 시작하기
        </Link>
      </div>
    </div>
  );
}
