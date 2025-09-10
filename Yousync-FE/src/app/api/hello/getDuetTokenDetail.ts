
import axios from "axios";
import type { TokenDetailResponse } from "@/types/pitch";

export async function getDuetTokenDetail(tokenId1: string, tokenId2: string, selected: string) {
  const numericId1 = Number(tokenId1);
  const numericId2 = Number(tokenId2);
  
  const [res1, res2] = await Promise.all([
    axios.get<TokenDetailResponse>(
      `${process.env.NEXT_PUBLIC_API_BASE_URL}/tokens/${numericId1}`
    ),
    axios.get<TokenDetailResponse>(
      `${process.env.NEXT_PUBLIC_API_BASE_URL}/tokens/${numericId2}`
    ),
  ])
  const token1 = res1.data;
  const token2 = res2.data;
  console.log(">> ", token1, token2, selected);

  // selected(내가 고른 배우 id)가 token1.id와 같으면 token1이 '나', 아니면 token2가 '나'
  const isToken1Me = String(token1.id) === String(selected);

  // token1 captions
const captions1 = token1.scripts?.map((script) => ({
  id: script.id,
  movie_name: token1.token_name,
  movie_id: token1.id,
  actor_id: isToken1Me ? 1 : 2,
  script: script.script,
  translation: script.translation || "",
  start_time: script.start_time,
  end_time: script.end_time,
  url: null,
  actor_pitch_values: [],
  background_audio_url: token1.bgvoice_url || "",
  actor: {
    name: isToken1Me ? "나" : token1.actor_name,
    id: isToken1Me ? 1 : 2,
  },
  words: script.words || [],      
})) || [];

// token2 captions
const captions2 = token2.scripts?.map((script) => ({
  id: script.id,
  movie_name: token2.token_name,
  movie_id: token2.id,
  actor_id: !isToken1Me ? 1 : 2,
  script: script.script,
  translation: script.translation || "",
  start_time: script.start_time,
  end_time: script.end_time,
  url: null,
  actor_pitch_values: [],
  background_audio_url: token2.bgvoice_url || "",
  actor: {
    name: !isToken1Me ? "나" : token2.actor_name,
    id: !isToken1Me ? 1 : 2,
  },
  words: script.words || [],     
})) || [];


  // 두 배우의 대사를 합치고, start_time 기준으로 정렬
  const mergedCaptions = [...captions1, ...captions2].sort(
    (a, b) => a.start_time - b.start_time
  );

  // front_data에 mergedCaptions 적용
  const front_data = {
    id: (isToken1Me ? token1.id : token2.id),
    user_id: 1, // 임시
    movie_id: (isToken1Me ? token1.id : token2.id),
    score: 85, // 임시 점수
    accuracy: 88,
    fluency: 82,
    pronunciation: 90,
    created_at: new Date().toISOString(),
    user_pitch_data: Array.from({ length: 50 }, () => Math.random() * 100), // 임시 pitch
    server_pitch_data: (isToken1Me ? token1.pitch?.map(p => p.hz || 0) : token2.pitch?.map(p => p.hz || 0)) || [],
    audio_url: (isToken1Me ? token1.bgvoice_url : token2.bgvoice_url) || "",
    movie: {
      title: (isToken1Me ? token1.token_name : token2.token_name),
      youtube_url: (isToken1Me ? token1.youtube_url : token2.youtube_url),
      category: (isToken1Me ? token1.category : token2.category),
    },
    captions: mergedCaptions,
  };

  const mergedTokenData = {
    ...(isToken1Me ? token1 : token2),
    captions: mergedCaptions,
    scripts: mergedCaptions,
  };

  return { tokenData: mergedTokenData, front_data };}