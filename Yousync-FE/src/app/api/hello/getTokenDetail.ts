import axios from "axios";
import type { TokenDetailResponse } from "@/types/pitch";

export async function getTokenDetail(tokenId: string) {
  const numericId = Number(tokenId);
  const response = await axios.get<TokenDetailResponse>(
    `${process.env.NEXT_PUBLIC_API_BASE_URL}/tokens/${numericId}`
  );
  const token = response.data;

  // 서버에서 바로 사용할 데이터 가공
  const front_data = {
    id: token.id,
    user_id: 1, // 임시 값
    movie_id: token.id,
    score: 85, // 임시 점수
    accuracy: 88,
    fluency: 82,
    pronunciation: 90,
    created_at: new Date().toISOString(),
    user_pitch_data: Array.from({ length: 50 }, () => Math.random() * 100), // 임시
    server_pitch_data: token.pitch?.map(p => p.hz || 0) || [],
    audio_url: token.bgvoice_url || "",
    movie: {
      title: token.token_name,
      youtube_url: token.youtube_url,
      category: token.category,
    },
    captions: token.scripts?.map((script, index) => ({
      id: script.id,
      movie_name: token.token_name,
      movie_id: token.id,
      actor_id: 1, // 임시 값
      script: script.script,
      translation: script.translation || "",
      start_time: script.start_time,
      end_time: script.end_time,
      url: null,
      actor_pitch_values: [],
      background_audio_url: token.bgvoice_url || "",
      actor: {
        name: token.actor_name,
        id: 1
      }
    })) || [],
  };

  return {tokenData: token, front_data};
}