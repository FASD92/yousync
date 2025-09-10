
import axios from "axios";
import type { ServerPitch } from "@/types/pitch";

export async function getServerPitchData(tokenId: string): Promise<ServerPitch[]> {
  const numericId = Number(tokenId);

  try {
    const response = await axios.get<ServerPitch[]>(
      `${process.env.NEXT_PUBLIC_API_BASE_URL}/tokens/${numericId}`
    );
    return response.data;
  } catch (error) {
    console.error("❌ 서버 pitch 데이터 가져오기 실패:", error);
    throw error;
  }
}