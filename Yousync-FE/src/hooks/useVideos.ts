//이해완료
import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { extractYoutubeVideoId } from "@/utils/extractYoutubeVideoId";
import type { TokenDetailResponse, PitchItem, ServerPitch, ScriptItem } from "@/types/pitch";
import type { VideoType } from "@/types/video";
import type { Caption, CaptionState } from "@/types/caption";

const fetchVideos = async () => {
  const res = await axios.get<Omit<TokenDetailResponse, "youtubeId">[]>(
    `${process.env.NEXT_PUBLIC_API_BASE_URL}/tokens/`,
    {
      timeout: 10000, // 10초 타임아웃
    }
  );
  
  
  return res.data
    .map((item) => {
      const youtubeId = extractYoutubeVideoId(item.youtube_url);
      return youtubeId ? { ...item, youtubeId } : null;
    })
    .filter(Boolean) as (TokenDetailResponse & { youtubeId: string })[];
};
/**
 * 
 * @returns 비디오 객체 반환
 */
export function useVideos() {
  return useQuery({
    queryKey: ["videos"],
    queryFn: fetchVideos,
    staleTime: 5 * 60 * 1000, // 5분간 데이터를 fresh로 유지
    gcTime: 10 * 60 * 1000, // 10분간 캐시 유지
    refetchOnWindowFocus: false, // 윈도우 포커스 시 재요청 비활성화
    refetchOnMount: false, // 마운트 시 재요청 비활성화 (캐시 우선)
  });
}

const fetchPopularVideos = async () => {
  const res = await axios.get<Omit<TokenDetailResponse, "youtubeId">[]>(
    `${process.env.NEXT_PUBLIC_API_BASE_URL}/tokens/popular/`,
    {
      timeout: 10000, // 10초 타임아웃
    }
  );
  
  
  return res.data
    .map((item) => {
      const youtubeId = extractYoutubeVideoId(item.youtube_url);
      return youtubeId ? { ...item, youtubeId } : null;
    })
    .filter(Boolean) as (TokenDetailResponse & { youtubeId: string })[];
};

/**
 * 
 * @returns "베스트 더빙" 비디오 객체 반환
 */
export function usePopularVideos() {
  return useQuery({
    queryKey: ["videos", "popular"],
    queryFn: fetchPopularVideos,
    staleTime: 5 * 60 * 1000, // 5분간 데이터를 fresh로 유지
    gcTime: 10 * 60 * 1000, // 10분간 캐시 유지
    refetchOnWindowFocus: false, // 윈도우 포커스 시 재요청 비활성화
    refetchOnMount: false, // 마운트 시 재요청 비활성화 (캐시 우선)
  });
}

const fetchLatestVideos = async () => {
  const res = await axios.get<Omit<TokenDetailResponse, "youtubeId">[]>(
    `${process.env.NEXT_PUBLIC_API_BASE_URL}/tokens/latest/`,
    {
      timeout: 10000, // 10초 타임아웃
    }
  );
  
  
  return res.data
    .map((item) => {
      const youtubeId = extractYoutubeVideoId(item.youtube_url);
      return youtubeId ? { ...item, youtubeId } : null;
    })
    .filter(Boolean) as (TokenDetailResponse & { youtubeId: string })[];
};

/**
 * 
 * @returns "신상 더빙" 비디오 객체 반환
 */
export function useLatestVideos() {
  return useQuery({
    queryKey: ["videos", "latest"],
    queryFn: fetchLatestVideos,
    staleTime: 5 * 60 * 1000, // 5분간 데이터를 fresh로 유지
    gcTime: 10 * 60 * 1000, // 10분간 캐시 유지
    refetchOnWindowFocus: false, // 윈도우 포커스 시 재요청 비활성화
    refetchOnMount: false, // 마운트 시 재요청 비활성화 (캐시 우선)
  });
}
// 로맨스
const fetchRomanticVideos = async () => {
  const res = await axios.get<Omit<TokenDetailResponse, "youtubeId">[]>(
    `${process.env.NEXT_PUBLIC_API_BASE_URL}/tokens/category/로맨스/`,
    {
      timeout: 10000, // 10초 타임아웃
    }
  );
  
  
  return res.data
    .map((item) => {
      const youtubeId = extractYoutubeVideoId(item.youtube_url);
      return youtubeId ? { ...item, youtubeId } : null;
    })
    .filter(Boolean) as (TokenDetailResponse & { youtubeId: string })[];
};
/**
 * 
 * @returns "로맨틱 더빙" 비디오 객체 반환
 */
export function useRomanticVideos() {
  return useQuery({
    queryKey: ["videos", "romantic"],
    queryFn: fetchRomanticVideos,
    staleTime: 5 * 60 * 1000, // 5분간 데이터를 fresh로 유지
    gcTime: 10 * 60 * 1000, // 10분간 캐시 유지
    refetchOnWindowFocus: false, // 윈도우 포커스 시 재요청 비활성화
    refetchOnMount: false, // 마운트 시 재요청 비활성화 (캐시 우선)
  });
}

// 액션
const fetchActionVideos = async () => {
  const res = await axios.get<Omit<TokenDetailResponse, "youtubeId">[]>(
    `${process.env.NEXT_PUBLIC_API_BASE_URL}/tokens/category/액션/`,
    { timeout: 10000 }
  );


  return res.data
    .map((item) => {
      const youtubeId = extractYoutubeVideoId(item.youtube_url);
      return youtubeId ? { ...item, youtubeId } : null;
    })
    .filter(Boolean) as (TokenDetailResponse & { youtubeId: string })[];
};
/**
 * 
 * @returns "액션 더빙" 비디오 객체 반환
 */
export function useActionVideos() {
  return useQuery({
    queryKey: ["videos", "action"],
    queryFn: fetchActionVideos,
    staleTime: 5 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
    refetchOnWindowFocus: false,
    refetchOnMount: false,
  });
}

// 애니메이션
const fetchAnimationVideos = async () => {
  const res = await axios.get<Omit<TokenDetailResponse, "youtubeId">[]>(
    `${process.env.NEXT_PUBLIC_API_BASE_URL}/tokens/category/애니메이션/`,
    { timeout: 10000 }
  );


  return res.data
    .map((item) => {
      const youtubeId = extractYoutubeVideoId(item.youtube_url);
      return youtubeId ? { ...item, youtubeId } : null;
    })
    .filter(Boolean) as (TokenDetailResponse & { youtubeId: string })[];
};
/**
 * 
 * @returns "애니메이션 더빙" 비디오 객체 반환
 */
export function useAnimationVideos() {
  return useQuery({
    queryKey: ["videos", "animation"],
    queryFn: fetchAnimationVideos,
    staleTime: 5 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
    refetchOnWindowFocus: false,
    refetchOnMount: false,
  });
}
// 코미디
const fetchComedyVideos = async () => {
  const res = await axios.get<Omit<TokenDetailResponse, "youtubeId">[]>(
    `${process.env.NEXT_PUBLIC_API_BASE_URL}/tokens/category/코미디/`,
    { timeout: 10000 }
  );


  return res.data
    .map((item) => {
      const youtubeId = extractYoutubeVideoId(item.youtube_url);
      return youtubeId ? { ...item, youtubeId } : null;
    })
    .filter(Boolean) as (TokenDetailResponse & { youtubeId: string })[];
};
/**
 * 
 * @returns "코미디 더빙" 비디오 객체 반환
 */
export function useComedyVideos() {
  return useQuery({
    queryKey: ["videos", "comedy"],
    queryFn: fetchComedyVideos,
    staleTime: 5 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
    refetchOnWindowFocus: false,
    refetchOnMount: false,
  });
}
// 판타지
const fetchFantasyVideos = async () => {
  const res = await axios.get<Omit<TokenDetailResponse, "youtubeId">[]>(
    `${process.env.NEXT_PUBLIC_API_BASE_URL}/tokens/category/판타지/`,
    { timeout: 10000 }
  );


  return res.data
    .map((item) => {
      const youtubeId = extractYoutubeVideoId(item.youtube_url);
      return youtubeId ? { ...item, youtubeId } : null;
    })
    .filter(Boolean) as (TokenDetailResponse & { youtubeId: string })[];
};
/**
 * 
 * @returns "판타지 더빙" 비디오 객체 반환
 */
export function useFantasyVideos() {
  return useQuery({
    queryKey: ["videos", "fantasy"],
    queryFn: fetchFantasyVideos,
    staleTime: 5 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
    refetchOnWindowFocus: false,
    refetchOnMount: false,
  });
}
// 드라마
const fetchDramaVideos = async () => {
  const res = await axios.get<Omit<TokenDetailResponse, "youtubeId">[]>(
    `${process.env.NEXT_PUBLIC_API_BASE_URL}/tokens/category/드라마/`,
    { timeout: 10000 }
  );


  return res.data
    .map((item) => {
      const youtubeId = extractYoutubeVideoId(item.youtube_url);
      return youtubeId ? { ...item, youtubeId } : null;
    })
    .filter(Boolean) as (TokenDetailResponse & { youtubeId: string })[];
};
/**
 * 
 * @returns "드라마 더빙" 비디오 객체 반환
 */
export function useDramaVideos() {
  return useQuery({
    queryKey: ["videos", "drama"],
    queryFn: fetchDramaVideos,
    staleTime: 5 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
    refetchOnWindowFocus: false,
    refetchOnMount: false,
  });
}

const fetchSyncCollection = async () => {
  const res = await axios.get<Omit<TokenDetailResponse, "youtubeId">[]>(
    `${process.env.NEXT_PUBLIC_API_BASE_URL}/tokens/sync-collection/`,
    { timeout: 10000 }
  );


  return res.data
    .map((item) => {
      const youtubeId = extractYoutubeVideoId(item.youtube_url);
      return youtubeId ? { ...item, youtubeId } : null;
    })
    .filter(Boolean) as (TokenDetailResponse & { youtubeId: string })[];

}
export function useSyncCollection() {
  return useQuery({
    queryKey: ["syncCollection"],
    queryFn: fetchSyncCollection,
    staleTime: 5 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
    refetchOnWindowFocus: false,
    refetchOnMount: false,
  });
}
export function useDuetScenes() {
  return useQuery({
    queryKey: ["duetScenes"],
    queryFn: async () => {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/duet/scenes`);
      if (!res.ok) throw new Error("듀엣 더빙 목록을 불러오지 못했습니다.");
      return res.json();
    },
    staleTime: 1000 * 60 * 5, // 5분 캐싱
  });
}