// src/types/dubbing.ts

// src/types/dubbing.ts

export interface Caption {
  // --- 기존 속성 ---
  script: string;
  start_time: number;
  end_time: number;
  actor?: { name: string };
  
  // --- [추가] PitchComparison이 요구하는 속성들 ---
  id: number; // 또는 string, 실제 타입에 맞게 수정
  movie_id: number; // 또는 string
  translation: string;
  // 만약 다른 속성도 필요하다면 여기에 추가합니다.
}

export interface FrontData {
  movie: {
    title: string;
    category: string;
    youtube_url: string;
  };
  captions: Caption[];
}