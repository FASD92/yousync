export interface PitchItem {
  movie_id: number;
  actor_id: number;
  start_time: number;
  end_time: number;
  script: string;
  translation: string;
  url: string;
  actor_pitch_values: number[]; // 서버 데이터 추가본
  time_values: number[]; // 서버 데이터 추가본
  background_audio_url: string;
  id: number;
  actor: {
    name: string;
    id: number;
  };
}

export interface ServerPitch {
  time: number;
  hz: number | null;
}

export interface ScriptItem {
  token_id: number;
  start_time: number;
  end_time: number;
  script: string;
  translation: string | null;
  id: number;
  words?: Array<{
    script_id: number;
    start_time: number;
    end_time: number;
    word: string;
    probability: number;
    id: number;
  }>;
}

export interface TokenDetailResponse {
  token_name: string;
  actor_name: string;
  category: string;
  start_time: number;
  end_time: number;
  s3_textgrid_url: string;
  s3_pitch_url: string;
  s3_bgvoice_url: string;
  youtube_url: string;
  id: number;
  pitch: ServerPitch[];
  bgvoice_url: string;
  scripts: ScriptItem[];
  youtubeId: string;
} 