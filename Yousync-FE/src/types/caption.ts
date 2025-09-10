export interface CaptionState {
  currentIdx: number;
  captions: Caption[];
}

// 통합된 Caption 인터페이스
export interface Caption {
  id: number;
  movie_id: number;
  actor_id: number;
  start_time: number;
  end_time: number;
  script: string;
  translation: string;
  url: string | null;
  actor_pitch_values: number[];
  background_audio_url: string;
  actor: {
    name: string;
    id: number;
  };
} 