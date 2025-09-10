export interface ApiError {
  code: string;
  message: string;
  status?: number;
  details?: any;
}

export interface ApiResponse<T = any> {
  data?: T;
  error?: ApiError;
  success: boolean;
}

// 토큰 관련 타입들
export interface Token {
  token_name: string;
  actor_name: string;
  category: string;
  start_time: number;
  end_time: number;
  s3_textgrid_url: string;
  s3_pitch_url: string;
  s3_bgvoice_url: string;
  thumbnail_url: string;
  youtube_url: string;
  view_count: number;
  id: number;
}

export interface Word {
  script_id: number;
  start_time: number;
  end_time: number;
  word: string;
  probability: number;
  id: number;
}

export interface Script {
  token_id: number;
  start_time: number;
  end_time: number;
  script: string;
  translation: string;
  id: number;
  words: Word[];
}

export interface TokenDetail extends Token {
  bgvoice_url: string;
  pitch: string;
  scripts: Script[];
}

export interface UserAudio {
  script_id: number;
  url: string;
}

export interface UserAudiosResponse {
  audios: UserAudio[];
}

export interface ViewIncrementResponse {
  token_id: number;
  view_count: number;
}

export interface PaginationParams {
  skip?: number;
  limit?: number;
}

// export interface TokensResponse extends Array<Token> {}
// export interface TokenDetailResponse extends TokenDetail {} 