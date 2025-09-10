// 사용자 정보 타입
export interface UserInfo {
  id: number;
  email: string;
  google_id: string;
  full_name: string;
  profile_picture: string;
  is_active: boolean;
  login_type: string;
  created_at: string;
  updated_at: string;
}

// 북마크 토큰 정보 타입
export interface BookmarkToken {
  id: number;
  token_name: string;
  actor_name: string;
  category: string;
  thumbnail_url?: string;
  youtube_url?: string;
}

// 북마크 정보 타입
export interface BookmarkInfo {
  id: number;
  user_id: number;
  token_id: number;
  created_at: string;
  token: BookmarkToken;
}

// 더빙한 토큰 정보 타입
export interface DubbedTokenInfo {
  token_id: number;
  token_name: string;
  actor_name: string;
  category: string;
  youtube_url?: string;
  last_dubbed_at: string;
  total_scripts: number;
  completed_scripts: number;
}

// 마이페이지 통합 정보 타입
export interface MyPageOverview {
  user_info: UserInfo;
  total_bookmarks: number;
  total_dubbed_tokens: number;
  total_practice_count: number;
  average_completion_rate: number;
  recent_bookmarks: BookmarkInfo[];
  recent_dubbed_tokens: DubbedTokenInfo[];
}