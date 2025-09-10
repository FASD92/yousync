export interface VideoType {
  videoId: number;
  youtubeId: string;
  actor_name?: string;
  category?: string;
  start_time?: number;
  end_time?: number;
  thumbnail_url?: string;
}

export interface MovieItemVideo {
  videoId: string;
  youtubeId: string;
  actor_name: string;
}

export interface MovieItemProps {
  video: MovieItemVideo;
  isPlayable?: boolean;
  isShorts?: boolean;
  playingVideo?: string | null;
  onPlay?: (youtubeId: string) => void;
  onOpenModal?: (youtubeId: string) => void;
  onStop?: () => void;
} 