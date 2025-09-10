export interface ActorMovie {
    id: number;
    token_name: string;
    actor_name: string;
    category: string;
    start_time: number;
    end_time: number;
    s3_textgrid_url: string;
    s3_pitch_url: string;
    s3_bgvoice_url: string;
    youtube_url: string;
    view_count: number;
} 