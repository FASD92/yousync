import { backendApi } from './api';

// 북마크 관련 타입
export interface BookmarkCreate {
  token_id: number;
}

export interface BookmarkOut {
  token_id: number;
}

export interface BookmarkListOut {
  id: number;
  user_id: number;
  token_id: number;
  created_at: string;
  token: {
    id: number;
    token_name: string;
    actor_name: string;
    category: string;
    thumbnail_url?: string;
  };
}

// 내가 더빙한 토큰 관련 타입
export interface MyDubbedTokenResponse {
  token_id: number;
  token_name: string;
  actor_name: string;
  category: string;
  last_dubbed_at: string;
  total_scripts: number;
  completed_scripts: number;
}

// 토큰 분석 상태 관련 타입
export interface ScriptResult {
  script_id: number;
  script_text: string;
  has_result: boolean;
  job_id?: string;
  status?: string;
  result?: any;
  created_at?: string;
}

export interface TokenAnalysisStatusResponse {
  token_id: number;
  has_analysis: boolean;
  script_results: ScriptResult[];
}

export const mypageService = {
  // 북마크 생성
  async createBookmark(tokenId: number): Promise<BookmarkOut> {
    try {
      const response = await backendApi.post<BookmarkOut>('/mypage/bookmarks', {
        token_id: tokenId
      });
      return response;
    } catch (error) {
      console.error('북마크 생성 실패:', error);
      throw error;
    }
  },

  // 북마크 삭제
  async deleteBookmark(tokenId: number): Promise<void> {
    try {
      await backendApi.delete(`/mypage/bookmarks/${tokenId}`);
    } catch (error) {
      console.error('북마크 삭제 실패:', error);
      throw error;
    }
  },

  // 북마크 목록 조회
  async getBookmarks(limit: number = 20, offset: number = 0): Promise<BookmarkListOut[]> {
    try {
      const response = await backendApi.get<BookmarkListOut[]>(
        `/mypage/bookmarks/?limit=${limit}&offset=${offset}`
      );
      return response;
    } catch (error) {
      console.error('북마크 목록 조회 실패:', error);
      throw error;
    }
  },

  // 내가 더빙한 토큰 목록 조회
  async getMyDubbedTokens(limit: number = 20, offset: number = 0): Promise<MyDubbedTokenResponse[]> {
    try {
      const response = await backendApi.get<MyDubbedTokenResponse[]>(
        `/mypage/my-dubbed-tokens?limit=${limit}&offset=${offset}`
      );
      return response;
    } catch (error) {
      console.error('내가 더빙한 토큰 목록 조회 실패:', error);
      throw error;
    }
  },

  // 특정 토큰의 내 분석 상태 확인
  async getTokenAnalysisStatus(tokenId: number): Promise<TokenAnalysisStatusResponse> {
    try {
      const response = await backendApi.get<TokenAnalysisStatusResponse>(
        `/mypage/tokens/${tokenId}/analysis-status`
      );
      return response;
    } catch (error) {
      console.error('토큰 분석 상태 조회 실패:', error);
      throw error;
    }
  },

  // 재더빙을 위한 기존 분석 결과 삭제
  async deleteMyTokenResults(tokenId: number): Promise<void> {
    try {
      await backendApi.delete(`/mypage/tokens/${tokenId}/my-results`);
    } catch (error) {
      console.error('분석 결과 삭제 실패:', error);
      throw error;
    }
  }
}
