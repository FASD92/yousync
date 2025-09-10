import axios from 'axios';
import { API_ENDPOINTS } from '@/lib/constants';
import { PaginationParams, Token, TokenDetail, UserAudiosResponse, ViewIncrementResponse } from '@/types/api';

// 타입 정의 (순환 참조 방지를 위해 여기서 직접 정의)
interface RefreshTokenResponse {
  access_token: string;
  token_type: string;
}

// ✅ Next.js API 라우트용 클라이언트 (상대 경로)
const apiClient = axios.create({
  baseURL: '',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ✅ 백엔드 API용 클라이언트 (절대 경로)
const backendClient = axios.create({
  baseURL: API_ENDPOINTS.BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ✅ 토큰 갱신 전용 클라이언트
const refreshClient = axios.create({
  baseURL: API_ENDPOINTS.BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ✅ 요청 인터셉터 - 토큰 자동 추가
const attachAuthToken = (config: any) => {
  if (typeof window !== 'undefined') {
    const accessToken = localStorage.getItem('access_token');
    if (accessToken && config.headers) {
      config.headers.Authorization = `Bearer ${accessToken}`;
    }
  }
  return config;
};

apiClient.interceptors.request.use(attachAuthToken, Promise.reject);
backendClient.interceptors.request.use(attachAuthToken, Promise.reject);
refreshClient.interceptors.request.use(attachAuthToken, Promise.reject);

// ✅ 응답 인터셉터 - 에러 로깅 및 401 처리
const handleResponseError = async (error: any) => {
  // 타임아웃 에러인 경우 간단한 메시지만 출력
  if (error.code === 'ECONNABORTED') {
    console.warn('API 요청 타임아웃 발생');
    return Promise.reject(error);
  }
  
  console.error('API Error:', error);
  
  if (error.response?.status === 401) {
    console.warn('인증 실패 - 토큰 갱신 시도');
    
    // 토큰 갱신 시도
    try {
      const accessToken = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
      
      if (accessToken) {
        // 백엔드의 refresh 엔드포인트 호출
        const refreshResponse = await refreshClient.post<RefreshTokenResponse>('/auth/refresh');
        
        // 새로운 토큰으로 업데이트
        if (typeof window !== 'undefined') {
          localStorage.setItem('access_token', refreshResponse.data.access_token);
        }
        
        // 원래 요청에 새 토큰을 추가하여 재시도
        if (error.config && error.config.headers) {
          error.config.headers.Authorization = `Bearer ${refreshResponse.data.access_token}`;
          return backendClient.request(error.config);
        }
      }
    } catch (refreshError) {
      console.error('토큰 갱신 실패:', refreshError);
      
      // 토큰 갱신 실패 시 모든 토큰 제거 및 로그아웃 처리
      if (typeof window !== 'undefined') {
        localStorage.removeItem('access_token');
        localStorage.removeItem('google_user');
        window.dispatchEvent(new Event('auth-change'));
      }
    }
  }
  
  return Promise.reject(error);
};

// ✅ Next.js API 라우트용 응답 인터셉터
const handleApiClientResponseError = async (error: any) => {
  // 타임아웃 에러인 경우 간단한 메시지만 출력
  if (error.code === 'ECONNABORTED') {
    console.warn('API 요청 타임아웃 발생');
    return Promise.reject(error);
  }
  
  console.error('API Client Error:', error);
  
  if (error.response?.status === 401) {
    console.warn('인증 실패 - 토큰 갱신 시도');
    
    // 토큰 갱신 시도
    try {
      const accessToken = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
      
      if (accessToken) {
        // 백엔드에서 토큰 갱신
        const refreshResponse = await refreshClient.post<RefreshTokenResponse>('/auth/refresh');
        
        // 새로운 토큰으로 업데이트
        if (typeof window !== 'undefined') {
          localStorage.setItem('access_token', refreshResponse.data.access_token);
        }
        
        // 원래 요청에 새 토큰을 추가하여 재시도
        if (error.config && error.config.headers) {
          error.config.headers.Authorization = `Bearer ${refreshResponse.data.access_token}`;
          return apiClient.request(error.config);
        }
      }
    } catch (refreshError) {
      console.error('토큰 갱신 실패:', refreshError);
      
      // 토큰 갱신 실패 시 모든 토큰 제거 및 로그아웃 처리
      if (typeof window !== 'undefined') {
        localStorage.removeItem('access_token');
        localStorage.removeItem('google_user');
        window.dispatchEvent(new Event('auth-change'));
      }
    }
  }
  
  return Promise.reject(error);
};

apiClient.interceptors.response.use((res) => res, handleApiClientResponseError);
backendClient.interceptors.response.use((res) => res, handleResponseError);

// ✅ API 메서드 (Next.js API 라우트용)
export const api = {
  get: async <T>(url: string) => (await apiClient.get<T>(url)).data,
  post: async <T>(url: string, data?: unknown) => (await apiClient.post<T>(url, data)).data,
  put: async <T>(url: string, data?: unknown) => (await apiClient.put<T>(url, data)).data,
  delete: async <T>(url: string) => (await apiClient.delete<T>(url)).data,
  patch: async <T>(url: string, data?: unknown) => (await apiClient.patch<T>(url, data)).data,
};

// ✅ 백엔드 API 메서드
export const backendApi = {
  get: async <T>(url: string) => (await backendClient.get<T>(url)).data,
  post: async <T>(url: string, data?: unknown) => (await backendClient.post<T>(url, data)).data,
  put: async <T>(url: string, data?: unknown) => (await backendClient.put<T>(url, data)).data,
  delete: async <T>(url: string) => (await backendClient.delete<T>(url)).data,
  patch: async <T>(url: string, data?: unknown) => (await backendClient.patch<T>(url, data)).data,
};

// 토큰 관련 API 함수들
export const tokenApi = {
  // 모든 토큰 목록 조회
  getTokens: async (params?: PaginationParams): Promise<Token[]> => {
    const queryParams = new URLSearchParams();
    if (params?.skip !== undefined) queryParams.append('skip', params.skip.toString());
    if (params?.limit !== undefined) queryParams.append('limit', params.limit.toString());
    
    const url = `${API_ENDPOINTS.TOKENS}${queryParams.toString() ? `?${queryParams.toString()}` : ''}`;
    return backendApi.get<Token[]>(url);
  },

  // 최신순으로 정렬된 토큰 목록 조회
  getLatestTokens: async (params?: PaginationParams): Promise<Token[]> => {
    const queryParams = new URLSearchParams();
    if (params?.skip !== undefined) queryParams.append('skip', params.skip.toString());
    if (params?.limit !== undefined) queryParams.append('limit', params.limit.toString());
    
    const url = `${API_ENDPOINTS.TOKENS}/latest${queryParams.toString() ? `?${queryParams.toString()}` : ''}`;
    return backendApi.get<Token[]>(url);
  },

  // 인기순으로 정렬된 토큰 목록 조회
  getPopularTokens: async (params?: PaginationParams): Promise<Token[]> => {
    const queryParams = new URLSearchParams();
    if (params?.skip !== undefined) queryParams.append('skip', params.skip.toString());
    if (params?.limit !== undefined) queryParams.append('limit', params.limit.toString());
    
    const url = `${API_ENDPOINTS.TOKENS}/popular${queryParams.toString() ? `?${queryParams.toString()}` : ''}`;
    return backendApi.get<Token[]>(url);
  },

  // 특정 카테고리의 토큰들 조회
  getTokensByCategory: async (category: string, params?: PaginationParams): Promise<Token[]> => {
    const queryParams = new URLSearchParams();
    if (params?.skip !== undefined) queryParams.append('skip', params.skip.toString());
    if (params?.limit !== undefined) queryParams.append('limit', params.limit.toString());
    
    const url = `${API_ENDPOINTS.TOKENS}/category/${category}${queryParams.toString() ? `?${queryParams.toString()}` : ''}`;
    return backendApi.get<Token[]>(url);
  },

  // 특정 토큰 상세 정보 조회
  getTokenDetail: async (tokenId: number): Promise<TokenDetail> => {
    const url = `${API_ENDPOINTS.TOKENS}/${tokenId}`;
    return backendApi.get<TokenDetail>(url);
  },

  // 토큰 생성
  createToken: async (tokenData: Partial<Token>): Promise<Token> => {
    const url = API_ENDPOINTS.TOKENS;
    return backendApi.post<Token>(url, tokenData);
  },

  // 토큰 업데이트
  updateToken: async (tokenId: number, tokenData: Partial<Token>): Promise<Token> => {
    const url = `${API_ENDPOINTS.TOKENS}/${tokenId}`;
    return backendApi.put<Token>(url, tokenData);
  },

  // 토큰 삭제
  deleteToken: async (tokenId: number): Promise<void> => {
    const url = `${API_ENDPOINTS.TOKENS}/${tokenId}`;
    return backendApi.delete<void>(url);
  },

  // 관련 토큰 조회 (같은 배우의 다른 토큰들)
  getRelatedTokens: async (tokenId: number, params?: PaginationParams): Promise<Token[]> => {
    const queryParams = new URLSearchParams();
    if (params?.skip !== undefined) queryParams.append('skip', params.skip.toString());
    if (params?.limit !== undefined) queryParams.append('limit', params.limit.toString());
    
    const url = `${API_ENDPOINTS.TOKENS}/${tokenId}/related${queryParams.toString() ? `?${queryParams.toString()}` : ''}`;
    return backendApi.get<Token[]>(url);
  },

  // 조회수 증가
  incrementView: async (tokenId: number): Promise<ViewIncrementResponse> => {
    const url = `${API_ENDPOINTS.TOKENS}/${tokenId}/view`;
    return backendApi.post<ViewIncrementResponse>(url);
  },

  // 사용자 오디오 목록 조회
  getUserAudios: async (tokenId: number): Promise<UserAudiosResponse> => {
    const url = `${API_ENDPOINTS.TOKENS}/${tokenId}/user-audios`;
    return backendApi.get<UserAudiosResponse>(url);
  },

  // 오디오 업로드
  uploadAudio: async (tokenId: number, audioData: FormData): Promise<any> => {
    const url = `${API_ENDPOINTS.TOKENS}/${tokenId}/upload-audio`;
    return backendApi.post<any>(url, audioData);
  },

  // 분석 결과 조회
  getAnalysisResult: async (jobId: string): Promise<string> => {
    const url = `${API_ENDPOINTS.TOKENS}/analysis-result/${jobId}`;
    return backendApi.get<string>(url);
  },

  // 분석 진행 상황 스트리밍 (SSE)
  getAnalysisProgress: async (jobId: string): Promise<string> => {
    const url = `${API_ENDPOINTS.TOKENS}/analysis-progress/${jobId}`;
    return backendApi.get<string>(url);
  },

  // SQS 큐 상태 조회
  getQueueStatus: async (): Promise<string> => {
    const url = `${API_ENDPOINTS.TOKENS}/queue/status`;
    return backendApi.get<string>(url);
  },
};
