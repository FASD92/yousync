import { backendApi } from './api';

export interface LogoutResponse {
  message: string;
}

export interface GoogleLoginRequest {
  access_token: string;
}

export interface GoogleLoginResponse {
  access_token: string;
  token_type: string;
}

export interface UserInfo {
  id: number;
  email: string;
  full_name: string;
  profile_picture?: string;
  is_active: boolean;
  login_type: string;
  created_at: string;
  updated_at: string;
}

export const authService = {
  // 구글 로그인 - 구글 토큰을 백엔드로 전송
  async googleLogin(googleToken: string): Promise<void> {
    try {
      const response = await backendApi.post<GoogleLoginResponse>('/auth/google-login', {
        id_token: googleToken  // 백엔드가 기대하는 필드명으로 변경
      });
      
      // 백엔드에서 받은 토큰을 localStorage에 저장
      if (typeof window !== 'undefined') {
        localStorage.setItem('access_token', response.access_token);
      }
      
      // 사용자 정보 가져오기
      const userInfo = await this.getCurrentUser();
      
      // 사용자 정보 저장
      if (typeof window !== 'undefined') {
        localStorage.setItem('google_user', JSON.stringify({
          id: userInfo.id,
          email: userInfo.email,
          name: userInfo.full_name,
          picture: userInfo.profile_picture
        }));
        
        // 인증 상태 변경 이벤트 발생
        window.dispatchEvent(new Event('auth-change'));
      }
      
    } catch (error) {
      console.error('구글 로그인 실패:', error);
      throw error;
    }
  },

  // 현재 사용자 정보 조회
  async getCurrentUser(): Promise<UserInfo> {
    try {
      const response = await backendApi.get<UserInfo>('/auth/me');
      return response;
    } catch (error) {
      console.error('사용자 정보 조회 실패:', error);
      throw error;
    }
  },

  // 토큰 갱신
  async refreshToken(): Promise<GoogleLoginResponse> {
    try {
      // 백엔드의 refresh 엔드포인트는 현재 토큰으로 새 토큰을 발급
      const response = await backendApi.post<GoogleLoginResponse>('/auth/refresh');

      // 새로운 토큰으로 업데이트
      if (typeof window !== 'undefined') {
        localStorage.setItem('access_token', response.access_token);
      }

      return response;
    } catch (error) {
      console.error('토큰 갱신 실패:', error);
      
      // 토큰 갱신 실패 시 모든 토큰 제거
      if (typeof window !== 'undefined') {
        localStorage.removeItem('access_token');
        localStorage.removeItem('google_user');
        window.dispatchEvent(new Event('auth-change'));
      }
      
      throw error;
    }
  },

  // 로그아웃
  async logout(): Promise<LogoutResponse> {
    try {
      const response = await backendApi.post<LogoutResponse>('/auth/logout');
      
      // 클라이언트 측에서도 토큰 제거
      if (typeof window !== 'undefined') {
        localStorage.removeItem('access_token');
        localStorage.removeItem('google_user');
        window.dispatchEvent(new Event('auth-change'));
      }
      
      return response;
    } catch (error) {
      console.error('로그아웃 실패:', error);
      
      // 에러가 발생해도 클라이언트 측 토큰은 제거
      if (typeof window !== 'undefined') {
        localStorage.removeItem('access_token');
        localStorage.removeItem('google_user');
        window.dispatchEvent(new Event('auth-change'));
      }
      
      throw error;
    }
  },
}; 