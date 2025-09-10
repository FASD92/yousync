import { useState, useEffect } from 'react';

export interface User {
  id: number;
  email: string;
  name: string;
  picture?: string;
}

export const useUser = () => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  useEffect(() => {
    const loadUserData = () => {
      try {
        if (typeof window !== 'undefined') {
          const userData = localStorage.getItem('google_user');
          const accessToken = localStorage.getItem('access_token');
          
          if (userData && accessToken) {
            const parsedUser = JSON.parse(userData);
            setUser(parsedUser);
            setIsLoggedIn(true);
          } else {
            setUser(null);
            setIsLoggedIn(false);
          }
        }
      } catch (error) {
        console.error('사용자 데이터 로드 실패:', error);
        setUser(null);
        setIsLoggedIn(false);
      } finally {
        setIsLoading(false);
      }
    };

    // 초기 로드
    loadUserData();

    // localStorage 변경 감지
    const handleStorageChange = () => {
      loadUserData();
    };

    // storage 이벤트 리스너 추가
    window.addEventListener('storage', handleStorageChange);

    // 커스텀 이벤트 리스너 추가 (같은 탭에서의 변경사항 감지)
    const handleAuthChange = () => {
      loadUserData();
    };
    
    window.addEventListener('auth-change', handleAuthChange);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('auth-change', handleAuthChange);
    };
  }, []);

  // 로그아웃 함수
  const logout = () => {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('access_token');
      localStorage.removeItem('google_user');
      
      setUser(null);
      setIsLoggedIn(false);
      
      // 커스텀 이벤트 발생 (다른 컴포넌트에 알림)
      window.dispatchEvent(new Event('auth-change'));
    }
  };

  // 사용자 정보 업데이트 함수
  const updateUser = (newUserData: User) => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('google_user', JSON.stringify(newUserData));
      setUser(newUserData);
      
      // 커스텀 이벤트 발생
      window.dispatchEvent(new Event('auth-change'));
    }
  };

  return {
    user,
    isLoading,
    isLoggedIn,
    logout,
    updateUser,
  };
};
