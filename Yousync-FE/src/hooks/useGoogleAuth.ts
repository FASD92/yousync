import { useEffect } from "react";
import { useUser } from "./useUser";

export function useGoogleAuth(onSuccess?: (token: string) => void) {
  const { user, isLoggedIn, isLoading } = useUser();

  useEffect(() => {
    // 로그인 상태가 변경되었을 때 콜백 실행
    if (isLoggedIn && user && onSuccess) {
      const accessToken = localStorage.getItem('access_token');
      if (accessToken) {
        onSuccess(accessToken);
      }
    }
  }, [isLoggedIn, user, onSuccess]);

  return {
    user,
    isLoggedIn,
    isLoading,
  };
} 