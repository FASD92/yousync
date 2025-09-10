"use client";

import { useRouter } from "next/navigation";
import { GoogleOAuthProvider, GoogleLogin } from "@react-oauth/google";
import { useState } from "react";
import { authService } from "@/services/auth";

export default function LoginPage() {
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();

  // 구글 로그인 성공 처리
  const handleGoogleSuccess = async (credentialResponse: any) => {
    try {
      setIsLoading(true);
      
      if (!credentialResponse.credential) {
        throw new Error('구글 인증 정보를 받지 못했습니다.');
      }
      
      // id_token을 백엔드로 전송
      await authService.googleLogin(credentialResponse.credential);
      
      // 로그인 성공 시 홈으로 이동
      router.push("/home");
    } catch (error) {
      console.error("로그인 실패:", error);
      if (error instanceof Error) {
        console.error("에러 메시지:", error.message);
      }
      alert("로그인에 실패했습니다. 다시 시도해주세요.");
    } finally {
      setIsLoading(false);
    }
  };

  // 구글 로그인 실패 처리
  const handleGoogleError = () => {
    console.error("구글 로그인 실패");
    setIsLoading(false);
  };

  return (
    <GoogleOAuthProvider clientId={process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID!}>
      <div className="min-h-screen w-full flex items-center justify-center bg-gradient-to-br from-[#0A1A16] via-[#102F26] to-black">
        <div className="w-full max-w-sm sm:max-w-md mx-auto p-4">
          {/* 카드 */}
          <div className="bg-black/60 backdrop-blur-2xl rounded-3xl shadow-2xl border border-emerald-900/40 px-8 py-10 flex flex-col items-center">
            {/* 로고/타이틀 */}
            <div className="mb-8 text-center">
              <h1 className="text-4xl sm:text-5xl font-extrabold text-white mb-2 tracking-tight drop-shadow-lg">
                <span className="text-emerald-400">You</span>Sync
              </h1>
              <p className="text-gray-300 text-lg sm:text-xl font-medium mt-1">
                AI 더빙 & 피치 분석 서비스
              </p>
            </div>
            {/* 로그인 안내 */}
            <div className="mb-7 w-full flex justify-center">
              <div className="w-full min-w-[240px] flex justify-center">
                 {!isLoading ? (
                   <div className="flex justify-center w-full">
                     <GoogleLogin
                       onSuccess={handleGoogleSuccess}
                       onError={handleGoogleError}
                       useOneTap={false}
                       size="large"
                       text="signin_with"
                       shape="rectangular"
                       theme="outline"
                       width="240px"
                     />
                   </div>
                 ) : (
                   <div className="w-full py-3 px-4 rounded-xl bg-gray-300 text-center text-gray-600 font-semibold">
                     로그인 중...
                   </div>
                 )}
               </div>
            </div>
            {/* 안내문구 */}
            <div className="w-full text-center mt-2">
              <p className="text-gray-400 text-sm mb-2">
                로그인하지 않아도 일부 기능을 체험할 수 있습니다.
              </p>
              <button
                onClick={() => router.push('/home')} 
                className="text-emerald-300 hover:text-white text-sm underline transition font-medium"
              >
                홈으로 돌아가기
              </button>
            </div>
          </div>
        </div>
      </div>
    </GoogleOAuthProvider>
  );
}