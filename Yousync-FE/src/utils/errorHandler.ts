// import type { ApiError } from '../types/api';

// // 에러 타입 정의
// export interface AppError {
//   message: string;
//   code?: string;
//   status?: number;
//   details?: unknown;
// }

// // API 에러를 앱 에러로 변환
// export function handleApiError(error: unknown): AppError {
//   if (error && typeof error === 'object' && 'response' in error) {
//     const apiError = error as { response: { data: ApiError; status: number } };
//     return {
//       message: apiError.response.data.message || '서버 에러가 발생했습니다',
//       code: apiError.response.data.code,
//       status: apiError.response.status,
//       details: apiError.response.data,
//     };
//   }

//   if (error instanceof Error) {
//     return {
//       message: error.message,
//       details: error,
//     };
//   }

//   return {
//     message: '알 수 없는 에러가 발생했습니다',
//     details: error,
//   };
// }

// // 네트워크 에러 처리
// export function handleNetworkError(error: unknown): AppError {
//   if (error && typeof error === 'object' && 'code' in error) {
//     const networkError = error as { code: string };
    
//     switch (networkError.code) {
//       case 'NETWORK_ERROR':
//         return {
//           message: '네트워크 연결을 확인해주세요',
//           code: 'NETWORK_ERROR',
//         };
//       case 'TIMEOUT':
//         return {
//           message: '요청 시간이 초과되었습니다',
//           code: 'TIMEOUT',
//         };
//       default:
//         return {
//           message: '네트워크 에러가 발생했습니다',
//           code: networkError.code,
//         };
//     }
//   }

//   return {
//     message: '네트워크 에러가 발생했습니다',
//     details: error,
//   };
// }

// // 유효성 검사 에러 처리
// export function handleValidationError(errors: Record<string, string[]>): AppError {
//   const messages = Object.values(errors).flat();
//   return {
//     message: messages.join(', '),
//     code: 'VALIDATION_ERROR',
//     details: errors,
//   };
// }

// // 에러 메시지 포맷팅
// export function formatErrorMessage(error: AppError): string {
//   if (error.status) {
//     return `[${error.status}] ${error.message}`;
//   }
//   return error.message;
// }

// // 콘솔에 에러 로깅
// export function logError(error: AppError, context?: string): void {
//   console.error(`[${context || 'App'}] Error:`, {
//     message: error.message,
//     code: error.code,
//     status: error.status,
//     details: error.details,
//   });
// } 