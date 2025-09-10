import { z } from "zod"

// 사용자 입력 유효성 검사 스키마
export const userSchema = z.object({
  name: z.string().min(2, "이름은 2자 이상이어야 합니다"),
  email: z.string().email("올바른 이메일 형식이 아닙니다"),
  password: z.string().min(6, "비밀번호는 6자 이상이어야 합니다"),
})

// YouTube URL 유효성 검사
export function isValidYouTubeUrl(url: string): boolean {
  const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+/
  return youtubeRegex.test(url)
}

// 점수 유효성 검사
export function isValidScore(score: number): boolean {
  return score >= 0 && score <= 100
}

// 파일 크기 유효성 검사 (MB 단위)
export function isValidFileSize(fileSize: number, maxSizeMB: number): boolean {
  const maxSizeBytes = maxSizeMB * 1024 * 1024
  return fileSize <= maxSizeBytes
}

// 파일 타입 유효성 검사
export function isValidFileType(fileName: string, allowedTypes: string[]): boolean {
  const fileExtension = fileName.split('.').pop()?.toLowerCase()
  return fileExtension ? allowedTypes.includes(fileExtension) : false
} 