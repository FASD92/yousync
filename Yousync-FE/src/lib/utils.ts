import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// 기존 utils 함수들을 여기로 이동
export * from "../utils/delayPlay"
export * from "../utils/encodeWav"
export * from "../utils/extractYoutubeVideoId"
export * from "../utils/mergeWavBlobs" 