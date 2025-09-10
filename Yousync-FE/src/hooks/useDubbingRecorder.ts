import { useState, useRef, useEffect } from 'react';
import { useVoiceRecorder } from './useVoiceRecorder';
import { useAudioStore } from '@/store/useAudioStore';
import { useJobIdsStore } from '@/store/useJobIdsStore';
import axios from 'axios';

// 타입 정의 추가
interface AxiosErrorInterface {
  isAxiosError?: boolean;
  code?: string;
  message?: string;
  response?: {
    status: number;
    data: any;
  };
}

// 타입 가드 함수 추가
function isAxiosError(error: unknown): error is AxiosErrorInterface {
  return (error as any)?.isAxiosError === true;
}
import { ScriptItem } from '@/types/pitch';

interface UseDubbingRecorderProps {
  captions?: { start_time: number; end_time: number }[];
  tokenId: string;
  scripts?: ScriptItem[];
  onUploadComplete?: (success: boolean, jobIds: string[]) => void;
}

interface UploadAudioResponse {
  job_id?: string;
  [key: string]: string | number | boolean | undefined;
}

export function useDubbingRecorder({
  captions = [],
  tokenId,
  scripts,
  onUploadComplete,
}: UseDubbingRecorderProps) {
  const { audioCtx } = useAudioStore();
  const { startRecording, stopRecording, recording, getAllBlobs } = useVoiceRecorder();

  const [recordedScripts, setRecordedScripts] = useState<boolean[]>(
    () => Array(Array.isArray(captions) ? captions.length : 0).fill(false)
  );
  const [uploading, setUploading] = useState(false);
  const [jobIds, setJobIds] = useState<string[]>([]);
  
  const addJobId = useJobIdsStore((state) => state.addJobId);

  const recordingRef = useRef(false);

  const startScriptRecording = (scriptIdx: number) => {
    // 스크립트 정보 가져오기
    const caption = captions[scriptIdx];
    if (caption) {
      startRecording(scriptIdx, caption.start_time, caption.end_time);
    } else {
      console.warn(`[WARN] 스크립트 ${scriptIdx} 정보를 찾을 수 없습니다.`);
      startRecording(scriptIdx, 0, 0);
    }
  };
  useEffect(() => {recordingRef.current = recording;}, [recording])
  // 단일 문장 업로드 함수
  const uploadScript = async (idx: number) => {
    console.log(`[DEBUG][uploadScript] 업로드 시작 idx=${idx}`);
    console.log(`[DEBUG][uploadScript] API_BASE_URL: ${process.env.NEXT_PUBLIC_API_BASE_URL}`);
    
    if (!audioCtx) {
      console.error('[ERROR][uploadScript] audioCtx is undefined');
      return;
    }
    if (!scripts || !scripts[idx]) {
      console.error(`[ERROR][uploadScript] scripts or scripts[${idx}] is undefined`);
      return;
    }
    
    const blobs = getAllBlobs();
    console.log(`[DEBUG][uploadScript] blobs:`, blobs);
    console.log(`[DEBUG][uploadScript] idx: ${idx}`);
    
    const blob = blobs[idx];
    if (!blob) {
      console.error(`[ERROR][uploadScript] blob is undefined for idx=${idx}`);
      console.error(`[ERROR][uploadScript] available keys:`, Object.keys(blobs));
      return;
    }
    
    const scriptId = scripts[idx].id;
    console.log(`[DEBUG][uploadScript] scriptId: ${scriptId}`);

    try {
      console.log(`[🔄 변환 중] 문장 ${idx + 1}번 오디오 변환 시작`);
      const arrayBuffer = await blob.arrayBuffer();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      const { encodeWav } = await import('@/utils/encodeWav');
      const wavBlob = encodeWav(audioBuffer);
      console.log(`[✅ 변환 완료] 문장 ${idx + 1}번 WAV 변환 완료, 크기: ${wavBlob.size} bytes`);

      const formData = new FormData();
      // 백엔드 파일명 형식에 맞게 수정 (확장자 포함)
      formData.append('file', wavBlob, `script_${scriptId}.wav`);

      console.log(`[DEBUG][uploadScript] axios.post 시작 idx=${idx}, scriptId=${scriptId}`);
      
      // Authorization 헤더 가져오기
      const accessToken = localStorage.getItem('access_token');
      console.log(`[DEBUG] accessToken 존재:`, !!accessToken);
      
      const headers: Record<string, string> = {};
      
      if (accessToken) {
        headers.Authorization = `Bearer ${accessToken}`;
        console.log(`[DEBUG] Authorization 헤더 추가됨`);
      } else {
        console.log(`[DEBUG] 비로그인 상태로 요청 전송 (user_id는 NULL로 저장됨)`);
      }
      
      console.log(`[DEBUG] 전송할 헤더:`, headers);

      const url = `${process.env.NEXT_PUBLIC_API_BASE_URL}/scripts/${scriptId}/upload-audio`;
      console.log(`[DEBUG] 요청 URL:`, url);
      
      // 타임아웃 설정 및 재시도 로직 추가
      const maxRetries = 3;
      let retryCount = 0;
      let res = null;
      
      while (retryCount < maxRetries) {
        try {
          res = await axios.post<UploadAudioResponse>(
            url,
            formData,
            { 
              headers,
              timeout: 30000 // 30초 타임아웃 설정 (파일 업로드는 시간이 더 필요)
            }
          );
          console.log(`[📥 서버 응답] 문장 ${idx + 1}번 서버 응답:`, res.data);
          break; // 성공하면 반복 중단
        } catch (retryError) {
          retryCount++;
          console.warn(`[⚠️ 업로드 재시도] 문장 ${idx + 1}번 업로드 실패 (${retryCount}/${maxRetries}):`, retryError);
          
          if (retryCount >= maxRetries) {
            throw retryError; // 최대 재시도 횟수 초과 시 에러 발생
          }
          
          // 재시도 전 잠시 대기 (지수 백오프)
          await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, retryCount)));
        }
      }
      
      console.log(`[🆔 Job ID 수신] 문장 ${idx + 1}번 job_id: ${res?.data?.job_id}`);

      if (res?.data && res.data.job_id) {
        addJobId(res.data.job_id);
        setJobIds(prev => [...prev, res.data.job_id!]);
        console.log(`[✅ 업로드 성공] 문장 ${idx + 1}번 업로드 완료!`);
        console.log(`[📊 Job ID 추가] 총 ${jobIds.length + 1}개의 Job ID 수집됨`);
        
        // 먼저 업로드 성공 알림
        if (onUploadComplete) {
          console.log(`[🔄 콜백 호출] 문장 ${idx + 1}번 onUploadComplete 호출`);
          onUploadComplete(true, [res.data.job_id]);
        }
        
        // 🆕 폴링 방식으로 분석 결과 조회
        const getAnalysisResult = async (jobId: string, maxAttempts = 10) => {
          console.log(`[🔍 분석 폴링 시작] 문장 ${idx + 1}번 분석 결과 폴링 시작`);
          
          for (let attempt = 0; attempt < maxAttempts; attempt++) {
            try {
              console.log(`[🔍 분석 폴링] 시도 ${attempt + 1}/${maxAttempts}`);
              
              const response = await axios.get(
                `${process.env.NEXT_PUBLIC_API_BASE_URL}/tokens/analysis-result/${jobId}/`, // 끝에 슬래시 추가
                { 
                  timeout: 10000,
                  headers: headers
                }
              );
              
              // 분석 결과가 있으면 반환
              if (response.data) {
                console.log(`[✅ 분석 완료] 문장 ${idx + 1}번 분석 결과:`, response.data);
                return response.data;
              }
              
              // 분석 중이면 대기 후 재시도
              console.log(`[⏳ 분석 중] 대기 후 재시도...`);
              await new Promise(resolve => setTimeout(resolve, 2000)); // 2초 대기
            } catch (error: unknown) {
              // 에러 객체 로깅
              console.warn(`[⚠️ 분석 폴링 실패] 시도 ${attempt + 1}/${maxAttempts}:`, error);
              
              // 에러 타입 체크 및 상세 정보 로깅
              if (isAxiosError(error)) {
                if (error.code === 'ECONNABORTED') {
                  console.error('요청 시간 초과 - 서버 응답이 너무 느립니다.');
                } else if (error.message === 'Network Error') {
                  console.error('네트워크 연결 문제 - API 서버에 접근할 수 없습니다.');
                  console.error('API URL:', `${process.env.NEXT_PUBLIC_API_BASE_URL}/tokens/analysis-result/${jobId}/`);
                } else if (error.response) {
                  console.error('서버 응답 에러:', error.response.status, error.response.data);
                }
              } else {
                console.error('알 수 없는 에러 발생:', (error as Error)?.message || '상세 정보 없음');
              }
              
              // 마지막 시도가 아니면 대기 후 재시도
              if (attempt < maxAttempts - 1) {
                await new Promise(resolve => setTimeout(resolve, 2000));
              }
            }
          }
          
          console.error(`[❌ 분석 폴링 실패] 문장 ${idx + 1}번 최대 시도 횟수 초과`);
          return null; // 최대 시도 횟수를 초과하면 null 반환
        };
        
        // 백그라운드에서 분석 결과 폴링 (결과를 기다리지 않음)
        getAnalysisResult(res.data.job_id)
          .then(result => {
            if (result) {
              console.log(`[✅ 분석 완료] 문장 ${idx + 1}번 분석 결과 폴링 성공`);
              // 필요한 경우 여기서 추가 처리
            }
          })
          .catch(error => {
            console.error(`[❌ 분석 실패] 문장 ${idx + 1}번 분석 폴링 중 예외 발생:`, error);
          });
      } else {
        console.error(`[❌ 업로드 실패] 문장 ${idx + 1}번 job_id가 응답에 없습니다.`);
        if (onUploadComplete) onUploadComplete(false, []);
      }
    } catch (e) {
      console.error(`[❌ 업로드 실패] 문장 ${idx + 1}번 업로드 중 에러:`, e);
      if (onUploadComplete) onUploadComplete(false, []);
    }
  };

  const stopScriptRecording = async (scriptIdx: number) => {
    if (!recordingRef.current) {
      console.warn(`[⚠️ 녹음 중지] 문장 ${scriptIdx + 1}번 - 녹음 중이 아니므로 건너뜀`);
      return;
    }
    console.log(`[🛑 녹음 중지] 문장 ${scriptIdx + 1}번 녹음 중지 시작`);
    try {
      console.log(`[🎤 녹음 종료] 문장 ${scriptIdx + 1}번 녹음 종료 처리`);
      await stopRecording(scriptIdx);
      
      console.log(`[📝 상태 업데이트] 문장 ${scriptIdx + 1}번 녹음 완료 상태로 변경`);
      setRecordedScripts((prev) => {
        const next = [...prev];
        next[scriptIdx] = true;
        return next;
      });
      
      // 분석 중 상태 설정 (이 부분이 중요)
      if (onUploadComplete) {
        onUploadComplete(false, []); // 분석 중 상태로 설정
      }
      
      await uploadScript(scriptIdx); // 각 문장별로 업로드
      console.log(`[DEBUG][stopScriptRecording] 업로드 완료 idx=${scriptIdx}`);
    } catch (e) {
      console.error('[ERROR][stopScriptRecording] in useDubbingRecorder failed', e);
      // 에러 발생 시에도 분석 중 상태 해제
      if (onUploadComplete) {
        onUploadComplete(true, []); // 분석 완료 상태로 설정
      }
    }
  };

  return {
    recording,
    recordedScripts,
    uploading,
    startScriptRecording,
    stopScriptRecording,
    getAllBlobs, // 녹음된 Blob들에 접근할 수 있도록 추가
    // allRecorded, // 제거
    // uploadAllRecordings, // 제거
  };
}