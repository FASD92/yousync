import { useRef, useState } from 'react';
import { useAudioStore } from '@/store/useAudioStore';

export function useVoiceRecorder() {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const allBlobsRef = useRef<Record<number, Blob>>({});
  const [recording, setRecording] = useState(false);
  const {stream} = useAudioStore();
  
  // 타이밍 추적을 위한 ref
  const recordingStartTimeRef = useRef<number | null>(null);
  const recordingStopTimeRef = useRef<number | null>(null);
  
  // 스크립트 정보 저장용 ref
  const scriptInfoRef = useRef<{ startTime: number; endTime: number; scriptIndex: number } | null>(null);
  
  // 동적 보정값 저장용 ref (실제 측정된 지연들의 평균)
  const dynamicOffsetRef = useRef<number>(0.1); // 초기값 100ms
  const measuredDelaysRef = useRef<number[]>([]); // 측정된 지연들

  // 오디오 길이 분석 및 스크립트 시간과 비교
  const analyzeRecordingTiming = async (blob: Blob, scriptIndex: number, stopDelay: number) => {
    try {
      // AudioContext를 사용하여 오디오 길이 측정
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const arrayBuffer = await blob.arrayBuffer();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      
      const actualDuration = audioBuffer.duration; // 실제 오디오 길이 (초)
      const scriptInfo = scriptInfoRef.current;
      
      if (scriptInfo && scriptInfo.scriptIndex === scriptIndex) {
        const expectedDuration = scriptInfo.endTime - scriptInfo.startTime;
        const timeDifference = Math.abs(actualDuration - expectedDuration);
        const accuracyPercentage = ((1 - timeDifference / expectedDuration) * 100).toFixed(2);
        
        // 타이밍 지연 분석
        const recordingRequestTime = recordingStartTimeRef.current;
        const recordingStopTime = recordingStopTimeRef.current;
        const totalRecordingTime = recordingStopTime && recordingRequestTime 
          ? (recordingStopTime - recordingRequestTime) / 1000 
          : 0;
        
        console.log('🎵 === 녹음 타이밍 분석 ===');
        console.log(`📝 문장 ${scriptIndex}:`);
        console.log(`   스크립트 시간: ${scriptInfo.startTime}s ~ ${scriptInfo.endTime}s`);
        console.log(`   예상 길이: ${expectedDuration.toFixed(3)}s`);
        console.log(`   실제 녹음 길이: ${actualDuration.toFixed(3)}s`);
        console.log(`   차이: ${timeDifference.toFixed(3)}s`);
        console.log(`   정확도: ${accuracyPercentage}%`);
        console.log(`   전체 녹음 시간: ${totalRecordingTime.toFixed(3)}s`);
        console.log(`   MediaRecorder.stop() 지연: ${stopDelay}ms`);
        
        // WAV 파일 자르기 (후처리 방식)
        if (actualDuration > expectedDuration) {
          console.log('✂️  WAV 파일 자르기 시작...');
          const trimmedBlob = await trimAudioBlob(blob, expectedDuration, audioContext);
          
          // 자른 파일로 교체
          allBlobsRef.current[scriptIndex] = trimmedBlob;
          
          // 자른 파일의 길이 확인
          const trimmedArrayBuffer = await trimmedBlob.arrayBuffer();
          const trimmedAudioBuffer = await audioContext.decodeAudioData(trimmedArrayBuffer);
          const trimmedDuration = trimmedAudioBuffer.duration;
          
          console.log(`   자른 후 길이: ${trimmedDuration.toFixed(3)}s`);
          console.log(`   자르기 후 차이: ${Math.abs(trimmedDuration - expectedDuration).toFixed(3)}s`);
          
          if (Math.abs(trimmedDuration - expectedDuration) < 0.1) {
            console.log('✅ WAV 파일 자르기 성공!');
          } else {
            console.warn('⚠️  WAV 파일 자르기 후에도 차이가 있습니다.');
          }
          
          audioContext.close();
        } else {
          console.log('✅ 녹음 길이가 정확합니다. 자르기 불필요.');
          audioContext.close();
        }
        
        // 지연 원인 분석
        if (timeDifference > 0.1) {
          console.log('🔍 지연 원인 분석:');
          if (actualDuration > expectedDuration) {
            console.log(`   - MediaRecorder.stop() 지연: ${stopDelay}ms`);
            console.log(`   - 초과 녹음: ${(actualDuration - expectedDuration).toFixed(3)}s`);
            console.log(`   - 후처리로 정확한 길이로 자르기 완료`);
          } else {
            console.log(`   - 녹음이 예상보다 짧음 (조기 종료 가능성)`);
          }
        }
        
        if (timeDifference > 0.5) {
          console.warn(`⚠️  타이밍 차이가 큽니다! (${timeDifference.toFixed(3)}s)`);
        } else if (timeDifference > 0.2) {
          console.log(`⚠️  타이밍 차이가 있습니다. (${timeDifference.toFixed(3)}s)`);
        } else {
          console.log(`✅ 타이밍이 정확합니다! (차이: ${timeDifference.toFixed(3)}s)`);
        }
        console.log('========================');
      }
      
    } catch (error) {
      console.error('[ERROR] 오디오 길이 분석 실패:', error);
    }
  };

  // WAV 파일 자르기 함수
  const trimAudioBlob = async (blob: Blob, targetDuration: number, audioContext: AudioContext): Promise<Blob> => {
    try {
      const arrayBuffer = await blob.arrayBuffer();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      
      // 목표 샘플 수 계산
      const targetSamples = Math.floor(targetDuration * audioBuffer.sampleRate);
      
      // 앞에서 자를 샘플 수 계산 (초과분 - 0.2초)
      const actualDuration = audioBuffer.duration;
      let startOffsetSamples = 0;
      if (actualDuration > targetDuration) {
        const excess = actualDuration - targetDuration;
        startOffsetSamples = Math.max(0, Math.floor((excess - 0.2) * audioBuffer.sampleRate));
      }

      // 새로운 AudioBuffer 생성 (목표 길이만큼)
      const trimmedBuffer = audioContext.createBuffer(
        audioBuffer.numberOfChannels,
        targetSamples,
        audioBuffer.sampleRate
      );

      // 각 채널의 데이터를 복사 (앞에서 자르기)
      for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
        const originalData = audioBuffer.getChannelData(channel);
        const trimmedData = trimmedBuffer.getChannelData(channel);
        for (let i = 0; i < targetSamples; i++) {
          trimmedData[i] = originalData[startOffsetSamples + i] || 0;
        }
      }

      // AudioBuffer를 Blob으로 변환
      const trimmedBlob = await audioBufferToBlob(trimmedBuffer);
      return trimmedBlob;
    } catch (error) {
      console.error('[ERROR] WAV 파일 자르기 실패:', error);
      return blob; // 실패 시 원본 반환
    }
  };

  // AudioBuffer를 Blob으로 변환하는 함수
  const audioBufferToBlob = async (audioBuffer: AudioBuffer): Promise<Blob> => {
    // 간단한 WAV 인코딩 (기존 encodeWav 함수 활용)
    const { encodeWav } = await import('@/utils/encodeWav');
    return encodeWav(audioBuffer);
  };

  const startRecording = async (scriptIndex: number, startTime: number, endTime: number) => {
    // const stream = await navigator.mediaDevices.getUserMedia({ audio: true }); // 마이크 접근 권한 요청
    if(!stream){
      console.warn("Stream not initialized");
      return;
    }
    
    // 스크립트 정보 저장
    scriptInfoRef.current = { startTime, endTime, scriptIndex };
    
    // 녹음 시작 시간 기록
    recordingStartTimeRef.current = Date.now();
    console.log(`[TIMING] 녹음 시작 요청: ${new Date(recordingStartTimeRef.current).toISOString()}`);
    console.log(`[SCRIPT] 문장 ${scriptIndex}: ${startTime}s ~ ${endTime}s (예상 길이: ${endTime - startTime}s)`);
    console.log(`[OFFSET] 현재 동적 보정값: ${dynamicOffsetRef.current.toFixed(3)}s`);
    
    const supportsOpus = MediaRecorder.isTypeSupported('audio/webm;codecs=opus');
    const supportsMp4 = MediaRecorder.isTypeSupported('audio/mp4');

    // 최적의 설정 선택
    const recorderOptions = supportsOpus
      ? { mimeType: 'audio/webm;codecs=opus', audioBitsPerSecond: 320000 }
      : supportsMp4
        ? { mimeType: 'audio/mp4', audioBitsPerSecond: 320000 }
        : { audioBitsPerSecond: 320000 };  // 기본 설정

    console.log(`[RECORDER] 사용 중인 녹음 설정:`, recorderOptions);
    const recorder = new MediaRecorder(stream, recorderOptions);

    mediaRecorderRef.current = recorder;  // 이후에 중지 같은 작업에 쓸 수 있음.
    chunksRef.current = []; // 녹음된 데이터 담을 배열(chunks)을 초기화

    recorder.ondataavailable = (e) => {// 녹음 중 일정 주기마다 오디오 데이터를 e.data 형태로 전달
      chunksRef.current.push(e.data); //chunkRef.current 배열에 순서대로 저장
    };

    recorder.start(); // 녹음 시작
    console.log('녹음 시작');
    setRecording(true); // 현재 녹음 중이라는 상태를 true로
  };

  // Blob(녹음된 오디오)를 Promise로 반환, 녹음이 안된 경우에는 null 반환
  const stopRecording = async (idx: number): Promise<Blob | null> => {
    console.log('[DEBUG] stopRecording called', idx);
    if (mediaRecorderRef.current) {
      console.log('[DEBUG] mediaRecorderRef.current.state:', mediaRecorderRef.current.state);
    } else {
      console.warn('[DEBUG] mediaRecorderRef.current is null');
    }
    return new Promise((resolve) => {
      if (!mediaRecorderRef.current) {
        console.warn('[WARN] stopRecording: mediaRecorderRef.current is null');
        setRecording(false);
        console.log('[DEBUG] stopRecording resolve(null) - mediaRecorderRef.current is null');
        return resolve(null);
      }
      mediaRecorderRef.current.onstop = async () => {
        console.log('[DEBUG] mediaRecorderRef.current.onstop fired');
        try {
          const blob = new Blob(chunksRef.current, { type: 'audio/wav' });
          allBlobsRef.current[idx] = blob;
          console.log('[DEBUG] Blob 생성됨', blob);
          
          // 녹음 종료 시간 기록
          recordingStopTimeRef.current = Date.now();
          const stopDelay = recordingStopTimeRef.current - recordingStartTimeRef.current!;
          
          // 녹음 타이밍 분석 실행
          await analyzeRecordingTiming(blob, idx, stopDelay);

          // 🔽 실제 서버로 보낼 Blob의 길이(초) 콘솔 출력
          const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
          const arrayBuffer = await allBlobsRef.current[idx].arrayBuffer();
          const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
          console.log(`🚀 [업로드 전] 최종 Blob 길이: ${audioBuffer.duration.toFixed(3)}s (문장 ${idx})`);
          audioContext.close();
          
          resolve(allBlobsRef.current[idx]);
          setRecording(false);
          // 여기서 wav파일로 변환해야
          console.log('[DEBUG] setRecording(false) called in onstop');
        } catch (e) {
          console.error('[ERROR] onstop handler failed', e);
          setRecording(false);
          console.log('[DEBUG] stopRecording resolve(null) - onstop handler failed');
          resolve(null);
        }
      };
      try {
        mediaRecorderRef.current.stop();
        console.log('[DEBUG] mediaRecorderRef.current.stop() called');
      } catch (e) {
        console.error('[ERROR] mediaRecorderRef.current.stop() threw', e);
        setRecording(false);
        console.log('[DEBUG] stopRecording resolve(null) - stop() threw');
        resolve(null);
      }
      setTimeout(() => {
        if (recording) {
          console.warn('[WARN] stopRecording: onstop not called in 1s, forcing setRecording(false)');
          setRecording(false);
          console.log('[DEBUG] stopRecording resolve(null) - onstop timeout');
          resolve(null);
        }
      }, 1000);
    });
  };

  return {
    startRecording,
    stopRecording,
    recording,
    getAllBlobs: () => allBlobsRef.current,  // Record<number, Blob> 형태로 반환
  };
}