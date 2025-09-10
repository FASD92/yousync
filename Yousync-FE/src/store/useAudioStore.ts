import {create} from 'zustand';

interface AudioStore {
    audioCtx: AudioContext | null;
    analyser: AnalyserNode | null;
    source: MediaStreamAudioSourceNode | null;
    stream: MediaStream | null;
    setAudioCtx: (payload: {
        audioCtx: AudioContext;
        analyser: AnalyserNode;
        source: MediaStreamAudioSourceNode;
        stream: MediaStream;
      }) => void;
      reset: () => void;
}

interface JobIdState {
  jobId: string | null;
  setJobId: (jobId: string | null) => void;
  resetJobId: () => void;
}

// 전역적으로 공유할 값: audioCtx, analyser, source, stream
// setAudioCtx(): 초기화해서 전역에 저장하는 함수
// reset(): 컴포넌트 언마운트 시 리소스 해제용
export const useAudioStore = create<AudioStore>((set) => ({
    audioCtx: null,
    analyser: null,
    source: null,
    stream: null,
    setAudioCtx: ({ audioCtx, analyser, source, stream }) =>
      set({ audioCtx, analyser, source, stream }),
    reset: () => set({ audioCtx: null, analyser: null, source: null, stream: null }),
}))

export const useJobIdStore = create<JobIdState>((set) => ({
  jobId: null,
  setJobId: (jobId) => set({ jobId }),
  resetJobId: () => set({ jobId: null }),
}));


