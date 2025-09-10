import { create } from 'zustand';

interface DuetTokenState {
  tokenId: string | null;
  frontDataId: string | null;
  startTime: number | null;
  endTime: number | null;
  actorInfo: {
    name: string;
    character: string;
  } | null;
  setTokenId: (tokenId: string) => void;
  setFrontDataId: (frontDataId: string) => void;
  setStartTime: (startTime: number) => void;
  setEndTime: (endTime: number) => void;
  setActorInfo: (actorInfo: { name: string; character: string }) => void;
  reset: () => void;
}

export const useDuetTokenStore = create<DuetTokenState>((set) => ({
  tokenId: null,
  frontDataId: null,
  startTime: null,
  endTime: null,
  actorInfo: null,
  
  setTokenId: (tokenId: string) => set({ tokenId }),
  setFrontDataId: (frontDataId: string) => set({ frontDataId }),
  setStartTime: (startTime: number) => set({ startTime }),
  setEndTime: (endTime: number) => set({ endTime }),
  setActorInfo: (actorInfo: { name: string; character: string }) => set({ actorInfo }),
  
  reset: () => set({
    tokenId: null,
    frontDataId: null,
    startTime: null,
    endTime: null,
    actorInfo: null,
  }),
}));
