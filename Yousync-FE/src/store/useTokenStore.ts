import { create } from 'zustand';

interface TokenInfoState {
  id: number | null;
  actor_name: string | null;
  start_time: number | null;
  end_time: number | null;
  bgvoice_url: string | null;
  setTokenInfo: (info: { id: number; actor_name: string; start_time: number; end_time: number; bgvoice_url: string }) => void;
  reset: () => void;
}

export const useTokenStore = create<TokenInfoState>((set) => ({
  id: null,
  actor_name: null,
  start_time: null,
  end_time: null,
  bgvoice_url: null,
  setTokenInfo: (info) => set({
    id: info.id,
    actor_name: info.actor_name,
    start_time: info.start_time,
    end_time: info.end_time,
    bgvoice_url: info.bgvoice_url,
  }),
  reset: () => set({
    id: null,
    actor_name: null,
    start_time: null,
    end_time: null,
    bgvoice_url: null,
  }),
}));
