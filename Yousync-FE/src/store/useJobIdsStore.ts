import { create } from 'zustand';

interface JobIdsState {
  multiJobIds: string[];
  setMultiJobIds: (ids: string[]) => void;
  addJobId: (id: string) => void;
  resetJobIds: () => void;
}

export const useJobIdsStore = create<JobIdsState>((set) => ({
  multiJobIds: [],
  setMultiJobIds: (ids) => set({ multiJobIds: ids }),
  addJobId: (id) => set((state) => ({ 
    multiJobIds: state.multiJobIds.includes(id) 
      ? state.multiJobIds 
      : [...state.multiJobIds, id] 
  })),
  resetJobIds: () => set({ multiJobIds: [] }),
})); 