// store/useResultStore.ts
import { create } from 'zustand'

interface WordAnalysis {
  word: string;
  text_status: 'fail' | 'pass';
  time_match: boolean;
  overlap_ratio: number;
  mfcc_similarity: number;
  word_score: number;
}

interface Summary {
  text_accuracy: number;
  time_accuracy: number;
  mfcc_average: number;
  total_words: number;
  passed_words: number;
  time_matched_words: number;
}

interface FailureAnalysis {
  stt_failures: string[];
  time_failures: string[];
  mfcc_low_quality: string[];
}

interface UserSTT {
  text: string;
  word_timestamps: Array<{
    word: string;
    start: number;
    end: number;
  }>;
}

export interface PitchResult {
  overall_score: number;
  word_analysis: WordAnalysis[];
  summary: Summary;
  failure_analysis?: FailureAnalysis;
  user_stt?: UserSTT;
}

interface ResultStore {
  finalResults: PitchResult[];
  setFinalResults: (results: PitchResult[] | ((prev: PitchResult[]) => PitchResult[])) => void;
}

export const useResultStore = create<ResultStore>((set) => ({
  finalResults: [],
  setFinalResults: (results) =>
    set((state) => ({
      finalResults: typeof results === 'function' ? results(state.finalResults) : results,
    })),
}));
