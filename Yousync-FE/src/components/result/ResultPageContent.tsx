'use client';

import { useEffect, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import ResultContainer from '@/components/result/ResultContainer';
import { backendApi } from '@/services/api';

// 스크립트 결과 아이템에 대한 타입 정의
interface ScriptResultItem {
  has_result: boolean;
  result: any;
}

// 분석 응답 타입 정의
interface AnalysisResponse {
  script_results?: Array<ScriptResultItem>;
  has_analysis: boolean;
}

export default function ResultPageContent() {
  const searchParams = useSearchParams();
  const tokenId = searchParams.get('token_id');
  const [resultData, setResultData] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchResultData = async () => {
      if (!tokenId) {
        setError('토큰 ID가 없습니다.');
        setLoading(false);
        return;
      }

      try {
        console.log(`토큰 ID ${tokenId}에 대한 분석 결과 가져오기 시작`);
        // 토큰 ID에 해당하는 분석 상태 정보 가져오기
        const response = await backendApi.get<AnalysisResponse>(`/mypage/tokens/${tokenId}/analysis-status`);
        console.log('분석 결과 데이터:', JSON.stringify(response, null, 2)); // 전체 응답 데이터 구조 확인
        
        // 데이터 구조 디버깅
        if (response?.script_results) {
          console.log('script_results 개수:', response.script_results.length);
          
          // 각 스크립트 결과 확인
          response.script_results.forEach((item, index) => {
            console.log(`script_result[${index}] has_result:`, item.has_result);
            if (item.has_result && item.result) {
              console.log(`script_result[${index}] result 키:`, Object.keys(item.result));
              
              // 중첩된 result 객체 확인
              if (item.result.result) {
                console.log(`script_result[${index}] 중첩된 result 키:`, Object.keys(item.result.result));
                console.log(`script_result[${index}] word_analysis 존재 여부:`, !!item.result.result.word_analysis);
                if (item.result.result.word_analysis) {
                  console.log(`script_result[${index}] word_analysis 길이:`, item.result.result.word_analysis.length);
                }
              }
            }
          });
          
          // 유효한 결과가 있는지 확인 (중첩된 result 구조 고려)
          const validResults = response.script_results.filter(item => 
            item.has_result && 
            item.result && 
            item.result.result && 
            item.result.result.word_analysis && 
            Array.isArray(item.result.result.word_analysis) && 
            item.result.result.word_analysis.length > 0
          );
          
          console.log('유효한 결과 개수:', validResults.length);
        }
        
        // 백엔드에서 이미 result 데이터를 포함하여 반환하므로 추가 API 호출 불필요
        setResultData(response);
        setLoading(false);
      } catch (err) {
        console.error('결과 데이터 가져오기 실패:', err);
        setError('결과 데이터를 가져오는데 실패했습니다.');
        setLoading(false);
      }
    };

    fetchResultData();
  }, [tokenId]);

  if (loading) {
    return (
      <div className="min-h-screen bg-neutral-950 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-4" />
          <p>결과 데이터를 불러오는 중...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-neutral-950 text-white flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-400 mb-4">{error}</p>
          <button
            onClick={() => window.history.back()}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition"
          >
            뒤로 가기
          </button>
        </div>
      </div>
    );
  }

  // 분석 결과가 없는 경우
  if (!resultData?.has_analysis) {
    return (
      <div className="min-h-screen bg-neutral-950 text-white flex items-center justify-center">
        <div className="text-center max-w-md mx-auto p-6 bg-gray-900 rounded-xl border border-gray-800">
          <div className="text-5xl mb-4">🎬</div>
          <h2 className="text-2xl font-bold mb-4">분석 결과가 없습니다</h2>
          <p className="text-gray-400 mb-6">
            아직 이 토큰에 대한 분석 결과가 없습니다. 먼저 더빙을 진행해주세요.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button
              onClick={() => window.location.href = `/dubbing/${tokenId}`}
              className="px-4 py-2 bg-gradient-to-r from-blue-500 to-green-500 text-white font-semibold rounded-lg shadow-lg hover:from-blue-600 hover:to-green-600 transition-all"
            >
              🎙️ 더빙하러 가기
            </button>
            <button
              onClick={() => window.history.back()}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white font-semibold rounded-lg transition"
            >
              ↩️ 뒤로가기
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ResultContainer에 전달할 결과 데이터 준비
  const finalResultsData = resultData?.script_results
    ?.filter((item: ScriptResultItem) => 
      item.has_result && 
      item.result && 
      item.result.result && 
      item.result.result.word_analysis && 
      Array.isArray(item.result.result.word_analysis) && 
      item.result.result.word_analysis.length > 0
    )
    ?.map((item: ScriptResultItem) => item.result.result) || [];
  
  // 전달할 데이터 로깅
  console.log('ResultContainer에 전달할 데이터:', finalResultsData);
  console.log('데이터 길이:', finalResultsData.length);
  if (finalResultsData.length > 0) {
    console.log('첫 번째 항목:', finalResultsData[0]);
    console.log('word_analysis 존재 여부:', !!finalResultsData[0]?.word_analysis);
    console.log('word_analysis 길이:', finalResultsData[0]?.word_analysis?.length);
  }

  return (
    <ResultContainer 
      finalResults={finalResultsData}
      hasAnalysisResults={resultData?.has_analysis && finalResultsData.length > 0} 
      showResults={true} 
    />
  );
}
