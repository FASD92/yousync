// src/components/urlsearch/LoadingIndicator.tsx

export default function LoadingIndicator() {
  return (
    <div className="flex items-center justify-center py-20">
      <div className="flex flex-col items-center space-y-4">
        <div className="animate-spin w-8 h-8 border-3 border-green-400 border-t-transparent rounded-full" />
        <div className="text-gray-500 font-medium">검색 결과를 불러오는 중...</div>
      </div>
    </div>
  );
}
