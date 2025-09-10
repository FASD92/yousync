// app/urlsearch/page.tsx
import { backendApi } from '@/services/api';
import UrlSearchContainer from '@/components/urlsearch/UrlSearchContainer';
import type { TokenDetailResponse } from '@/types/pitch';
import { extractYoutubeVideoId } from '@/utils/extractYoutubeVideoId';

interface UrlSearchPageProps {
  searchParams: { [key: string]: string | string[] | undefined };
}

async function getTokens(url: string): Promise<TokenDetailResponse[]> {
  try {
    const data = await backendApi.get<TokenDetailResponse[]>(url);
    if (Array.isArray(data) && data.length > 0) {
      return data.map(token => ({
        ...token,
        youtubeId: extractYoutubeVideoId(token.youtube_url) || '',
      }));
    }
    return [];
  } catch (error) {
    console.error('API 호출 에러:', error);
    return [];
  }
}

export default async function Page(props: any) {
  const searchParams = props.searchParams;

  const videoId = searchParams?.videoId;
  let tokens: TokenDetailResponse[] = [];
  let errorMessage: string | null = null;

  if (typeof videoId === 'string' && videoId) {
    const youtubeUrl = `https://www.youtube.com/watch?v=${videoId}`;
    const url = `/urls/tokens?youtube_url=${encodeURIComponent(youtubeUrl)}`;
    tokens = await getTokens(url);
    if (tokens.length === 0) {
      errorMessage = '검색 결과가 없거나 URL을 처리하는 중 오류가 발생했습니다.';
    }
  } else {
    errorMessage = '검색할 URL이 제공되지 않았습니다.';
  }

  return <UrlSearchContainer initialTokens={tokens} errorMessage={errorMessage} />;
}
