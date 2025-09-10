
// export function extractYoutubeVideoId(url: string) {
//   try {
//     const parsed = new URL(url);

//     // 유튜브 도메인 (서브도메인 처리)
//     const hostname = parsed.hostname.replace(/^www\./, '');

//     // youtube.com/watch?v=xxxx 형식
//     if (hostname === "youtube.com" || hostname === "m.youtube.com") {
//       const id = parsed.searchParams.get("v");
//       console.log('추출된 YouTube ID:', id);
//       return id;
//     }

//     // youtu.be/xxxx 형식
//     if (hostname === "youtu.be") {
//       const id = parsed.pathname.slice(1);
//       console.log('추출된 YouTube ID:', id);
//       return id;
//     }

//     return null;
    
//   } catch {
//     console.error('유효하지 않은 URL:', url);
//     return null;
//   }
// }

export const extractYoutubeVideoId = (url: string): string | null => {
  const match = url.match(/(?:v=|youtu.be\/)([\w-]+)/);
  return match ? match[1] : null;
};

export const getYoutubeThumbnail = (url: string) => {
  const match = url.match(/(?:v=|youtu.be\/)([\w-]+)/);
  return match ? `https://img.youtube.com/vi/${match[1]}/mqdefault.jpg` : '';
};