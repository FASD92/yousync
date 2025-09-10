// 서버 컴포넌트 (기본 틀)
import DubbingContainer from "@/components/dubbing/DubbingContainer";
import type { TokenDetailResponse, ServerPitch } from "@/types/pitch";
import { getServerPitchData } from "@/app/api/hello/getServerpitchData";
import { getDuetTokenDetail } from "@/app/api/hello/getDuetTokenDetail";
// 통합된 DubbingContainer 사용
// import DuetDubbingContainer from "@/components/duetdubbing/DuetDubbingContainer";

// 서버에서 데이터 fetch하는 함수들
export default async function Page(props: any) {
  const { params, searchParams }: any = props;
  const { id } = params;
  const modalId = searchParams.modalId as string | undefined;
  const actor1 = searchParams.actor1 as string | undefined;
  const actor2 = searchParams.actor2 as string | undefined;
  const selected = searchParams.selected as string | undefined;

  if (!actor1 || !actor2 || !selected) {
    // 에러 처리
    return <div>배우 정보가 없습니다.</div>;
  }
  
  const { tokenData, front_data } = await getDuetTokenDetail(actor1, actor2, selected);//id
  const serverPitchData = await getServerPitchData(actor1);

  return (
    <DubbingContainer
      tokenData={tokenData}
      front_data={front_data}
      serverPitchData={serverPitchData}
      id={id}
      modalId={modalId}
      isDuet={true} // 듀엣 모드 활성화
      selectedActor={selected} // 선택한 배우 정보 전달
    />
  );
}
  