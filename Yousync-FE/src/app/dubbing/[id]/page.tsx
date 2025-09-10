// 서버 컴포넌트 (기본 틀)
// import DubbingContainer from "@/components/dubbing/DubbingContainer";
import { getServerPitchData } from "@/app/api/hello/getServerpitchData";
import { getTokenDetail } from "@/app/api/hello/getTokenDetail";
import DubbingContainer from "@/components/dubbing/DubbingContainer";
// 서버에서 데이터 fetch하는 함수들
export default async function Page({ params, searchParams }: any) {
    const { id } = params;
    const modalId = searchParams.modalId as string | undefined;
  
    console.log("더빙 페이지 받은 데이터:", {
      id: id,
      modalId: modalId,
      searchParams: searchParams
    });
  
    const { tokenData, front_data } = await getTokenDetail(id);
    const serverPitchData = await getServerPitchData(id);
  
    return (
      <DubbingContainer
        tokenData={tokenData}
        front_data={front_data}
        serverPitchData={serverPitchData}
        id={id}
        modalId={modalId}
      />
    );
  }
  