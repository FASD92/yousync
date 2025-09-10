import UploadRequestContainer from "@/components/uploadrequest/UploadRequestContainer";
const UploadRequestPage = () => {

  return (
    <div className="bg-neutral-950 min-h-screen px-6 py-8 text-white">
      <h1 className="text-2xl font-bold mb-6">요청 게시판</h1>
      <UploadRequestContainer />
    </div>
  );
};

export default UploadRequestPage;