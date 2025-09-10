import { useState } from "react";
import LanderSequence from "@/components/graph/LanderSequence";
import MainStartButton from "@/components/lending/MainStartButton";

export default function Page() {
  const [showMain, setShowMain] = useState(false);

  return (
    <>
      {!showMain ? (
        <LanderSequence onEnd={() => setShowMain(true)} />
      ) : (
        <MainStartButton />
      )}
    </>
  );
}
