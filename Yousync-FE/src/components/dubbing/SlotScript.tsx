// /components/dubbing/SlotScript.tsx
"use client";

import { FC, useMemo } from "react";
import { Caption } from "@/types/caption";

interface SlotScriptProps {
  captions: Caption[];
  currentIdx: number;
}

const SlotScript: FC<SlotScriptProps> = ({ captions, currentIdx }) => {
  // activeIdx 기준으로 위아래 2줄, 총 5줄을 가져오기
  const lines = useMemo<(Caption | null)[]>(() => {
    const result: (Caption | null)[] = [];
    for (let offset = -2; offset <= 2; offset++) {
      const idx = currentIdx + offset;
      result.push(idx >= 0 && idx < captions.length ? captions[idx] : null);
    }
    return result;
  }, [captions, currentIdx]);

  return (
    <div className="flex-1 flex items-center justify-center">
      <div className="w-full relative flex items-center" style={{ height: 280 }}>
        {/* 윗줄 2개 */}
        <div className="absolute left-0 w-full flex flex-col items-center" style={{ top: 0 }}>
          {[0, 1].map(i =>
            lines[i] ? (
              <div
                key={i}
                className="text-gray-400 blur-sm text-lg font-medium max-w-full text-center"
                style={{ minHeight: 34, marginTop: i === 1 ? 2 : 0 }}
              >
                {(() => {
                  const textarea = document.createElement('textarea');
                  textarea.innerHTML = lines[i]!.script;
                  return textarea.value;
                })()}
              </div>
            ) : (
              <div key={i} style={{ minHeight: 34 }} />
            )
          )}
        </div>

        {/* 중앙(고정) 자막 */}
        <div
          className="absolute left-0 right-0 flex flex-col items-center"
          style={{ top: "50%", transform: "translateY(-50%)" }}
        >
          {lines[2] && (
            <div
              key="center"
              className="text-2xl font-extrabold text-[#fffde0] px-5 py-2 max-w-full text-center"
              style={{
                textShadow: "0 0 8px #be8cff, 0 0 18px #be8cff, 0 0 32px #be8cff",
                minHeight: 44,
              }}
            >
              {(() => {
                const textarea = document.createElement('textarea');
                textarea.innerHTML = lines[2].script;
                return textarea.value;
              })()}
            </div>
          )}
        </div>

        {/* 아랫줄 2개 */}
        <div className="absolute left-0 w-full flex flex-col items-center" style={{ bottom: 0 }}>
          {[3, 4].map(i =>
            lines[i] ? (
              <div
                key={i}
                className="text-gray-400 blur-sm text-lg font-medium max-w-full text-center"
                style={{ minHeight: 34, marginTop: 2 }}
              >
                {(() => {
                  const textarea = document.createElement('textarea');
                  textarea.innerHTML = lines[i]!.script;
                  return textarea.value;
                })()}
              </div>
            ) : (
              <div key={i} style={{ minHeight: 34 }} />
            )
          )}
        </div>
      </div>
    </div>
  );
};

export default SlotScript;