// components/ActorNameModal.tsx
'use client';

import React from 'react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ActorNameModal: React.FC<ModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  const maleActors = [
    '브래드 피트', '레오나르도 디카프리오', '톰 행크스',
    '로버트 다우니 주니어', '크리스 헴스워스', '크리스 에반스',
    '조니 뎁', '조지 클루니', '베네딕트 컴버배치', '이드리스 엘바',
    '모건 프리먼', '윌 스미스', '덴젤 워싱턴', '라이언 레이놀즈',
    '휴 잭맨', '크리스찬 베일', '키아누 리브스', '해리슨 포드', '맷 데이먼'
  ];

  const femaleActors = [
    '안젤리나 졸리', '제니퍼 로렌스', '엠마 왓슨', '스칼렛 요한슨',
    '줄리아 로버츠', '메릴 스트립', '나탈리 포트먼', '산드라 블록', '케이트 블란쳇'
  ];

  const britishActors = [
    '엠마 왓슨', '베네딕트 컴버배치', '이드리스 엘바', '크리스찬 베일'
  ];

  const americanActors = [
    '안젤리나 졸리', '브래드 피트', '레오나르도 디카프리오', '제니퍼 로렌스', '톰 행크스',
    '로버트 다우니 주니어', '스칼렛 요한슨', '크리스 에반스', '조니 뎁', '줄리아 로버츠',
    '메릴 스트립', '조지 클루니', '나탈리 포트먼', '모건 프리먼', '윌 스미스', '덴젤 워싱턴',
    '산드라 블록', '라이언 레이놀즈', '휴 잭맨', '키아누 리브스', '해리슨 포드', '맷 데이먼'
  ];

  const groups = [
    { title: '남자배우', list: maleActors },
    { title: '여자배우', list: femaleActors },
    { title: '영국배우', list: britishActors },
    { title: '미국배우', list: americanActors },
  ];

  return (
    <div className="fixed inset-0 bg-black/40 flex items-start justify-center pt-2 z-50">
      <div className="relative bg-white w-full max-w-none p-8 shadow-lg overflow max-h-[90vh]">
        <button
          onClick={onClose}
          className="absolute -top-1 right-2 text-2xl text-gray-600 hover:text-gray-800"
        >
          ×
        </button>

        <div className="flex divide-x divide-gray-300">
          {groups.map(({ title, list }) => (
            <div key={title} className="flex-1 px-4">
              <h3 className="text-lg font-bold mb-3 text-center">{title}</h3>
              <ul className="flex flex-wrap gap-2 justify-center text-gray-700">
                {list.map((name) => (
                  <li key={name} className="hover:text-black cursor-pointer">
                    {name}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};