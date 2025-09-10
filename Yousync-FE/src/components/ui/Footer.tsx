import React from "react";

const Footer: React.FC = () => {
  return (
    <footer className="py-12 px-2 border-t border-gray-800 bg-black">
      <div className="max-w-7xl mx-auto">
        <div className="grid md:grid-cols-4 gap-8">
          <div>
            <h3 className="text-3xl font-bold bg-gradient-to-r from-green-400 via-emerald-400 to-teal-400 bg-clip-text text-transparent mb-4">
              YouSync
            </h3>
            <p className="text-gray-500 font-medium">
              AI와 함께 더빙의 새로운 재미를 발견하세요!
            </p>
          </div>
          <div>
            <h4 className="font-bold mb-4 text-white">제품</h4>
            <ul className="space-y-2 text-gray-500">
              <li><a href="#" className="hover:text-green-400 transition-colors font-medium">더빙 연습</a></li>
              <li><a href="#" className="hover:text-emerald-400 transition-colors font-medium">성과 분석</a></li>
              <li><a href="#" className="hover:text-teal-400 transition-colors font-medium">결과 다운로드</a></li>
            </ul>
          </div>
          <div>
            <h4 className="font-bold mb-4 text-white">회사</h4>
            <ul className="space-y-2 text-gray-500">
              <li><a href="#" className="hover:text-green-400 transition-colors font-medium">소개</a></li>
              <li><a href="#" className="hover:text-emerald-400 transition-colors font-medium">팀</a></li>
              <li><a href="#" className="hover:text-teal-400 transition-colors font-medium">채용</a></li>
            </ul>
          </div>
          <div>
            <h4 className="font-bold mb-4 text-white">지원</h4>
            <ul className="space-y-2 text-gray-500">
              <li><a href="#" className="hover:text-green-400 transition-colors font-medium">도움말</a></li>
              <li><a href="#" className="hover:text-emerald-400 transition-colors font-medium">문의하기</a></li>
              <li><a href="#" className="hover:text-teal-400 transition-colors font-medium">개인정보처리방침</a></li>
            </ul>
          </div>
        </div>
        <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-600">
          <p className="font-medium">&copy; 2024 YouSync. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
