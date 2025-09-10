"use client";

interface StatsGridProps {
  stats: {
    totalBookmarks: number;
    totalDubbedTokens: number;
    averageScore: number;
    totalPracticeCount: number;
  };
  loading?: boolean;
}

export default function StatsGrid({ stats, loading }: StatsGridProps) {
  if (loading) {
    return (
      <section>
        <h2 className="text-2xl font-bold mb-6">통계</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {[...Array(4)].map((_, index) => (
            <div
              key={index}
              className="bg-neutral-900 rounded-xl p-6 border border-neutral-800 text-center animate-pulse"
            >
              <div className="w-8 h-8 bg-neutral-700 rounded mb-2 mx-auto"></div>
              <div className="h-6 bg-neutral-700 rounded mb-1"></div>
              <div className="h-4 bg-neutral-700 rounded w-2/3 mx-auto"></div>
            </div>
          ))}
        </div>
      </section>
    );
  }

  const statItems = [
    {
      label: "총 북마크",
      value: `${stats.totalBookmarks}개`,
      icon: "📌",
      color: "text-blue-400"
    },
    {
      label: "더빙한 토큰",
      value: `${stats.totalDubbedTokens}개`,
      icon: "🎬",
      color: "text-green-400"
    },
    {
      label: "평균 완성도",
      value: `${stats.averageScore}%`,
      icon: "⭐",
      color: "text-yellow-400"
    },
    {
      label: "총 연습 횟수",
      value: `${stats.totalPracticeCount}회`,
      icon: "🔥",
      color: "text-red-400"
    }
  ];

  return (
    <section>
      <h2 className="text-2xl font-bold mb-6">통계</h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
        {statItems.map((stat) => (
          <div
            key={stat.label}
            className="bg-neutral-900 rounded-xl p-6 border border-neutral-800 text-center hover:border-neutral-700 transition-colors duration-300"
          >
            <div className="text-3xl mb-2">{stat.icon}</div>
            <div className={`text-2xl font-bold mb-1 ${stat.color}`}>{stat.value}</div>
            <div className="text-gray-400 text-sm">{stat.label}</div>
          </div>
        ))}
      </div>
    </section>
  );
} 