"use client";
import React from "react";

export interface ScoreCardsProps {
  syncRate: number;
  pronunciation: number;
  intonation: number;
  timing: number;
}

const ScoreCards: React.FC<ScoreCardsProps> = ({ syncRate, pronunciation, intonation, timing }) => {
  const scoreCards = [
    { 
      icon: "⭐️", 
      label: "싱크율", 
      value: Math.round(syncRate * 100),
      gradient: "from-amber-400 to-yellow-500",
      shadow: "rgba(251, 191, 36, 0.4)"
    },
    { 
      icon: "🎯", 
      label: "발음", 
      value: Math.round(pronunciation * 100),
      gradient: "from-emerald-400 to-teal-500",
      shadow: "rgba(16, 185, 129, 0.4)"
    },
    { 
      icon: "⏱️", 
      label: "발화타임", 
      value: Math.round(timing * 100),
      gradient: "from-blue-400 to-indigo-500",
      shadow: "rgba(79, 70, 229, 0.4)"
    },
    { 
      icon: "🎵", 
      label: "억양", 
      value: Math.round(intonation * 100),
      gradient: "from-purple-400 to-pink-500",
      shadow: "rgba(219, 39, 119, 0.4)"
    }
  ];
  
  return (
    <div className="w-full max-w-4xl mx-auto mb-16">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6">
        {scoreCards.map((card, idx) => (
          <div
            key={idx}
            className="bg-gray-900/80 backdrop-blur-sm rounded-2xl p-6 flex flex-col items-center shadow-xl transition-all duration-300 transform hover:scale-105 border border-gray-800/50"
            style={{ 
              minWidth: 140,
              boxShadow: `0 10px 25px -5px ${card.shadow}, 0 0 10px rgba(0, 0, 0, 0.1) inset`
            }}
          >
            <div className="text-4xl mb-3">
              {card.icon}
            </div>
            <div className={`text-4xl font-extrabold mb-2 text-transparent bg-clip-text bg-gradient-to-br ${card.gradient}`}>
              {isNaN(card.value) ? 0 : card.value}
            </div>
            <div className="text-base text-gray-400 font-medium">
              {card.label}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ScoreCards;