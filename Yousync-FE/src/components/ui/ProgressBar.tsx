interface ProgressBarProps {
  progress: number;
  label?: string;
  color?: 'emerald' | 'blue' | 'indigo' | 'purple';
  size?: 'sm' | 'md' | 'lg';
  isDuet?: boolean;
  isMyLine?: boolean;
}

export default function ProgressBar({ 
  progress, 
  label, 
  color = 'emerald', 
  size = 'md',
  isDuet = false,
  isMyLine = true
}: ProgressBarProps) {
  const getColorClasses = () => {
    if (isDuet) {
      return isMyLine 
        ? 'bg-gradient-to-r from-emerald-500 via-teal-500 to-green-500'
        : 'bg-gradient-to-r from-blue-600 via-indigo-500 to-violet-500';
    }
    
    switch (color) {
      case 'blue':
        return 'bg-gradient-to-r from-blue-500 via-blue-600 to-blue-700';
      case 'indigo':
        return 'bg-gradient-to-r from-indigo-500 via-indigo-600 to-indigo-700';
      case 'purple':
        return 'bg-gradient-to-r from-purple-500 via-purple-600 to-purple-700';
      default:
        return 'bg-gradient-to-r from-emerald-500 via-teal-500 to-green-500';
    }
  };

  const getSizeClasses = () => {
    switch (size) {
      case 'sm':
        return 'h-1.5';
      case 'lg':
        return 'h-4';
      default:
        return 'h-2.5';
    }
  };

  const getGlowEffect = () => {
    if (isDuet) {
      return isMyLine 
        ? '0 0 10px rgba(16, 185, 129, 0.5)'
        : '0 0 10px rgba(59, 130, 246, 0.5)';
    }
    
    switch (color) {
      case 'blue':
        return '0 0 10px rgba(59, 130, 246, 0.5)';
      case 'indigo':
        return '0 0 10px rgba(99, 102, 241, 0.5)';
      case 'purple':
        return '0 0 10px rgba(147, 51, 234, 0.5)';
      default:
        return '0 0 10px rgba(16, 185, 129, 0.5)';
    }
  };

  return (
    <div className="w-full">
      {label && (
        <div className="flex justify-between items-center mb-2">
          <span className={`text-sm font-medium ${
            isDuet && !isMyLine ? 'text-blue-300' : 'text-emerald-300'
          }`}>
            {label}
          </span>
          <span className={`text-sm font-bold ${
            isDuet && !isMyLine ? 'text-blue-400' : 'text-emerald-400'
          }`}>
            {Math.round(progress)}%
          </span>
        </div>
      )}
      <div className={`relative w-full ${getSizeClasses()} bg-gray-800/90 rounded-lg overflow-hidden shadow-inner border border-gray-700/30`}>
        <div
          className={`absolute top-0 left-0 h-full ${getColorClasses()} transition-all duration-500 ease-out`}
          style={{ 
            width: `${Math.min(100, Math.max(0, progress))}%`,
            boxShadow: getGlowEffect()
          }}
        >
          {!label && (
            <span 
              className="absolute right-2 font-bold text-white drop-shadow-md flex items-center h-full text-xs"
            >
              {Math.round(progress)}%
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
