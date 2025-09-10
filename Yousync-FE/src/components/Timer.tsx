import React, { useEffect, useState, useRef } from 'react'
import {
    CircularProgressbarWithChildren,
    buildStyles,
} from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

export const Timer = () => {
    //1. 카운트 다운 변수 필요할듯
    const timerRef = useRef<number | null>(null);
    const [percentage, setPercentage] = useState<number>(100);
    useEffect(() => {
        console.log("Timer 마운트됨")
        setPercentage(100);
      
        const timeout = setTimeout(() => {
          requestAnimationFrame(() => {
            // 이 안에서 시작하면, 반드시 100으로 렌더링된 다음에 애니메이션 시작됨
            const start = performance.now();
            const duration = 3000;
      
            function animateFrame(now: number) {
              const elapsed = now - start;
              const progress = Math.min(1, elapsed / duration);
              const value = 100 * (1 - progress);
      
              setPercentage(value);
      
              if (progress < 1) {
                timerRef.current = requestAnimationFrame(animateFrame);
              }
            }
      
            if (timerRef.current !== null) cancelAnimationFrame(timerRef.current);
            timerRef.current = requestAnimationFrame(animateFrame);
          });
        }, 0);
      
        return () => {
          console.log("Timer 언마운트됨")
          clearTimeout(timeout);
          if (timerRef.current !== null) cancelAnimationFrame(timerRef.current);
        };
      }, []);

   
  return (
    <div className='flex flex-row '>
        <div className='w-24 h-24'>
        <CircularProgressbarWithChildren
        strokeWidth={12}
        value={percentage} 
        styles={buildStyles({
          pathColor: '#46c288', // emerald-500: #46c288
          trailColor: '#1f2937', // gray-800
          pathTransition: 'none',
        })}
      >
        <div>3초</div>
    </CircularProgressbarWithChildren>

        </div>
    </div>
  )
}