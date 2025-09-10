import type { NextConfig } from 'next'
/** @type {import('next').NextConfig} */

const nextConfig: NextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['img.youtube.com', 'phinf.pstatic.net', 
            'encrypted-tbn0.gstatic.com'],
  },
  
  allowedDevOrigins: [
    // 프로덕션 도메인, 개발 시점에 로드하는 도메인
    'https://yousync.duckdns.org',  
    // 'https://yousync.duckdns.org:3000'
  ],
  
  
}

export default nextConfig
