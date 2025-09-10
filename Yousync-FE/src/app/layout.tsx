import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import '../styles/globals.css'
import QueryProvider from './providers/QueryProvider'
import { GoogleOAuthProvider } from '@react-oauth/google'
import React, { ReactNode } from 'react'
import ConditionalNavBar from '@/components/ui/ConditionalNavBar'
import FloatingRequestButton from '@/components/ui/FloatingRequestButton'
const inter = Inter({ subsets: ['latin'] });
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="ko">
      <body className={inter.className}>
        <GoogleOAuthProvider clientId={process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID!}>
          <QueryProvider>
            <ConditionalNavBar />
            {children}
            <FloatingRequestButton />
          </QueryProvider>
        </GoogleOAuthProvider>
      </body>
    </html>
  );
}