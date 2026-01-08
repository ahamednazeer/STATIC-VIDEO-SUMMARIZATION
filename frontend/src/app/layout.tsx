import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  display: 'swap',
  variable: '--font-inter',
});

export const metadata: Metadata = {
  title: "Video Summarizer | Static Keyframe Extraction",
  description: "Transform long videos into meaningful static summaries using K-means clustering and histogram-based feature extraction.",
  keywords: ["video summarization", "keyframe extraction", "k-means clustering", "computer vision"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.variable} style={{ backgroundColor: '#0a0a0b', colorScheme: 'dark' }}>
      <body className="antialiased" style={{ backgroundColor: '#0a0a0b', color: '#f5f5f7', minHeight: '100vh' }}>
        {children}
      </body>
    </html>
  );
}
