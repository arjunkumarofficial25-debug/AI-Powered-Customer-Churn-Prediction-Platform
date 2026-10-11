import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { SiteHeader } from "@/components/site-header";
import "./globals.css";

const geistSans = Geist({ subsets: ["latin"], variable: "--font-geist-sans" });
const geistMono = Geist_Mono({ subsets: ["latin"], variable: "--font-geist-mono" });

export const metadata: Metadata = {
  title: "Churn Prediction Platform",
  description:
    "Predict which telecom customers are likely to churn, understand why, and act on it with retention recommendations.",
};

export const viewport: Viewport = {
  themeColor: "#14171c",
  colorScheme: "dark",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} bg-background`}>
      <body className="min-h-screen font-sans antialiased">
        <SiteHeader />
        <main className="mx-auto w-full max-w-6xl px-4 py-8 md:px-6">{children}</main>
        <footer className="mx-auto max-w-6xl px-4 pb-8 text-xs text-muted-foreground md:px-6">
          Dataset: IBM Telco Customer Churn sample data (7,043 customers).
        </footer>
      </body>
    </html>
  );
}
