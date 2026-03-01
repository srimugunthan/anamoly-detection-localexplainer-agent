import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
    ];
  },
  images: {
    // Allow next/image to load plots proxied through the Next.js dev server
    unoptimized: true,
  },
};

export default nextConfig;
