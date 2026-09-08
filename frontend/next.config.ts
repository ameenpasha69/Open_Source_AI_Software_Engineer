import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // "standalone" emits a server with only the modules it imports, which is what
  // the Docker image copies instead of the whole node_modules tree.
  //
  // Not on Vercel, though: Vercel builds its own output and tracing the
  // standalone server on top of that fails the build on .next/*.nft.json. It
  // sets VERCEL=1, so the mode is chosen from that rather than being pinned.
  output: process.env.VERCEL ? undefined : "standalone",
};

export default nextConfig;
