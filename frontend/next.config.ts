import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emits .next/standalone: a self-contained server with only the modules it
  // actually imports, so the production image carries that instead of the whole
  // node_modules tree. Additive -- `next dev` and `next start` are unaffected.
  output: "standalone",
};

export default nextConfig;
