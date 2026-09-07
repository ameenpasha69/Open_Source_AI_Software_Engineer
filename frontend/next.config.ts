import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Next.js blocks cross-origin requests for dev resources (JS chunks, HMR)
  // by default, so loading the UI from another device on the LAN leaves the
  // page stuck on "Loading…" while the chunks are refused. The host serving
  // the dev server has to be named explicitly.
  //
  // Dev-only, and deliberately a specific address rather than a wildcard:
  // this opens the dev server's own resources to that origin, so it should
  // stay as narrow as the setup actually requires. Change it if this
  // machine's LAN address changes.
  allowedDevOrigins: ["192.168.0.6"],
};

export default nextConfig;
