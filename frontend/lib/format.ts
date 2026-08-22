/** Human-readable byte size. Model sizes span three orders of magnitude
 * (46 MB to 20 GB), so the unit has to adapt rather than being fixed. */
export function formatBytes(bytes: number): string {
  if (!bytes) return "—";
  const gb = bytes / 1024 ** 3;
  if (gb >= 1) return `${gb.toFixed(gb >= 10 ? 0 : 1)} GB`;
  return `${Math.round(bytes / 1024 ** 2)} MB`;
}

/** Context windows are quoted in tokens and are always round numbers. */
export function formatTokens(tokens: number): string {
  return tokens >= 1000 ? `${Math.round(tokens / 1024)}K` : String(tokens);
}

export function formatRelativeTime(iso: string | null): string {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const seconds = Math.round((Date.now() - then) / 1000);
  if (seconds < 60) return "just now";
  const units: [number, Intl.RelativeTimeFormatUnit][] = [
    [60, "minute"],
    [24, "hour"],
    [7, "day"],
    [4.35, "week"],
    [12, "month"],
  ];
  let value = seconds / 60;
  let unit: Intl.RelativeTimeFormatUnit = "minute";
  for (const [divisor, nextUnit] of units.slice(1)) {
    if (Math.abs(value) < divisor) break;
    value /= divisor;
    unit = nextUnit;
  }
  return new Intl.RelativeTimeFormat("en", { numeric: "auto" }).format(-Math.round(value), unit);
}

/** Strips the implicit `:latest` the backend reports but nobody types. */
export function displayModelName(name: string): string {
  return name.replace(/:latest$/, "");
}
