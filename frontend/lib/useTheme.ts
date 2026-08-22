"use client";

import { useCallback, useEffect, useState } from "react";

export type Theme = "light" | "dark" | "system";

export const THEME_STORAGE_KEY = "laise-theme";

/** Inlined into <head> so the stored theme is applied before first paint —
 * without it, a dark-mode user gets a white flash on every navigation. */
export const THEME_INIT_SCRIPT = `
try {
  var t = localStorage.getItem("${THEME_STORAGE_KEY}");
  if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
} catch (e) {}
`;

function apply(theme: Theme) {
  const root = document.documentElement;
  if (theme === "system") delete root.dataset.theme;
  else root.dataset.theme = theme;
}

export function useTheme() {
  // Always "system" on the server and the first client render; the real
  // value is read in the effect below. The <head> script has already put
  // the right colors on screen, so there's nothing to flash.
  const [theme, setTheme] = useState<Theme>("system");

  useEffect(() => {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (stored === "light" || stored === "dark") setTheme(stored);
  }, []);

  const change = useCallback((next: Theme) => {
    setTheme(next);
    apply(next);
    if (next === "system") localStorage.removeItem(THEME_STORAGE_KEY);
    else localStorage.setItem(THEME_STORAGE_KEY, next);
  }, []);

  return { theme, setTheme: change };
}
