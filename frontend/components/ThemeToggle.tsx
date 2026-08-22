"use client";

import { type Theme, useTheme } from "@/lib/useTheme";

const OPTIONS: { value: Theme; label: string; icon: string }[] = [
  { value: "light", label: "Light", icon: "☀" },
  { value: "dark", label: "Dark", icon: "☾" },
  { value: "system", label: "Match system", icon: "◐" },
];

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();

  return (
    <div
      role="radiogroup"
      aria-label="Color theme"
      className="flex items-center gap-0.5 rounded-lg border border-border bg-background p-0.5"
    >
      {OPTIONS.map((option) => (
        <button
          key={option.value}
          role="radio"
          aria-checked={theme === option.value}
          aria-label={option.label}
          title={option.label}
          onClick={() => setTheme(option.value)}
          className={`h-6 w-6 rounded-md text-xs transition-colors ${
            theme === option.value
              ? "bg-surface text-foreground shadow-sm"
              : "text-muted-subtle hover:text-foreground"
          }`}
        >
          {option.icon}
        </button>
      ))}
    </div>
  );
}
