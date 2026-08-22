const STATUS_STYLES: Record<string, string> = {
  running: "bg-accent-soft text-accent border-accent/40",
  done: "bg-success-soft text-success border-success/40",
  verified: "bg-success-soft text-success border-success/40",
  passed: "bg-success-soft text-success border-success/40",
  completed: "bg-success-soft text-success border-success/40",
  failed: "bg-danger-soft text-danger border-danger/40",
  cancelled: "bg-background-subtle text-muted border-border",
  max_iterations_reached: "bg-warning-soft text-warning border-warning/40",
  no_progress: "bg-warning-soft text-warning border-warning/40",
  unverified: "bg-warning-soft text-warning border-warning/40",
  partially_verified: "bg-warning-soft text-warning border-warning/40",
  not_applicable: "bg-background-subtle text-muted border-border",
};

export function StatusBadge({ status }: { status: string }) {
  const style = STATUS_STYLES[status] ?? "bg-background-subtle text-muted border-border";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium whitespace-nowrap ${style}`}
    >
      {status === "running" && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />}
      {status.replaceAll("_", " ")}
    </span>
  );
}
