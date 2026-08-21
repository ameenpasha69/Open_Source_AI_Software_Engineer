const STATUS_STYLES: Record<string, string> = {
  running: "bg-accent/10 text-accent border-accent/30",
  done: "bg-success/10 text-success border-success/30",
  verified: "bg-success/10 text-success border-success/30",
  passed: "bg-success/10 text-success border-success/30",
  completed: "bg-success/10 text-success border-success/30",
  failed: "bg-danger/10 text-danger border-danger/30",
  cancelled: "bg-muted/10 text-muted border-muted/30",
  max_iterations_reached: "bg-warning/10 text-warning border-warning/30",
  unverified: "bg-warning/10 text-warning border-warning/30",
  partially_verified: "bg-warning/10 text-warning border-warning/30",
  not_applicable: "bg-muted/10 text-muted border-muted/30",
};

export function StatusBadge({ status }: { status: string }) {
  const style = STATUS_STYLES[status] ?? "bg-muted/10 text-muted border-muted/30";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${style}`}
    >
      {status === "running" && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />}
      {status.replaceAll("_", " ")}
    </span>
  );
}
