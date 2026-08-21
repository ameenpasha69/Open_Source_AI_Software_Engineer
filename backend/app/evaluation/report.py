from app.evaluation.models import EvalReport


def format_report(report: EvalReport) -> str:
    lines = [
        "Local AI Software Engineer Evaluation",
        "─" * 42,
        "",
        f"Config:                {', '.join(f'{k}={v}' for k, v in report.config.items())}",
        "",
        f"Tasks:                 {report.task_count}",
        f"Successful:            {report.success_count}",
        f"Success rate:          {report.success_rate:.0%}",
        "",
        f"First attempt:         {report.first_attempt_count}",
        f"First-attempt rate:    {report.first_attempt_rate:.0%}",
        "",
        f"Average iterations:    {report.average_iterations:.1f}",
        f"Average tool calls:    {report.average_tool_calls:.1f}",
        f"Average runtime:       {report.average_duration_seconds:.1f}s",
        "",
        f"Regression rate:       {report.regression_rate:.0%}",
    ]
    if report.retrieval_hit_rate is not None:
        lines.append(f"Retrieval hit rate:    {report.retrieval_hit_rate:.0%}")
    lines.append("")
    lines.append("Per-task detail:")
    for r in report.results:
        marker = "✓" if r.success else "✗"
        lines.append(
            f"  {marker} {r.task_id:<35} status={r.agent_status:<22} "
            f"verification={r.verification_status:<18} iters={r.iterations} "
            f"tools={r.tool_call_count} {r.duration_seconds:.1f}s"
        )
        if r.regressed_tests:
            lines.append(f"      regressed: {', '.join(r.regressed_tests)}")
        if r.error:
            lines.append(f"      error: {r.error}")
    return "\n".join(lines)
