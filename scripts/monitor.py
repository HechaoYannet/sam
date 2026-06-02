"""SAM Training Monitor — AI Training Companion.

Watches a training run and provides intelligent progress analysis:
  - Detects convergence, plateaus, and anomalies
  - Generates periodic checkpoint summaries
  - Alerts on NaN, loss spikes, and stalled training

Usage:
    # Watch a running training session (poll every 60s)
    python scripts/monitor.py --output_dir outputs/run1

    # One-shot report
    python scripts/monitor.py --output_dir outputs/run1 --once

    # Watch with custom interval
    python scripts/monitor.py --output_dir outputs/run1 --interval 30
"""

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def load_log(log_path: Path) -> list[dict]:
    """Load the structured training log."""
    if not log_path.exists():
        return []
    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def analyze_metrics(entries: list[dict]) -> dict:
    """Analyze training metrics for trends and issues.

    Returns a report dict with findings and recommendations.
    """
    report = {
        "status": "unknown",
        "findings": [],
        "warnings": [],
        "recommendations": [],
    }

    checkpoints = [e for e in entries if e.get("type") == "checkpoint"]
    alerts = [e for e in entries if e.get("type") == "alert"]

    if not checkpoints:
        report["status"] = "no_data"
        report["findings"].append("No checkpoints recorded yet — training may not have started.")
        return report

    # Extract metrics over time
    epochs = []
    metrics_by_epoch = {}

    for ckpt in checkpoints:
        ep = ckpt.get("epoch", 0)
        epochs.append(ep)
        if "metrics" in ckpt:
            metrics_by_epoch[ep] = ckpt["metrics"]

    if not epochs:
        report["status"] = "no_data"
        return report

    latest_epoch = max(epochs)
    first_epoch = min(epochs)
    total_epochs = len(epochs)
    report["latest_epoch"] = latest_epoch
    report["total_checkpoints"] = total_epochs
    report["elapsed_epochs"] = latest_epoch - first_epoch

    # Trend analysis on key metrics
    loss_keys = ["loss_align", "loss_rel", "loss_analogy"]
    for key in loss_keys:
        values = []
        for ep in sorted(epochs):
            if ep in metrics_by_epoch and key in metrics_by_epoch[ep]:
                values.append(metrics_by_epoch[ep][key])

        if len(values) >= 3:
            # Check if loss is decreasing
            first_vals = np.mean(values[:max(3, len(values)//5)])
            last_vals = np.mean(values[-max(3, len(values)//5):])
            trend = "decreasing" if last_vals < first_vals * 0.95 else \
                    "increasing" if last_vals > first_vals * 1.05 else "flat"

            if trend == "flat" and len(values) >= 10:
                report["warnings"].append(
                    f"{key} has been flat for {len(values)} checkpoints "
                    f"(from {first_vals:.4f} to {last_vals:.4f})"
                )

    # Check for NaN alerts
    nan_alerts = [a for a in alerts if a.get("alert") == "NaN_detected"]
    if nan_alerts:
        report["warnings"].append(
            f"NaN detected {len(nan_alerts)} time(s) — check learning rate and gradient clipping"
        )
        report["recommendations"].append("Consider reducing LR or increasing gradient clipping")

    # Progress toward total epochs
    config_path = Path(str(checkpoints[0].get("path", ""))).parent.parent / "run_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        total_epochs_config = config.get("epochs", 60)
        progress_pct = latest_epoch / total_epochs_config * 100
        report["progress_pct"] = progress_pct
        report["total_planned_epochs"] = total_epochs_config

        if progress_pct >= 100:
            report["status"] = "completed"
            report["findings"].append("Training has reached the planned number of epochs.")
        elif progress_pct >= 50:
            report["status"] = "in_progress"
            remaining = total_epochs_config - latest_epoch
            report["findings"].append(
                f"Progress: {progress_pct:.0f}% ({latest_epoch}/{total_epochs_config} epochs). "
                f"~{remaining} epochs remaining."
            )
        else:
            report["status"] = "early"
            report["findings"].append(
                f"Early stage: {progress_pct:.0f}% complete. Still warming up."
            )

    # Best model tracking
    best_path = Path(str(checkpoints[0].get("path", ""))).parent.parent / "checkpoint_best.pt"
    if best_path.exists():
        report["best_checkpoint_exists"] = True

    # Check for recent activity
    last_checkpoint_time = checkpoints[-1].get("timestamp", "")
    if last_checkpoint_time:
        try:
            last_time = datetime.fromisoformat(last_checkpoint_time)
            since = datetime.now() - last_time
            if since > timedelta(minutes=30):
                report["warnings"].append(
                    f"No checkpoint in {since.total_seconds()/60:.0f} minutes — training may be stalled."
                )
        except (ValueError, TypeError):
            pass

    # Validation loss trend (best indicator of convergence)
    val_losses = []
    for ep in sorted(epochs):
        if ep in metrics_by_epoch and "val_align" in metrics_by_epoch[ep]:
            val_losses.append((ep, metrics_by_epoch[ep]["val_align"]))

    if len(val_losses) >= 3:
        best_ep, best_val = min(val_losses, key=lambda x: x[1])
        latest_ep, latest_val = val_losses[-1]

        if best_ep < latest_ep - 5:
            report["warnings"].append(
                f"Best validation loss was at epoch {best_ep} ({best_val:.4f}), "
                f"current: {latest_val:.4f} — possible overfitting."
            )
            report["recommendations"].append(
                "Consider loading checkpoint_best.pt or enabling early stopping"
            )

    # Summary
    if not report["warnings"] and report["status"] == "in_progress":
        report["findings"].append("Training is proceeding normally — no issues detected.")

    return report


def print_report(report: dict, output_dir: str):
    """Pretty-print a monitoring report."""
    print(f"\n{'='*60}")
    print(f"SAM Monitor Report — {output_dir}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    if report["status"] == "no_data":
        print("No training data found yet. Waiting for checkpoints...")
        return

    # Status badge
    status_badges = {
        "completed": "[DONE]",
        "in_progress": "[RUNNING]",
        "early": "[STARTING]",
        "no_data": "[WAITING]",
    }
    badge = status_badges.get(report["status"], "[?]")
    epoch_info = f"Epoch {report.get('latest_epoch', '?')}"
    if "progress_pct" in report:
        epoch_info += f" ({report['progress_pct']:.0f}%)"
    print(f"\n  Status: {badge} {epoch_info}")

    # Findings
    if report["findings"]:
        print(f"\n  Findings:")
        for f in report["findings"]:
            print(f"    • {f}")

    # Warnings
    if report["warnings"]:
        print(f"\n  Warnings:")
        for w in report["warnings"]:
            print(f"    ⚠ {w}")

    # Recommendations
    if report["recommendations"]:
        print(f"\n  Recommendations:")
        for r in report["recommendations"]:
            print(f"    → {r}")

    # Best checkpoint
    if report.get("best_checkpoint_exists"):
        print(f"\n  Best checkpoint available: checkpoint_best.pt")

    print(f"\n{'='*60}\n")


def watch(output_dir: str, interval: int = 60):
    """Continuously watch a training run."""
    log_path = Path(output_dir) / "training_log.jsonl"
    last_epoch = -1

    print(f"Watching {output_dir} every {interval}s...")
    print(f"(Press Ctrl+C to stop)\n")

    try:
        while True:
            entries = load_log(log_path)
            if entries:
                checkpoints = [e for e in entries if e.get("type") == "checkpoint"]
                if checkpoints:
                    latest = max(e.get("epoch", 0) for e in checkpoints)
                    if latest != last_epoch:
                        last_epoch = latest
                        report = analyze_metrics(entries)
                        print_report(report, output_dir)

                        # Write report to file for programmatic access
                        report_path = Path(output_dir) / "monitor_report.json"
                        with open(report_path, "w") as f:
                            json.dump(report, f, indent=2, default=str)

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="SAM Training Monitor — AI Training Companion"
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Training output directory to watch")
    parser.add_argument("--interval", type=int, default=60,
                        help="Polling interval in seconds (default: 60)")
    parser.add_argument("--once", action="store_true",
                        help="Generate a single report and exit")
    args = parser.parse_args()

    log_path = Path(args.output_dir) / "training_log.jsonl"

    if args.once:
        entries = load_log(log_path)
        report = analyze_metrics(entries)
        print_report(report, args.output_dir)
    else:
        watch(args.output_dir, args.interval)


if __name__ == "__main__":
    main()
