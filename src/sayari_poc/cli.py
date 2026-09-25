"""Argument parsing and command coordination for the approved run command."""

import argparse
import json
import logging
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from sayari_poc.config import Settings, load_settings
from sayari_poc.excel import InputError
from sayari_poc.pipeline import OUTPUT_DIR, plan_work, run_pipeline
from sayari_poc.risk_taxonomy import OntologySnapshotError
from sayari_poc.transport import SayariError


def _positive_int(value: str) -> int:
    """Parse a positive entity limit without echoing invalid input."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("limit must be a positive integer") from None
    if number <= 0:
        raise argparse.ArgumentTypeError("limit must be a positive integer")
    return number


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and restore the caller's logging state."""
    logger = logging.getLogger("sayari_poc")
    previous_level = logger.level
    handler = logging.StreamHandler(sys.stderr)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        return _run_command(argv)
    finally:
        # If main() is embedded or called again, don't leave a handler pointing at an old stderr.
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous_level)


def _run_command(argv: Sequence[str] | None) -> int:
    """Translate planning and execution failures into safe diagnostics."""
    parser = argparse.ArgumentParser(prog="python -m sayari_poc")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Screen suppliers and render the static report")
    run.add_argument("--limit", type=_positive_int, help="Global limit on named input entities")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--refresh", action="store_true", help="Refresh cached API responses")
    mode.add_argument(
        "--offline", action="store_true", help="Cache only; fail on a missing response"
    )
    run.add_argument("--dry-run", action="store_true", help="Estimate requests without API access")
    selection = run.add_mutually_exclusive_group()
    selection.add_argument(
        "--sheet",
        action="append",
        dest="sheets",
        help="Sheet name; repeatable. Defaults to the list_3 supplier portfolio",
    )
    selection.add_argument(
        "--all-sheets", action="store_true", help="Every sheet in workbook order"
    )
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
        if args.dry_run:
            plan = plan_work(settings, args.sheets, args.limit, all_sheets=args.all_sheets)
            print(json.dumps(plan, ensure_ascii=True))
            print(
                f"Planned calls: up to {plan['planned_calls']} logical data requests (estimate). "
                "Cache hits, duplicate identities and unresolved rows can reduce calls; "
                "retries and OAuth are excluded. No API calls made."
            )
            requested_mode = "--refresh" if args.refresh else "--offline" if args.offline else None
            if requested_mode:
                print(
                    f"{requested_mode} is not executed during dry-run; "
                    "cache availability is not checked and responses are not refreshed."
                )
            return 0
        findings = run_pipeline(
            settings,
            args.sheets,
            args.limit,
            all_sheets=args.all_sheets,
            offline=args.offline,
            refresh=args.refresh,
        )
    except ValidationError as exc:
        # Show numeric positions and known top-level setting names; redact any other key.
        fields = ", ".join(
            ".".join(
                str(part)
                if isinstance(part, int)
                or (len(error["loc"]) == 1 and part in Settings.model_fields)
                else "<redacted>"
                for part in error["loc"]
            )
            for error in exc.errors(include_url=False, include_context=False, include_input=False)
        )
        print(f"Configuration error: check {fields}", file=sys.stderr)
        return 1
    except (SayariError, InputError, OntologySnapshotError) as exc:
        # These error types are built with safe, user-facing messages, so we can print them.
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1
    except ValueError:
        # Hide the message: an arbitrary ValueError may contain source data or private paths.
        print("Run failed: invalid input or data (ValueError); details withheld.", file=sys.stderr)
        return 1
    except OSError:
        # Give fixed file-access advice rather than print the OS error, which includes paths.
        print("Run failed: check input, output, cache and CA file access.", file=sys.stderr)
        return 1
    except Exception as exc:
        # An unexpected exception's message can contain payload data, so show only its class.
        print(
            f"Run failed unexpectedly ({type(exc).__name__}); no report success claimed.",
            file=sys.stderr,
        )
        return 1
    budget_errors = sum(e.get("error_type") == "BudgetExceeded" for e in findings.exceptions)
    # Call out an exhausted budget, even though the partial artifacts have already been saved.
    if budget_errors:
        print(
            f"Budget exhausted: {budget_errors} error rows report BudgetExceeded; "
            "partial results and run manifest preserved.",
            file=sys.stderr,
        )
    print(f"Suppliers: {len(findings.suppliers)}; exceptions: {len(findings.exceptions)}")
    print(f"Findings: {OUTPUT_DIR / 'findings.json'}")
    print(f"Report: {OUTPUT_DIR / 'report.html'}")
    return (
        1
        if any(
            row["status"] == "error" or row.get("coverage_status") == "error"
            for row in findings.suppliers
        )
        else 0
    )
