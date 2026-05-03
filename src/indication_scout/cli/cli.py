"""IndicationScout CLI entry point.

Exposes the `scout` command declared in `pyproject.toml` under `[project.scripts]`.

Usage:
    scout find -d <drug> [--out-dir DIR] [--no-write]
"""

import asyncio
import logging
import os
from datetime import date, datetime
from pathlib import Path

import click
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "snapshots"


def _load_env() -> None:
    """Load .env files before importing modules that read settings at import time."""
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / ".env.constants")


async def _run_for_drug(
    drug: str, out_dir: Path, write: bool, date_before: date | None = None
) -> None:
    # Imports are deferred until after _load_env() runs in main(), because
    # base_client.py calls get_settings() at import time.
    from langchain_anthropic import ChatAnthropic

    from indication_scout.agents.supervisor.supervisor_agent import (
        build_supervisor_agent,
        run_supervisor_agent,
    )
    from indication_scout.constants import DEFAULT_CACHE_DIR
    from indication_scout.db.session import get_db
    from indication_scout.helpers.drug_helpers import normalize_drug_name
    from indication_scout.report.format_report import format_report
    from indication_scout.services.retrieval import RetrievalService

    # Normalize at the entry point so every downstream consumer (cache keys,
    # tools, sub-agents, snapshot filename, logs) sees a consistent lowercased
    # form. Anything that needs the original casing must be captured before
    # this point.
    drug = normalize_drug_name(drug)

    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=0,
        max_tokens=4096,
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    db = next(get_db())
    svc = RetrievalService(DEFAULT_CACHE_DIR)

    logger.info("Starting %s (date_before=%s)", drug, date_before)
    agent, get_merged_allowlist, get_auto_findings = build_supervisor_agent(
        llm=llm, svc=svc, db=db, date_before=date_before
    )
    output = await run_supervisor_agent(
        agent, get_merged_allowlist, drug, get_auto_findings=get_auto_findings,
    )

    if not write:
        click.echo(format_report(output))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cutoff_tag = f"_holdout_{date_before.isoformat()}" if date_before else ""
    md_path = out_dir / f"{drug}{cutoff_tag}_{timestamp}.md"
    json_path = out_dir / f"{drug}{cutoff_tag}_{timestamp}.json"
    md_path.write_text(format_report(output), encoding="utf-8")
    json_path.write_text(output.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Finished %s -> %s, %s", drug, md_path, json_path)
    click.echo(f"Report:    {md_path}")
    click.echo(f"Structured: {json_path}")


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """IndicationScout — agentic drug repurposing."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


@cli.command()
@click.option(
    "-d",
    "--drug",
    required=True,
    help="Drug name to run the supervisor pipeline on.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_OUT_DIR,
    show_default=True,
    help="Directory to write the report into.",
)
@click.option(
    "--no-write",
    is_flag=True,
    help="Print the markdown report to stdout instead of writing to disk.",
)
@click.option(
    "--date-before",
    "date_before",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help=(
        "Temporal cutoff (YYYY-MM-DD). PubMed and ClinicalTrials.gov queries are restricted to "
        "records dated strictly before this date. Mechanism (OpenTargets) data has no date "
        "filter and is always current."
    ),
)
def find(
    drug: str, out_dir: Path, no_write: bool, date_before: datetime | None
) -> None:
    """Run the supervisor pipeline on DRUG and produce a repurposing report."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise click.ClickException(
            "ANTHROPIC_API_KEY is not set. Add it to your .env or environment."
        )
    cutoff = date_before.date() if date_before is not None else None
    asyncio.run(_run_for_drug(drug, out_dir, write=not no_write, date_before=cutoff))


def main() -> None:
    """Console-script entry point referenced by `pyproject.toml`."""
    _load_env()
    cli()


if __name__ == "__main__":
    main()
