"""Command-line interface for IndicationScout."""

import click


@click.group()
@click.version_option(package_name="indication-scout")
def main():
    """IndicationScout: Discover new indications for drugs."""
    pass


@main.command()
@click.option("-d", "--drug", required=True, help="Drug name to search for indications")
@click.option(
    "-n",
    "--top-n",
    default=10,
    show_default=True,
    help="Number of top indications to return",
)
@click.option("-o", "--output", type=click.Path(), help="Output file path (JSON)")
def find(drug: str, top_n: int, output: str | None):
    """Find potential new indications for a given drug."""
    click.echo(f"Searching indications for: {drug}")
    click.echo(f"Top {top_n} results:")

    # Placeholder results - will be replaced with actual ML inference
    results = [
        {"indication": "Chronic Pain", "score": 0.92},
        {"indication": "Migraine", "score": 0.87},
        {"indication": "Arthritis", "score": 0.81},
    ]

    for i, result in enumerate(results[:top_n], 1):
        click.echo(f"  {i}. {result['indication']} (score: {result['score']:.2f})")

    if output:
        import json
        from pathlib import Path

        Path(output).write_text(
            json.dumps({"drug": drug, "indications": results}, indent=2)
        )
        click.echo(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
