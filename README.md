# IndicationScout

A machine learning toolkit for drug repurposing and indication discovery.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

### CLI

```bash
# Find indications for a drug
scout find -d "Tylenol"

# Limit results
scout find -d "Metformin" -n 5

# Save results to JSON
scout find -d "Aspirin" -o results.json
```

### API

```bash
uvicorn indication_scout.api.main:app --reload
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/

# Lint code
ruff check src/
```

## Project Structure

```
indication-scout/
├── src/indication_scout/
│   ├── agents/          # AI agents for analysis
│   ├── api/             # FastAPI application
│   ├── data_sources/    # External API clients
│   ├── db/              # Database layer
│   ├── models/          # Data models
│   └── services/        # Business logic
├── tests/               # Test suite
├── notebooks/           # Jupyter notebooks
└── scripts/             # Utility scripts
```

## License

MIT
