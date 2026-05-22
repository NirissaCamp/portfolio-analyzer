# Portfolio Analyzer

A Python-based portfolio analysis tool for U.S. stocks.

## Status

Under active development. See [design spec](docs/superpowers/specs/) for the plan.

## Tech Stack

Python · Streamlit · Pandas · Plotly · yfinance · SQLite · pytest

## Run Locally

```bash
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
# source venv/bin/activate    # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

## Project Structure

```
portfolio-analyzer/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── src/
│   ├── data/               # Data fetching + caching
│   ├── analytics/          # Financial metric calculations
│   └── ui/                 # Streamlit UI components
├── tests/                  # pytest unit tests
└── docs/                   # Design docs and notes
```

## Roadmap

- [ ] Phase 1: Portfolio dashboard (this MVP)
- [ ] Phase 2: ML-based return prediction
- [ ] Phase 3: AI Q&A with RAG

## License

MIT
