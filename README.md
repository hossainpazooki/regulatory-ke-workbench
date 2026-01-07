# Regulatory Knowledge Engineering Workbench

A computational law platform for MiCA, RWA tokenization, and stablecoin frameworks. Transforms regulatory documents into executable rules with traceable decision logic.

**Live Demo:** [pazooki.streamlit.app](https://pazooki.streamlit.app)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/hossainpazooki/RWAs.git
cd RWAs
python -m venv .venv
.\.venv\Scripts\Activate  # Windows
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Launch app
streamlit run frontend/Home.py
```

Opens at `http://localhost:8501` with:
- **Home** — Overview and instructions
- **KE Workbench** — Rule inspection, decision trees, trace testing
- **Production Demo** — Compiled IR, premise index, benchmarks
- **Navigator** — Cross-border compliance analysis

## API Server (Optional)

```bash
uvicorn backend.main:app --reload
```

Endpoints at `http://localhost:8000`:
- `POST /decide` — Evaluate scenarios
- `POST /navigate` — Cross-border compliance
- `GET /rules` — List rules
- `POST /v2/evaluate` — Production evaluation with O(1) lookup

## Project Structure

```
backend/
├── ontology/       # Domain types (Actor, Instrument, Provision)
├── rules/          # YAML rules + decision engine
├── jurisdiction/   # Multi-jurisdiction support (EU, UK, US)
├── compiler/       # YAML → IR compilation
├── verify/         # Semantic consistency engine
└── api/            # FastAPI routes

frontend/
├── Home.py         # Landing page
└── pages/          # KE Workbench, Production Demo, Navigator

data/legal/         # Legal corpus (MiCA, FCA, GENIUS Act)
docs/               # Design documentation
```

## Regulatory Frameworks

| Framework | Jurisdiction | Status |
|-----------|--------------|--------|
| MiCA | EU | Modeled (9 rules) |
| FCA Crypto | UK | Modeled (5 rules) |
| GENIUS Act | US | Illustrative (6 rules) |
| RWA Tokenization | EU | Illustrative (2 rules) |

## Documentation

- [Knowledge Model](docs/knowledge_model.md) — Ontology design
- [Rule DSL](docs/rule_dsl.md) — YAML rule specification
- [Engine Design](docs/engine_design.md) — Architecture details

## Disclaimer

Research/demo project, not legal advice. Rules are interpretive models—consult qualified legal counsel for compliance decisions.

## License

MIT License. See [LICENSE](LICENSE).

---

Built with [Claude Code](https://claude.ai/code)
