# Contributing to ROPE

Thanks for your interest in contributing to ROPE! Here's how you can help.

## Ways to Contribute

### Add New Attacks
1. Edit `data/attacks.json` to add your attack variant
2. Follow the schema: `{"task_id": int, "type": string, "text": string, "goal": string}`
3. Valid attack types: `hijack`, `extract`, `obfuscate`, `poison`
4. Run tests: `pytest tests/test_all.py::TestDataValidation -v`
5. Submit a PR

### Add New Defenses
1. Add your defense function to `rope/defenses.py`
2. Register it in the `DEFENSES` dict
3. Write tests in `tests/test_all.py`
4. Submit a PR

### Add New Models
1. Add your model to the `MODELS` dict in `rope/models.py`
2. Ensure it works with 4-bit quantization
3. Document VRAM requirements
4. Submit a PR

### Report Bugs
- Use GitHub Issues
- Include: Python version, GPU info, error traceback
- Minimal reproduction steps

## Development Setup

```bash
git clone https://github.com/yourusername/rope-bench.git
cd rope-bench
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

## Code Standards

- **Formatting**: Black (line length 100)
- **Linting**: Ruff
- **Type hints**: All public functions
- **Docstrings**: Google style
- **Tests**: pytest, target 80%+ coverage

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-attack-type`)
3. Write tests for your changes
4. Ensure all tests pass: `pytest tests/ -v`
5. Format code: `black rope/ tests/`
6. Submit PR with clear description

## Code of Conduct

Be respectful and constructive. We're all here to make AI safer.
