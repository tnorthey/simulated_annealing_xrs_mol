# Tests

This directory contains unit tests for the simulated annealing XRS molecular codebase.

## Running Tests

To run all tests:
```bash
pytest tests/
```

To run a specific test file:
```bash
pytest tests/test_mol.py
```

To run with verbose output:
```bash
pytest tests/ -v
```

To run with coverage:
```bash
pytest tests/ --cov=modules --cov-report=html
```

To run performance tests (slow):
```bash
pytest tests/test_performance.py -v -s
```

To skip slow tests:
```bash
pytest tests/ -m "not slow"
```

## Test Structure

- `conftest.py`: Pytest configuration and fixtures
- `test_mol.py`: Tests for `modules/mol.py` (XYZ manipulation, periodic table, distances, RMSD, etc.)
- `test_x.py`: Tests for `modules/x.py` (X-ray scattering calculations)
- `test_read_input.py`: Tests for `modules/read_input.py` (TOML input parsing)
- `test_wrap.py`: Tests for `modules/wrap.py` (Wrapper functions)
- `test_performance.py`: Performance tests for simulated annealing (marked with `@pytest.mark.slow`)

## Requirements

Tests require:
- pytest
- numpy
- scipy

Install with:
```bash
pip install pytest numpy scipy
```

## Note

Some tests may be skipped if required data files are not present (e.g., Compton scattering data, XYZ files). These are marked with `@pytest.mark.skipif` decorators.
