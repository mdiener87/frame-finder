# Contributing to Frame Finder

Thanks for your interest in contributing! We welcome issues, feature requests, and pull requests.

## Quick Start

- Fork the repo and create a feature branch.
- Use a virtual environment with a recent Python (3.9+ recommended).
- Install deps: `pip install -r requirements.txt`
- Run the app locally: `python app.py` and open http://localhost:5000
- Sanity check the environment: `python test_setup.py`

## Development Setup

1. Clone and create a venv
   ```bash
   git clone https://github.com/<your-username>/frame-finder.git
   cd frame-finder
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run locally
   ```bash
   python app.py
   ```

## Project Conventions

- Keep the style consistent with existing code.
- Prefer clear names and small, focused functions.
- Add docstrings where behavior isn’t obvious.
- Use type hints where they add clarity (optional).
- Avoid adding heavy dependencies unless necessary.

## Pull Requests

- Open a draft PR early for feedback if helpful.
- Include a concise description of the change and rationale.
- Add screenshots/GIFs for UI changes where relevant.
- Update README or inline help if behavior/usage changes.
- Ensure the app runs locally and `python test_setup.py` succeeds.

## Issues and Feature Requests

- Search existing issues before filing a new one.
- For bugs, include repro steps, expected vs actual behavior, and logs/console output.
- For features, describe the use case and desired behavior.

## Code of Conduct

By participating in this project you agree to abide by our Code of Conduct (see `CODE_OF_CONDUCT.md`).

## License

By contributing, you agree that your contributions will be licensed under the MIT License (see `LICENSE`).

