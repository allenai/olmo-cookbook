.PHONY: lint format typecheck check install

# Install development dependencies
install:
	uv sync --extra dev

# Run linting (check only)
lint:
	uv run ruff check . --exclude scripts/

# Auto-fix linting issues where possible
lint-fix:
	uv run ruff check . --fix --exclude scripts/

# Check code formatting
format-check:
	uv run ruff format --check . --exclude scripts/

# Auto-format code
format:
	uv run ruff format . --exclude scripts/

# Run type checking
typecheck:
	uv run mypy src/ tests/

# Run all checks (lint + format + typecheck)
check: lint format-check typecheck

# Fix formatting and auto-fixable linting issues
fix: format lint-fix

# Help
help:
	@echo "Available targets:"
	@echo "  install      - Install development dependencies"
	@echo "  lint         - Run linting (check only)"
	@echo "  lint-fix      - Auto-fix linting issues where possible"
	@echo "  format-check - Check code formatting"
	@echo "  format       - Auto-format code"
	@echo "  typecheck    - Run type checking"
	@echo "  check        - Run all checks (lint + format + typecheck)"
	@echo "  fix           - Fix formatting and auto-fixable linting issues"
