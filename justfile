# Run tests with uv and pytest for multiple python versions.
test:
  uv run --python 3.10 --all-extras pytest
  uv run --python 3.11 --all-extras pytest
  uv run --python 3.12 --all-extras pytest
  uv run --python 3.13 --all-extras pytest
  uv run --python 3.14 --all-extras pytest

test-lowest:
  uv run --python 3.10 --resolution lowest pytest
  uv run --python 3.11 --resolution lowest pytest
  uv run --python 3.12 --resolution lowest pytest
  uv run --python 3.13 --resolution lowest pytest
  uv run --python 3.14 --resolution lowest pytest

test-lowest-direct:
  uv run --python 3.10 --resolution lowest-direct pytest
  uv run --python 3.11 --resolution lowest-direct pytest
  uv run --python 3.12 --resolution lowest-direct pytest
  uv run --python 3.13 --resolution lowest-direct pytest
  uv run --python 3.14 --resolution lowest-direct pytest
