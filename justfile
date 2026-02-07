# Run all tests against multiple Python versions.
test:
  uv run --python 3.10 --group test pytest
  uv run --python 3.11 --group test pytest
