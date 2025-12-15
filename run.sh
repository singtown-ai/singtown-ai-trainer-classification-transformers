set -e
(
  cd singtown-ai-trainer-classification-transformers
  uv run main.py
)
(
  cd rknn2
  uv run main.py
)