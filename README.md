# SingTown AI Trainer Classification Keras

## Support Models

- timm/mobilenetv3_large_100.ra_in1k

## Test

```
# test
unset SINGTOWN_AI_HOST
unset SINGTOWN_AI_TOKEN
unset SINGTOWN_AI_TASK_ID
export SINGTOWN_AI_MOCK_TASK_PATH="../mock-task.json"
export SINGTOWN_AI_MOCK_DATASET_PATH="../classification-20.json"
uv run main.py
```

```
# test
export SINGTOWN_AI_HOST="https://ai.singtown.com"
export SINGTOWN_AI_TOKEN="your token"
export SINGTOWN_AI_TASK_ID="your id"
unset SINGTOWN_AI_MOCK_TASK_PATH
unset SINGTOWN_AI_MOCK_DATASET_PATH
uv run main.py
```

