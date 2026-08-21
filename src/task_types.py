"""Shared semantic task IDs used by collation, modeling, and metrics."""

TASK_NAMES = ("asr", "tts", "s2s", "s2tt", "ser", "asc")
TASK_TO_ID = {name: task_id for task_id, name in enumerate(TASK_NAMES)}
NUM_TASKS = len(TASK_NAMES)
