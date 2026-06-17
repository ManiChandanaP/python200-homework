# warmup_11.py
# Week 11 Warmup: Prefect Orchestration + Production Patterns
from prefect import task
from prefect.logging import get_run_logger
# =============================================================================
# PREFECT QUESTION 1
# =============================================================================
# A @flow is the top-level orchestration unit in Prefect. It coordinates the
# execution of tasks, tracks the overall run state, and is what you call to
# kick off your pipeline. Prefect treats it as the "job."
#
# A @task is a discrete unit of work within a flow. Tasks get individual state
# tracking (Pending, Running, Completed, Failed), retry logic, and logging.
# They run inside a flow.
#
# For a pure in-memory helper like Celsius-to-Fahrenheit conversion:
# NO -- I would NOT decorate it with @task. Tasks add overhead: Prefect has to
# track state, serialize arguments, and log the call. A pure calculation with
# no I/O, no retries needed, and no external dependency doesn't benefit from
# that overhead. It should just be a plain Python function called from within
# a task that does need tracking.

# =============================================================================
# PREFECT QUESTION 2
# =============================================================================
@task(name="call_api", retries=3, retry_delay_seconds=30)

# =============================================================================
# PREFECT QUESTION 3
# =============================================================================
# In the Prefect UI at http://localhost:4200, I would:
# 1. Click on the failed flow run to open its detail page.
# 2. Click on the "transform" task in the task run list -- it will show as
#    Failed with a red indicator.
# 3. Open the "Logs" tab for that task run. I would expect to find:
#    - The full Python traceback (e.g., KeyError, API timeout, JSON parse error)
#    - Any log messages I emitted before the crash
#    - The exact timestamp of failure
# 4. The "load" task will show as "Not Run" because Prefect skips downstream
#    tasks when an upstream task fails -- it never even gets scheduled.

# =============================================================================
# PRODUCTION QUESTION 1
# =============================================================================
# raise_for_status() is a requests method that raises an HTTPError exception
# automatically if the response status code is 4xx or 5xx.
#
# Why it's better than a manual check:
# - It raises an actual exception, which Prefect catches and marks the task
#   as Failed. The exception propagates and downstream tasks do not run.
# - The manual `print("error")` approach does NOT raise an exception, so the
#   task continues executing with bad/empty data, likely causing a confusing
#   crash later -- or silently loading garbage into storage.
#
# When the API returns a 500:
# - With raise_for_status(): task immediately raises HTTPError -> task is
#   marked Failed -> retries kick in if configured -> downstream tasks are
#   blocked. Clean failure, correct behavior.
# - With print("error"): task continues, returns None or broken data, next
#   task crashes with an AttributeError or KeyError, and the actual root
#   cause (the 500) is buried or invisible in logs.

# =============================================================================
# PRODUCTION QUESTION 2
# =============================================================================
# Scenario: pipeline crashes halfway through transform, blob may or may not
# have been partially written. You fix the bug and re-run from the beginning.
#
# With overwrite=True: on re-run, the load task simply writes the new complete
# file over whatever is at that path. No error, no stale partial data left
# behind. You always end up with exactly one clean result file.
#
# Without overwrite=True (the default raises an error if the blob exists):
# the load task would crash with a ResourceExistsError on re-run because the
# blob path already exists from the previous attempt. You would have to
# manually delete the blob in Azure Portal before every re-run -- painful and
# error-prone in a production pipeline.

# =============================================================================
# PRODUCTION QUESTION 3
# =============================================================================


@task
def load_to_blob(records: list, blob_path: str):
    logger = get_run_logger()
    logger.info(f"Loading {len(records)} records to {blob_path}")