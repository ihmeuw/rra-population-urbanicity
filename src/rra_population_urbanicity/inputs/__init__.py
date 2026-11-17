from rra_population_urbanicity.inputs.aggregate import (
    aggregate_block_task,
    aggregate_run,
)

RUNNERS = {"aggregate": aggregate_run}
TASK_RUNNERS = {"aggregate-block": aggregate_block_task}
