from rra_population_urbanicity.classify.kernel import kernel_block_task
from rra_population_urbanicity.classify.runner import classify_run

RUNNERS = {"run": classify_run}
TASK_RUNNERS = {"kernel-block": kernel_block_task}
