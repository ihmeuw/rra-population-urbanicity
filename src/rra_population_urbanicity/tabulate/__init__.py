from rra_population_urbanicity.tabulate.admin_rasters import admin_rasters_run
from rra_population_urbanicity.tabulate.classes import classes_run
from rra_population_urbanicity.tabulate.population import (
    population_block_task,
    population_run,
)

RUNNERS = {
    "admin-rasters": admin_rasters_run,
    "population": population_run,
    "classes": classes_run,
}
TASK_RUNNERS = {"population-block": population_block_task}
