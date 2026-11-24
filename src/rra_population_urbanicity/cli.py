import click

from rra_population_urbanicity import classify, inputs, reference, tabulate, validate


@click.group()
def purun() -> None:
    """Run a stage of the population urbanicity pipeline."""


@click.group()
def putask() -> None:
    """Run an individual task in the population urbanicity pipeline."""


# Each stage module registers its workflow commands in RUNNERS and its
# individual tasks in TASK_RUNNERS. A stage may have only one of the two.
for module in [reference, inputs, classify, tabulate, validate]:
    stage_name = module.__name__.split(".")[-1]
    for parent, commands in (
        (purun, getattr(module, "RUNNERS", {})),
        (putask, getattr(module, "TASK_RUNNERS", {})),
    ):
        if commands:
            group = click.Group(name=stage_name)
            for name, command in commands.items():
                group.add_command(command, name)
            parent.add_command(group)
