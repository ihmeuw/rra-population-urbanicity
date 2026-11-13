def __getattr__(name: str) -> object:
    # The runner imports the specification, which imports the method parameters
    # from this package, so the runner is loaded only when RUNNERS is first read.
    if name == "RUNNERS":
        from rra_population_urbanicity.classify.runner import classify_run

        return {"run": classify_run}
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
