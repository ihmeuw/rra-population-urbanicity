from rra_population_urbanicity.validate.classification_diff import validate_diff
from rra_population_urbanicity.validate.ghs_smod_reproduction import validate_ghs_smod

RUNNERS = {"ghs-smod": validate_ghs_smod, "diff": validate_diff}
