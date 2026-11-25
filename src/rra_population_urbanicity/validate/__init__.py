from rra_population_urbanicity.validate.classification_diff import validate_diff
from rra_population_urbanicity.validate.ghs_smod_reproduction import validate_ghs_smod
from rra_population_urbanicity.validate.national_definition_gap import (
    validate_national_gap,
)
from rra_population_urbanicity.validate.settlements import validate_settlements
from rra_population_urbanicity.validate.wup2025_shares import validate_wup2025

RUNNERS = {
    "ghs-smod": validate_ghs_smod,
    "diff": validate_diff,
    "wup2025": validate_wup2025,
    "settlements": validate_settlements,
    "national-gap": validate_national_gap,
}
