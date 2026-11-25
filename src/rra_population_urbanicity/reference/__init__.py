from rra_population_urbanicity.reference.ghsl import download_ghsl
from rra_population_urbanicity.reference.us_census import download_us_census
from rra_population_urbanicity.reference.wup import copy_wup

RUNNERS = {
    "ghsl": download_ghsl,
    "us-census": download_us_census,
    "wup": copy_wup,
}
