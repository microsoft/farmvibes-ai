from copy import copy
from dataclasses import asdict, dataclass
from enum import Enum, IntEnum, auto
from typing import Dict, List, Optional, Tuple, Union

import geopandas
from pyproj import Geod
from shapely import geometry as shpg

from vibe_core.data import GHGFlux, GHGProtocolVibe, gen_hash_id


@dataclass(frozen=True, unsafe_hash=True)
class Fertilizer:
    source: str
    details: str
    co2: float
    n2o: float
    nitrogen_ratio: float
    unit: str


@dataclass
class GHG:
    name: str
    details: str
    factor: float


@dataclass
class EmissionFactor:
    value: float
    unit: str
    details: Optional[str] = None


class FertilizerType(Enum):
    SYNTHETIC = auto()
    UREA = auto()
    LIMESTONE_CALCITE = auto()
    LIMESTONE_DOLOMITE = auto()
    GYPSUM = auto()
    MANURE = auto()
    MANURE_BIRDS = auto()
    ORGANIC_COMPOUND = auto()
    GENERIC_ORGANIC_COMPOUND = auto()
    FILTER_CAKE = auto()
    VINASSE = auto()


class CropType(Enum):
    SOYBEAN = auto()
    CORN = auto()
    BEANS = auto()
    RICE = auto()
    WHEAT = auto()
    SUGARCANE = auto()
    SUGARCANE_WITH_BURNING = auto()
    COTTON = auto()
    GREEN_MANURE_LEGUMES = auto()
    GREEN_MANURE_GRASSES = auto()
    GREEN_MANURE = auto()


class Biome(Enum):
    US_FOREST = 1
    BRAZIL_AMAZON_FOREST = 2
    BRAZIL_AMAZON_SAVANNA = 3
    BRAZIL_CERRADO = 4
    BRAZIL_PANTANAL = 5
    BRAZIL_CAATINGA = 6
    BRAZIL_MATA_ATLANTICA = 7
    BRAZIL_PAMPA = 8


class CurrentLandUse(Enum):
    CONVENTIONAL_CROPS = auto()
    DIRECT_SEEDING = auto()
    SUGARCANE_WITH_BURNING = auto()
    SUGARCANE_WITHOUT_BURNING = auto()


class PreviousLandUse(Enum):
    CONVENTIONAL_CROPS = auto()
    DIRECT_SEEDING = auto()
    NATIVE = auto()
    SUGARCANE_WITH_BURNING = auto()
    SUGARCANE_WITHOUT_BURNING = auto()


BIOME_TO_CARBON_STOCK = {
    Biome.US_FOREST: 88.39,  # Source: EPA
    Biome.BRAZIL_AMAZON_FOREST: 573.16,
    Biome.BRAZIL_AMAZON_SAVANNA: 86.38,
    Biome.BRAZIL_CERRADO: 115.92,
    Biome.BRAZIL_PANTANAL: 150.52,
    Biome.BRAZIL_CAATINGA: 159.57,
    Biome.BRAZIL_MATA_ATLANTICA: 468.5,  # average value
    Biome.BRAZIL_PAMPA: 92.10,
}


GLOBAL_HEATING_POTENTIAL_GHG = {
    "CO2": GHG("CO2", "Carbon dioxide", 1.0),
    "N2O": GHG("N2O", "Nitrous oxide", 298.0),
    "CH4": GHG("CH4", "Methane", 25.0),
}

GHG_CONVERSION = {
    "C_CO2": 3.66,
    "CO_CO2": 1.57,
    "N-N2O_N2O": 1.57,
    "NOX_N2O": 0.96,
}

WORLD_GASOLINE_MIXTURE = 1 - 0.1
GASOLINE_MIXTURES = {  # % gasoline
    "Argentina": 1 - 0.05,
    "Australia": 1 - 0.1,
    "Brazil": 1 - 0.27,
    "Canada": 1 - 0.05,
    "China": 1 - 0.1,
    "Colombia": 1 - 0.1,
    "Costa Rica": 1 - 0.07,
    "India": 1 - 0.2,
    "Jamica": 1 - 0.1,
    "Malawi": 1 - 0.1,
    "Mexico": 1 - 0.6,
    "New Zealand": 1 - 0.1,
    "Pakistan": 1 - 0.1,
    "Paraguay": 1 - 0.24,
    "Peru": 1 - 0.08,
    "Philippines": 1 - 0.1,
    "Thailand": 1 - 0.2,
    "Vietnam": 1 - 0.05,
    "Austria": 1 - 0.1,
    "Denmark": 1 - 0.05,
    "Finland": 1 - 0.1,
    "France": 1 - 0.1,
    "Germany": 1 - 0.1,
    "Ireland": 1 - 0.04,
    "Netherlands": 1 - 0.15,
    "Romania": 1 - 0.04,
    "Sweden": 1 - 0.1,
    "United States of America": 1 - 0.1,
    "World": WORLD_GASOLINE_MIXTURE,
}

# Emission factors {{{

FERTILIZER_SYNTHETIC = Fertilizer(
    "Synthetic", "Except urea", 0.0, 0.01130, 0.0, "kg N2O/kg applied nitrogen"
)
FERTILIZER_UREA = Fertilizer("Urea", "", 0.73300, 0.00880, 45.0 / 100, "kg N2O/kg applied nitrogen")
FERTILIZER_LIMESTONE_CALCITE = Fertilizer(
    "Limestone", "Calcite", 0.44000, 0, 0, "kg CO2/kg limestone"
)
FERTILIZER_LIMESTONE_DOLOMITE = Fertilizer(
    "Limestone", "Dolomite", 0.47667, 0, 0, "kg CO2/kg limestone"
)
FERTILIZER_GYPSUM = Fertilizer("Agricultural Gypsum", "", 0.40000, 0, 0, "kg CO2/kg gypsum")
FERTILIZER_MANURE = Fertilizer(
    "Manure", "Bovine, horse, pig, sheep", 0, 0.00020, 1.6 / 100, "kg N2O/kg manure"
)
FERTILIZER_MANURE_BIRDS = Fertilizer("Manure", "Birds", 0, 0.00038, 3.0 / 100, "kg N2O/kg manure")
FERTILIZER_ORGANIC_COMPOUND = Fertilizer(
    "Organic compound", "", 0, 0.000176, 1.4 / 100, "kg N2O/kg manure"
)
FERTILIZER_GENERIC_ORGANIC = Fertilizer(
    "Generic organic fertilizer",
    "",
    0,
    0.000226285714285714,
    1.8 / 100,
    "kg N2O/kg manure",
)
FERTILIZER_FILTER_CAKE = Fertilizer("Filter cake", "", 0, 2.35723, 1.4 / 100, "kg N2O/hectare-year")
FERTILIZER_VINASSE = Fertilizer("Vinasse", "", 0, 0.00001, 0.0313 / 100, "kg N2O/filter")

C_N2O_FLOW_RATE = 0.0075  # kg N2O/kg N applied
C_FRAC_GAS_F = 0.1  # Fraction of N2O emitted as gas
C_FRAC_LEACH = 0.3  # Fraction of N leached
C_N2O_VOLATILIZATION = 0.02  # kg N2O/kg N applied
N2O_RESIDUE = 0.20  # Ratio
N2O_ATMOSPHERIC_VOLATIZATION_RATE = 0.01  # kg N2O-N/kg N
N2O_SOIL_LOSS = 0.0188571428571429  # N2O tonnes / ha / year
CO2EQ_SOIL_EMISSIONS = 73.3333333333  # CO2eq tonnes / ha -- tropical / subtropical

FOREST_TO_CROPLAND_CARBON_STOCK = 88.39  # tonnes CO2 / ha -- reference: EPA
# https://www.epa.gov/energy/greenhouse-gases-equivalencies-calculator-calculations-and-references

HIGH_CLAY_CONTENT_EMISSION_FACTOR = 0.16  # tonnes CO2 / ha year
LOW_CLAY_CONTENT_EMISSION_FACTOR = 0.92  # tonnes CO2 / ha year
CLAY_CONTENT_THRESHOLD = 0.6
FOREST_STR = "forest"

RESIDUES = {
    CropType.SOYBEAN: EmissionFactor(0.000243624857142857, "kg N2O/kg product"),
    CropType.CORN: EmissionFactor(0.000162963428571429, "kg N2O/kg product"),
    CropType.BEANS: EmissionFactor(0.000346297285714286, "kg N2O/kg product"),
    CropType.RICE: EmissionFactor(0.00011484, "kg N2O/kg product"),
    CropType.WHEAT: EmissionFactor(0.000177728571428571, "kg N2O/kg product"),
    CropType.SUGARCANE: EmissionFactor(0.0000170657142857143, "kg N2O/kg product"),
    CropType.SUGARCANE_WITH_BURNING: EmissionFactor(0.00000341314285714286, "kg N2O/kg product"),
    CropType.COTTON: EmissionFactor(0.000361428571428571, "kg N2O/kg product"),
    CropType.GREEN_MANURE_LEGUMES: EmissionFactor(0.000382380952380952, "kg N2O/kg product"),
    CropType.GREEN_MANURE_GRASSES: EmissionFactor(0.000158015873015873, "kg N2O/kg product"),
    CropType.GREEN_MANURE: EmissionFactor(0.000247761904761905, "kg N2O/kg product"),
}

ENERGY_FACTORS_BY_COUNTRY = {  # {{{
    "Albania": 0.003095364,
    "Algeria": 0.159542831,
    "Angola": 0.065773567,
    "Argentina": 0.098421175,
    "Armenia": 0.029916277,
    "Australia": 0.236261887,
    "Austria": 0.045215264,
    "Azerbaijan": 0.12282867,
    "Bahrain": 0.184169045,
    "Bangladesh": 0.162124582,
    "Belarus": 0.083745465,
    "Belgium": 0.060356361,
    "Benin": 0.200827188,
    "Bolivia": 0.108945513,
    "Bosnia & Herzegovina": 0.214942416,
    "Brazil": 0.017763401,
    "Brunei Darussalam": 0.209197436,
    "Bulgaria": 0.128374791,
    "Cameroon": 0.067222554,
    "Canada": 0.046323264,
    "Chile": 0.10327114,
    "China": 0.205740504,
    "Chinese Taipei": 0.175834586,
    "Colombia": 0.048550178,
    "Congo Dem. Rep.": 0.000812663,
    "Costa Rica": 0.011027675,
    "Cote d'Ivoire": 0.118112301,
    "Croatia": 0.078498393,
    "Cuba": 0.208176358,
    "Cyprus": 0.206163455,
    "Czech Republic": 0.142448081,
    "Denmark": 0.083861473,
    "Dominican Republic": 0.163578361,
    "Ecuador": 0.080231471,
    "Egypt": 0.128953112,
    "El Salvador": 0.088302753,
    "Eritrea": 0.186029433,
    "Estonia": 0.194967281,
    "Ethiopia": 0.032771621,
    "Finland": 0.056897185,
    "France": 0.024888727,
    "Gabon": 0.089232088,
    "Georgia": 0.035633834,
    "Germany": 0.119247392,
    "Ghana": 0.051669336,
    "Greece": 0.200104523,
    "Guatemala": 0.096702916,
    "Haiti": 0.15155357,
    "Honduras": 0.095377471,
    "Hong Kong (China)": 0.211343355,
    "Hungary": 0.08367062,
    "Iceland": 0.000117448,
    "India": 0.263539434,
    "Indonesia": 0.206556241,
    "Iran Islamic Rep.": 0.174499668,
    "Iraq": 0.18949185,
    "Ireland": 0.128871203,
    "Israel": 0.192482591,
    "Italy": 0.107035847,
    "Jamaica": 0.150815254,
    "Japan": 0.114874393,
    "Jordan": 0.160811796,
    "Kazakhstan": 0.132976177,
    "Kenya": 0.109399904,
    "Korea, Dem Rep. of": 0.138185411,
    "Korea, Rep. of": 0.137862623,
    "Kuwait": 0.24088064,
    "Kyrgyzstan": 0.02242761,
    "Latvia": 0.042401553,
    "Lebanon": 0.198518255,
    "Libya": 0.241484223,
    "Lithuania": 0.030789353,
    "Luxembourg": 0.106447222,
    "Malaysia": 0.179675413,
    "Malta": 0.235565038,
    "Mexico": 0.126030291,
    "Moldova": 0.110827257,
    "Morocco": 0.176849154,
    "Mozambique": 0.000139414,
    "Myanmar": 0.054249536,
    "Namibia": 0.06562166,
    "Nepal": 0.001203039,
    "Netherlands": 0.103732345,
    "Netherlands Antilles": 0.195810829,
    "New Zealand": 0.046121331,
    "Nicaragua": 0.140110007,
    "Nigeria": 0.115158373,
    "Norway": 0.004788776,
    "Oman": 0.233369148,
    "Pakistan": 0.126789742,
    "Panama": 0.08370217,
    "Paraguay": 0,
    "Peru": 0.065456928,
    "Philippines": 0.132450182,
    "Poland": 0.177334846,
    "Portugal": 0.102001926,
    "Qatar": 0.136840465,
    "Romania": 0.114776003,
    "Russia": 0.087920908,
    "Saudi Arabia": 0.209753403,
    "Senegal": 0.170202733,
    "Serbia & Montenegro": 0.188450468,
    "Singapore": 0.143723334,
    "Slovak Republic": 0.061415332,
    "Slovenia": 0.087538925,
    "South Africa": 0.256475657,
    "Spain": 0.082763168,
    "Sri Lanka": 0.127450138,
    "Sudan": 0.098500037,
    "Sweden": 0.011948395,
    "Switzerland": 0.011061718,
    "Syria": 0.177526918,
    "Tajikistan": 0.008095713,
    "Tanzania United Rep.": 0.077898522,
    "Thailand": 0.142206509,
    "The former Yugoslav Republic of Macedonia": 0.196643297,
    "Togo": 0.055835278,
    "Trinidad & Tobago": 0.19910965,
    "Tunisia": 0.149048022,
    "Turkey": 0.132940333,
    "Turkmenistan": 0.218678315,
    "Ukraine": 0.103585535,
    "United Arab Emirates": 0.174855004,
    "United Kingdom": 0.124511777,
    "United States": 0.14076309,
    "Uruguay": 0.070130528,
    "Uzbekistan": 0.127828824,
    "Venezuela": 0.055011591,
    "Vietnam": 0.106396642,
    "Yemen": 0.174635675,
    "Zambia": 0.000899308,
    "Zimbabwe": 0.171449399,
    "Africa": 0.178111,
    "Asia": 0.206365,
    "Central and Eastern Europe": 0.093903,
    "China (including Hong Kong)": 0.205811,
    "Former USSR": 0.096388889,
    "Latin America": 0.048475,
    "Middle East": 0.19113,
    "Rest of Europe": 0.107222222,
}  # }}}


# }}} Emission factors


class Scope(IntEnum):
    SCOPE_1 = 1
    SCOPE_2 = 2
    SCOPE_3 = 3


@dataclass
class Emissions:
    scope: Scope
    source: str
    co2: float = 0.0
    n2o: float = 0.0
    ch4: float = 0.0

    CO2_CO2EQ = GHG("CO2", "Carbon dioxide", 1.0)
    N2O_CO2EQ = GHG("N2O", "Nitrous oxide", 298.0)
    CH4_CO2EQ = GHG("CH4", "Methane", 25.0)

    @property
    def total(self):
        # co2 equivalent
        return self.co2 + self.n2o * self.N2O_CO2EQ.factor + self.ch4 * self.CH4_CO2EQ.factor

    def __add__(self, other: "Emissions") -> "Emissions":
        return Emissions(
            scope=self.scope,
            source=self.source + " / " + other.source,
            co2=self.co2 + other.co2,
            n2o=self.n2o + other.n2o,
            ch4=self.ch4 + other.ch4,
        )

    def __rmul__(self, scalar: float) -> "Emissions":
        return Emissions(
            scope=self.scope,
            source=self.source,
            co2=self.co2 * scalar,
            n2o=self.n2o * scalar,
            ch4=self.ch4 * scalar,
        )


class FuelType(Enum):
    DIESEL = 1
    DIESEL_B2 = 2
    DIESEL_B5 = 3
    DIESEL_B6 = 4
    DIESEL_B7 = 5
    DIESEL_B8 = 6
    DIESEL_B9 = 7
    DIESEL_B10 = 8
    GASOLINE = 9
    BIODIESEL = 10
    ETHANOL_ANHYDROUS = 11
    ETHANOL_HYDRATED = 12


FUEL_COMPOSITION = {  # 1 - Diesel = Biodiesel
    FuelType.DIESEL: 1.0,
    FuelType.DIESEL_B2: 0.98,
    FuelType.DIESEL_B5: 0.95,
    FuelType.DIESEL_B6: 0.94,
    FuelType.DIESEL_B7: 0.93,
    FuelType.DIESEL_B8: 0.92,
    FuelType.DIESEL_B9: 0.91,
    FuelType.DIESEL_B10: 0.9,
}

AVERAGE_FUEL_CONSUMPTION = 20  # liters per hour
FUEL_EMISSION_FACTORS: Dict[FuelType, Emissions] = {
    k: v * Emissions(Scope.SCOPE_1, k.name, co2=0.002681, n2o=0.00000002, ch4=0.00000030)
    for k, v in FUEL_COMPOSITION.items()
    if k != FuelType.GASOLINE
}
FUEL_EMISSION_FACTORS[FuelType.GASOLINE] = Emissions(
    Scope.SCOPE_1, "Gasoline", co2=0.002212, n2o=0.0, ch4=0.0
)
FUEL_EMISSION_FACTORS[FuelType.ETHANOL_ANHYDROUS] = Emissions(
    Scope.SCOPE_1, "Ethanol anhydrous", co2=0.001526, n2o=0.0, ch4=0.0
)
FUEL_EMISSION_FACTORS[FuelType.ETHANOL_HYDRATED] = Emissions(
    Scope.SCOPE_1, "Ethanol hydrated", co2=0.001457, n2o=0.0, ch4=0.0
)
FUEL_EMISSION_FACTORS[FuelType.BIODIESEL] = Emissions(
    Scope.SCOPE_1, "Biodiesel", co2=0.002499, n2o=0.0, ch4=0.0
)

BURNING_EMISSION_FACTORS = {
    CropType.BEANS: Emissions(
        Scope.SCOPE_1,
        "Biomass Burning (Beans)",
        co2=GHG_CONVERSION["CO_CO2"] * 0.0734272,
        n2o=0.000288464 + GHG_CONVERSION["NOX_N2O"] * 0.0104259131428571,
        ch4=0.00349653333333333,
    ),
    CropType.CORN: Emissions(
        Scope.SCOPE_1,
        "Biomass Burning (Corn)",
        co2=GHG_CONVERSION["CO_CO2"] * 0.078583792,
        n2o=0.000123488816 + GHG_CONVERSION["NOX_N2O"] * 0.00446323863542857,
        ch4=0.00374208533333333,
    ),
    CropType.COTTON: Emissions(
        Scope.SCOPE_1,
        "Biomass Burning (Cotton)",
        co2=GHG_CONVERSION["CO_CO2"] * 0.10773,
        n2o=0.000355509 + GHG_CONVERSION["NOX_N2O"] * 0.012849111,
        ch4=0.00513,
    ),
    CropType.RICE: Emissions(
        Scope.SCOPE_1,
        "Biomass Burning (Rice)",
        co2=GHG_CONVERSION["CO_CO2"] * 0.04873344,
        n2o=0.000053606784 + GHG_CONVERSION["NOX_N2O"] * 0.001937502336,
        ch4=0.00232064,
    ),
    CropType.SOYBEAN: Emissions(
        Scope.SCOPE_1,
        "Biomass Burning (Soybeans)",
        co2=GHG_CONVERSION["CO_CO2"] * 0.0975744,
        n2o=0.000383328 + GHG_CONVERSION["NOX_N2O"] * 0.0138545691428571,
        ch4=0.0046464,
    ),
    CropType.SUGARCANE: Emissions(
        Scope.SCOPE_1,
        "Biomass Burning (Sugarcane)",
        co2=GHG_CONVERSION["CO_CO2"] * 0.00793636844,
        n2o=0.0000186425657631827 + GHG_CONVERSION["NOX_N2O"] * 0.000673795591155031,
        ch4=0.000377922306666667,
    ),
    CropType.WHEAT: Emissions(
        Scope.SCOPE_1,
        "Biomass Burning (Wheat)",
        co2=GHG_CONVERSION["CO_CO2"] * 0.058212,
        n2o=0.0000548856 + GHG_CONVERSION["NOX_N2O"] * 0.0019837224,
        ch4=0.002772,
    ),
}

GREEN_MANURE_CAPTURE_FACTOR = -1.835  # tonnes of CO2 per hectare


def geometry_to_country_name(
    polygon: Union[
        shpg.Polygon,
        shpg.MultiPolygon,
        shpg.Point,
        shpg.LineString,
        shpg.LinearRing,
        shpg.MultiLineString,
        shpg.GeometryCollection,
    ],
) -> str:
    # Use geopandas "naturalearth_lowres" dataset
    df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))  # type: ignore
    df = df[df.geometry.intersects(polygon)]

    assert df is not None, "There is not intersection between the geometry, and any country"
    if len(df) == 0:
        return "World"
    return df.iloc[0]["name"]


def get_land_use_change_factor(
    previous_land_use: PreviousLandUse,
    current_land_use: CurrentLandUse,
    biome: Biome,
    high_clay_content: bool,
):
    if previous_land_use.name == current_land_use.name:
        return 0.0
    if previous_land_use == PreviousLandUse.DIRECT_SEEDING:
        if current_land_use == CurrentLandUse.CONVENTIONAL_CROPS:
            return 0.9167
    elif previous_land_use == PreviousLandUse.CONVENTIONAL_CROPS:
        if current_land_use == CurrentLandUse.SUGARCANE_WITH_BURNING:
            return -2.09
        elif current_land_use == CurrentLandUse.DIRECT_SEEDING:
            return -1.52
    elif previous_land_use == PreviousLandUse.NATIVE:
        if current_land_use == CurrentLandUse.CONVENTIONAL_CROPS and high_clay_content:
            return 0.1613
        elif current_land_use == CurrentLandUse.CONVENTIONAL_CROPS and not high_clay_content:
            return 0.9167
        elif current_land_use == CurrentLandUse.SUGARCANE_WITH_BURNING:
            return 3.1203
        elif current_land_use == CurrentLandUse.DIRECT_SEEDING:
            if biome == Biome.BRAZIL_CERRADO:
                return -0.44
            elif biome == Biome.BRAZIL_AMAZON_SAVANNA or biome == Biome.BRAZIL_AMAZON_FOREST:
                return 0.88
    return 0.0  # we don't know what this is, so we return 0


class CropEmission:
    """General calculation method for emissions from a crop type.

    Computation should be correct for the following crops:
        - wheat
        - corn
        - cotton
        - soybeans

    :param crop_type: Crop type
    :param cultivation_area: Cultivation area in hectares
    """

    def __init__(self, crop_type: CropType, cultivation_area: float):
        self.cultivation_area = cultivation_area / 1000.0
        self.crop_type = crop_type

        if crop_type not in [
            CropType.WHEAT,
            CropType.CORN,
            CropType.COTTON,
            CropType.SOYBEAN,
        ]:
            raise ValueError("Crop type not supported")

    def fuel_emissions(
        self,
        fuel_consumptions: List[Tuple[FuelType, float]],
        scope: Scope = Scope.SCOPE_1,
        desc: str = "",
        gasoline_mixture: float = WORLD_GASOLINE_MIXTURE,
    ) -> Emissions:
        emissions = Emissions(scope, desc)
        for fuel_type, fuel_consumption in fuel_consumptions:
            tmp = copy(FUEL_EMISSION_FACTORS[fuel_type])
            tmp.scope = scope
            emissions += fuel_consumption * tmp
            if "DIESEL" in fuel_type.name:
                emissions += (
                    fuel_consumption
                    * (1 - FUEL_COMPOSITION[fuel_type])
                    * FUEL_EMISSION_FACTORS[FuelType.BIODIESEL]
                )
            elif "GASOLINE" in fuel_type.name:
                emissions += (
                    fuel_consumption
                    * (1 - gasoline_mixture)
                    * FUEL_EMISSION_FACTORS[FuelType.ETHANOL_ANHYDROUS]
                )
        return emissions

    def biomass_burning_emissions(
        self, average_yield: float, burn_area: float, scope: Scope = Scope.SCOPE_1
    ) -> Emissions:
        tmp = copy(BURNING_EMISSION_FACTORS[self.crop_type])
        tmp.scope = scope
        return average_yield * burn_area * tmp

    def initial_carbon_stock(self, biome: str = "", previous_land_use: str = "") -> Emissions:
        if biome.upper() not in Biome.__members__ or "native" not in previous_land_use.lower():
            return Emissions(Scope.SCOPE_1, "Initial carbon stock")
        stock = BIOME_TO_CARBON_STOCK[Biome[biome.upper()]]
        return Emissions(
            Scope.SCOPE_1,
            "Initial carbon stock",
            co2=(stock * self.cultivation_area * 1000),
        )

    def carbon_capture(
        self,
        cultivation_area: float,
        green_manure_amount: float = 0.0,
        green_manure_grass_amount: float = 0.0,
        freen_fertilizer_legumes_amount: float = 0.0,
    ) -> Emissions:
        total_capture = (
            cultivation_area
            * GREEN_MANURE_CAPTURE_FACTOR
            * any(
                (
                    green_manure_amount,
                    green_manure_grass_amount,
                    freen_fertilizer_legumes_amount,
                )
            )
        )
        return Emissions(
            Scope.SCOPE_1,
            "Carbon captured by Green Manure",
            co2=total_capture,
        )

    def land_use_emissions(
        self,
        biome: str = "",
        previous_land_use: str = "",
        cultivation_area: float = 0.0,
        current_land_use: str = "",
        clay_content: float = 0.0,
    ) -> Emissions:
        try:
            previous = PreviousLandUse[previous_land_use.upper()]
        except Exception:
            for land_use in PreviousLandUse:
                if previous_land_use.upper() in land_use.name:
                    previous = land_use
                    break
            raise ValueError(
                f"Previous land use {previous_land_use} not supported. "
                f"Supported values: {PreviousLandUse.__members__}"
            )
        try:
            current = CurrentLandUse[current_land_use.upper()]
        except Exception:
            for land_use in CurrentLandUse:
                if current_land_use.upper() in land_use.name:
                    current = land_use
                    break
            raise ValueError(
                f"Current land use {current_land_use} not supported. "
                f"Supported values: {CurrentLandUse.__members__}"
            )
        return (
            cultivation_area
            * get_land_use_change_factor(
                previous,
                current,
                Biome[biome.upper()],
                clay_content > CLAY_CONTENT_THRESHOLD,
            )
            * Emissions(Scope.SCOPE_1, "Land use change", co2=1.0)
        )

    def fertilizer_emissions(
        self,
        average_yield: float = 0.0,
        urea_amount: float = 0.0,
        gypsum_amount: float = 0.0,
        limestone_calcite_amount: float = 0.0,
        limestone_dolomite_amount: float = 0.0,
        synthetic_fertilizer_amount: float = 0.0,
        synthetic_fertilizer_nitrogen_ratio: float = 0.0,
        manure_amount: float = 0.0,
        manure_birds_amount: float = 0.0,
        organic_compound_amount: float = 0.0,
        organic_other_amount: float = 0.0,
        green_manure_amount: float = 0.0,
        green_manure_grass_amount: float = 0.0,
        green_manure_legumes_amount: float = 0.0,
        soil_management_area: float = 0.0,
    ) -> Dict[str, Emissions]:
        leached_rate = C_N2O_FLOW_RATE * GHG_CONVERSION["N-N2O_N2O"] * C_FRAC_LEACH
        return {
            "Urea": Emissions(  # ✅
                scope=Scope.SCOPE_1,
                source="Fertilizer emissions, urea",
                co2=FERTILIZER_UREA.co2 * urea_amount * self.cultivation_area,
                n2o=FERTILIZER_UREA.n2o
                * (urea_amount * FERTILIZER_UREA.nitrogen_ratio)
                * self.cultivation_area,
            ),
            "Liming, gypsum": (
                Emissions(  # ✅
                    scope=Scope.SCOPE_1,
                    source="Fertilizer emissions, gypsum",
                    co2=gypsum_amount * FERTILIZER_GYPSUM.co2 * self.cultivation_area,
                )
                + Emissions(
                    scope=Scope.SCOPE_1,
                    source="Fertilizer emissions, limestone, calcite",
                    co2=limestone_calcite_amount
                    * FERTILIZER_LIMESTONE_CALCITE.co2
                    * self.cultivation_area,
                )
                + Emissions(
                    scope=Scope.SCOPE_1,
                    source="Fertilizer emissions, limestone, dolomite",
                    co2=limestone_dolomite_amount
                    * FERTILIZER_LIMESTONE_DOLOMITE.co2
                    * self.cultivation_area,
                )
            ),
            "Synthetic nitrogen fertilizer": Emissions(  # ✅
                scope=Scope.SCOPE_1,
                source="Fertilizer emissions, synthetic nitrogen fertilizer",
                n2o=FERTILIZER_SYNTHETIC.n2o
                * (synthetic_fertilizer_amount * synthetic_fertilizer_nitrogen_ratio)
                * self.cultivation_area,
            ),
            "Organic fertilizers": (
                Emissions(  # ✅
                    scope=Scope.SCOPE_1,
                    source="Fertilizer emissions, manure",
                    n2o=manure_amount * FERTILIZER_MANURE.n2o * self.cultivation_area,
                )
                + Emissions(
                    scope=Scope.SCOPE_1,
                    source="Fertilizer emissions, bird manure",
                    n2o=manure_birds_amount * FERTILIZER_MANURE_BIRDS.n2o * self.cultivation_area,
                )
                + Emissions(
                    scope=Scope.SCOPE_1,
                    source="Fertilizer emissions, organic fertilizer",
                    n2o=organic_compound_amount
                    * FERTILIZER_ORGANIC_COMPOUND.n2o
                    * self.cultivation_area,
                )
                + Emissions(
                    scope=Scope.SCOPE_1,
                    source="Fertilizer emissions, organic others",
                    n2o=organic_other_amount
                    * FERTILIZER_GENERIC_ORGANIC.n2o
                    * self.cultivation_area,
                )
            ),
            "Leaching / Surface runoff": (
                Emissions(  # ✅
                    scope=Scope.SCOPE_1,
                    source="Flow emissions, surface runoff, urea",
                    n2o=(urea_amount * FERTILIZER_UREA.nitrogen_ratio)
                    * leached_rate
                    * self.cultivation_area,
                )
                + Emissions(
                    scope=Scope.SCOPE_1,
                    source="Flow emissions, surface runoff, synthetic fertilizer",
                    n2o=(synthetic_fertilizer_amount * synthetic_fertilizer_nitrogen_ratio)
                    * leached_rate
                    * self.cultivation_area,
                )
                + Emissions(
                    scope=Scope.SCOPE_1,
                    source="Flow emissions, surface runoff, organic fertilizer",
                    n2o=(organic_compound_amount * FERTILIZER_ORGANIC_COMPOUND.nitrogen_ratio)
                    * leached_rate
                    * self.cultivation_area,
                )
                + Emissions(
                    scope=Scope.SCOPE_1,
                    source="Flow emissions, surface runoff, manure",
                    n2o=(manure_amount * FERTILIZER_MANURE.nitrogen_ratio)
                    * leached_rate
                    * self.cultivation_area,
                )
                + Emissions(
                    scope=Scope.SCOPE_1,
                    source="Flow emissions, surface runoff, manure, bird",
                    n2o=(manure_birds_amount * FERTILIZER_MANURE_BIRDS.nitrogen_ratio)
                    * leached_rate
                    * self.cultivation_area,
                )
                + Emissions(
                    scope=Scope.SCOPE_1,
                    source="Flow emissions, surface runoff, organic, other",
                    n2o=(organic_other_amount * FERTILIZER_GENERIC_ORGANIC.nitrogen_ratio)
                    * leached_rate
                    * self.cultivation_area,
                )
            ),
            "Atmospheric emissions, N2O": (
                Emissions(
                    scope=Scope.SCOPE_1,
                    source="Atmospheric emissions, N2O, Urea",
                    n2o=urea_amount
                    * FERTILIZER_UREA.nitrogen_ratio
                    * C_FRAC_GAS_F
                    * N2O_ATMOSPHERIC_VOLATIZATION_RATE
                    * GHG_CONVERSION["N-N2O_N2O"]
                    * self.cultivation_area,
                )
                + Emissions(
                    scope=Scope.SCOPE_1,
                    source="Atmospheric emissions, N2O, Synthetic nitrogen fertilizer",
                    n2o=synthetic_fertilizer_amount
                    * synthetic_fertilizer_nitrogen_ratio
                    * C_FRAC_GAS_F
                    * N2O_ATMOSPHERIC_VOLATIZATION_RATE
                    * GHG_CONVERSION["N-N2O_N2O"]
                    * self.cultivation_area,
                )
                + Emissions(
                    scope=Scope.SCOPE_1,
                    source="Atmospheric emissions, N2O, Organic fertilizer",
                    n2o=organic_compound_amount
                    * FERTILIZER_ORGANIC_COMPOUND.nitrogen_ratio
                    * C_FRAC_GAS_F
                    * C_N2O_VOLATILIZATION
                    * GHG_CONVERSION["N-N2O_N2O"]
                    * self.cultivation_area,
                )
                + Emissions(
                    scope=Scope.SCOPE_1,
                    source="Atmospheric emissions, N2O, Manure",
                    n2o=manure_amount
                    * FERTILIZER_MANURE.nitrogen_ratio
                    * C_FRAC_GAS_F
                    * C_N2O_VOLATILIZATION
                    * GHG_CONVERSION["N-N2O_N2O"]
                    * self.cultivation_area,
                )
                + Emissions(
                    scope=Scope.SCOPE_1,
                    source="Atmospheric emissions, N2O, Manure, Birds",
                    n2o=manure_birds_amount
                    * FERTILIZER_MANURE_BIRDS.nitrogen_ratio
                    * C_FRAC_GAS_F
                    * C_N2O_VOLATILIZATION
                    * GHG_CONVERSION["N-N2O_N2O"]
                    * self.cultivation_area,
                )
                + Emissions(
                    scope=Scope.SCOPE_1,
                    source="Atmospheric emissions, N2O, Organic, other",
                    n2o=organic_other_amount
                    * FERTILIZER_GENERIC_ORGANIC.nitrogen_ratio
                    * C_FRAC_GAS_F
                    * C_N2O_VOLATILIZATION
                    * GHG_CONVERSION["N-N2O_N2O"]
                    * self.cultivation_area,
                )
            ),
            "Residue decomposition": (
                Emissions(
                    scope=Scope.SCOPE_1,
                    source="Residue decomposition",
                    n2o=(
                        (average_yield * RESIDUES[self.crop_type].value)
                        + (green_manure_amount / 1000 * RESIDUES[CropType.GREEN_MANURE].value)
                        + (
                            green_manure_grass_amount
                            / 1000
                            * RESIDUES[CropType.GREEN_MANURE_GRASSES].value
                        )
                        + (
                            green_manure_legumes_amount
                            / 1000
                            * RESIDUES[CropType.GREEN_MANURE_LEGUMES].value
                        )
                    )
                    * 10,
                )
            ),
            "Soil management": (
                Emissions(
                    scope=Scope.SCOPE_1,
                    source="Soil management",
                    co2=soil_management_area * CO2EQ_SOIL_EMISSIONS,
                )
            ),
        }


class CallbackBuilder:
    def __init__(
        self,
        crop_type: str,
    ):
        if crop_type.upper() not in CropType.__members__:
            raise ValueError(f"Unsupported crop type: {crop_type}")
        self.crop_type = CropType[crop_type.upper()]

    def __call__(self):
        def emissions_callback(ghg: GHGProtocolVibe) -> Dict[str, List[GHGFlux]]:
            geometry = shpg.shape(ghg.geometry)
            country_name = geometry_to_country_name(geometry)  # type: ignore
            gasoline_mixture = GASOLINE_MIXTURES.get(country_name, GASOLINE_MIXTURES["World"])

            if ghg.cultivation_area:
                area_ha = ghg.cultivation_area
            else:
                geod = Geod(ellps="WGS84")
                area = abs(geod.geometry_area_perimeter(geometry)[0])  # in m^2
                area_ha = area / 10000  # in ha

            fuel_consumptions = []
            if ghg.diesel_amount != 0:
                if ghg.diesel_type is None:
                    raise ValueError("Diesel amount is not zero, but diesel type is not specified")
                fuel_consumptions.append(
                    (
                        FuelType[ghg.diesel_type.upper()],
                        ghg.diesel_amount,
                    )
                )
            if ghg.gasoline_amount != 0:
                fuel_consumptions.append(
                    (
                        FuelType.GASOLINE,
                        ghg.gasoline_amount * gasoline_mixture if ghg.gasoline_amount else 0.0,
                        # The above can be done because all equations are linear
                    )
                )

            if not ghg.total_yield:
                raise ValueError("Total yield is not specified")

            crop_emission = CropEmission(self.crop_type, area_ha)
            internal_operations_emissions = crop_emission.fuel_emissions(
                fuel_consumptions,
                Scope.SCOPE_1,
                "Internal operations",
                gasoline_mixture,
            )
            transport_emissions = crop_emission.fuel_emissions(
                [
                    (
                        FuelType[
                            ghg.transport_diesel_type.upper()
                            if ghg.transport_diesel_type
                            else "DIESEL"
                        ],
                        ghg.transport_diesel_amount if ghg.transport_diesel_amount else 0.0,
                    )
                ],
                Scope.SCOPE_3,
                "Transportation",
                gasoline_mixture,
            )
            fertilizer_parameters = dict(
                average_yield=ghg.total_yield / area_ha,
                urea_amount=ghg.urea_amount if ghg.urea_amount else 0,
                gypsum_amount=ghg.gypsum_amount if ghg.gypsum_amount else 0,
                limestone_calcite_amount=ghg.limestone_calcite_amount,
                limestone_dolomite_amount=ghg.limestone_dolomite_amount,
                synthetic_fertilizer_amount=ghg.synthetic_fertilizer_amount,
                synthetic_fertilizer_nitrogen_ratio=ghg.synthetic_fertilizer_nitrogen_ratio,
                manure_amount=ghg.manure_amount,
                manure_birds_amount=ghg.manure_birds_amount,
                organic_compound_amount=ghg.organic_compound_amount,
                organic_other_amount=ghg.organic_other_amount,
                green_manure_amount=ghg.green_manure_amount,
                green_manure_grass_amount=ghg.green_manure_grass_amount,
                green_manure_legumes_amount=ghg.green_manure_legumes_amount,
                soil_management_area=ghg.soil_management_area
                if ghg.soil_management_area
                else area_ha,
            )
            fertilizer_parameters = {
                k: v if v is not None else 0.0 for k, v in fertilizer_parameters.items()
            }

            fertilizer_emissions = crop_emission.fertilizer_emissions(**fertilizer_parameters)
            initial_carbon_stock = crop_emission.initial_carbon_stock(
                ghg.biome, ghg.previous_land_use
            )
            biomass_burning_emissions = crop_emission.biomass_burning_emissions(
                average_yield=ghg.total_yield / area_ha,
                burn_area=ghg.burn_area if ghg.burn_area else 0.0,
            )
            carbon_capture = crop_emission.carbon_capture(
                area_ha,
                ghg.green_manure_amount if ghg.green_manure_amount else 0.0,
                ghg.green_manure_grass_amount if ghg.green_manure_grass_amount else 0.0,
                ghg.green_manure_legumes_amount if ghg.green_manure_legumes_amount else 0.0,
            )
            land_use_emissions = crop_emission.land_use_emissions(
                ghg.biome,
                ghg.previous_land_use,
                area_ha,
                ghg.current_land_use,
                ghg.soil_clay_content if ghg.soil_clay_content else 0.0,
            )

            emissions = (
                [internal_operations_emissions]
                + [e for e in fertilizer_emissions.values()]
                + [initial_carbon_stock]
                + [transport_emissions]
                + [biomass_burning_emissions]
                + [carbon_capture]
                + [land_use_emissions]
            )
            return {
                "fluxes": [
                    GHGFlux(
                        id=gen_hash_id(
                            f"ghg_{e.scope}_{e.source}_{asdict(ghg)}",
                            ghg.geometry,
                            ghg.time_range,
                        ),
                        time_range=ghg.time_range,
                        geometry=ghg.geometry,
                        scope=str(e.scope.value),
                        value=e.total,
                        description=e.source,
                        assets=[],
                    )
                    for e in emissions
                ]
            }

        return emissions_callback
