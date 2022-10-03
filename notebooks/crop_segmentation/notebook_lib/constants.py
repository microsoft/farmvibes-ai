from itertools import chain

CROP_INDICES = [e for e in chain(range(1, 81), range(196, 256))]

# 24 - Winter Wheat: 1127411
# 36 - Alfalfa: 877421
# 43 - Potatoes: 497398
# 37 - Other Hay/Non Alfalfa: 460732
# 68 - Apples: 416528
# 1 - Corn: 396329
# 23 - Spring Wheat: 150973
# 69 - Grapes: 124028
# 42 - Dry Beans: 118422
# 59 - Sod/Grass Seed: 115036
# 12 - Sweet Corn: 100565
WINTER_WHEAT_INDEX = [24]
ALFALFA_INDEX = [36]
POTATO_INDEX = [43]
OTHER_HAY_INDEX = [37]
APPELS_INDEX = [68]
CORN_INDEX = [1]
SPRING_WHEAT_INDEX = [23]
GRAPES_INDEX = [69]
DRY_BEANS_INDEX = [42]
SOD_INDEX = [59]
SWEET_CORN_INDEX = [12]
