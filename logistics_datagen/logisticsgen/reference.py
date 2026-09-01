"""Static reference data: geography, taxonomies and name pools.

Kept dependency-free (no Faker) so the generator stays fast and fully vectorised
at 1M+ rows. Name synthesis is combinatorial rather than dictionary-based.
"""

from __future__ import annotations

import numpy as np

# --- Geography ---------------------------------------------------------------
# (city, country, region, lat, lon, urban_share) - a spread of real metro hubs
# so coordinates and distances land in plausible places on a map.
CITIES: list[tuple[str, str, str, float, float, float]] = [
    ("London",        "GB", "EMEA",     51.5074,  -0.1278, 0.82),
    ("Manchester",    "GB", "EMEA",     53.4808,  -2.2426, 0.71),
    ("Berlin",        "DE", "EMEA",     52.5200,  13.4050, 0.78),
    ("Munich",        "DE", "EMEA",     48.1351,  11.5820, 0.70),
    ("Paris",         "FR", "EMEA",     48.8566,   2.3522, 0.85),
    ("Lyon",          "FR", "EMEA",     45.7640,   4.8357, 0.66),
    ("Madrid",        "ES", "EMEA",     40.4168,  -3.7038, 0.79),
    ("Barcelona",     "ES", "EMEA",     41.3851,   2.1734, 0.80),
    ("Milan",         "IT", "EMEA",     45.4642,   9.1900, 0.77),
    ("Rome",          "IT", "EMEA",     41.9028,  12.4964, 0.75),
    ("Warsaw",        "PL", "EMEA",     52.2297,  21.0122, 0.68),
    ("Krakow",        "PL", "EMEA",     50.0647,  19.9450, 0.63),
    ("Kyiv",          "UA", "EMEA",     50.4501,  30.5234, 0.72),
    ("Amsterdam",     "NL", "EMEA",     52.3676,   4.9041, 0.81),
    ("Rotterdam",     "NL", "EMEA",     51.9244,   4.4777, 0.74),
    ("Stockholm",     "SE", "EMEA",     59.3293,  18.0686, 0.69),
    ("Dubai",         "AE", "EMEA",     25.2048,  55.2708, 0.88),
    ("Istanbul",      "TR", "EMEA",     41.0082,  28.9784, 0.84),
    ("New York",      "US", "NA",       40.7128, -74.0060, 0.90),
    ("Chicago",       "US", "NA",       41.8781, -87.6298, 0.76),
    ("Dallas",        "US", "NA",       32.7767, -96.7970, 0.61),
    ("Los Angeles",   "US", "NA",       34.0522,-118.2437, 0.83),
    ("Atlanta",       "US", "NA",       33.7490, -84.3880, 0.58),
    ("Seattle",       "US", "NA",       47.6062,-122.3321, 0.72),
    ("Toronto",       "CA", "NA",       43.6532, -79.3832, 0.79),
    ("Vancouver",     "CA", "NA",       49.2827,-123.1207, 0.70),
    ("Mexico City",   "MX", "LATAM",    19.4326, -99.1332, 0.86),
    ("Sao Paulo",     "BR", "LATAM",   -23.5505, -46.6333, 0.87),
    ("Buenos Aires",  "AR", "LATAM",   -34.6037, -58.3816, 0.80),
    ("Bogota",        "CO", "LATAM",     4.7110, -74.0721, 0.82),
    ("Singapore",     "SG", "APAC",      1.3521, 103.8198, 0.95),
    ("Tokyo",         "JP", "APAC",     35.6762, 139.6503, 0.91),
    ("Osaka",         "JP", "APAC",     34.6937, 135.5023, 0.85),
    ("Seoul",         "KR", "APAC",     37.5665, 126.9780, 0.89),
    ("Shanghai",      "CN", "APAC",     31.2304, 121.4737, 0.90),
    ("Shenzhen",      "CN", "APAC",     22.5431, 114.0579, 0.88),
    ("Mumbai",        "IN", "APAC",     19.0760,  72.8777, 0.88),
    ("Delhi",         "IN", "APAC",     28.7041,  77.1025, 0.85),
    ("Bengaluru",     "IN", "APAC",     12.9716,  77.5946, 0.79),
    ("Jakarta",       "ID", "APAC",     -6.2088, 106.8456, 0.86),
    ("Sydney",        "AU", "APAC",    -33.8688, 151.2093, 0.78),
    ("Melbourne",     "AU", "APAC",    -37.8136, 144.9631, 0.75),
    ("Johannesburg",  "ZA", "EMEA",    -26.2041,  28.0473, 0.74),
    ("Cairo",         "EG", "EMEA",     30.0444,  31.2357, 0.83),
    ("Lagos",         "NG", "EMEA",      6.5244,   3.3792, 0.87),
]

CITY_NAMES = [c[0] for c in CITIES]
COUNTRIES = [c[1] for c in CITIES]
REGIONS = [c[2] for c in CITIES]
CITY_LAT = np.array([c[3] for c in CITIES])
CITY_LON = np.array([c[4] for c in CITIES])
CITY_URBAN_SHARE = np.array([c[5] for c in CITIES])

# Rough purchasing-power / cost multiplier per region - feeds fuel, labour, tolls.
REGION_COST_INDEX = {"EMEA": 1.00, "NA": 1.12, "LATAM": 0.68, "APAC": 0.79}

# --- Fleet -------------------------------------------------------------------
VEHICLE_TYPES = ["bike", "car", "van", "ev_van", "truck", "refrigerated_truck"]
VEHICLE_TYPE_WEIGHTS = [0.10, 0.14, 0.34, 0.10, 0.22, 0.10]
VEHICLE_CAPACITY_KG = {
    "bike": 25, "car": 350, "van": 1400, "ev_van": 1200,
    "truck": 12000, "refrigerated_truck": 9000,
}
VEHICLE_CAPACITY_M3 = {
    "bike": 0.15, "car": 1.6, "van": 9.0, "ev_van": 8.0,
    "truck": 48.0, "refrigerated_truck": 38.0,
}
VEHICLE_MAKES = [
    "Mercedes", "Ford", "Volkswagen", "Renault", "Iveco", "Scania",
    "Volvo", "MAN", "Toyota", "Nissan", "Rivian", "BYD",
]
FUEL_TYPES = ["diesel", "petrol", "electric", "hybrid", "cng"]

# --- Workforce ---------------------------------------------------------------
EMPLOYMENT_TYPES = ["full_time", "part_time", "contractor", "gig"]
EMPLOYMENT_WEIGHTS = [0.44, 0.18, 0.22, 0.16]
LICENCE_CLASSES = ["B", "C", "C+E", "D", "none"]
SHIFT_TYPES = ["morning", "afternoon", "night", "split", "on_call"]

FIRST_NAMES = [
    "Alex", "Maria", "Daniel", "Sofia", "Lucas", "Anna", "Omar", "Yuki",
    "Priya", "Carlos", "Elena", "Marcus", "Chen", "Ivan", "Fatima", "Hugo",
    "Nadia", "Peter", "Grace", "Tomas", "Leila", "Viktor", "Amara", "Jun",
    "Isabella", "Mateo", "Zara", "Nikolai", "Aisha", "Diego", "Freya", "Rahul",
]
LAST_NAMES = [
    "Smith", "Muller", "Dubois", "Garcia", "Rossi", "Kowalski", "Petrov",
    "Nakamura", "Sharma", "Silva", "Nguyen", "Okafor", "Hansen", "Novak",
    "Ali", "Fernandez", "Weber", "Lindqvist", "Brown", "Kim", "Zhang",
    "Ivanov", "Costa", "Bakker", "Moreau", "Rahman", "Torres", "Yilmaz",
]
STREET_TYPES = ["St", "Ave", "Rd", "Blvd", "Ln", "Way", "Sq", "Dr"]
STREET_STEMS = [
    "Maple", "Oak", "Harbour", "Station", "Market", "Church", "Industrial",
    "Riverside", "Park", "Victoria", "Cedar", "Bridge", "Kings", "Union",
    "Central", "Warehouse", "Elm", "Pine", "Grove", "Depot",
]

# --- Order taxonomy ----------------------------------------------------------
PRIORITIES = ["standard", "express", "same_day", "scheduled", "economy"]
PRIORITY_WEIGHTS = [0.46, 0.21, 0.09, 0.14, 0.10]
PRIORITY_SLA_FACTOR = {
    "same_day": 0.25, "express": 0.5, "scheduled": 1.0,
    "standard": 1.0, "economy": 1.6,
}
ORDER_STATUSES = ["delivered", "delivered_late", "failed", "returned", "cancelled", "in_transit"]
PAYMENT_METHODS = ["card", "cash_on_delivery", "wallet", "invoice", "prepaid"]
CUSTOMER_SEGMENTS = ["retail_consumer", "smb", "enterprise", "public_sector", "marketplace_seller"]
CUSTOMER_SEGMENT_WEIGHTS = [0.55, 0.22, 0.12, 0.04, 0.07]
LOYALTY_TIERS = ["none", "bronze", "silver", "gold", "platinum"]
PACKAGE_TYPES = ["envelope", "small_box", "medium_box", "large_box", "pallet", "crate", "cold_box"]
AREA_TYPES = ["urban", "suburban", "rural"]

# --- Warehouse ---------------------------------------------------------------
WAREHOUSE_TYPES = ["fulfilment_centre", "cross_dock", "cold_storage", "micro_hub", "regional_dc"]
WAREHOUSE_TYPE_WEIGHTS = [0.30, 0.20, 0.12, 0.23, 0.15]
AUTOMATION_LEVELS = ["manual", "semi_automated", "automated", "lights_out"]
SKU_CATEGORIES = [
    "electronics", "apparel", "grocery_ambient", "grocery_chilled", "pharma",
    "spare_parts", "furniture", "cosmetics", "books", "industrial_chemicals",
]

# --- Environment -------------------------------------------------------------
WEATHER_CONDITIONS = ["clear", "cloudy", "rain", "heavy_rain", "snow", "fog", "storm", "heat_wave"]
WEATHER_WEIGHTS = [0.36, 0.27, 0.16, 0.06, 0.05, 0.05, 0.03, 0.02]
WEATHER_DELAY_FACTOR = {
    "clear": 1.00, "cloudy": 1.01, "rain": 1.09, "heavy_rain": 1.22,
    "snow": 1.45, "fog": 1.18, "storm": 1.55, "heat_wave": 1.08,
}
TRAFFIC_LEVELS = ["free_flow", "light", "moderate", "heavy", "gridlock"]
TRAFFIC_DELAY_FACTOR = {
    "free_flow": 0.92, "light": 1.00, "moderate": 1.14,
    "heavy": 1.38, "gridlock": 1.85,
}
INCIDENT_TYPES = ["none", "collision", "roadworks", "closure", "protest", "flooding"]

MAINTENANCE_TYPES = [
    "scheduled_service", "tyre_replacement", "brake_repair", "engine_repair",
    "battery_replacement", "refrigeration_unit", "bodywork", "inspection",
]
FEEDBACK_CHANNELS = ["app", "email", "sms", "call_centre", "web"]
DELIVERY_EVENTS = [
    "order_created", "picked", "packed", "dispatched", "in_transit",
    "out_for_delivery", "delivery_attempted", "delivered", "exception", "returned",
]
COST_CATEGORIES = ["fuel", "labour", "maintenance", "tolls", "insurance", "warehousing", "overhead", "penalties"]

# Holiday templates - (month, day, name, scope) applied per country/region.
HOLIDAY_TEMPLATES = [
    (1, 1, "New Year", "global"), (12, 25, "Christmas", "christian"),
    (12, 26, "Boxing Day", "christian"), (11, 28, "Thanksgiving", "US"),
    (7, 4, "Independence Day", "US"), (5, 1, "Labour Day", "global"),
    (11, 11, "Singles Day", "APAC"), (10, 3, "Unity Day", "DE"),
    (8, 24, "Independence Day", "UA"), (1, 26, "Republic Day", "IN"),
    (11, 29, "Black Friday", "global"), (12, 31, "New Year Eve", "global"),
]
