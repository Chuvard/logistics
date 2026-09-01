"""Table generators, split by role.

``entities``    - dimensions (warehouses, customers, vehicles, drivers, zones)
``environment`` - exogenous panels (traffic, weather, fuel prices, holidays)
``orders``      - the order/route fact core
``operations``  - operational facts hanging off the core
"""

from .entities import (
    build_customers, build_delivery_zones, build_drivers,
    build_pickup_locations, build_vehicles, build_warehouses,
)
from .environment import (
    build_fuel_prices, build_regional_holidays, build_traffic, build_weather,
)
from .operations import (
    build_courier_performance, build_customer_feedback, build_delivery_history,
    build_gps_tracking, build_inventory, build_operating_costs,
    build_shift_planning, build_vehicle_maintenance,
)
from .orders import build_orders, build_routes

__all__ = [
    "build_warehouses", "build_pickup_locations", "build_delivery_zones",
    "build_customers", "build_vehicles", "build_drivers",
    "build_traffic", "build_weather", "build_fuel_prices", "build_regional_holidays",
    "build_orders", "build_routes",
    "build_gps_tracking", "build_delivery_history", "build_customer_feedback",
    "build_vehicle_maintenance", "build_inventory", "build_shift_planning",
    "build_operating_costs", "build_courier_performance",
]
