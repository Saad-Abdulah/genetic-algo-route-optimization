import csv
import io
import math
from collections import defaultdict


class City:
    """Represents a city with geographic coordinates."""

    def __init__(self, name, latitude, longitude):
        self.name = name
        self.lat = latitude
        self.lon = longitude

    def __repr__(self):
        return f"City({self.name}, {self.lat}, {self.lon})"


def haversine(city_a, city_b):
    """
    Calculate the great-circle distance between two cities
    using the Haversine formula. Returns distance in kilometers.
    """
    R = 6371.0

    lat1 = math.radians(city_a.lat)
    lat2 = math.radians(city_b.lat)
    dlat = math.radians(city_b.lat - city_a.lat)
    dlon = math.radians(city_b.lon - city_a.lon)

    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def route_distance(route, cities):
    """
    Compute total distance of a closed route (returns to starting city).
    route: list of city indices representing the visit order.
    """
    total = 0.0
    n = len(route)
    for i in range(n):
        from_city = cities[route[i]]
        to_city = cities[route[(i + 1) % n]]
        total += haversine(from_city, to_city)
    return total


def _city_from_csv_row(row, idx):
    name = row.get("City") or row.get("city")
    lat = row.get("Latitude") or row.get("latitude")
    lon = row.get("Longitude") or row.get("longitude")

    if name is None or lat is None or lon is None:
        raise ValueError(f"Invalid row at line {idx}: {row}")

    clean_name = name.strip()
    if clean_name == "":
        raise ValueError(f"Empty city name at line {idx}.")

    try:
        lat_val = float(lat)
        lon_val = float(lon)
    except ValueError:
        raise ValueError(
            f"Invalid numeric value at line {idx}: "
            f"Latitude='{lat}', Longitude='{lon}'"
        )

    if not (-90.0 <= lat_val <= 90.0):
        raise ValueError(f"Latitude out of range at line {idx}: {lat_val}")
    if not (-180.0 <= lon_val <= 180.0):
        raise ValueError(f"Longitude out of range at line {idx}: {lon_val}")

    if not math.isfinite(lat_val) or not math.isfinite(lon_val):
        raise ValueError(f"Non-finite coordinates at line {idx}.")

    return City(clean_name, lat_val, lon_val)


def load_cities_with_roads(filepath):
    """
    Load cities and optional road edges from one CSV file.

    City rows use columns: City, Latitude, Longitude (case variants allowed).
    After a line whose first cell starts with # and contains ROAD (e.g. ``# ROADS``),
    rows are ``From,To`` pairs (city names matching the city section). Roads are undirected.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    split_idx = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") and "ROAD" in stripped.upper():
            split_idx = i
            break

    city_text = "".join(lines[:split_idx])
    road_lines = lines[split_idx + 1 :] if split_idx < len(lines) else []

    cities = []
    reader = csv.DictReader(io.StringIO(city_text))
    if reader.fieldnames is None:
        raise ValueError("CSV file has no header row for cities.")

    for idx, row in enumerate(reader, start=2):
        cities.append(_city_from_csv_row(row, idx))

    if len(cities) == 0:
        raise ValueError("CSV file is empty or has no valid city data.")

    road_pairs = []
    if road_lines:
        rr = csv.reader(io.StringIO("".join(road_lines)))
        header_skipped = False
        for row in rr:
            if not row or all(not (c or "").strip() for c in row):
                continue
            a = row[0].strip()
            if a.startswith("#"):
                continue
            if not header_skipped:
                if (
                    a.lower() == "from"
                    and len(row) > 1
                    and row[1].strip().lower() == "to"
                ):
                    header_skipped = True
                    continue
                header_skipped = True
            if len(row) < 2:
                continue
            b = row[1].strip()
            if not a or not b:
                continue
            road_pairs.append((a, b))

    return cities, road_pairs


def road_pairs_to_canonical_edges(cities, pairs):
    """Map (name, name) road pairs to sorted (i, j) index pairs."""
    name_to_idx = {c.name: i for i, c in enumerate(cities)}
    edges = set()
    for a, b in pairs:
        if a not in name_to_idx or b not in name_to_idx:
            raise ValueError(f"Road references unknown city: {a!r} — {b!r}")
        if a == b:
            continue
        i, j = name_to_idx[a], name_to_idx[b]
        if i > j:
            i, j = j, i
        edges.add((i, j))
    return edges


def load_cities(filepath):
    """
    Load city data from a CSV file (optional ``# ROADS`` section is ignored).
    Expects columns: City (or city), Latitude (or latitude), Longitude (or longitude).
    """
    cities, _ = load_cities_with_roads(filepath)
    return cities


def find_data_inconsistencies(cities):
    """
    Return a list of data consistency errors that should block execution.
    """
    errors = []
    if not cities:
        errors.append("No city data loaded.")
        return errors

    # Duplicate city names (case-insensitive)
    name_to_indices = defaultdict(list)
    for i, c in enumerate(cities):
        key = c.name.strip().casefold()
        name_to_indices[key].append(i + 1)

    for norm_name, positions in name_to_indices.items():
        if len(positions) > 1:
            errors.append(
                f"Duplicate city name detected ('{norm_name}') at rows: {positions}"
            )

    # Duplicate exact coordinates
    coord_to_indices = defaultdict(list)
    for i, c in enumerate(cities):
        coord_to_indices[(c.lat, c.lon)].append(i + 1)

    for coord, positions in coord_to_indices.items():
        if len(positions) > 1:
            errors.append(
                f"Duplicate coordinates detected {coord} at rows: {positions}"
            )

    return errors


def validate_route(route, num_cities):
    """Verify that a route is a valid permutation of all city indices."""
    if len(route) != num_cities:
        return False
    if sorted(route) != list(range(num_cities)):
        return False
    return True


def validate_open_route(route, num_cities, start_idx, end_idx):
    """Verify open path: permutation of all indices, fixed start and end."""
    if len(route) != num_cities:
        return False
    if sorted(route) != list(range(num_cities)):
        return False
    if route[0] != start_idx or route[-1] != end_idx:
        return False
    return True


def validate_corridor_path(route, src_idx, dst_idx):
    """Route starts at src, ends at dst, no repeated cities."""
    if not route or route[0] != src_idx or route[-1] != dst_idx:
        return False
    if len(route) != len(set(route)):
        return False
    return True
