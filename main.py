"""
Route Optimization using Genetic Algorithm
AI2002 - Artificial Intelligence | Assignment 3
Spring 2026, FAST NUCES Islamabad

Solves the Traveling Salesman Problem (TSP) using a Genetic Algorithm
with tournament selection, order crossover (OX1), and swap mutation.
"""

import os
import sys
import time
import random

from route_utils import load_cities, validate_route, find_data_inconsistencies
from genetic_algorithm import GeneticAlgorithm
from visualization import plot_route, plot_fitness

RESULTS_DIR = "results"
DEFAULT_CSV = os.path.join("data", "cities.csv")
CSV_OPTIONS = {
    "1": os.path.join("data", "cities_20.csv"),
    "2": os.path.join("data", "cities_25.csv"),
    "3": os.path.join("data", "cities.csv"),
}

SEPARATOR = "=" * 62


def safe_input(prompt, default=""):
    try:
        return input(prompt)
    except EOFError:
        return default


def print_banner():
    print()
    print(SEPARATOR)
    print("   Route Optimization using Genetic Algorithm (TSP)")
    print("   AI2002 - Assignment 3  |  Spring 2026")
    print(SEPARATOR)
    print()


def read_positive_int(prompt, default):
    raw = safe_input(prompt, "").strip()
    if raw == "":
        return default
    try:
        val = int(raw)
        if val <= 0:
            raise ValueError
        return val
    except ValueError:
        print(f"    -> Invalid input, using default ({default})")
        return default


def read_positive_float(prompt, default):
    raw = safe_input(prompt, "").strip()
    if raw == "":
        return default
    try:
        val = float(raw)
        if val <= 0:
            raise ValueError
        return val
    except ValueError:
        print(f"    -> Invalid input, using default ({default})")
        return default


def read_probability(prompt, default):
    raw = safe_input(prompt, "").strip()
    if raw == "":
        return default
    try:
        val = float(raw)
        if not (0.0 < val <= 1.0):
            raise ValueError
        return val
    except ValueError:
        print(f"    -> Invalid input, using default ({default})")
        return default


def get_parameters():
    """Prompt the user for GA parameters (or accept defaults)."""
    print("  Configure Parameters (press Enter for defaults)")
    print("  " + "-" * 44)

    pop_size = read_positive_int("    Population size   [100] : ", 100)
    mut_rate = read_probability("    Mutation rate    [0.015] : ", 0.015)
    gens = read_positive_int("    Generations       [500] : ", 500)
    tourn = read_positive_int("    Tournament size     [5] : ", 5)

    seed_raw = safe_input("    Random seed        [none] : ", "").strip()
    seed = None
    if seed_raw != "":
        try:
            seed = int(seed_raw)
        except ValueError:
            print("    -> Invalid seed, randomness will remain non-deterministic")
            seed = None

    print()
    return pop_size, mut_rate, gens, tourn, seed


def choose_dataset():
    print("  Select Dataset:")
    print("  " + "-" * 44)
    print("    1) data/cities_20.csv")
    print("    2) data/cities_25.csv")
    print("    3) data/cities.csv")
    while True:
        choice = safe_input("  Enter choice [3]: ", "3").strip()
        if choice == "":
            choice = "3"
        if choice in CSV_OPTIONS:
            return CSV_OPTIONS[choice]
        print("    -> Invalid choice. Please enter only 1, 2, or 3.")


def show_cities(cities):
    """Print a numbered table of loaded cities."""
    print(f"  Loaded {len(cities)} cities:")
    print("  " + "-" * 44)
    for i, c in enumerate(cities):
        print(f"    {i + 1:>2}. {c.name:<22} ({c.lat:>9.5f}, {c.lon:>10.5f})")
    print()


def show_results(best_route, best_dist, cities, elapsed):
    """Print the final optimization results."""
    print()
    print(SEPARATOR)
    print("   OPTIMIZATION RESULTS")
    print(SEPARATOR)
    print(f"   Total Distance  : {best_dist:,.2f} km")
    print(f"   Cities Visited  : {len(cities)}")
    print(f"   Computation Time: {elapsed:.2f} seconds")
    print()
    print("   Optimized Route:")
    print("   " + "-" * 44)
    for idx, city_idx in enumerate(best_route):
        c = cities[city_idx]
        print(f"     {idx + 1:>2}. {c.name}")
    print(f"     --> Return to {cities[best_route[0]].name}")
    print(SEPARATOR)
    print()


def main():
    print_banner()

    # --- Load city data ---
    csv_path = choose_dataset()
    print(f"  Selected CSV: {csv_path}")

    if not os.path.isfile(csv_path):
        print(f"\n  Error: file '{csv_path}' does not exist.")
        sys.exit(1)

    try:
        cities = load_cities(csv_path)
    except Exception as e:
        print(f"\n  Error loading CSV: {e}")
        sys.exit(1)

    issues = find_data_inconsistencies(cities)
    if issues:
        print("\n  Data consistency errors detected:")
        for i, issue in enumerate(issues, start=1):
            print(f"    {i}. {issue}")
        print("  Please fix the CSV and run again.")
        sys.exit(1)

    if len(cities) < 4:
        print("\n  Error: at least 4 cities are needed for a meaningful route.")
        sys.exit(1)

    show_cities(cities)

    # --- GA parameters ---
    pop_size, mut_rate, gens, tourn, seed = get_parameters()

    if seed is not None:
        random.seed(seed)

    # --- Run the Genetic Algorithm ---
    print("  Running Genetic Algorithm...")
    print("  " + "-" * 44)

    if tourn > pop_size:
        print("  Error: tournament size cannot exceed population size.")
        sys.exit(1)

    try:
        ga = GeneticAlgorithm(
            cities=cities,
            pop_size=pop_size,
            mutation_rate=mut_rate,
            tournament_size=tourn,
            elite_count=2
        )

        t0 = time.time()
        best_route, best_dist = ga.run(generations=gens, verbose=True)
        elapsed = time.time() - t0
    except Exception as e:
        print(f"\n  Error during optimization: {e}")
        sys.exit(1)

    if not validate_route(best_route, len(cities)):
        print("  Warning: route validation failed - not a valid permutation!")

    show_results(best_route, best_dist, cities, elapsed)

    # --- Save visualizations ---
    os.makedirs(RESULTS_DIR, exist_ok=True)

    route_fig = os.path.join(RESULTS_DIR, "best_route.png")
    fitness_fig = os.path.join(RESULTS_DIR, "fitness_plot.png")

    plot_route(best_route, cities, save_path=route_fig)
    plot_fitness(ga.fitness_history, save_path=fitness_fig)

    print("  Done. Check the results/ folder for plots.\n")


if __name__ == "__main__":
    main()
