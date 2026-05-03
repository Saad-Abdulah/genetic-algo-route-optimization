import os
import time
import random

from route_utils import load_cities
from genetic_algorithm import GeneticAlgorithm


DEFAULT_CASES = [
    ("data/cities_20.csv", 100, 500, 0.015, 5, 2, 42),
    ("data/cities_25.csv", 120, 700, 0.02, 5, 2, 42),
    ("data/cities.csv", 200, 1500, 0.02, 5, 4, 42),
]


def run_case(csv_path, pop_size, generations, mutation_rate, tournament_size, elite_count, seed):
    cities = load_cities(csv_path)
    random.seed(seed)
    ga = GeneticAlgorithm(
        cities=cities,
        pop_size=pop_size,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
        elite_count=elite_count,
    )
    start = time.time()
    _, best_distance = ga.run(generations=generations, verbose=False)
    elapsed = time.time() - start
    return len(cities), best_distance, elapsed


def main():
    print("=" * 84)
    print("GA BENCHMARK REPORT")
    print("=" * 84)
    print(f"{'Dataset':<22} {'Cities':>6} {'Distance (km)':>18} {'Time (s)':>10}")
    print("-" * 84)
    for case in DEFAULT_CASES:
        dataset, pop_size, generations, mutation_rate, tournament_size, elite_count, seed = case
        n, dist, elapsed = run_case(
            dataset,
            pop_size,
            generations,
            mutation_rate,
            tournament_size,
            elite_count,
            seed,
        )
        print(f"{dataset:<22} {n:>6} {dist:>18,.2f} {elapsed:>10.2f}")
    print("-" * 84)


if __name__ == "__main__":
    main()
