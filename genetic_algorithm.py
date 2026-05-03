import random

from route_utils import haversine


class GeneticAlgorithm:
    """
    Genetic Algorithm for solving the Traveling Salesman Problem.

    Representation: each individual is a permutation of city indices (closed tour),
    or a permutation of middle cities when fixed_start and fixed_end are set
    (open path visiting every city once from start to end).

    Fitness: inverse of total route distance (shorter = fitter).
    Selection: tournament selection.
    Crossover: Order Crossover (OX1) - preserves relative city ordering.
    Mutation: swap mutation - randomly swaps two cities in the route.
    Elitism: top individuals carried to next generation unchanged.
    """

    def __init__(
        self,
        cities,
        pop_size=100,
        mutation_rate=0.015,
        tournament_size=5,
        elite_count=2,
        fixed_start=None,
        fixed_end=None,
    ):
        if pop_size <= 0:
            raise ValueError("Population size must be positive.")
        if not (0.0 < mutation_rate <= 1.0):
            raise ValueError("Mutation rate must be in range (0, 1].")
        if tournament_size <= 0:
            raise ValueError("Tournament size must be positive.")
        if tournament_size > pop_size:
            raise ValueError("Tournament size cannot exceed population size.")
        if elite_count < 0:
            raise ValueError("Elite count cannot be negative.")
        if elite_count >= pop_size:
            raise ValueError("Elite count must be smaller than population size.")

        self.cities = cities
        self.num_cities = len(cities)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_count = elite_count

        self.fixed_start = fixed_start
        self.fixed_end = fixed_end
        self._middle = None
        self._middle_n = None

        if fixed_start is not None or fixed_end is not None:
            if fixed_start is None or fixed_end is None:
                raise ValueError("fixed_start and fixed_end must both be set or both None.")
            if fixed_start == fixed_end:
                raise ValueError("fixed_start and fixed_end must be different city indices.")
            if not (0 <= fixed_start < self.num_cities and 0 <= fixed_end < self.num_cities):
                raise ValueError("fixed_start / fixed_end out of range.")
            self._middle = [i for i in range(self.num_cities) if i not in (fixed_start, fixed_end)]
            self._middle_n = len(self._middle)
            if self.num_cities < 2:
                raise ValueError("Need at least two cities for fixed endpoints.")

        self._dist = self._build_dist_matrix()

        self.population = []
        self.fitness_history = []

    def _build_dist_matrix(self):
        n = self.num_cities
        d = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                val = haversine(self.cities[i], self.cities[j])
                d[i][j] = val
                d[j][i] = val
        return d

    def _expand(self, chromosome):
        """Map internal chromosome to full city index list."""
        if self._middle is None:
            return chromosome
        return [self.fixed_start] + chromosome + [self.fixed_end]

    def _route_distance_fast(self, chromosome):
        """Total distance: closed tour or open path depending on mode."""
        route = chromosome if self._middle is None else self._expand(chromosome)
        n = len(route)
        d = self._dist
        if self._middle is None:
            total = 0.0
            for i in range(n):
                total += d[route[i]][route[(i + 1) % n]]
            return total
        total = 0.0
        for i in range(n - 1):
            total += d[route[i]][route[i + 1]]
        return total

    def _random_route(self):
        if self._middle is None:
            route = list(range(self.num_cities))
            random.shuffle(route)
            return route
        mid = self._middle[:]
        random.shuffle(mid)
        return mid

    def _init_population(self):
        self.population = [self._random_route() for _ in range(self.pop_size)]

    def _evaluate_population(self):
        distances = [self._route_distance_fast(r) for r in self.population]
        fitnesses = [1.0 / dist for dist in distances]
        ranked_idx = sorted(range(len(self.population)), key=lambda i: distances[i])
        return fitnesses, distances, ranked_idx

    def _tournament_select(self, fitnesses):
        contestants = random.sample(range(self.pop_size), self.tournament_size)
        best_i = max(contestants, key=lambda i: fitnesses[i])
        return self.population[best_i]

    def _order_crossover(self, parent1, parent2):
        if self._middle is None:
            size = self.num_cities
            start, end = sorted(random.sample(range(size), 2))
            child = [None] * size
            child[start : end + 1] = parent1[start : end + 1]
            inherited = set(child[start : end + 1])
            pos = (end + 1) % size
            for city in parent2:
                if city not in inherited:
                    child[pos] = city
                    pos = (pos + 1) % size
            return child

        size = self._middle_n
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start : end + 1] = parent1[start : end + 1]
        inherited = set(child[start : end + 1])
        pos = (end + 1) % size
        for gene in parent2:
            if gene not in inherited:
                child[pos] = gene
                pos = (pos + 1) % size
        return child

    def _swap_mutate(self, route):
        size = self.num_cities if self._middle is None else self._middle_n
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(size), 2)
            route[i], route[j] = route[j], route[i]
        return route

    def run(self, generations=500, verbose=True):
        """
        Execute the GA for a given number of generations.
        Returns the best route found and its total distance (full route indices).
        """
        best_route, best_distance, _, _ = self._run_loop(
            generations=generations,
            verbose=verbose,
            snapshot_stride=None,
            alt_tolerance=None,
            max_alternatives=0,
        )
        return best_route, best_distance

    def run_directions_mode(
        self,
        generations=500,
        verbose=False,
        snapshot_stride=None,
        alt_tolerance=0.12,
        max_routes=6,
    ):
        """
        Fixed endpoints mode: returns best full route, distance, history,
        evolution snapshots (gen, route, dist), and diverse near-best routes.
        """
        if self._middle is None:
            raise ValueError("Directions mode requires fixed_start and fixed_end when creating GeneticAlgorithm.")
        return self._run_loop(
            generations=generations,
            verbose=verbose,
            snapshot_stride=snapshot_stride,
            alt_tolerance=alt_tolerance,
            max_alternatives=max_routes,
        )

    def _run_loop(
        self,
        generations,
        verbose,
        snapshot_stride,
        alt_tolerance,
        max_alternatives,
    ):
        self._init_population()
        self.fitness_history = []

        best_route_full = None
        best_distance = float("inf")
        snapshots = []

        for gen in range(generations):
            fitnesses, distances, ranked_idx = self._evaluate_population()

            top_i = ranked_idx[0]
            top_chrom = self.population[top_i]
            top_dist = distances[top_i]
            top_full = top_chrom[:] if self._middle is None else self._expand(top_chrom)

            if top_dist < best_distance:
                best_distance = top_dist
                best_route_full = top_full[:]

            self.fitness_history.append(best_distance)

            if verbose and (gen % 50 == 0 or gen == generations - 1):
                print(f"  Gen {gen:>4d}  |  Best Distance: {best_distance:,.2f} km")

            if snapshot_stride is not None and snapshot_stride > 0:
                if gen % snapshot_stride == 0 or gen == generations - 1:
                    if best_route_full is not None:
                        snapshots.append((gen, best_route_full[:], best_distance))

            next_gen = [self.population[i][:] for i in ranked_idx[: self.elite_count]]

            while len(next_gen) < self.pop_size:
                p1 = self._tournament_select(fitnesses)
                p2 = self._tournament_select(fitnesses)
                child = self._order_crossover(p1, p2)
                child = self._swap_mutate(child)
                next_gen.append(child)

            self.population = next_gen

        if best_route_full is None:
            raise RuntimeError("GA finished without a best route.")

        alternatives = []
        if alt_tolerance is not None and max_alternatives > 0:
            fitnesses, distances, ranked_idx = self._evaluate_population()
            seen = set()
            cap = best_distance * (1.0 + float(alt_tolerance))
            for i in ranked_idx:
                chrom = self.population[i]
                full = chrom[:] if self._middle is None else self._expand(chrom)
                key = tuple(full)
                if key in seen:
                    continue
                seen.add(key)
                d_i = distances[i]
                if d_i <= cap:
                    alternatives.append((full[:], d_i))
                if len(alternatives) >= max_alternatives:
                    break

        return best_route_full, best_distance, snapshots, alternatives


class CorridorPathGA:
    """
    Genetic algorithm for source-to-destination routes visiting a subset of cities.

    Candidate cities lie near the logical corridor (detour length vs direct).
    With road edges from CSV, legs must follow that graph; otherwise valid legs use
    k-nearest-neighbor connectivity so routes chain through nearby hubs.

    Chromosome: ordered list of intermediate city indices (subset of candidates).
    Initialization uses only random feasible constructions (random walks and random
    intermediate lists)—no separate shortest-path engine.

    ``run()`` samples distinct feasible paths across generations into a path archive;
    the UI filters that archive by tolerance and min/max counts without re-running.

    Connectivity of the road / kNN constraint graph is checked with **stochastic random walks**
    (same neighbor-choice idea as chromosome repair walks—not BFS/DFS and not an optimizer).
    """

    _BAD_FIT = 1e-12

    def __init__(
        self,
        cities,
        src_idx,
        dst_idx,
        pop_size=100,
        mutation_rate=0.22,
        tournament_size=5,
        elite_count=2,
        max_intermediate=None,
        corridor_ratio=1.42,
        k_nn=None,
        road_edges=None,
    ):
        if src_idx == dst_idx:
            raise ValueError("Source and destination must differ.")

        self.cities = cities
        self.n = len(cities)
        self.src_idx = src_idx
        self.dst_idx = dst_idx
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_count = elite_count
        cap = self.n - 2
        if max_intermediate is None:
            self.max_intermediate = max(1, cap)
        else:
            self.max_intermediate = max(1, min(int(max_intermediate), cap))

        if pop_size <= 0 or elite_count >= pop_size:
            raise ValueError("Invalid population or elite settings.")

        self._dist = self._build_dist_matrix()

        direct = self._dist[src_idx][dst_idx]

        self.pool = []
        for i in range(self.n):
            if i in (src_idx, dst_idx):
                continue
            detour = self._dist[src_idx][i] + self._dist[i][dst_idx]
            if detour <= corridor_ratio * direct + 1e-9:
                self.pool.append(i)

        ratio = corridor_ratio
        while len(self.pool) < min(8, self.n - 2) and ratio < 3.0:
            ratio += 0.25
            self.pool = []
            for i in range(self.n):
                if i in (src_idx, dst_idx):
                    continue
                detour = self._dist[src_idx][i] + self._dist[i][dst_idx]
                if detour <= ratio * direct + 1e-9:
                    self.pool.append(i)

        if not self.pool:
            self.pool = [i for i in range(self.n) if i not in (src_idx, dst_idx)]

        self.pool_set = set(self.pool)

        self.road_mode = bool(road_edges)
        if self.road_mode:
            self._canonical_edges = frozenset(road_edges)
            self._road_neighbors = [set() for _ in range(self.n)]
            for i, j in self._canonical_edges:
                self._road_neighbors[i].add(j)
                self._road_neighbors[j].add(i)
            self.k_nn_used = None
            self._knn_sets = None
            if not self._road_pair_reachable_by_random_walks():
                raise ValueError(
                    "Road network appears to admit no route between the selected source and destination "
                    "(random-walk probe gave up)."
                )
        else:
            k_base = k_nn if k_nn is not None else min(self.n - 1, max(8, self.n // 4))
            self.k_nn_used = self._resolve_knn_connectivity(k_base)
            self._road_neighbors = None
            self._knn_sets = None

        self.population = []
        self.fitness_history = []
        self.evolution_snapshots = []

    def _build_dist_matrix(self):
        n = self.n
        d = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                v = haversine(self.cities[i], self.cities[j])
                d[i][j] = v
                d[j][i] = v
        return d

    def _compute_knn(self, k):
        knn = []
        for i in range(self.n):
            others = [(self._dist[i][j], j) for j in range(self.n) if j != i]
            others.sort(key=lambda x: x[0])
            knn.append({j for _, j in others[:k]})
        return knn

    def _single_road_probe_walk(self):
        """One random attempt to walk src→dst on the road graph (feasibility probe, not optimization)."""
        path = [self.src_idx]
        cap = max(120, min(600, 14 * self.n))
        for _ in range(cap):
            cur = path[-1]
            if cur == self.dst_idx:
                return True
            nbrs = [v for v in self._road_neighbors[cur] if v not in path]
            if not nbrs:
                break
            if self.dst_idx in nbrs and (len(nbrs) == 1 or random.random() < 0.45):
                return True
            choices = [v for v in nbrs if v != self.dst_idx]
            if not choices:
                return self.dst_idx in nbrs
            path.append(random.choice(choices))
        cur = path[-1]
        return cur == self.dst_idx or self.dst_idx in self._road_neighbors[cur]

    def _road_pair_reachable_by_random_walks(self):
        base = max(400, 55 * self.n)
        for scale in (1, 2, 4, 8):
            for _ in range(base * scale):
                if self._single_road_probe_walk():
                    return True
        return False

    def _single_knn_probe_walk(self, knn_sets):
        """Random walk on kNN adjacency to hit dst (connectivity probe only)."""
        path = [self.src_idx]
        cap = max(90, min(500, 12 * self.n))
        for _ in range(cap):
            cur = path[-1]
            if cur == self.dst_idx:
                return True
            nbrs = list(knn_sets[cur])
            if self.dst_idx in nbrs and (
                random.random() < 0.38 or len(path) >= min(40, 4 * self.n)
            ):
                return True
            opts = [v for v in nbrs if v not in path and v != self.dst_idx]
            if not opts:
                return self.dst_idx in nbrs
            path.append(random.choice(opts))
        cur = path[-1]
        return cur == self.dst_idx or self.dst_idx in knn_sets[cur]

    def _knn_pair_reachable_by_random_walks(self, knn_sets):
        base = max(280, 38 * self.n)
        for scale in (1, 2, 4):
            for _ in range(base * scale):
                if self._single_knn_probe_walk(knn_sets):
                    return True
        return False

    def _resolve_knn_connectivity(self, k_start):
        k = max(2, min(k_start, self.n - 1))
        while k < self.n:
            knn_sets = self._compute_knn(k)
            if self._knn_pair_reachable_by_random_walks(knn_sets):
                return k
            k += 1
        return self.n - 1

    def _edge_ok(self, a, b):
        if self.road_mode:
            i, j = (a, b) if a <= b else (b, a)
            return (i, j) in self._canonical_edges
        knn_sets = self._knn_sets
        if b == self.dst_idx:
            return self.dst_idx in knn_sets[a]
        if a == self.src_idx:
            if b in knn_sets[self.src_idx]:
                return True
            return b in self.pool_set
        return b in knn_sets[a]

    def _path_valid(self, path):
        if len(path) < 2:
            return False
        for i in range(len(path) - 1):
            if not self._edge_ok(path[i], path[i + 1]):
                return False
        return True

    def _path_distance(self, path):
        total = 0.0
        for i in range(len(path) - 1):
            total += self._dist[path[i]][path[i + 1]]
        return total

    def _normalize_mids(self, mids):
        out = []
        seen = set([self.src_idx, self.dst_idx])
        for x in mids:
            if x == self.src_idx or x == self.dst_idx:
                continue
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
            if len(out) >= self.max_intermediate:
                break
        return out

    def _to_path(self, mids):
        return [self.src_idx] + self._normalize_mids(mids) + [self.dst_idx]

    def _random_walk_chromosome(self):
        if self.road_mode:
            path = [self.src_idx]
            for _ in range(120):
                cur = path[-1]
                if cur == self.dst_idx:
                    break
                nbrs = [v for v in self._road_neighbors[cur] if v not in path]
                if not nbrs:
                    break
                if self.dst_idx in nbrs and (
                    len(nbrs) == 1 or random.random() < 0.28
                ):
                    path.append(self.dst_idx)
                    break
                choices = [v for v in nbrs if v != self.dst_idx]
                pool_choices = [v for v in choices if v in self.pool_set]
                pick_from = pool_choices if pool_choices else choices
                if not pick_from:
                    if self.dst_idx in nbrs:
                        path.append(self.dst_idx)
                    break
                path.append(random.choice(pick_from))
            if (
                path[-1] != self.dst_idx
                and self.dst_idx in self._road_neighbors[path[-1]]
            ):
                path.append(self.dst_idx)
            if path[-1] != self.dst_idx:
                return None
            mids = path[1:-1]
            mids = self._normalize_mids(mids)
            return mids if self._path_valid(self._to_path(mids)) else None

        knn_sets = self._knn_sets
        path = [self.src_idx]
        for _ in range(80):
            cur = path[-1]
            if self.dst_idx in knn_sets[cur] and random.random() < 0.22:
                path.append(self.dst_idx)
                break
            opts = []
            for v in knn_sets[cur]:
                if v == self.dst_idx:
                    continue
                if v not in path:
                    opts.append(v)
            pool_opts = [v for v in opts if v in self.pool_set]
            pick_from = pool_opts if pool_opts else opts
            if not pick_from:
                break
            path.append(random.choice(pick_from))
        if path[-1] != self.dst_idx and self.dst_idx in knn_sets[path[-1]]:
            path.append(self.dst_idx)
        if path[-1] != self.dst_idx:
            return None
        mids = path[1:-1]
        mids = self._normalize_mids(mids)
        return mids if self._path_valid(self._to_path(mids)) else None

    def _random_list_chromosome(self):
        if not self.pool:
            return None
        k_take = random.randint(1, min(len(self.pool), self.max_intermediate))
        mids = random.sample(self.pool, k_take)
        random.shuffle(mids)
        mids = self._normalize_mids(mids)
        path = self._to_path(mids)
        return mids if self._path_valid(path) else None

    def _init_population(self):
        self.population = []
        direct_ok = self._path_valid([self.src_idx, self.dst_idx])
        if direct_ok:
            for _ in range(max(4, min(self.pop_size // 8, 28))):
                self.population.append([])

        attempts = 0
        max_attempts = self.pop_size * 120
        while len(self.population) < self.pop_size and attempts < max_attempts:
            attempts += 1
            if random.random() < 0.55:
                c = self._random_walk_chromosome()
            else:
                c = self._random_list_chromosome()
            if c is not None:
                self.population.append(c[:])

        template = None
        for _ in range(500):
            c = self._random_walk_chromosome()
            if c is not None:
                template = c[:]
                break

        while len(self.population) < self.pop_size:
            if template is not None:
                jitter = template[:]
                if jitter and random.random() < 0.5:
                    jitter.pop(random.randrange(len(jitter)))
                self.population.append(self._normalize_mids(jitter))
            elif self.population:
                self.population.append(self.population[-1][:])
            elif direct_ok:
                self.population.append([])
            else:
                self.population.append([])

    def _evaluate(self):
        distances = []
        fitnesses = []
        for mids in self.population:
            path = self._to_path(mids)
            if not self._path_valid(path):
                distances.append(float("inf"))
                fitnesses.append(self._BAD_FIT)
            else:
                d = self._path_distance(path)
                distances.append(d)
                fitnesses.append(1.0 / d)
        ranked = sorted(range(len(self.population)), key=lambda i: distances[i])
        return fitnesses, distances, ranked

    def _tournament_select(self, fitnesses):
        contestants = random.sample(range(self.pop_size), self.tournament_size)
        best_i = max(contestants, key=lambda i: fitnesses[i])
        return self.population[best_i][:]

    def _crossover(self, p1, p2):
        if not p1:
            return p2[:]
        if not p2:
            return p1[:]
        cut = random.randint(0, len(p1))
        seen = set()
        child = []
        for x in p1[:cut]:
            if x not in seen and x not in (self.src_idx, self.dst_idx):
                seen.add(x)
                child.append(x)
        for x in p2:
            if x not in seen and x not in (self.src_idx, self.dst_idx):
                seen.add(x)
                child.append(x)
            if len(child) >= self.max_intermediate:
                break
        return child

    def _mutate(self, mids):
        mids = mids[:]
        if random.random() < self.mutation_rate:
            if mids and random.random() < 0.45:
                mids.pop(random.randrange(len(mids)))
            elif len(self.pool) > 0 and len(mids) < self.max_intermediate:
                cand = random.choice(self.pool)
                if cand not in mids:
                    insert_at = random.randint(0, len(mids))
                    mids.insert(insert_at, cand)
            elif len(mids) >= 2:
                i, j = random.sample(range(len(mids)), 2)
                mids[i], mids[j] = mids[j], mids[i]
        return self._normalize_mids(mids)

    def _merge_path_archive(self, archive, path, dist):
        """Keep best distance per unique path tuple."""
        key = tuple(path)
        prev = archive.get(key)
        if prev is None or dist < prev:
            archive[key] = float(dist)

    def _prune_path_archive(self, archive, max_entries):
        if len(archive) <= max_entries:
            return
        ranked = sorted(archive.items(), key=lambda kv: kv[1])
        keep = dict(ranked[:max_entries])
        archive.clear()
        archive.update(keep)

    def run(
        self,
        generations=400,
        verbose=False,
        path_archive_cap=320,
    ):
        if not self.road_mode:
            self._knn_sets = self._compute_knn(self.k_nn_used)
        self._init_population()

        best_path = None
        best_dist = float("inf")
        self.fitness_history = []
        self.evolution_snapshots = []
        path_archive = {}

        target_snaps = min(56, max(8, generations // 9))
        snapshot_stride = max(1, generations // target_snaps)

        for gen in range(generations):
            fitnesses, distances, ranked_idx = self._evaluate()

            top_i = ranked_idx[0]
            top_mids = self.population[top_i]
            path = self._to_path(top_mids)
            top_dist = distances[top_i]

            if top_dist < best_dist and top_dist < float("inf"):
                best_dist = top_dist
                best_path = path[:]

            pop_best_km = top_dist if top_dist < float("inf") else float("nan")
            elite_so_far_km = best_dist if best_dist < float("inf") else float("nan")
            valid_dists = [d for d in distances if d < float("inf")]
            mean_valid_km = (
                sum(valid_dists) / len(valid_dists) if valid_dists else float("nan")
            )
            self.fitness_history.append(
                {
                    "population_best_km": pop_best_km,
                    "elite_best_so_far_km": elite_so_far_km,
                    "population_mean_valid_km": mean_valid_km,
                }
            )

            scan_n = min(max(24, self.pop_size // 3), self.pop_size)
            for idx in ranked_idx[:scan_n]:
                pth = self._to_path(self.population[idx])
                if not self._path_valid(pth):
                    continue
                self._merge_path_archive(path_archive, pth, distances[idx])

            self._prune_path_archive(path_archive, path_archive_cap)

            if gen % snapshot_stride == 0 or gen == generations - 1:
                samples = []
                seen_paths = set()
                ranks_pool = list(ranked_idx)
                pick_positions = [0, 1, 2, 3]
                step = max(1, len(ranks_pool) // 9)
                for k in range(4, 14):
                    pick_positions.append(min(len(ranks_pool) - 1, k * step))
                sample_idx_order = []
                seen_i = set()
                for pos in pick_positions:
                    if pos < len(ranks_pool):
                        ix = ranks_pool[pos]
                        if ix not in seen_i:
                            seen_i.add(ix)
                            sample_idx_order.append(ix)
                for idx in sample_idx_order:
                    pth = self._to_path(self.population[idx])
                    if not self._path_valid(pth):
                        continue
                    key = tuple(pth)
                    if key in seen_paths:
                        continue
                    seen_paths.add(key)
                    samples.append([pth[:], float(distances[idx])])
                    if len(samples) >= 10:
                        break
                if len(samples) < 4:
                    for idx in ranks_pool:
                        pth = self._to_path(self.population[idx])
                        if not self._path_valid(pth):
                            continue
                        key = tuple(pth)
                        if key in seen_paths:
                            continue
                        seen_paths.add(key)
                        samples.append([pth[:], float(distances[idx])])
                        if len(samples) >= 10:
                            break
                self.evolution_snapshots.append(
                    {
                        "generation": gen,
                        "best_so_far_dist": float(best_dist)
                        if best_path is not None
                        else None,
                        "best_so_far_path": best_path[:] if best_path else None,
                        "population_samples": samples[:10],
                    }
                )

            if verbose and (gen % 80 == 0 or gen == generations - 1):
                bd = best_dist if best_path else float("nan")
                print(f"  Gen {gen:>4d}  |  Best: {bd:,.2f} km")

            next_gen = [self.population[i][:] for i in ranked_idx[: self.elite_count]]

            while len(next_gen) < self.pop_size:
                p1 = self._tournament_select(fitnesses)
                p2 = self._tournament_select(fitnesses)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                next_gen.append(child)

            self.population = next_gen

        if best_path is None:
            path_direct = [self.src_idx, self.dst_idx]
            if self._path_valid(path_direct):
                best_path = path_direct
                best_dist = self._path_distance(path_direct)
            else:
                raise RuntimeError(
                    "GA could not build a valid corridor path; try widening the corridor ratio "
                    "in the sidebar."
                )

        self._merge_path_archive(path_archive, best_path, best_dist)
        for mids in self.population:
            pth = self._to_path(mids)
            if self._path_valid(pth):
                self._merge_path_archive(path_archive, pth, self._path_distance(pth))

        ranked_routes = sorted(
            ([list(k), v] for k, v in path_archive.items()),
            key=lambda x: x[1],
        )
        route_archive = ranked_routes[:path_archive_cap]

        return best_path, best_dist, route_archive
