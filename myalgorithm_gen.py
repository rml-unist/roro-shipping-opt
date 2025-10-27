import time
import json
import random
import numpy as np
import networkx as nx
from copy import deepcopy
from itertools import islice
from collections import deque, Counter
import multiprocessing
import os
import util
import heapq

class PortOptimizerGA:
    # GA Parameters
    POPULATION_SIZE = 40
    ELITE_SIZE = 5
    TOURNAMENT_SIZE = 5
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.3
    MAX_GENERATIONS = 1000
    STAGNATION_LIMIT = 100

    def __init__(self, prob_info):
        self.start_time = time.time()
        self.prob_info = prob_info

        # Problem Data
        self.N = prob_info['N']
        self.E = set(map(tuple, prob_info['E']))
        self.K = prob_info['K']
        self.P = prob_info['P']
        self.F = prob_info['F']
        self.LB = prob_info['LB']

        # Graph and Path Pre-computation
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.N))
        self.G.add_edges_from(self.E)
        
        self.shortest_distances = np.zeros(self.N, dtype=int)
        max_num_paths = 5
        self.shortest_paths = [[] for _ in range(self.N)]
        for i in range(1, self.N):
            try:
                paths = list(islice(nx.shortest_simple_paths(self.G, 0, i), max_num_paths))
                if paths:
                    self.shortest_paths[i] = paths
                    self.shortest_distances[i] = len(paths[0]) - 1
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                self.shortest_distances[i] = float('inf')
                
        self.route_heatmaps = {d: {} for d in range(self.P)}
        self.adj = {u: list(self.G.neighbors(u)) for u in range(self.N)}
        
        try:
            bc = nx.betweenness_centrality(self.G, normalized=True)
            self.node_centrality = np.array([bc.get(i, 0.0) for i in range(self.N)], dtype=float)
        except Exception:
            self.node_centrality = np.zeros(self.N, dtype=float)

        self.population = []
        self.fitness_cache = {}

    def _r1_candidates_same_dest(self, reachable_nodes, occ, dest):
        reachable_set = set(reachable_nodes)
        seeds = []
        for n in range(self.N):
            k = occ[n]
            if k != -1 and self.K[k][0][1] == dest:
                seeds.append(n)
        if not seeds:
            return []

        cand = set()
        for u in seeds:
            for v in self.G.neighbors(u):
                if v != 0 and occ[v] == -1 and v in reachable_set:
                    cand.add(v)
        return list(cand)
    
    def _reachability_safe_candidates(self, cands, occ, k):
        bfs_out = util.bfs(self.G, occ)
        if not bfs_out or bfs_out[0] is None:
            return []
        before = set(bfs_out[0])

        safe = []
        for n in cands:
            if n not in before:
                continue
            occ_tmp = occ.copy()
            occ_tmp[n] = k
            bfs_after = util.bfs(self.G, occ_tmp)
            if not bfs_after or bfs_after[0] is None:
                continue
            after = set(bfs_after[0])

            if (before - {n}).issubset(after):
                safe.append(n)
        return safe
    
    def min_blocking_path(self, G_adj, node_allocations, target, gate=0):
        INF = (10**9, 10**9)
        n_nodes = len(G_adj)
        dist = {u: INF for u in range(n_nodes)}
        prev = {u: None for u in range(n_nodes)}

        dist[gate] = (0, 0)
        pq = [(0, 0, gate)]

        while pq:
            b, d, u = heapq.heappop(pq)
            if (b, d) != dist[u]:
                continue
            if u == target:
                break
            for v in G_adj[u]:
                occupied = (node_allocations[v] != -1) and (v != target)
                nb = b + (1 if occupied else 0)
                nd = d + 1
                if (nb, nd) < dist[v]:
                    dist[v] = (nb, nd)
                    prev[v] = u
                    heapq.heappush(pq, (nb, nd, v))

        path = []
        cur = target
        if dist[target] == INF:
            return [], []
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()

        blocking = [x for x in path[1:-1] if node_allocations[x] != -1]
        return path, blocking

    def _loading_heuristic(self, p, node_allocations, rehandling_demands, SA=False):
        K_load = {idx: r for idx, ((o,d),r) in enumerate(self.K) if o == p}

        if len(rehandling_demands) > 0:
            for k in rehandling_demands:
                if k in K_load:
                    K_load[k] += 1
                else:
                    K_load[k] = 1

        route_list = []
        last_rehandling_demands = []
        total_loading_demands = sum([r for k,r in K_load.items()])

        reachable_nodes, reachable_node_distances = util.bfs(self.G, node_allocations)
        available_nodes = util.get_available_nodes(node_allocations)

        if len(available_nodes) < total_loading_demands:
            return None, None

        if len(reachable_nodes) < total_loading_demands:
            available_but_not_reachable = [n for n in available_nodes if n not in reachable_nodes]

            while len(reachable_nodes) < total_loading_demands:
                if len(available_but_not_reachable) == 0:
                    return None, None
                
                n = available_but_not_reachable.pop(0)
                path, blocks = self.min_blocking_path(self.adj, node_allocations, n, gate=0)

                for idx, i in enumerate(path[:-1]):
                    if node_allocations[i] != -1:
                        k_block = node_allocations[i]
                        last_rehandling_demands.append(k_block)
                        cut = []
                        for x in path:
                            cut.append(x)
                            if x == i:
                                break
                        route_list.append((cut[::-1], k_block))
                        node_allocations[i] = -1
                        total_loading_demands += 1

                reachable_nodes, reachable_node_distances = util.bfs(self.G, node_allocations)

        for k in last_rehandling_demands:
            if k in K_load:
                K_load[k] += 1
            else:
                K_load[k] = 1

        if total_loading_demands > 0:
            all_demands_to_load = [k for k, r in K_load.items() for _ in range(r)]
            
            long_term_vehicles = [k for k in all_demands_to_load if self.K[k][0][1] > p + 1]
            next_port_vehicles = [k for k in all_demands_to_load if self.K[k][0][1] == p + 1]

            long_term_vehicles.sort(key=lambda k: self.K[k][0][1], reverse=True)
            reserve_cnt = len(next_port_vehicles)

            for k in long_term_vehicles:
                d_target = self.K[k][0][1]
                reachable_nodes, _ = util.bfs(self.G, node_allocations)
                avail_now = [n for n in reachable_nodes if n != 0 and node_allocations[n] == -1]
                reserved_nodes = set(sorted(avail_now, key=lambda n: self.shortest_distances[n])[:reserve_cnt])
                
                r1_cands = self._r1_candidates_same_dest(reachable_nodes, node_allocations, d_target) or []
                r1_cands = [n for n in r1_cands if n not in reserved_nodes]
                r1_safe = self._reachability_safe_candidates(r1_cands, node_allocations, k)

                if r1_safe:
                    n_sel = max(r1_safe, key=lambda n: self.shortest_distances[n])
                else:
                    all_cands = [n for n in reachable_nodes if n not in reserved_nodes]
                    if not all_cands:
                        continue

                    if SA:
                        safe_cands = self._reachability_safe_candidates(all_cands, node_allocations, k)
                        if safe_cands:
                            n_sel = random.choice(safe_cands)
                        else:
                            n_sel = random.choice(all_cands) if all_cands else None
                    else:
                        safe_cands = self._reachability_safe_candidates(all_cands, node_allocations, k)
                        cands_to_score = safe_cands if safe_cands else all_cands
                        
                        if not cands_to_score:
                            continue
                        
                        n_sel = max(cands_to_score, key=lambda n: self.shortest_distances[n])

                if n_sel is None:
                    continue

                _, prev_nodes = util.dijkstra(self.G, node_allocations)
                path = util.path_backtracking(prev_nodes, 0, n_sel)
                node_allocations[n_sel] = k
                route_list.append((path, k))

            reachable_nodes, reachable_node_distances = util.bfs(self.G, node_allocations)
            loading_nodes = reachable_nodes[:len(next_port_vehicles)][::-1]
            distances, previous_nodes = util.dijkstra(self.G, node_allocations)

            for n, k in zip(loading_nodes, next_port_vehicles):
                node_allocations[n] = k
                path = util.path_backtracking(previous_nodes, 0, n)
                route_list.append((path, k))

        return route_list, node_allocations

    def _unloading_heuristic(self, p, node_allocations):
        K_unload = {idx: r for idx, ((o, d), r) in enumerate(self.K) if d == p}
        route_list, rehandling_demands = [], []

        for k in K_unload.keys():
            for n in range(self.N):
                if node_allocations[n] == k:
                    path, blocks = self.min_blocking_path(self.adj, node_allocations, n, gate=0)
                    
                    if not blocks:
                        route_list.append((path[::-1], k))
                        node_allocations[n] = -1
                    else:
                        for bn in blocks:
                            cut = []
                            for x in path:
                                cut.append(x)
                                if x == bn:
                                    break
                            route_list.append((cut[::-1], node_allocations[bn]))
                            if node_allocations[bn] not in K_unload:
                                rehandling_demands.append(node_allocations[bn])
                            node_allocations[bn] = -1
                        
                        route_list.append((path[::-1], k))
                        node_allocations[n] = -1

        return route_list, rehandling_demands, node_allocations

    def _apply_routes_to_occupancy(self, routes, occ):
        for path, k in routes:
            if not path:
                continue
            if path[0] == 0 and path[-1] != 0:
                occ[path[-1]] = k
            elif path[-1] == 0 and path[0] != 0:
                occ[path[0]] = -1
            else:
                occ[path[0]] = -1
                occ[path[-1]] = k
        return occ

    def _create_initial_solution(self):
        node_allocations = np.full(self.N, -1, dtype=int)
        solution = {p: [] for p in range(self.P)}

        for p in range(self.P):
            rehandling_demands = []
            if p > 0:
                unload_routes, rehandling_demands, node_allocations = self._unloading_heuristic(p, node_allocations)
                solution[p].extend(unload_routes)

            if p < self.P - 1:
                load_routes, node_allocations = self._loading_heuristic(p, node_allocations, rehandling_demands)
                if load_routes is None:
                    return None, float('inf')
                solution[p].extend(load_routes)

        result = util.check_feasibility(self.prob_info, solution)
        cost = result['obj'] if result['feasible'] else float('inf')
        return solution, cost

    def _evaluate_fitness(self, solution):
        if solution is None:
            return float('inf')
        result = util.check_feasibility(self.prob_info, solution)
        return result['obj'] if result['feasible'] else float('inf')

    def _create_random_solution(self):
        node_allocations = np.full(self.N, -1, dtype=int)
        solution = {p: [] for p in range(self.P)}
        
        for p in range(self.P):
            rehandling_demands = []
            if p > 0:
                unload_routes, rehandling_demands, node_allocations = self._unloading_heuristic(p, node_allocations)
                solution[p].extend(unload_routes)
            
            if p < self.P - 1:
                load_routes, node_allocations = self._loading_heuristic(p, node_allocations, rehandling_demands, SA=True)
                if load_routes is None:
                    return None
                solution[p].extend(load_routes)
        
        return solution

    def _initialize_population(self):
        population = []
        initial_sol, _ = self._create_initial_solution()
        if initial_sol:
            population.append(initial_sol)
        
        attempts = 0
        while len(population) < self.POPULATION_SIZE and attempts < self.POPULATION_SIZE * 3:
            new_sol = self._create_random_solution()
            if new_sol is not None:
                population.append(new_sol)
            attempts += 1
        
        return population

    def _tournament_selection(self, population, fitness_values):
        tournament_size = min(self.TOURNAMENT_SIZE, len(population))
        tournament = random.sample(list(zip(population, fitness_values)), tournament_size)
        winner = min(tournament, key=lambda x: x[1])
        return winner[0]

    def _crossover(self, parent1, parent2):
        if random.random() > self.CROSSOVER_RATE or self.P <= 2:
            return deepcopy(parent1), deepcopy(parent2)
        
        crossover_point = random.randint(1, self.P - 2)
        
        child1 = self._create_child_from_crossover(parent1, crossover_point, True)
        child2 = self._create_child_from_crossover(parent2, crossover_point, False)
        
        return child1, child2

    def _create_child_from_crossover(self, parent, crossover_point, use_first_part):
        child = {p: [] for p in range(self.P)}
        
        if use_first_part:
            for p in range(crossover_point):
                child[p] = list(parent.get(p, []))
        
        occ = np.full(self.N, -1, dtype=int)
        for p in range(crossover_point if use_first_part else 0):
            routes = child.get(p, [])
            occ = self._apply_routes_to_occupancy(routes, occ)
        
        start_p = crossover_point if use_first_part else 0
        for p in range(start_p, self.P):
            rehandling_demands = []
            if p > 0:
                unload_routes, rehandling_demands, occ = self._unloading_heuristic(p, occ)
                if p not in child:
                    child[p] = []
                child[p].extend(unload_routes)
            
            if p < self.P - 1:
                load_routes, occ = self._loading_heuristic(p, occ, rehandling_demands, SA=True)
                if load_routes is None:
                    return None
                child[p].extend(load_routes)
        
        return child

    def _mutation(self, solution):
        if random.random() > self.MUTATION_RATE:
            return solution
        
        mutate_port = random.randint(0, self.P - 2)
        mutated = {p: list(solution.get(p, [])) for p in range(self.P)}
        
        occ = np.full(self.N, -1, dtype=int)
        for p in range(mutate_port):
            routes = mutated.get(p, [])
            occ = self._apply_routes_to_occupancy(routes, occ)
        
        for p in range(mutate_port, self.P):
            rehandling_demands = []
            if p > 0:
                unload_routes, rehandling_demands, occ = self._unloading_heuristic(p, occ)
                mutated[p] = unload_routes
            
            if p < self.P - 1:
                load_routes, occ = self._loading_heuristic(p, occ, rehandling_demands, SA=True)
                if load_routes is None:
                    return solution
                mutated[p].extend(load_routes)
        
        return mutated

    def solve_with_ga(self, timelimit=60, shared_data=None, lock=None):
        ga_start_time = time.time()
        
        population = self._initialize_population()
        if not population:
            return None, float('inf')
        
        best_solution = None
        best_fitness = float('inf')
        stagnation_count = 0
        generation = 0
        
        while (time.time() - ga_start_time) < timelimit and generation < self.MAX_GENERATIONS:
            generation += 1
            
            fitness_values = [self._evaluate_fitness(sol) for sol in population]
            
            min_idx = np.argmin(fitness_values)
            if fitness_values[min_idx] < best_fitness:
                best_fitness = fitness_values[min_idx]
                best_solution = deepcopy(population[min_idx])
                stagnation_count = 0
                
                if shared_data and lock:
                    with lock:
                        if best_fitness < shared_data['best_cost']:
                            shared_data['best_cost'] = best_fitness
                            shared_data['best_solution'] = deepcopy(best_solution)
            else:
                stagnation_count += 1
            
            if stagnation_count >= self.STAGNATION_LIMIT:
                population = self._initialize_population()
                stagnation_count = 0
                continue
            
            sorted_pop = sorted(zip(population, fitness_values), key=lambda x: x[1])
            elite = [sol for sol, _ in sorted_pop[:self.ELITE_SIZE]]
            
            new_population = elite.copy()
            
            while len(new_population) < self.POPULATION_SIZE:
                parent1 = self._tournament_selection(population, fitness_values)
                parent2 = self._tournament_selection(population, fitness_values)
                
                child1, child2 = self._crossover(parent1, parent2)
                
                if child1:
                    child1 = self._mutation(child1)
                    new_population.append(child1)
                
                if child2 and len(new_population) < self.POPULATION_SIZE:
                    child2 = self._mutation(child2)
                    new_population.append(child2)
            
            population = new_population[:self.POPULATION_SIZE]
            
            if shared_data and lock and generation % 20 == 0:
                with lock:
                    if shared_data.get('best_solution'):
                        population[random.randint(0, len(population)-1)] = deepcopy(shared_data['best_solution'])
        
        return best_solution, best_fitness

def run_single_algorithm_ga(prob_info, timelimit, shared_data, lock):
    optimizer = PortOptimizerGA(prob_info)
    solution, cost = optimizer.solve_with_ga(timelimit, shared_data, lock)
    
    if shared_data and lock:
        with lock:
            if cost < shared_data['best_cost']:
                shared_data['best_cost'] = cost
                shared_data['best_solution'] = solution

def algorithm(prob_info, timelimit=60):
    NUM_PARALLEL_RUNS = 4
    
    with multiprocessing.Manager() as manager:
        shared_data = manager.dict()
        shared_data['best_cost'] = float('inf')
        shared_data['best_solution'] = None
        lock = manager.Lock()
        
        args = [(prob_info, timelimit, shared_data, lock) for _ in range(NUM_PARALLEL_RUNS)]
        
        with multiprocessing.Pool(processes=NUM_PARALLEL_RUNS) as pool:
            pool.starmap(run_single_algorithm_ga, args)
        
        return shared_data['best_solution']

if __name__ == "__main__":
    import json, os, sys, csv, jsbeautifier

    NUM_PROBLEMS = 10
    PROBLEM_DIR = "stage2_exercise_problems"
    OUTPUT_CSV_FILE = "results_ga.csv"
    TIMELIMIT_PER_PROBLEM = 60

    results_for_csv = []

    print(f"--- Starting GA Batch Processing for {NUM_PROBLEMS} problems ---")

    for i in range(1, NUM_PROBLEMS + 1):
        prob_name = f"prob{i}"
        prob_file = os.path.join(PROBLEM_DIR, f"{prob_name}.json")
        
        print(f"\n{'='*20} Running {prob_name} {'='*20}")

        if not os.path.exists(prob_file):
            print(f"File not found: {prob_file}. Skipping.")
            results_for_csv.append([prob_name, "File Not Found"])
            continue

        try:
            with open(prob_file, 'r') as f:
                prob_info = json.load(f)

            solution = algorithm(prob_info, TIMELIMIT_PER_PROBLEM)

            checked_solution = util.check_feasibility(prob_info, solution)
            
            obj_value = 'Infeasible'
            if checked_solution.get('feasible', False):
                obj_value = checked_solution.get('obj', 'N/A')
                print(f"✅ {prob_name} successful. Objective: {obj_value}")
            else:
                print(f"❌ {prob_name} resulted in an infeasible solution.")

            results_for_csv.append([prob_name, obj_value])

        except Exception as e:
            import traceback
            print(f"❌ An exception occurred while running {prob_name}: {repr(e)}")
            traceback.print_exc()
            results_for_csv.append([prob_name, 'Error'])

    print(f"\n--- Batch processing complete. Writing results to {OUTPUT_CSV_FILE} ---")
    try:
        with open(OUTPUT_CSV_FILE, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['problem_name', 'objective_cost'])
            writer.writerows(results_for_csv)
        print(f"✅ Successfully saved results to {OUTPUT_CSV_FILE}")
    except Exception as e:
        print(f"❌ Failed to write CSV file: {repr(e)}")

    sys.exit(0)