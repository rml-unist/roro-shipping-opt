"""
Roll-on/Roll-off Ship Problem Generator

Generates feasible optimization problems with varying difficulty levels.
"""

import json
import random
import os
import networkx as nx
from collections import defaultdict
import numpy as np


class ProblemGenerator:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_grid_graph(self, width, height, num_layers, num_ramps, hole_ratio=0.15):
        """
        Generate a multi-layer grid graph with holes and ramps.

        Args:
            width: Grid width
            height: Grid height
            num_layers: Number of layers (floors)
            num_ramps: Number of ramps connecting layers
            hole_ratio: Ratio of cells to remove (0.0 to 0.3)

        Returns:
            nodes: List of [coord, info] pairs
            edges: List of edge pairs
            node_id_map: Dict mapping coord tuple to node id
        """
        # Generate base grid cells for each layer
        all_cells = []
        for z in range(num_layers):
            for x in range(width):
                for y in range(height):
                    all_cells.append((x, y, z))

        # Gate is always at (0, 0, 0)
        gate = (0, 0, 0)

        # Remove random cells (holes), but keep gate and ensure connectivity
        num_holes = int(len(all_cells) * hole_ratio)
        removable_cells = [c for c in all_cells if c != gate]

        # Remove cells while maintaining connectivity
        holes = set()
        attempts = 0
        max_attempts = num_holes * 3

        while len(holes) < num_holes and attempts < max_attempts:
            attempts += 1
            candidate = random.choice(removable_cells)
            if candidate in holes:
                continue

            # Check if removing this cell keeps the layer connected
            test_holes = holes | {candidate}
            if self._is_layer_connected(width, height, candidate[2], test_holes, gate):
                holes.add(candidate)

        # Final cells
        cells = [c for c in all_cells if c not in holes]

        # Assign node IDs
        node_id = 0
        node_id_map = {}
        nodes = []

        # Gate first
        y_offset_per_layer = height + 0.8  # Visual offset for layers

        for coord in cells:
            x, y, z = coord
            node_id_map[coord] = node_id

            # Determine node type
            if coord == gate:
                node_type = "gate"
            else:
                node_type = "hold"

            pos_y = y + z * y_offset_per_layer

            nodes.append([
                list(coord),
                {
                    "pos": [x, pos_y],
                    "type": node_type,
                    "id": node_id,
                    "distance": abs(x) + abs(y) + z * 2  # Approximate distance
                }
            ])
            node_id += 1

        # Generate edges within each layer
        edges = []
        edge_set = set()

        for coord in cells:
            x, y, z = coord
            # Connect to adjacent cells in same layer
            neighbors = [
                (x+1, y, z), (x-1, y, z),
                (x, y+1, z), (x, y-1, z)
            ]
            for nb in neighbors:
                if nb in node_id_map:
                    u, v = node_id_map[coord], node_id_map[nb]
                    edge_key = (min(u, v), max(u, v))
                    if edge_key not in edge_set:
                        edge_set.add(edge_key)
                        edges.append([u, v])

        # Add ramps between layers
        ramp_candidates = []
        for coord in cells:
            x, y, z = coord
            if z < num_layers - 1:
                above = (x, y, z + 1)
                if above in node_id_map:
                    ramp_candidates.append((coord, above))

        # Select ramps
        num_ramps = min(num_ramps, len(ramp_candidates))
        if num_ramps > 0 and ramp_candidates:
            selected_ramps = random.sample(ramp_candidates, num_ramps)

            for lower, upper in selected_ramps:
                u, v = node_id_map[lower], node_id_map[upper]
                edge_key = (min(u, v), max(u, v))
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    edges.append([u, v])

                # Mark as ramp type
                for i, (coord, info) in enumerate(nodes):
                    if tuple(coord) == lower or tuple(coord) == upper:
                        if info["type"] != "gate":
                            nodes[i][1]["type"] = "ramp"

        # Ensure all nodes are reachable from gate
        G = nx.Graph()
        G.add_nodes_from(range(len(nodes)))
        G.add_edges_from([tuple(e) for e in edges])

        # Find unreachable nodes and remove them
        if 0 in G:
            reachable = set(nx.single_source_shortest_path_length(G, 0).keys())
            unreachable = set(range(len(nodes))) - reachable

            if unreachable:
                # Rebuild without unreachable nodes
                new_nodes = []
                new_node_map = {}
                new_id = 0

                for old_id, (coord, info) in enumerate(nodes):
                    if old_id in reachable:
                        new_node_map[old_id] = new_id
                        info["id"] = new_id
                        new_nodes.append([coord, info])
                        new_id += 1

                new_edges = []
                for u, v in edges:
                    if u in reachable and v in reachable:
                        new_edges.append([new_node_map[u], new_node_map[v]])

                nodes = new_nodes
                edges = new_edges

        # Update distances using BFS from gate
        G = nx.Graph()
        G.add_nodes_from(range(len(nodes)))
        G.add_edges_from([tuple(e) for e in edges])

        if 0 in G:
            distances = nx.single_source_shortest_path_length(G, 0)
            for i, (coord, info) in enumerate(nodes):
                nodes[i][1]["distance"] = distances.get(i, 999)

        return nodes, edges, node_id_map

    def _is_layer_connected(self, width, height, z, holes, gate):
        """Check if a layer remains connected after removing holes."""
        cells = set()
        for x in range(width):
            for y in range(height):
                coord = (x, y, z)
                if coord not in holes:
                    cells.add(coord)

        if not cells:
            return z != 0  # Empty layer OK if not gate layer

        # BFS to check connectivity within layer
        start = next(iter(cells))
        visited = {start}
        queue = [start]

        while queue:
            x, y, _ = queue.pop(0)
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nb = (x+dx, y+dy, z)
                if nb in cells and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

        return len(visited) == len(cells)

    def generate_demands(self, num_ports, num_demand_types, total_demand, node_count,
                         target_utilization=0.98, avg_voyage_ratio=0.4):
        """
        Generate demand specifications that create blocking situations.

        Strategy for harder problems:
        1. Mix long-voyage cargo (blocks nodes) with short-voyage cargo (needs rehandling)
        2. Create peak loads at specific ports
        3. Overlap demands to maximize blocking
        """
        max_capacity = node_count - 1
        target_max_load = int(max_capacity * target_utilization)

        demands = []
        remaining = total_demand
        port_load = [0] * num_ports

        # Phase 1: Add long-voyage "blocking" cargo (50% of demand)
        # These stay on ship for long periods, blocking access to nodes
        long_voyage_demand = int(total_demand * 0.5)
        long_voyage_len = max(int(num_ports * 0.75), 5)  # 75% of total voyage

        long_attempts = 0
        while long_voyage_demand > 0 and remaining > 0 and long_attempts < 100:
            long_attempts += 1
            # Start from early ports, end at late ports
            origin = random.randint(0, max(0, num_ports - long_voyage_len - 2))
            dest = min(origin + long_voyage_len + random.randint(0, 3), num_ports - 1)
            if dest <= origin:
                dest = origin + 1

            max_additional = target_max_load - max(port_load[origin:dest])
            if max_additional <= 0:
                long_voyage_len = max(2, long_voyage_len - 1)
                if long_voyage_len <= 2:
                    break
                continue

            qty = min(random.randint(1, max(1, max_additional // 2)),
                     long_voyage_demand, remaining, max_additional)
            if qty <= 0:
                break

            demands.append([[origin, dest], qty])
            remaining -= qty
            long_voyage_demand -= qty
            for p in range(origin, dest):
                port_load[p] += qty

        # Phase 2: Add short-voyage cargo that must navigate around blockers (40% of demand)
        # These need to be loaded/unloaded while long cargo is present, creating conflicts
        short_voyage_demand = int(total_demand * 0.4)

        for _ in range(num_demand_types):
            if short_voyage_demand <= 0 or remaining <= 0:
                break

            # Short voyage: 1-3 ports
            voyage_len = random.randint(1, min(3, num_ports - 1))

            # Prefer origins where there's already high load (creates blocking)
            weighted_origins = []
            for o in range(num_ports - voyage_len):
                load = max(port_load[o:o+voyage_len]) if o + voyage_len <= num_ports else 0
                weight = load + 1  # Higher load = more likely to choose
                weighted_origins.extend([o] * weight)

            if not weighted_origins:
                continue

            origin = random.choice(weighted_origins)
            dest = origin + voyage_len

            max_additional = target_max_load - max(port_load[origin:dest])
            if max_additional <= 0:
                continue

            qty = min(random.randint(1, max(1, max_additional)),
                     short_voyage_demand, remaining, max_additional)
            if qty <= 0:
                continue

            demands.append([[origin, dest], qty])
            remaining -= qty
            short_voyage_demand -= qty
            for p in range(origin, dest):
                port_load[p] += qty

        # Phase 3: Fill remaining with mixed cargo, targeting high-load ports
        attempts = 0
        max_attempts = num_demand_types * 5

        while remaining > 0 and attempts < max_attempts:
            attempts += 1

            # Find port with some load but not full
            candidates = [(p, port_load[p]) for p in range(num_ports - 1)
                         if port_load[p] < target_max_load]
            if not candidates:
                break

            # Prefer ports with medium-high load
            candidates.sort(key=lambda x: -x[1])
            origin = candidates[0][0] if random.random() < 0.7 else random.choice(candidates)[0]

            # Variable voyage length
            voyage_len = random.randint(1, min(num_ports - origin - 1,
                                               int(num_ports * random.uniform(0.2, 0.6))))
            voyage_len = max(1, voyage_len)
            dest = origin + voyage_len

            if dest >= num_ports:
                continue

            max_additional = target_max_load - max(port_load[origin:dest])
            if max_additional <= 0:
                continue

            qty = min(remaining, max_additional, random.randint(1, max(1, max_additional)))
            demands.append([[origin, dest], qty])
            remaining -= qty
            for p in range(origin, dest):
                port_load[p] += qty

        return demands

    def calculate_lower_bound(self, N, edges, K, P, F):
        """
        Calculate the lower bound for the problem.

        LB = blocking을 무시했을 때의 최소 비용
        - 각 화물은 노드를 점유 (origin port ~ dest port)
        - 같은 시점에 같은 노드를 두 화물이 사용할 수 없음
        - 노드 재사용 가능 (다른 시점에)
        """
        G = nx.Graph()
        G.add_nodes_from(range(N))
        G.add_edges_from([tuple(e) for e in edges])

        # Get distances from gate
        distances = nx.single_source_shortest_path_length(G, 0)
        sorted_distances = sorted([d for n, d in distances.items() if n != 0])

        # 각 화물의 interval (origin, dest)
        cargo_intervals = []
        for (o, d), qty in K:
            for _ in range(qty):
                cargo_intervals.append((o, d))

        if not cargo_intervals:
            return 0

        # 각 port에서 concurrent load
        max_concurrent = max(
            sum(1 for (o, d) in cargo_intervals if o <= p < d)
            for p in range(P)
        )

        # 가장 가까운 노드들을 pool로 사용
        pool_distances = sorted_distances[:max_concurrent]

        # Interval scheduling: origin 순으로 처리
        cargo_intervals.sort(key=lambda x: (x[0], x[1]))

        # 각 노드의 free 시점
        node_free = [0] * max_concurrent

        total_cost = 0
        node_idx = max_concurrent  # 다음에 추가할 노드 인덱스

        for origin, dest in cargo_intervals:
            # 사용 가능한 노드 중 가장 가까운 것
            best_idx = None
            for i in range(len(node_free)):
                if node_free[i] <= origin:
                    best_idx = i
                    break

            if best_idx is None:
                # 새 노드 필요
                if node_idx < len(sorted_distances):
                    pool_distances.append(sorted_distances[node_idx])
                    node_free.append(0)
                    best_idx = len(node_free) - 1
                    node_idx += 1
                else:
                    # 노드 부족 - 불가능
                    return float('inf')

            node_free[best_idx] = dest
            total_cost += 2 * (F + pool_distances[best_idx])

        return total_cost

    def generate_problem(self, difficulty="medium", problem_id=1):
        """
        Generate a complete problem with specified difficulty.

        Difficulty levels match original competition problems:
        - MaxUtil ≈ 0.98-1.0 (almost all nodes used at peak)
        - Cargo/Node ≈ 1.3-2.5 (more total cargo than nodes, via turnover)
        - Avg voyage ≈ 3-7 ports (shorter = more turnover = harder)
        """

        # Diverse parameters for GNN+RL training
        # Cover wide ranges of: |N|, |A|, |P|, D(G), S(G), κ_avg
        if difficulty == "tiny":
            # |N|: 25-50, for fast training iterations
            width = random.randint(3, 5)
            height = random.randint(3, 5)
            num_layers = random.randint(1, 3)
            num_ramps = random.randint(1, 3)
            hole_ratio = random.uniform(0.05, 0.20)
            num_ports = random.randint(6, 10)
            num_demand_types = random.randint(8, 15)
            cargo_ratio = random.uniform(1.3, 1.8)
            target_util = random.uniform(0.95, 0.99)
            avg_voyage_ratio = random.uniform(0.35, 0.6)

        elif difficulty == "small":
            # |N|: 50-90
            width = random.randint(4, 7)
            height = random.randint(3, 5)
            num_layers = random.randint(2, 4)
            num_ramps = random.randint(1, 4)
            hole_ratio = random.uniform(0.08, 0.25)
            num_ports = random.randint(8, 14)
            num_demand_types = random.randint(10, 20)
            cargo_ratio = random.uniform(1.4, 2.0)
            target_util = random.uniform(0.96, 0.99)
            avg_voyage_ratio = random.uniform(0.35, 0.55)

        elif difficulty == "medium":
            # |N|: 90-150
            width = random.randint(6, 12)
            height = random.randint(4, 6)
            num_layers = random.randint(2, 4)
            num_ramps = random.randint(1, 5)
            hole_ratio = random.uniform(0.10, 0.28)
            num_ports = random.randint(10, 18)
            num_demand_types = random.randint(12, 28)
            cargo_ratio = random.uniform(1.5, 2.2)
            target_util = random.uniform(0.97, 0.99)
            avg_voyage_ratio = random.uniform(0.30, 0.50)

        elif difficulty == "large":
            # |N|: 150-250
            width = random.randint(10, 18)
            height = random.randint(4, 7)
            num_layers = random.randint(2, 4)
            num_ramps = random.randint(1, 5)
            hole_ratio = random.uniform(0.10, 0.25)
            num_ports = random.randint(12, 22)
            num_demand_types = random.randint(15, 35)
            cargo_ratio = random.uniform(1.6, 2.4)
            target_util = random.uniform(0.97, 0.99)
            avg_voyage_ratio = random.uniform(0.28, 0.45)

        else:  # xlarge
            # |N|: 250-450
            width = random.randint(15, 30)
            height = random.randint(5, 8)
            num_layers = random.randint(2, 4)
            num_ramps = random.randint(1, 5)
            hole_ratio = random.uniform(0.10, 0.25)
            num_ports = random.randint(14, 24)
            num_demand_types = random.randint(20, 45)
            cargo_ratio = random.uniform(1.8, 2.6)
            target_util = random.uniform(0.97, 0.99)
            avg_voyage_ratio = random.uniform(0.25, 0.40)

        # Generate grid graph
        nodes, edges, _ = self.generate_grid_graph(
            width, height, num_layers, num_ramps, hole_ratio
        )

        N = len(nodes)

        # Calculate total demand based on cargo ratio (like original problems)
        total_demand = int((N - 1) * cargo_ratio)

        # Generate demands with high utilization
        K = self.generate_demands(num_ports, num_demand_types, total_demand, N,
                                  target_utilization=target_util,
                                  avg_voyage_ratio=avg_voyage_ratio)

        # Recalculate actual total demand
        actual_total = sum(qty for (o, d), qty in K)

        # Fixed cost
        F = 100

        # Calculate LB
        LB = self.calculate_lower_bound(N, edges, K, num_ports, F)

        problem = {
            "N": N,
            "E": edges,
            "P": num_ports,
            "K": K,
            "F": F,
            "LB": LB,
            "grid_graph": {
                "nodes": nodes,
                "edges": self._format_grid_edges(nodes, edges)
            }
        }

        return problem

    def _format_grid_edges(self, nodes, edges):
        """Format edges for grid_graph structure."""
        grid_edges = []
        id_to_coord = {info["id"]: coord for coord, info in nodes}

        for u, v in edges:
            coord_u = id_to_coord.get(u)
            coord_v = id_to_coord.get(v)
            if coord_u and coord_v:
                # Check if ramp edge (different z)
                is_ramp = coord_u[2] != coord_v[2]
                edge_info = {"ramp": True} if is_ramp else {}
                grid_edges.append([coord_u, coord_v, edge_info])

        return grid_edges

    def verify_feasibility(self, problem):
        """Verify that the problem has a feasible solution."""
        N = problem["N"]
        edges = problem["E"]
        K = problem["K"]
        P = problem["P"]

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(N))
        G.add_edges_from([tuple(e) for e in edges])

        # Check connectivity from gate
        if not nx.is_connected(G):
            return False, "Graph is not connected"

        available_nodes = N - 1  # Exclude gate

        # Check that at each port, the concurrent load doesn't exceed capacity
        for p in range(P):
            concurrent_load = sum(
                qty for (o, d), qty in K if o <= p < d
            )
            if concurrent_load > available_nodes:
                return False, f"Port {p} load ({concurrent_load}) exceeds capacity ({available_nodes})"

        # Check total demand makes sense
        total_demand = sum(qty for (o, d), qty in K)
        if total_demand == 0:
            return False, "No demands"

        return True, "OK"

    def generate_problem_set(self, num_problems, output_dir, seed=None):
        """
        Generate a diverse set of problems with varying difficulties.

        Args:
            num_problems: Number of problems to generate
            output_dir: Directory to save problems
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        os.makedirs(output_dir, exist_ok=True)

        # Distribute difficulties across 5 levels for diverse GNN+RL training
        difficulties = ["tiny", "small", "medium", "large", "xlarge"]
        problems_per_difficulty = num_problems // len(difficulties)
        remainder = num_problems % len(difficulties)

        difficulty_counts = {d: problems_per_difficulty for d in difficulties}
        for i, d in enumerate(difficulties):
            if i < remainder:
                difficulty_counts[d] += 1

        # Generate problems
        problem_id = 1
        generated = []

        for difficulty in difficulties:
            count = difficulty_counts[difficulty]
            for _ in range(count):
                attempts = 0
                max_attempts = 10

                while attempts < max_attempts:
                    try:
                        problem = self.generate_problem(difficulty, problem_id)
                        is_feasible, msg = self.verify_feasibility(problem)

                        if is_feasible:
                            # Save problem
                            filename = f"prob{problem_id}.json"
                            filepath = os.path.join(output_dir, filename)

                            with open(filepath, 'w') as f:
                                json.dump(problem, f, indent=2)

                            # Summary info
                            total_demand = sum(qty for (o, d), qty in problem["K"])
                            generated.append({
                                "id": problem_id,
                                "difficulty": difficulty,
                                "N": problem["N"],
                                "E": len(problem["E"]),
                                "P": problem["P"],
                                "K": len(problem["K"]),
                                "demand": total_demand,
                                "LB": problem["LB"]
                            })

                            print(f"Generated prob{problem_id} ({difficulty}): "
                                  f"N={problem['N']}, P={problem['P']}, "
                                  f"K={len(problem['K'])}, demand={total_demand}")

                            problem_id += 1
                            break
                        else:
                            attempts += 1

                    except Exception as e:
                        attempts += 1
                        if attempts >= max_attempts:
                            print(f"Failed to generate problem {problem_id}: {e}")

        # Save summary
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(generated, f, indent=2)

        print(f"\nGenerated {len(generated)} problems in {output_dir}")
        return generated


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate RoRo ship optimization problems")
    parser.add_argument("--num", type=int, default=20, help="Number of problems to generate")
    parser.add_argument("--output", type=str, default="generated_problems", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    generator = ProblemGenerator(seed=args.seed)
    generator.generate_problem_set(args.num, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
