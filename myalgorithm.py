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
import util  # check_feasibility, bfs, dijkstra 등 유틸리티 함수가 포함된 모듈
import heapq

import engine

# ==================================================================
#  MAIN OPTIMIZER CLASS
# ==================================================================

class PortOptimizer:
    # --- SA Parameters ---
    INITIAL_TEMP = 2000.0  # 초기 온도 증가
    MIN_TEMP = 1e-5
    COOLING_RATE = 0.995  # 더 천천히 냉각
    SA_RESTARTS = 500
    MAX_NO_IMPROVEMENT_ITER = 1500  # 정체 허용 증가
    REHEAT_FACTOR = 0.7  # 재가열 강도 증가
    # INITIAL_TEMP = 1000.0
    # MIN_TEMP = 1e-4
    # COOLING_RATE = 0.99
    # SA_RESTARTS = 300
    # MAX_NO_IMPROVEMENT_ITER = 1000
    # REHEAT_FACTOR = 0.5

    def __init__(self, prob_info):
        self.start_time = time.time()
        self.prob_info = prob_info

        # --- Problem Data ---
        self.N = prob_info['N']
        self.E = set(map(tuple, prob_info['E']))
        self.K = prob_info['K']
        self.P = prob_info['P']
        self.F = prob_info['F']
        self.LB = prob_info['LB']

        # --- Graph and Path Pre-computation ---
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
        self.adj_list = [[] for _ in range(self.N)]
        for u, v in self.G.edges():
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)
        try:
            bc = nx.betweenness_centrality(self.G, normalized=True)
            self.node_centrality = np.array([bc.get(i, 0.0) for i in range(self.N)], dtype=float)
        except Exception:
            self.node_centrality = np.zeros(self.N, dtype=float)

        # ✨✨ 추가
        self._precompute_articulation_impact()

    # ==================================================================
    #  Transplanted Heuristics (From Template Code)
    # ==================================================================

    # ✨✨ 추가
    def _precompute_articulation_impact(self):
        """articulation node를 제거했을 때 0에서 도달 불가해지는 노드 수를 0~1로 정규화."""
        try:
            arts = set(nx.articulation_points(self.G))
        except Exception:
            arts = set()
        self.articulation = arts
        self.cut_impact = np.zeros(self.N, dtype=float)

        if not arts or 0 not in self.G:
            return

        for v in arts:
            if v == 0:
                # 게이트 자체는 제외(문제 정의상 보통 제거 불가, 영향력 계산도 무의미)
                continue
            Gtmp = self.G.copy()
            if v in Gtmp:
                Gtmp.remove_node(v)
            # 0에서 도달 가능한 노드 수
            try:
                reach = nx.single_source_shortest_path_length(Gtmp, 0).keys()
                reach_cnt = sum(1 for _ in reach)
            except Exception:
                reach_cnt = 0
            # v 제거로 0에서 잃는 노드 수(자기 자신 제외)
            lost = max(0, (self.N - 1) - reach_cnt)
            self.cut_impact[v] = float(lost)

        mx = self.cut_impact.max()
        if mx > 0:
            self.cut_impact /= mx

    def _r1_candidates_same_dest(self, reachable_nodes, occ, dest):
        """
        R1: 이미 배치된 같은 목적지(dest) 차량의 1-hop 이웃 중,
            현재 reachable이고 비어있는(-1) 노드 후보를 반환.
        """
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
        """
        cands: 평가할 후보 노드 리스트
        occ:   현재 node_allocations
        k:     배치하려는 차량 id (점유 표시에만 사용)

        반환: 배치해도 기존 reachable 노드(후보 자신 제외)가 하나도 사라지지 않는 안전 후보 목록
        """
        bfs_out = util.bfs(self.G, occ)
        if not bfs_out or bfs_out[0] is None:
            return []
        before = set(bfs_out[0])

        safe = []
        for n in cands:
            if n not in before:
                # 원래부터 reachable이 아니면 스킵(정상적으로는 올 일 없음)
                continue
            occ_tmp = occ.copy()
            occ_tmp[n] = k  # n을 점유했다고 가정
            bfs_after = util.bfs(self.G, occ_tmp)
            if not bfs_after or bfs_after[0] is None:
                continue
            after = set(bfs_after[0])

            # 후보 자신(n)이 사라지는 건 허용, 그 외 노드는 사라지면 안 됨
            if (before - {n}).issubset(after):
                safe.append(n)
        return safe

    def get_dest_port(self, demand_idx, all_demands):
        """수요 인덱스로 목적지 항구를 반환하는 헬퍼 함수"""
        if demand_idx is None or demand_idx < 0 or demand_idx >= len(all_demands):
            return float('inf')  # 유효하지 않은 경우 무한대 반환
        return all_demands[demand_idx][0][1]

    def min_blocking_path(self, G_adj, node_allocations, target, gate=0):
        """
        gate->target으로 가는 경로 중, (중간 점유 노드 수, 경로 길이)를
        사전식으로 최소화하는 경로를 반환.
        반환: (path_0_to_target, blocking_nodes_list)
        """
        # 비용: (blocks, dist)
        INF = (10 ** 9, 10 ** 9)
        n_nodes = len(G_adj)
        dist = {u: INF for u in range(n_nodes)}
        prev = {u: None for u in range(n_nodes)}

        dist[gate] = (0, 0)
        pq = [(0, 0, gate)]  # (blocks, dist, node)

        while pq:
            b, d, u = heapq.heappop(pq)
            if (b, d) != dist[u]:
                continue
            if u == target:
                break
            for v in G_adj[u]:
                # target으로 진입 비용은 0, 그 외 점유면 1
                occupied = (node_allocations[v] != -1) and (v != target)
                nb = b + (1 if occupied else 0)
                nd = d + 1
                if (nb, nd) < dist[v]:
                    dist[v] = (nb, nd)
                    prev[v] = u
                    heapq.heappush(pq, (nb, nd, v))

        # 경로 복원
        path = []
        cur = target
        if dist[target] == INF:
            return [], []  # 경로 없음
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()  # gate->target

        # 차단 노드: gate/target 제외, 현재 점유된 노드
        blocking = [x for x in path[1:-1] if node_allocations[x] != -1]
        return path, blocking

    def _loading_heuristic(self, p, node_allocations, rehandling_demands, SA=False):

        # All demands that should be loaded at port p
        K_load = {idx: r for idx, ((o, d), r) in enumerate(self.K) if o == p}

        if len(rehandling_demands) > 0:

            # Merge the rehandling demands with the loading demands
            for k in rehandling_demands:
                if k in K_load:
                    K_load[k] += 1
                else:
                    K_load[k] = 1

        route_list = []

        last_rehandling_demands = []

        # Total number of demands to load (including rehandling demands)
        total_loading_demands = sum([r for k, r in K_load.items()])

        # Get reachable nodes from the gate
        reachable_nodes, reachable_node_distances = util.bfs(self.G, node_allocations)

        # Get not occupied nodes
        available_nodes = util.get_available_nodes(node_allocations)

        if len(available_nodes) < total_loading_demands:
            return None, None

        if len(reachable_nodes) < total_loading_demands:

            # A very simple rehandling heuristic
            # 1. Get nodes that are available but not reachable
            # 2. Loop until we have enough reachable nodes
            # 2-1. Pick a node from the available but not reachable nodes
            # 2-2. Get the shortest path to the node from the gate
            # 2-3. Roll-off demands occupied on the path by order of distance from the gate. (and push to rehandling_demands stack for the later reloading)
            # 2-4. Check if the number of reachable nodes is enough to load the demand
            available_but_not_reachable = [n for n in available_nodes if n not in reachable_nodes]

            while len(reachable_nodes) < total_loading_demands:

                if len(available_but_not_reachable) == 0:
                    return None, None

                # Pick a node from the available but not reachable nodes
                n = available_but_not_reachable.pop(0)

                # Get the shortest path to the node from the gate
                distances, previous_nodes = util.dijkstra(self.G, node_allocations=None)

                path, blocks = self.min_blocking_path(self.adj, node_allocations, n, gate=0)

                # 차단 해제(리핸들링 루트) 생성
                for idx, i in enumerate(path[:-1]):  # target 직전까지
                    if node_allocations[i] != -1:
                        k_block = node_allocations[i]
                        last_rehandling_demands.append(k_block)
                        # bn까지 역방향 경로 생성
                        # i까지 잘라서 i->0
                        cut = []
                        for x in path:
                            cut.append(x)
                            if x == i:
                                break
                        route_list.append((cut[::-1], k_block))
                        node_allocations[i] = -1
                        total_loading_demands += 1

                # Check if the number of reachable nodes is enough to load the demand
                reachable_nodes, reachable_node_distances = util.bfs(self.G, node_allocations)

        # Merge the rehandling demands with the loading demands
        for k in last_rehandling_demands:
            if k in K_load:
                K_load[k] += 1
            else:
                K_load[k] = 1

        if total_loading_demands > 0:
            # 1. 수요 분리: 전체 수요를 '장기'와 '단기'로 나눕니다.
            all_demands_to_load = [k for k, r in K_load.items() for _ in range(r)]

            long_term_vehicles = [k for k in all_demands_to_load if self.K[k][0][1] > p + 1]
            next_port_vehicles = [k for k in all_demands_to_load if self.K[k][0][1] == p + 1]

            # 장기 화물은 목적지가 먼 순서대로 정렬하여 가장 깊은 곳부터 채웁니다.
            long_term_vehicles.sort(key=lambda k: self.K[k][0][1], reverse=True)

            # 장기 차량을 목적지 내림차순(먼 곳 먼저)으로 순회
            long_term_vehicles_sorted = sorted(long_term_vehicles, key=lambda k: self.K[k][0][1], reverse=True)

            # ✅ 단기차량을 위한 예약 존 계산: 게이트에 가까운 빈 노드들 중에서 next_port_vehicles 수만큼
            reserve_cnt = len(next_port_vehicles)

            for k in long_term_vehicles:
                d_target = self.K[k][0][1]

                # 최신 reachable
                reachable_nodes, _ = util.bfs(self.G, node_allocations)
                avail_now = [n for n in reachable_nodes if n != 0 and node_allocations[n] == -1]
                reserved_nodes = set(sorted(avail_now, key=lambda n: self.shortest_distances[n])[:reserve_cnt])

                # --- R1: 같은 목적지 클러스터 1-hop 후보 → 예약 존 제외 → 가장 깊은 노드
                r1_cands = self._r1_candidates_same_dest(reachable_nodes, node_allocations, d_target) or []
                r1_cands = [n for n in r1_cands if n not in reserved_nodes]

                r1_safe = self._reachability_safe_candidates(r1_cands, node_allocations, k)

                if r1_safe:
                    # 동률이면 기존 정책 유지: 가장 깊은 노드 선택
                    n_sel = max(r1_safe, key=lambda n: self.shortest_distances[n])
                else:
                    # ================================================================= #
                    # =================== R2 로직 시작 (생략 없음) ==================== #
                    # ================================================================= #
                    if SA:
                        all_cands = [n for n in reachable_nodes if n not in reserved_nodes]
                        if not all_cands:
                            continue

                        # 1. 먼저 안전한 후보 노드를 필터링합니다.
                        safe_cands = self._reachability_safe_candidates(all_cands, node_allocations, k)
                        if safe_cands:
                            n_sel = random.choice(safe_cands)
                        else:
                            n_sel = random.choice(all_cands)
                    else:
                        all_cands = [n for n in reachable_nodes if n not in reserved_nodes]
                        if not all_cands:
                            continue

                        # 1. 먼저 안전한 후보 노드를 필터링합니다.
                        safe_cands = self._reachability_safe_candidates(all_cands, node_allocations, k)

                        # 안전한 후보가 있으면 그 안에서 탐색, 없으면 전체 후보를 대상으로 탐색 (안정성 확보)
                        cands_to_score = safe_cands if safe_cands else all_cands

                        if not cands_to_score:
                            continue

                        # 2. 해당 목적지의 첫 차량인지 확인합니다.
                        is_first_vehicle_for_dest = not self.route_heatmaps[d_target]

                        if is_first_vehicle_for_dest:
                            # --- 전략 A: 첫 차량일 경우 'Attractiveness' 점수가 가장 높은 곳에 배치 ---

                            att_scores = self._calculate_attractiveness(cands_to_score, k, p)

                            best_node_info = max(zip(cands_to_score, att_scores), key=lambda item: item[1])
                            n_sel = best_node_info[0]

                        else:
                            # --- 전략 B: 첫 차량이 아닐 경우 히트맵 점수로 클러스터링 강화 ---
                            best_cand_node = -1
                            max_score = -float('inf')

                            for n_cand in cands_to_score:
                                _, prev_nodes = util.dijkstra(self.G, node_allocations)
                                path_cand = util.path_backtracking(prev_nodes, 0, n_cand)

                                if not path_cand: continue

                                current_score = 0.0

                                # 긍정 점수: 같은 목적지 히트맵 경로를 따라갈수록 점수 상승
                                for node_on_path in path_cand[:-1]:
                                    current_score += self.route_heatmaps[d_target].get(node_on_path, 0)

                                    # 부정 점수: 다른 목적지 히트맵 경로와 겹치면 점수 하락
                                    for other_dest, heatmap in self.route_heatmaps.items():
                                        if other_dest != d_target:
                                            current_score -= heatmap.get(node_on_path, 0) * 0.5

                                # 깊이에 따른 추가 점수 (tie-breaker 역할)
                                current_score += self.shortest_distances[n_cand] * 0.1

                                if current_score > max_score:
                                    max_score = current_score
                                    best_cand_node = n_cand

                            n_sel = best_cand_node if best_cand_node != -1 else cands_to_score[-1]

                # 경로 계산(현 점유 기준)
                _, prev_nodes = util.dijkstra(self.G, node_allocations)
                path = util.path_backtracking(prev_nodes, 0, n_sel)

                # 배치 반영
                node_allocations[n_sel] = k
                route_list.append((path, k))

                if d_target > p + 1:
                    for node_on_path in path[:-1]:  # 마지막 노드 제외
                        heatmap = self.route_heatmaps[d_target]
                        heatmap[node_on_path] = heatmap.get(node_on_path, 0) + 1

            # 갱신된 점유 기준으로 reachable 재계산(단기 적재를 위해)
            reachable_nodes, reachable_node_distances = util.bfs(self.G, node_allocations)

            # 3. 단기 화물 배치

            loading_nodes = reachable_nodes[:len(next_port_vehicles)][::-1]

            # Get the shortest path to the node from the gate
            distances, previous_nodes = util.dijkstra(self.G, node_allocations)

            # Allocate the nodes starting from behind so that there is no blocking
            for n, k in zip(loading_nodes, next_port_vehicles):
                node_allocations[n] = k

                path = util.path_backtracking(previous_nodes, 0, n)

                route_list.append((path, k))

        return route_list, node_allocations

    # ✨✨ 추가
    def _lexi_dijkstra_all(self, occ, gate=0):
        """
        gate에서 모든 노드까지 (차단수, 거리) 사전식 최소 비용과 prev를 한 번에 계산.
        '어떤 target이어도' 동일 prev로 최적 경로를 재구성 가능.
        주의: target 노드 자체의 점유는 블로킹 카운트에서 빼줘야 하므로,
            후보 평가 시 blocks = dist[n][0] - 1 로 보정.
        """
        INF = (10 ** 9, 10 ** 9)
        dist = [(10 ** 9, 10 ** 9)] * self.N  # (blocks, dist)
        prev = [-1] * self.N
        dist[gate] = (0, 0)
        pq = [(0, 0, gate)]
        while pq:
            b, d, u = heapq.heappop(pq)
            if (b, d) != dist[u]:
                continue
            for v in self.adj[u]:
                nb = b + (1 if occ[v] != -1 else 0)  # v가 점유면 +1
                nd = d + 1
                if (nb, nd) < dist[v]:
                    dist[v] = (nb, nd)
                    prev[v] = u
                    heapq.heappush(pq, (nb, nd, v))
        return dist, prev

    # ✨✨ 추가
    def _bfs_path_free_between(self, occ, src: int, dst: int) -> list | None:
        """
        현재 점유 occ 기준으로 '빈칸만' 통과해서 src->dst 경로를 BFS로 찾는다.
        - src와 dst는 서로 달라야 함
        - dst는 반드시 빈칸이어야 함 (occ[dst] == -1)
        - 찾으면 [src, ..., dst] 반환, 없으면 None
        """
        if src == dst or occ[dst] != -1:
            return None
        N = self.N
        seen = [False] * N
        prev = [-1] * N
        dq = deque([src])
        seen[src] = True

        while dq:
            u = dq.popleft()
            for v in self.adj[u]:
                # 출발(src)은 점유되어 있어도 OK, 그 외는 빈칸만 허용
                if v != dst and occ[v] != -1:
                    continue
                if not seen[v]:
                    seen[v] = True
                    prev[v] = u
                    if v == dst:
                        # 경로 복원
                        path = []
                        cur = v
                        while cur != -1:
                            path.append(cur)
                            cur = prev[cur]
                        path.reverse()
                        return path
                    dq.append(v)
        return None

    # ✨✨ 추가
    def _try_random_side_move(self, path_sel: list[int], occ: np.ndarray, idx_bn: int,
                              rng: random.Random) -> list | None:
        """
        블로킹 노드 bn = path_sel[idx_bn]를 '옆으로' 잠시 치우기 위한 경로를 찾는다.
        후보 조건:
        - 빈칸(occ[u] == -1)
        - 금지영역 제외: 게이트(0), 현재 하역경로(path_sel), 현재 블로킹 집합
        - bn -> u 가 '빈칸만' 경유로 실제 reachable ( _bfs_path_free_between )
        - u 가 reachability-safe ( _reachability_safe_candidates )

        성공 시 bn->u 경로를 반환, 실패 시 None
        """
        bn = path_sel[idx_bn]

        # 금지영역 구성
        ban = set(path_sel) | {0}
        # 현재 경로상 블로커들도 ban에 추가해서 경로와 겹치지 않게
        for x in path_sel[1:-1]:
            if occ[x] != -1:
                ban.add(x)

        # 후보 풀: 빈칸 & 금지영역 제외
        raw_pool = [u for u in range(self.N) if (occ[u] == -1 and u not in ban)]
        if not raw_pool:
            return None

        # 속도-품질 절충을 위한 소표본(표본 크기 조절 가능)
        EMPTY_CAP = 64
        rng.shuffle(raw_pool)
        raw_pool = raw_pool[:EMPTY_CAP]

        # 1) 실제로 bn->u가 빈칸만으로 이동가능한 u만 필터링
        reachables = []
        path_cache = {}
        for u in raw_pool:
            path_bn_u = self._bfs_path_free_between(occ, bn, u)
            if path_bn_u:
                reachables.append(u)
                path_cache[u] = path_bn_u
        if not reachables:
            return None

        # 2) reachability-safe 필터 적용 (게이트 도달성 악화 방지)
        # safe_cands = self._reachability_safe_candidates(reachables, occ, occ[bn])
        # if not safe_cands:
        #     return None

        # 3) 최종 선택 (무작위 or 가중치 선택 가능)
        s = rng.choice(reachables)
        return path_cache.get(s) or self._bfs_path_free_between(occ, bn, s)

    # ✨✨ 통째로 변경
    def _unloading_heuristic(self, p, node_allocations, SA: bool = False):
        """
        하역 순서:
        - 0→n에 대해 (차단수, 거리) 사전식 최소 경로로 가장 쉬운 후보부터 처리.
        - SA=True면, 블로커 중 일부를 '프리픽스 옆자리'로 0-블록 사이드 이동(랜덤) 시도.
            실패 시 즉시 임시하역으로 폴백 -> 항상 완주 보장.
        """
        from collections import Counter
        rng = random.Random()  # 필요 시 상위에서 seed 주입 가능

        K_unload = {idx: r for idx, ((o, d), r) in enumerate(self.K) if d == p}
        route_list, rehandling_demands = [], []

        # 하역 대상들의 위치 수집
        pending = []
        for k in K_unload.keys():
            for n in range(self.N):
                if node_allocations[n] == k:
                    pending.append((k, n))

        # SA 모드에서 사이드 이동 예산(너무 과도하면 역효과)
        # 한 경로에서 시도할 블로커 비율과 경로별 최대 개수
        SA_TRY_RATIO = 0.5 if SA else 0.0  # 블로커의 50% 정도를 랜덤 시도
        SA_TRY_CAP = 2 if SA else 0  # 경로별 최대 2개까지만 시도

        while pending:
            dist, prev = self._lexi_dijkstra_all(node_allocations, gate=0)

            cand, zero_block = [], []
            for (k, n) in list(pending):
                b, d = dist[n]
                if b >= 10 ** 9:
                    continue
                blocks = b - (1 if node_allocations[n] != -1 else 0)

                # 경로 복원(0->n)
                path = []
                cur = n
                while cur != -1:
                    path.append(cur)
                    cur = prev[cur]
                path.reverse()
                if not path or path[0] != 0:
                    continue

                if blocks == 0:
                    zero_block.append((k, n, path))
                else:
                    B = [x for x in path[1:-1] if node_allocations[x] != -1]
                    cand.append((k, n, path, blocks, B))

            # 0-블록은 바로 하역
            if zero_block:
                for (k_sel, n_sel, path_sel) in zero_block:
                    if node_allocations[n_sel] != k_sel:
                        continue
                    route_list.append((path_sel[::-1], k_sel))
                    node_allocations[n_sel] = -1
                    pending = [(k, n) for (k, n) in pending if not (k == k_sel and n == n_sel)]
                continue

            if not cand:
                break

            # 가장 쉬운 후보 선택 (차단수, 경로길이)
            cand.sort(key=lambda t: (t[3], len(t[2])))
            k_sel, n_sel, path_sel, blocks_sel, B_sel = cand[0]

            # 블로커 처리
            pos = {node: i for i, node in enumerate(path_sel)}
            B_work = list(B_sel)

            # SA=True면 일부 블로커를 사이드 이동 시도 (랜덤 크기)
            if SA and B_work:
                rng.shuffle(B_work)
                k_try = min(SA_TRY_CAP, max(1, int(len(B_work) * SA_TRY_RATIO)))
                try_set = B_work[:k_try]
            else:
                try_set = []

            for bn in list(B_work):
                if node_allocations[bn] == -1:
                    continue  # 이미 처리됨
                idx_bn = pos.get(bn, None)
                if idx_bn is None:
                    continue

                k_block = node_allocations[bn]

                # 이번 포트 하역 대상이면 그냥 하역
                if self.K[k_block][0][1] == p:
                    cut = path_sel[:idx_bn + 1][::-1]  # bn->0
                    route_list.append((cut, k_block))
                    node_allocations[bn] = -1
                    pending = [(k, n) for (k, n) in pending if not (k == k_block and n == bn)]
                    continue

                moved = False
                if bn in try_set:
                    # 프리픽스 옆자리로 0-블록 사이드 이동 시도
                    side_path = self._try_random_side_move(path_sel, node_allocations, idx_bn, rng)
                    if side_path:
                        # intra-hold move
                        route_list.append((side_path, k_block))
                        node_allocations[bn] = -1
                        node_allocations[side_path[-1]] = k_block
                        moved = True

                if not moved:
                    # 폴백: 임시하역(bn->0)
                    cut = path_sel[:idx_bn + 1][::-1]
                    route_list.append((cut, k_block))
                    if k_block not in K_unload:
                        rehandling_demands.append(k_block)
                    node_allocations[bn] = -1

            # 본체 하역
            if node_allocations[n_sel] == k_sel:
                route_list.append((path_sel[::-1], k_sel))
                node_allocations[n_sel] = -1
                pending = [(k, n) for (k, n) in pending if not (k == k_sel and n == n_sel)]

        return route_list, rehandling_demands, node_allocations

    # ==================================================================
    #  SA Framework Core Logic
    # ==================================================================
    def _calculate_attractiveness(self, nodes, car_k, p):
        scores = []
        d_target = self.K[car_k][0][1]
        # 깊이 타겟
        remoteness_ratio = (d_target - (p + 1)) / max(1, (self.P - 1) - (p + 1))
        finite = self.shortest_distances[np.isfinite(self.shortest_distances)]
        max_d = np.max(finite) if finite.size > 0 else 1.0
        target_depth = max_d * remoteness_ratio

        # 가중치 (권장 초기값; 이후 자동튜닝)
        alpha = 1.0  # 깊이 적합도
        beta = 1.0  # 중심성 페널티
        delta = 0.6  # 동일 목적지 히트맵 보너스
        eta = 0.4  # 혼잡 대리지표 페널티

        # 간단 혼잡 대리지표: 이웃들의 중심성 합
        for node in nodes:
            depth_fit = 1.0 / (abs(self.shortest_distances[node] - target_depth) + 1.0)
            cent_pen = self.node_centrality[node]

            # 동일 목적지 히트맵(스파인) 혜택
            spine_bonus = 0.0
            hm = self.route_heatmaps.get(d_target, {})
            for nb in self.adj[node]:
                spine_bonus += hm.get(nb, 0)
            # 정규화
            spine_bonus = np.tanh(spine_bonus)

            # 혼잡 대리지표: 이웃 중심성 합
            cong = sum(self.node_centrality[nb] for nb in self.adj[node])

            score = (
                    alpha * depth_fit
                    - beta * cent_pen
                # + delta * spine_bonus
                # - eta * cong
            )
            scores.append(score)
        return scores

    def _apply_routes_to_occupancy(self, routes, occ):
        """
        routes: [(path, k), ...] 를 차례로 적용하여 occ를 갱신.
        - 로딩(path[0]==0): 마지막 노드에 k 적재
        - 언로딩(path[-1]==0): 첫 노드 비우기
        - 리핸들링(중간->중간): 첫 노드 비우고 마지막 노드에 k 적재
        """
        for path, k in routes:
            if not path:
                continue
            if path[0] == 0 and path[-1] != 0:
                # loading
                occ[path[-1]] = k
            elif path[-1] == 0 and path[0] != 0:
                # unloading
                occ[path[0]] = -1
            else:
                # rehandling (intra-hold move)
                occ[path[0]] = -1
                occ[path[-1]] = k
        return occ

    def _create_initial_solution(self):

        node_allocations = np.full(self.N, -1, dtype=int)
        solution = {p: [] for p in range(self.P)}
        self.route_heatmaps = {d: {} for d in range(self.P)}  # 여기서도 초기화

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

        self.route_heatmaps = {d: {} for d in range(self.P)}
        result = util.check_feasibility(self.prob_info, solution)
        cost = result['obj'] if result['feasible'] else float('inf')
        return solution, cost

    def _create_neighbor_from_port(self, base_solution):
        """
        base_solution에서 0..start_port-1은 고정.
        start_port부터 P-1까지를 언로딩 재계산 + 로딩은 SA=True로 재구성.
        """
        # ✨✨ 여기서 랜덤으로 선택이 더 좋은 결과가 나오는것 같은데 확인해볼필요있음.
        start_port = random.choice(range(self.P - 1))

        # ① new_sol은 기본적으로 base_solution을 복사
        new_sol = {p: list(base_solution.get(p, [])) for p in range(self.P)}

        # ② start_port 직전까지의 점유 상태 복원
        occ = np.full(self.N, -1, dtype=int)
        for t in range(start_port):
            routes_t = new_sol.get(t, [])
            occ = self._apply_routes_to_occupancy(routes_t, occ)

        # ③ start_port에서 언로딩을 새로 계산
        unload_routes, rehandling, occ = self._unloading_heuristic(start_port, occ)

        # ④ start_port 로딩을 SA=True로 새로 구성
        load_routes, occ = self._loading_heuristic(start_port, occ, rehandling, SA=True)
        if load_routes is None:
            return None  # 불능 이웃

        new_sol[start_port] = unload_routes + load_routes

        # ⑤ start_port+1 .. P-1까지 순차 재시뮬 (언로딩→로딩(SA=True))
        for t in range(start_port + 1, self.P):
            unload_routes_t, rehandling_t, occ = self._unloading_heuristic(t, occ)
            if t < self.P - 1:
                load_routes_t, occ = self._loading_heuristic(t, occ, rehandling_t, SA=False)
                if load_routes_t is None:
                    return None
                new_sol[t] = unload_routes_t + load_routes_t
            else:
                new_sol[t] = unload_routes_t  # 마지막 포트는 로딩 없음

        return new_sol

    def _rebuild_heatmaps(self, solution):
        """
        주어진 solution 딕셔너리를 기반으로 self.route_heatmaps를 완전히 재생성합니다.
        """
        # 히트맵을 깨끗하게 초기화
        self.route_heatmaps = {d: {} for d in range(self.P)}

        # solution의 모든 포트를 순회하며 히트맵을 다시 쌓음
        for p in range(self.P - 1):  # 마지막 포트는 선적 작업이 없으므로 제외
            routes_at_p = solution.get(p, [])
            for path, k in routes_at_p:
                # 선적 경로(path[0] == 0)이고 장기 화물인 경우에만 히트맵에 기록
                is_loading = (path[0] == 0)
                if is_loading:
                    d_target = self.K[k][0][1]
                    if d_target > p + 1:
                        for node_on_path in path[:-1]:  # 마지막 노드 제외
                            heatmap = self.route_heatmaps[d_target]
                            heatmap[node_on_path] = heatmap.get(node_on_path, 0) + 1

    # ✨✨ 추가
    def _create_neighbor_from_unload(self, base_solution):
        """
        base_solution에서 0..start_port-1은 고정.
        start_port부터 P-1까지를 언로딩 재계산 + 로딩은 SA=True로 재구성.
        """

        start_port = random.choice(range(self.P - 1))

        self._last_start_port = start_port
        # ① new_sol은 기본적으로 base_solution을 복사
        new_sol = {p: list(base_solution.get(p, [])) for p in range(self.P)}

        # ② start_port 직전까지의 점유 상태 복원
        occ = np.full(self.N, -1, dtype=int)
        for t in range(start_port):
            routes_t = new_sol.get(t, [])
            occ = self._apply_routes_to_occupancy(routes_t, occ)

        # ③ start_port에서 언로딩을 새로 계산
        unload_routes, rehandling, occ = self._unloading_heuristic(start_port, occ, SA=True)

        # ④ start_port 로딩을 SA=True로 새로 구성
        load_routes, occ = self._loading_heuristic(start_port, occ, rehandling)
        if load_routes is None:
            return None  # 불능 이웃

        new_sol[start_port] = unload_routes + load_routes

        # ⑤ start_port+1 .. P-1까지 순차 재시뮬 (언로딩→로딩(SA=True))
        for t in range(start_port + 1, self.P):
            unload_routes_t, rehandling_t, occ = self._unloading_heuristic(t, occ)
            if t < self.P - 1:
                load_routes_t, occ = self._loading_heuristic(t, occ, rehandling_t, SA=False)
                if load_routes_t is None:
                    return None
                new_sol[t] = unload_routes_t + load_routes_t
            else:
                new_sol[t] = unload_routes_t  # 마지막 포트는 로딩 없음

        return new_sol

    def _choose_neighbor_operator(self, no_improvement_count, T, iteration):
        """
        개선 정체 정도(no_improvement_count)와 온도(T)에 따라
        이웃 생성 오퍼레이터를 선택한다.
        - 초반/개선 빠름: from_port 선호(큰 폭 재배치)
        - 중반/정체 시작: 혼합
        - 장기 정체/저온: change_node 선호(미세 조정)
        """
        SLOW1 = 200  # 정체 1단계
        SLOW2 = 600  # 정체 2단계 (길게 정체되면 더 로컬)
        LOWT = 1e-2  # 충분히 식었음

        if no_improvement_count < SLOW1 and T > LOWT:
            # 탐색 초반: 큰 폭 변경 위주
            return 'from_port'
        elif no_improvement_count < SLOW2:
            # 중간: 반반 섞기
            return 'mix'
        else:
            # 장기 정체 또는 저온: 미세 조정 위주
            return 'change_node'

    # ✨✨ 통째로 변경
    def _try_neighbor(self, current_sol, primary='from_port', p_try_alt=0.35):
        """
        1차 오퍼레이터로 이웃 생성, 실패하거나 None이면
        확률적으로 대체 오퍼레이터도 시도.
        """
        neighbor_sol = None
        # 1차
        if primary == 'from_port':
            neighbor_sol = engine.run(self.P,
                                      self.N,
                                      self.K,
                                      self.adj_list,
                                      list(self.shortest_distances),
                                      self.shortest_paths,
                                      current_sol)
            if neighbor_sol is None and random.random() < 0.8:
                neighbor_sol = engine.run3(self.P,
                                          self.N,
                                          self.K,
                                          self.adj_list,
                                          list(self.shortest_distances),
                                          self.shortest_paths,
                                          current_sol)
        elif primary == 'change_node':
            neighbor_sol = engine.run3(self.P,
                                      self.N,
                                      self.K,
                                      self.adj_list,
                                      list(self.shortest_distances),
                                      self.shortest_paths,
                                      current_sol)
            if neighbor_sol is None and random.random() < 0.8:
                neighbor_sol = engine.run(self.P,
                                          self.N,
                                          self.K,
                                          self.adj_list,
                                          list(self.shortest_distances),
                                          self.shortest_paths,
                                          current_sol)
        else:  # primary == 'mix'
            if random.random() < 0.5:
                neighbor_sol = engine.run(self.P,
                                          self.N,
                                          self.K,
                                          self.adj_list,
                                          list(self.shortest_distances),
                                          self.shortest_paths,
                                          current_sol)
                if neighbor_sol is None and random.random() < p_try_alt:
                    neighbor_sol = engine.run3(self.P,
                                              self.N,
                                              self.K,
                                              self.adj_list,
                                              list(self.shortest_distances),
                                              self.shortest_paths,
                                              current_sol)
            else:
                neighbor_sol = engine.run3(self.P,
                                          self.N,
                                          self.K,
                                          self.adj_list,
                                          list(self.shortest_distances),
                                          self.shortest_paths,
                                          current_sol)
                if neighbor_sol is None and random.random() < p_try_alt:
                    neighbor_sol = engine.run(self.P,
                                              self.N,
                                              self.K,
                                              self.adj_list,
                                              list(self.shortest_distances),
                                              self.shortest_paths,
                                              current_sol)

        return neighbor_sol

    def _run_simulated_annealing(self, initial_solution, initial_cost, timelimit, shared_data, lock):
        sa_start_time = time.time()
        T = self.INITIAL_TEMP

        current_sol = deepcopy(initial_solution)
        current_cost = initial_cost

        self._rebuild_heatmaps(current_sol)

        best_sol = deepcopy(initial_solution)
        best_cost = initial_cost

        no_improvement_count = 0
        iteration = 0
        reheats_used = 0  # (선택) 재가열 횟수 제한용

        while (time.time() - sa_start_time) < timelimit:
            iteration += 1

            # neighbor_sol = self._create_neighbor_from_port(current_sol)
            # if random.random() < 0.5:
            #     neighbor_sol = self._create_neighbor_from_port(current_sol)
            # else:
            #     neighbor_sol = self._change_loading_node(current_sol)

            op = self._choose_neighbor_operator(no_improvement_count, T, iteration)
            neighbor_sol = self._try_neighbor(current_sol, primary=op)

            # neighbor_sol = self._create_neighbor_from_port(current_sol)

            if neighbor_sol is not None:
                checked = util.check_feasibility(self.prob_info, neighbor_sol)
                if checked['feasible']:
                    neighbor_cost = checked['obj']
                    cost_diff = neighbor_cost - current_cost

                    # 메트로폴리스 기준
                    if cost_diff < 0 or random.random() < np.exp(-cost_diff / T):
                        current_sol = neighbor_sol
                        current_cost = neighbor_cost

                        self._rebuild_heatmaps(current_sol)

                        if current_cost < best_cost:
                            best_sol = current_sol
                            best_cost = current_cost
                            no_improvement_count = 0
                        else:
                            no_improvement_count += 1
                    else:
                        no_improvement_count += 1
                else:
                    # 이웃 해가 불능이면 개선 없음으로 간주
                    no_improvement_count += 1
            else:
                # 이웃 생성 실패 → 개선 없음으로 간주
                no_improvement_count += 1

            # === (A) 한 번만 냉각 ===
            T = max(T * self.COOLING_RATE, self.MIN_TEMP)

            # === (B) 협력 공유(200회마다) ===
            if shared_data and lock and iteration % 200 == 0:
                pid = os.getpid()
                with lock:
                    shared_data['worker_costs'][pid] = current_cost

                    if best_cost < shared_data['best_cost']:
                        # print(f"🎉 Worker {os.getpid()}: 새로운 전역 최적해 발견! Cost: {best_cost}")
                        shared_data['best_cost'] = best_cost
                        shared_data['best_solution'] = deepcopy(best_sol)

                    if len(shared_data['worker_costs']) > 0:
                        worst_pid, worst_cost = max(shared_data['worker_costs'].items(), key=lambda kv: kv[1])
                        # 오직 최악 워커만 전역해 흡수
                        if (pid == worst_pid
                                and shared_data['best_solution'] is not None
                                and shared_data['best_cost'] < current_cost):
                            current_sol = deepcopy(shared_data['best_solution'])
                            current_cost = shared_data['best_cost']
                            # 흡수 직후 온도는 살짝 올려 다양성 확보
                            T = self.INITIAL_TEMP * (self.REHEAT_FACTOR ** (reheats_used ** 0.5))
                            no_improvement_count = 0

                    # elif shared_data['best_cost'] < current_cost and shared_data['best_solution'] is not None:
                    #     # print(f"🌊 Worker {os.getpid()}: 다른 섬의 우수해 흡수! Global Cost: {shared_data['best_cost']}")
                    #     current_sol = deepcopy(shared_data['best_solution'])
                    #     current_cost = shared_data['best_cost']
                    #     # 흡수 직후 온도 살짝 올려 다양성 확보
                    #     T = max(self.INITIAL_TEMP * self.REHEAT_FACTOR, self.MIN_TEMP)
                    #     no_improvement_count = 0

            # === (C) 재가열 조건 ===
            need_reheat = (no_improvement_count >= self.MAX_NO_IMPROVEMENT_ITER or T <= self.MIN_TEMP)
            can_reheat = (reheats_used < self.SA_RESTARTS)  # (선택) 횟수 제한

            if need_reheat and can_reheat:
                pid = os.getpid()
                reheats_used += 1
                # 최고해 주변으로 점프 (협력 모드면 전역해 우선)
                if shared_data and lock:
                    with lock:
                        # 내 최신 비용 반영
                        shared_data['worker_costs'][pid] = current_cost

                        # 현재 최악 워커 식별
                        if len(shared_data['worker_costs']) > 0:
                            worst_pid, _ = max(shared_data['worker_costs'].items(), key=lambda kv: kv[1])
                        else:
                            worst_pid = pid

                        if pid == worst_pid and shared_data['best_solution'] is not None and shared_data[
                            'best_cost'] <= best_cost:
                            # 최악 워커만 전역해로 재가열
                            current_sol = deepcopy(shared_data['best_solution'])
                            current_cost = shared_data['best_cost']
                        else:
                            # 나머지는 자기 best 근처로만 재가열
                            current_sol = deepcopy(best_sol)
                            current_cost = best_cost
                else:
                    current_sol = deepcopy(best_sol)
                    current_cost = best_cost

                # 온도 재가열 + 카운터 리셋
                # T = max(self.INITIAL_TEMP * self.REHEAT_FACTOR, self.MIN_TEMP)
                T = self.INITIAL_TEMP * (self.REHEAT_FACTOR ** (reheats_used ** 0.5))
                no_improvement_count = 0
                # print(f"🔥 Reheat #{reheats_used}: T={T:.4f}, best_cost={best_cost}")

        # 종료 직전 전역해 갱신
        if shared_data and lock:
            with lock:
                if best_cost < shared_data['best_cost']:
                    # print(f"🏁 Worker {os.getpid()}: 최종 결과 보고. Cost: {best_cost}")
                    shared_data['best_cost'] = best_cost
                    shared_data['best_solution'] = deepcopy(best_sol)

        return best_sol, best_cost

    def solve(self, timelimit=60, shared_data=None, lock=None):  # <-- shared_data, lock 추가
        # 초기 해 생성
        initial_sol, initial_cost = self._create_initial_solution()

        # 만약 병렬 실행이 아니라면(공유 객체가 없으면) 기존 로직대로 작동
        if shared_data is None:
            best_overall_sol = initial_sol
            best_overall_cost = initial_cost
        else:
            # 병렬 실행 시, 초기 해를 공유 저장소에 업데이트 시도
            with lock:
                if initial_cost < shared_data['best_cost']:
                    shared_data['best_cost'] = initial_cost
                    shared_data['best_solution'] = deepcopy(initial_sol)
            # 공유 저장소의 값을 현재 최적해로 사용
            best_overall_sol = shared_data['best_solution']
            best_overall_cost = shared_data['best_cost']

        # SA를 여러 번 실행하여 해 개선
        # (기존에는 for loop였으나, 협력 모델에서는 단일 실행으로도 충분)
        remaining_time = timelimit - (time.time() - self.start_time)
        if remaining_time > 5:  # SA 실행을 위한 최소 시간
            # SA 실행 시 공유 객체 전달
            sa_sol, sa_cost = self._run_simulated_annealing(
                best_overall_sol,
                best_overall_cost,
                remaining_time - 2,
                shared_data,
                lock
            )
            if sa_cost < best_overall_cost:
                best_overall_sol = sa_sol
                best_overall_cost = sa_cost

        # 최종 결과를 반환할 필요는 없지만, 로깅을 위해 남겨둘 수 있음

        return best_overall_sol, best_overall_cost


# --- Main Execution Logic ---
def run_single_algorithm(prob_info, timelimit, shared_data, lock):  # <-- shared_data, lock 추가
    """
    단일 워커가 실행하는 알고리즘.
    공유 객체를 받아서 Optimizer의 solve 메서드로 전달합니다.
    """
    optimizer = PortOptimizer(prob_info)
    # solve 메서드에 공유 객체 전달
    optimizer.solve(timelimit, shared_data, lock)
    # 이제 이 함수는 결과를 반환할 필요가 없습니다. 결과는 shared_data에 저장됩니다.


def algorithm(prob_info, timelimit=60):
    """
    협력적 병렬 처리를 총괄하는 함수.
    """
    NUM_PARALLEL_RUNS = 4

    # Manager를 통해 공유 객체 생성
    with multiprocessing.Manager() as manager:
        shared_data = manager.dict()
        shared_data['best_cost'] = float('inf')
        shared_data['best_solution'] = None
        shared_data["num_workers"] = NUM_PARALLEL_RUNS
        shared_data["worker_costs"] = manager.dict()
        lock = manager.Lock()

        # 각 워커에 공유 객체를 인자로 전달
        args = [(prob_info, timelimit, shared_data, lock) for _ in range(NUM_PARALLEL_RUNS)]

        # 워커 풀 실행
        with multiprocessing.Pool(processes=NUM_PARALLEL_RUNS) as pool:
            pool.starmap(run_single_algorithm, args)

        # 모든 작업이 끝난 후, 공유 저장소에 저장된 최종 결과를 반환

        return shared_data['best_solution']


if __name__ == "__main__":
    # You can run this file to test your algorithm from terminal.

    import json
    import os
    import sys
    import jsbeautifier


    def numpy_to_python(obj):
        if isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.float64) or isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        raise TypeError(f"Type {type(obj)} not serializable")


    # Arguments list should be problem_name, problem_file, timelimit (in seconds)
    if len(sys.argv) == 4:
        prob_name = sys.argv[1]
        prob_file = sys.argv[2]
        timelimit = int(sys.argv[3])

        with open(prob_file, 'r') as f:
            prob_info = json.load(f)

        exception = None
        solution = None

        try:

            alg_start_time = time.time()

            # Run algorithm!
            solution = algorithm(prob_info, timelimit)

            alg_end_time = time.time()


            checked_solution = util.check_feasibility(prob_info, solution)

            checked_solution['time'] = alg_end_time - alg_start_time
            checked_solution['timelimit_exception'] = (alg_end_time - alg_start_time) > timelimit + 2 # allowing additional 2 second!
            checked_solution['exception'] = exception

            checked_solution['prob_name'] = prob_name
            checked_solution['prob_file'] = prob_file


            with open('results.json', 'w') as f:
                opts = jsbeautifier.default_options()
                opts.indent_size = 2
                f.write(jsbeautifier.beautify(json.dumps(checked_solution, default=numpy_to_python), opts))
                print(f'Results are saved as file results.json')

            sys.exit(0)

        except Exception as e:
            print(f"Exception: {repr(e)}")
            sys.exit(1)

    else:
        print("Usage: python myalgorithm.py <problem_name> <problem_file> <timelimit_in_seconds>")
        sys.exit(2)

# if __name__ == "__main__":
#     # --- 라이브러리 임포트 ---
#     import json, os, sys, csv, jsbeautifier
#
#     # --- 배치 실행 설정 ---
#     # exercise_problems 디렉토리에 있는 prob1.json ~ prob10.json을 실행합니다.
#     NUM_PROBLEMS = 10
#     PROBLEM_DIR = "stage2_exercise_problems"
#     OUTPUT_CSV_FILE = "results.csv"
#     TIMELIMIT_PER_PROBLEM = 60  # 문제당 시간 제한(초)
#
#     # 결과를 저장할 리스트
#     results_for_csv = []
#
#     print(f"--- Starting Batch Processing for {NUM_PROBLEMS} problems ---")
#
#     # --- 문제 파일 순회 루프 ---
#     for i in range(1, NUM_PROBLEMS + 1):
#         prob_name = f"prob{i}"
#         prob_file = os.path.join(PROBLEM_DIR, f"{prob_name}.json")
#
#         print(f"\n{'=' * 20} Running {prob_name} {'=' * 20}")
#
#         # 파일 존재 여부 확인
#         if not os.path.exists(prob_file):
#             print(f"File not found: {prob_file}. Skipping.")
#             results_for_csv.append([prob_name, "File Not Found"])
#             continue
#
#         try:
#             # 문제 파일 로드
#             with open(prob_file, 'r') as f:
#                 prob_info = json.load(f)
#
#             # 알고리즘 실행
#             solution = algorithm(prob_info, TIMELIMIT_PER_PROBLEM)
#
#             # 결과 검증 및 obj 값 추출
#             checked_solution = util.check_feasibility(prob_info, solution)
#
#             obj_value = 'Infeasible'  # 기본값
#             if checked_solution.get('feasible', False):
#                 obj_value = checked_solution.get('obj', 'N/A')
#                 print(f"✅ {prob_name} successful. Objective: {obj_value}")
#             else:
#                 print(f"❌ {prob_name} resulted in an infeasible solution.")
#                 # 비현실적인 이유를 출력하고 싶다면 아래 주석을 해제하세요.
#                 # print(f"   Reason: {checked_solution.get('infeasibility')}")
#
#             results_for_csv.append([prob_name, obj_value])
#
#         except Exception as e:
#             import traceback
#
#             print(f"❌ An exception occurred while running {prob_name}: {repr(e)}")
#             traceback.print_exc()
#             results_for_csv.append([prob_name, 'Error'])
#
#     # --- CSV 파일로 결과 저장 ---
#     print(f"\n--- Batch processing complete. Writing results to {OUTPUT_CSV_FILE} ---")
#     try:
#         with open(OUTPUT_CSV_FILE, 'w', newline='') as csvfile:
#             writer = csv.writer(csvfile)
#             # CSV 헤더 작성
#             writer.writerow(['problem_name', 'objective_cost'])
#             # 데이터 작성
#             writer.writerows(results_for_csv)
#         print(f"✅ Successfully saved results to {OUTPUT_CSV_FILE}")
#     except Exception as e:
#         print(f"❌ Failed to write CSV file: {repr(e)}")
#
#     sys.exit(0)