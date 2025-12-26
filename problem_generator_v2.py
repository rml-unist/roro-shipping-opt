"""
Stage 2 형식 RoRo Ship Stowage 문제 생성기

Stage 2와 동일한 3D Multi-deck Grid 구조:
- 다중 Deck (1-4개)
- Ramp로 연결
- 불완전한 Grid (빈 노드)
- 노드 타입: gate, hold, ramp
"""

import json
import random
import numpy as np
from collections import defaultdict
import os


def generate_stage2_graph(
    num_decks=None,
    grid_width=None,
    grid_height=None,
    missing_ratio=None,
    ramps_per_connection=None
):
    """
    Stage 2 형식의 3D Multi-deck Grid 그래프 생성

    Returns:
        N: 노드 수
        E: 엣지 리스트 [[a, b], ...]
        grid_graph: 상세 그래프 정보
    """
    # 파라미터 랜덤 설정 (Stage 2 분포 기반)
    if num_decks is None:
        num_decks = random.choices([1, 2, 3, 4], weights=[0.1, 0.25, 0.35, 0.3])[0]

    if grid_width is None:
        if num_decks == 1:
            grid_width = random.randint(15, 30)  # 1 deck은 넓게
        else:
            grid_width = random.randint(4, 12)

    if grid_height is None:
        grid_height = random.randint(3, 6)

    if missing_ratio is None:
        missing_ratio = random.uniform(0.05, 0.25)  # 5-25% 노드 누락

    if ramps_per_connection is None:
        ramps_per_connection = random.randint(2, 4)

    # 노드 생성
    nodes = []  # [[x, y, deck], {info}]
    node_id = 0
    coord_to_id = {}  # (x, y, deck) -> node_id

    # Gate (항상 [0, 0, 0])
    gate_info = {
        "pos": [0, 0.0],
        "type": "gate",
        "id": 0,
        "distance": 0
    }
    nodes.append([[0, 0, 0], gate_info])
    coord_to_id[(0, 0, 0)] = 0
    node_id = 1

    # 각 deck별로 grid 생성
    deck_configs = []
    for deck in range(num_decks):
        # Deck별 크기 변동 (약간의 불규칙성)
        if deck == 0:
            w, h = grid_width, grid_height
        else:
            # 다른 deck은 약간 다른 크기 가능
            w = grid_width + random.randint(-2, 2)
            h = grid_height + random.randint(-1, 1)
            w = max(3, min(w, grid_width + 2))
            h = max(2, min(h, grid_height + 1))

        # 이 deck에서 누락할 위치들
        total_positions = w * h
        num_missing = int(total_positions * missing_ratio * random.uniform(0.5, 1.5))

        # 가장자리 노드는 덜 누락되도록
        missing_positions = set()
        attempts = 0
        while len(missing_positions) < num_missing and attempts < total_positions * 2:
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            # Gate 위치는 누락 불가
            if deck == 0 and x == 0 and y == 0:
                attempts += 1
                continue
            # 가장자리는 50% 확률로만 누락
            if x == 0 or x == w - 1 or y == 0 or y == h - 1:
                if random.random() < 0.5:
                    attempts += 1
                    continue
            missing_positions.add((x, y))
            attempts += 1

        deck_configs.append({
            'width': w,
            'height': h,
            'missing': missing_positions
        })

        # 노드 생성
        for x in range(w):
            for y in range(h):
                if (x, y) in missing_positions:
                    continue
                if deck == 0 and x == 0 and y == 0:
                    continue  # Gate는 이미 추가됨

                # 거리 계산 (BFS로 나중에 재계산)
                distance = x + y + deck * 5  # 임시 거리

                # 노드 타입 결정 (나중에 ramp 설정)
                node_type = "hold"

                info = {
                    "pos": [x, y + deck * (grid_height + 1.8)],  # y offset for visualization
                    "type": node_type,
                    "id": node_id,
                    "distance": distance
                }
                nodes.append([[x, y, deck], info])
                coord_to_id[(x, y, deck)] = node_id
                node_id += 1

    # 엣지 생성 (같은 deck 내)
    edges = []  # [[coord1, coord2, {}]]
    E = []  # [[id1, id2], ...]

    for deck in range(num_decks):
        config = deck_configs[deck]
        w, h = config['width'], config['height']
        missing = config['missing']

        for x in range(w):
            for y in range(h):
                if (x, y) in missing:
                    continue
                if (x, y, deck) not in coord_to_id:
                    continue

                current_id = coord_to_id[(x, y, deck)]

                # 오른쪽 연결
                if x + 1 < w and (x + 1, y) not in missing:
                    if (x + 1, y, deck) in coord_to_id:
                        neighbor_id = coord_to_id[(x + 1, y, deck)]
                        edges.append([[x, y, deck], [x + 1, y, deck], {}])
                        E.append([current_id, neighbor_id])

                # 아래쪽 연결
                if y + 1 < h and (x, y + 1) not in missing:
                    if (x, y + 1, deck) in coord_to_id:
                        neighbor_id = coord_to_id[(x, y + 1, deck)]
                        edges.append([[x, y, deck], [x, y + 1, deck], {}])
                        E.append([current_id, neighbor_id])

    # Ramp 생성 (deck 간 연결) - 반드시 연결 보장
    ramp_nodes = []
    for deck in range(num_decks - 1):
        # 두 deck 모두에 존재하는 위치 찾기
        config_lower = deck_configs[deck]
        config_upper = deck_configs[deck + 1]

        common_positions = []
        for x in range(min(config_lower['width'], config_upper['width'])):
            for y in range(min(config_lower['height'], config_upper['height'])):
                if (x, y) not in config_lower['missing'] and (x, y) not in config_upper['missing']:
                    if (x, y, deck) in coord_to_id and (x, y, deck + 1) in coord_to_id:
                        # Gate 제외
                        if deck == 0 and x == 0 and y == 0:
                            continue
                        common_positions.append((x, y))

        # 공통 위치가 없으면 강제로 하나 생성
        if not common_positions:
            # 두 deck에서 가장 가까운 노드 쌍 찾기
            for x in range(max(config_lower['width'], config_upper['width'])):
                for y in range(max(config_lower['height'], config_upper['height'])):
                    if (x, y, deck) in coord_to_id and (x, y, deck + 1) in coord_to_id:
                        common_positions.append((x, y))
                        break
                if common_positions:
                    break

            # 여전히 없으면 missing 노드를 복원해서라도 연결
            if not common_positions:
                # lower deck의 첫 번째 노드와 upper deck의 첫 번째 노드를 찾아 연결
                lower_nodes = [(x, y) for (x, y, d) in coord_to_id if d == deck]
                upper_nodes = [(x, y) for (x, y, d) in coord_to_id if d == deck + 1]
                if lower_nodes and upper_nodes:
                    # 같은 x,y 위치 중 하나에 노드 추가
                    for lx, ly in lower_nodes:
                        if (lx, ly) in upper_nodes:
                            common_positions.append((lx, ly))
                            break
                    if not common_positions and lower_nodes:
                        # 강제로 upper deck에 노드 추가
                        lx, ly = lower_nodes[0]
                        info = {
                            "pos": [lx, ly + (deck + 1) * (grid_height + 1.8)],
                            "type": "hold",
                            "id": node_id,
                            "distance": 0
                        }
                        nodes.append([[lx, ly, deck + 1], info])
                        coord_to_id[(lx, ly, deck + 1)] = node_id
                        node_id += 1
                        common_positions.append((lx, ly))

        # Ramp 위치 선택 (최소 1개 보장)
        num_ramps = max(1, min(ramps_per_connection, len(common_positions)))
        if common_positions:
            ramp_positions = random.sample(common_positions, min(num_ramps, len(common_positions)))

            for (x, y) in ramp_positions:
                lower_id = coord_to_id[(x, y, deck)]
                upper_id = coord_to_id[(x, y, deck + 1)]

                # Ramp 엣지 추가
                edges.append([[x, y, deck], [x, y, deck + 1], {"ramp": True}])
                E.append([lower_id, upper_id])

                # 노드 타입을 ramp로 변경
                ramp_nodes.append(lower_id)
                ramp_nodes.append(upper_id)

    # 노드 타입 업데이트
    for node in nodes:
        if node[1]['id'] in ramp_nodes:
            node[1]['type'] = 'ramp'

    # 거리 재계산 (BFS from gate) 및 연결성 보장
    from collections import deque

    def rebuild_adj():
        adj = defaultdict(list)
        for a, b in E:
            adj[a].append(b)
            adj[b].append(a)
        return adj

    def compute_distances():
        adj = rebuild_adj()
        dists = {0: 0}
        queue = deque([0])
        while queue:
            current = queue.popleft()
            for neighbor in adj[current]:
                if neighbor not in dists:
                    dists[neighbor] = dists[current] + 1
                    queue.append(neighbor)
        return dists

    distances = compute_distances()

    # 연결되지 않은 노드 확인 및 연결
    all_node_ids = set(coord_to_id.values())
    disconnected = all_node_ids - set(distances.keys())

    max_iterations = len(disconnected) + 10
    iteration = 0
    while disconnected and iteration < max_iterations:
        iteration += 1

        # 연결되지 않은 노드 중 하나 선택
        disc_id = next(iter(disconnected))

        # 이 노드의 좌표 찾기
        disc_coord = None
        for coord, nid in coord_to_id.items():
            if nid == disc_id:
                disc_coord = coord
                break

        if disc_coord is None:
            break

        x, y, deck = disc_coord

        # 연결된 노드 중 가장 가까운 같은 deck 노드 찾기
        connected_same_deck = []
        for coord, nid in coord_to_id.items():
            if coord[2] == deck and nid in distances:
                dist = abs(coord[0] - x) + abs(coord[1] - y)
                connected_same_deck.append((dist, coord, nid))

        if connected_same_deck:
            connected_same_deck.sort()
            _, closest_coord, closest_id = connected_same_deck[0]

            # 엣지 추가
            E.append([disc_id, closest_id])
            edges.append([[x, y, deck], list(closest_coord), {}])
        else:
            # 다른 deck의 연결된 노드와 연결 (ramp 추가)
            connected_any = []
            for coord, nid in coord_to_id.items():
                if nid in distances:
                    dist = abs(coord[0] - x) + abs(coord[1] - y) + abs(coord[2] - deck) * 10
                    connected_any.append((dist, coord, nid))

            if connected_any:
                connected_any.sort()
                _, closest_coord, closest_id = connected_any[0]
                E.append([disc_id, closest_id])
                if closest_coord[2] != deck:
                    edges.append([[x, y, deck], list(closest_coord), {"ramp": True}])
                    ramp_nodes.append(disc_id)
                    ramp_nodes.append(closest_id)
                else:
                    edges.append([[x, y, deck], list(closest_coord), {}])

        # BFS 재실행
        distances = compute_distances()
        disconnected = all_node_ids - set(distances.keys())

    # 노드 타입 업데이트 (ramp_nodes 반영)
    for node in nodes:
        if node[1]['id'] in ramp_nodes:
            node[1]['type'] = 'ramp'

    # 거리 업데이트
    for node in nodes:
        nid = node[1]['id']
        if nid in distances:
            node[1]['distance'] = distances[nid]

    N = len(nodes)

    grid_graph = {
        "nodes": nodes,
        "edges": edges
    }

    return N, E, grid_graph


def generate_demands_stage2(P, capacity, situation='normal'):
    """
    Stage 2 스타일 수요 생성

    Args:
        P: 포트 수
        capacity: 최대 수용량 (N-1)
        situation: 상황 타입
    """
    K = []

    # 상황별 파라미터
    if situation == 'normal':
        fill_ratio = random.uniform(1.0, 1.5)
        long_trip_ratio = 0.3
    elif situation == 'tight':
        fill_ratio = random.uniform(1.5, 2.2)
        long_trip_ratio = 0.4
    elif situation == 'crossing':
        fill_ratio = random.uniform(1.2, 1.8)
        long_trip_ratio = 0.6
    elif situation == 'rehandling':
        fill_ratio = random.uniform(1.3, 1.7)
        long_trip_ratio = 0.2
    elif situation == 'sparse':
        fill_ratio = random.uniform(0.5, 0.8)
        long_trip_ratio = 0.3
    else:
        fill_ratio = random.uniform(0.8, 1.5)
        long_trip_ratio = 0.3

    target_total = int(capacity * fill_ratio)
    current_total = 0

    while current_total < target_total:
        if situation == 'rehandling':
            # 중간 포트 집중
            mid = P // 2
            if random.random() < 0.4:
                origin = random.randint(0, max(0, mid - 1))
                dest = mid
            elif random.random() < 0.6:
                origin = mid
                dest = random.randint(min(P - 1, mid + 1), P - 1)
            else:
                origin = random.randint(0, P - 2)
                dest = random.randint(origin + 1, P - 1)
        elif situation == 'crossing' and random.random() < long_trip_ratio:
            # 긴 여정
            origin = random.randint(0, max(0, P // 3 - 1))
            dest = random.randint(min(P - 1, P * 2 // 3), P - 1)
        else:
            # 일반
            origin = random.randint(0, P - 2)
            dest = random.randint(origin + 1, P - 1)

        # 수량 결정
        if situation == 'tight':
            count = random.randint(3, max(3, int(capacity * 0.15)))
        else:
            count = random.randint(1, max(1, int(capacity * 0.1)))

        if current_total + count > target_total:
            count = target_total - current_total

        if count > 0:
            K.append([[origin, dest], count])
            current_total += count

    # 동일한 OD 합치기
    demand_dict = defaultdict(int)
    for (origin, dest), count in K:
        demand_dict[(origin, dest)] += count

    K = [[[o, d], c] for (o, d), c in demand_dict.items()]

    return K


def generate_problem_stage2(
    situation='normal',
    num_decks=None,
    grid_width=None,
    grid_height=None,
    P=None
):
    """
    Stage 2 형식 문제 생성

    Args:
        situation: 'normal', 'tight', 'crossing', 'rehandling', 'sparse'
        num_decks: deck 수 (None이면 랜덤)
        grid_width: grid 너비 (None이면 랜덤)
        grid_height: grid 높이 (None이면 랜덤)
        P: 포트 수 (None이면 랜덤)
    """
    # 포트 수
    if P is None:
        P = random.randint(8, 20)

    # 그래프 생성
    N, E, grid_graph = generate_stage2_graph(
        num_decks=num_decks,
        grid_width=grid_width,
        grid_height=grid_height
    )

    # 수요 생성
    capacity = N - 1
    K = generate_demands_stage2(P, capacity, situation)

    if not K:
        K = [[[0, P - 1], max(1, capacity // 4)]]

    # LB 추정
    total_demand = sum(c for _, c in K)
    avg_distance = sum(n[1]['distance'] for n in grid_graph['nodes']) / N
    LB = int(total_demand * avg_distance * 1.5)

    return {
        'N': N,
        'E': E,
        'P': P,
        'K': K,
        'F': 100,
        'LB': LB,
        'grid_graph': grid_graph,
        'situation': situation,
        'num_decks': len(set(n[0][2] for n in grid_graph['nodes']))
    }


def check_feasibility(prob):
    """myalgorithm_orin으로 feasibility 체크"""
    import io
    from contextlib import redirect_stdout

    try:
        # Import here to avoid circular dependency
        from myalgorithm_orin import algorithm
        import util

        with redirect_stdout(io.StringIO()):
            solution = algorithm(prob, timelimit=30)
        result = util.check_feasibility(prob, solution)
        return result.get('feasible', False)
    except Exception:
        return False


def generate_diverse_dataset_v2(
    num_problems=500,
    output_dir='diverse_training_data_v2',
    situation_weights=None,
    validate_feasibility=True
):
    """
    다양한 Stage 2 형식 데이터셋 생성
    validate_feasibility=True면 각 문제를 myalgorithm_orin으로 검증
    """
    if situation_weights is None:
        situation_weights = {
            'normal': 0.25,
            'tight': 0.20,
            'crossing': 0.20,
            'rehandling': 0.20,
            'sparse': 0.15
        }

    os.makedirs(output_dir, exist_ok=True)

    situations = list(situation_weights.keys())
    weights = list(situation_weights.values())

    stats = defaultdict(list)
    total_attempts = 0

    print(f"Generating {num_problems} Stage 2 format problems...")
    print(f"Situation weights: {situation_weights}")
    if validate_feasibility:
        print("Feasibility validation: ENABLED (using myalgorithm_orin)")

    i = 1
    while i <= num_problems:
        situation = random.choices(situations, weights=weights)[0]

        # 다양한 크기 분포
        size_type = random.choices(['small', 'medium', 'large'], weights=[0.3, 0.4, 0.3])[0]

        if size_type == 'small':
            num_decks = random.choices([1, 2, 3], weights=[0.2, 0.5, 0.3])[0]
            grid_width = random.randint(4, 8)
            grid_height = random.randint(3, 4)
        elif size_type == 'medium':
            num_decks = random.choices([2, 3, 4], weights=[0.3, 0.4, 0.3])[0]
            grid_width = random.randint(6, 12)
            grid_height = random.randint(4, 6)
        else:  # large
            num_decks = random.choices([1, 2, 3, 4], weights=[0.15, 0.25, 0.35, 0.25])[0]
            if num_decks == 1:
                grid_width = random.randint(20, 30)
                grid_height = random.randint(4, 5)
            else:
                grid_width = random.randint(8, 15)
                grid_height = random.randint(5, 7)

        prob = generate_problem_stage2(
            situation=situation,
            num_decks=num_decks,
            grid_width=grid_width,
            grid_height=grid_height
        )

        total_attempts += 1

        # Feasibility 검증
        if validate_feasibility:
            if not check_feasibility(prob):
                # 실패시 재생성
                continue

        # 저장
        filepath = os.path.join(output_dir, f'prob{i}.json')
        with open(filepath, 'w') as f:
            json.dump(prob, f, indent=2)

        # 통계 수집
        stats['N'].append(prob['N'])
        stats['P'].append(prob['P'])
        stats['K_count'].append(len(prob['K']))
        stats['num_decks'].append(prob['num_decks'])
        stats['situation'].append(situation)

        total_demand = sum(c for _, c in prob['K'])
        stats['capacity_ratio'].append(total_demand / (prob['N'] - 1) * 100)

        # Ramp 수 계산
        ramp_count = sum(1 for n in prob['grid_graph']['nodes'] if n[1]['type'] == 'ramp')
        stats['ramp_count'].append(ramp_count)

        if i % 100 == 0:
            success_rate = i / total_attempts * 100
            print(f"  Generated {i}/{num_problems} problems (attempts: {total_attempts}, success rate: {success_rate:.1f}%)")

        i += 1

    # 통계 출력
    print("\n" + "=" * 60)
    print("Generated Dataset Statistics (Stage 2 Format)")
    print("=" * 60)
    if validate_feasibility:
        print(f"Total attempts: {total_attempts}, Success rate: {num_problems/total_attempts*100:.1f}%")
    print(f"N: min={min(stats['N'])}, max={max(stats['N'])}, avg={np.mean(stats['N']):.1f}")
    print(f"P: min={min(stats['P'])}, max={max(stats['P'])}, avg={np.mean(stats['P']):.1f}")
    print(f"K: min={min(stats['K_count'])}, max={max(stats['K_count'])}, avg={np.mean(stats['K_count']):.1f}")
    print(f"Decks: min={min(stats['num_decks'])}, max={max(stats['num_decks'])}, avg={np.mean(stats['num_decks']):.1f}")
    print(f"Ramps: min={min(stats['ramp_count'])}, max={max(stats['ramp_count'])}, avg={np.mean(stats['ramp_count']):.1f}")
    print(f"Capacity: min={min(stats['capacity_ratio']):.1f}%, max={max(stats['capacity_ratio']):.1f}%, avg={np.mean(stats['capacity_ratio']):.1f}%")

    # 상황별 분포
    from collections import Counter
    situation_counts = Counter(stats['situation'])
    print(f"\nSituation breakdown:")
    for sit, count in sorted(situation_counts.items()):
        print(f"  {sit}: {count} ({count/num_problems*100:.1f}%)")

    # Deck 분포
    deck_counts = Counter(stats['num_decks'])
    print(f"\nDeck count breakdown:")
    for deck, count in sorted(deck_counts.items()):
        print(f"  {deck} decks: {count} ({count/num_problems*100:.1f}%)")

    print(f"\nSaved to {output_dir}/")

    return stats


if __name__ == "__main__":
    import sys

    num_problems = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'diverse_training_data_v2'

    generate_diverse_dataset_v2(num_problems, output_dir)
