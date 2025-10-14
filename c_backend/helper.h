#ifndef HELPER_H
#define HELPER_H

#include <map>
#include <queue>
#include <mutex>
#include <vector>
#include <iostream>
#include <optional>
#include <unordered_set>
#include <limits>
#include <algorithm>
#include <functional>
#include <unordered_map>

// --- 타입 별칭 ---
using Path = std::vector<int>;
using Move = std::pair<Path, int>;
using PortSchedule = std::vector<Move>;
using SolutionType = std::map<int, PortSchedule>;
using DemandInfo = std::vector<std::pair<std::pair<int, int>, int>>;
using StrategicTunnelNodes = std::unordered_set<int>;
using ShortestPaths = std::vector<std::vector<std::vector<int>>>;
using Edge = std::vector<std::pair<int, int>>;

// 히트맵 타입 정의
using Heatmap = std::unordered_map<int, std::unordered_map<int, int>>;

struct LexiDijkstraResult {
    std::vector<std::pair<int, int>> dist; // {blocks, dist}
    std::vector<int> prev;
};

// min_blocking_path 결과 타입
struct BlockingPathResult {
    Path path;
    std::vector<int> blocking_nodes;
};

// 재핸들링 결과 타입
struct UnloadingResult {
    PortSchedule routes;
    std::vector<int> rehandling_demands;
    std::vector<int> final_occupation;
};

// 로딩 결과 타입
struct LoadingResult {
    std::optional<PortSchedule> routes;
    std::vector<int> final_occupation;
};

// 수정된 LoadingResult 구조체
struct LoadingResultWithHeatmap {
    std::optional<PortSchedule> routes;
    std::vector<int> final_occupation;
    Heatmap updated_heatmaps;  // 추가
};

// Feasibility Check 시 필요
struct FeasibilityResult {
    bool feasible;
    std::optional<double> obj;
    std::optional<std::vector<std::string>> infeasibility;
    SolutionType solution;
};

struct PairHash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        // 간단한 해시 조합 방식
        return h1 ^ (h2 << 1);
    }
};

struct SharedData {
    std::mutex mtx; // 데이터 접근을 제어할 뮤텍스
    double best_cost = std::numeric_limits<double>::infinity();
    std::optional<SolutionType> best_solution;
};

// --- C++ Graph 유틸리티 ---
using AdjacencyList = std::vector<std::vector<int>>;
using ShortestDistances = std::vector<int>;
const int INF = std::numeric_limits<int>::max();


void print_set(const std::unordered_set<int> &s) {
    std::cout << "{ ";
    for (int element : s) {
        std::cout << element << " ";
    }
    std::cout << "}" << std::endl;
}

// 어떤 타입(T)의 벡터라도 출력할 수 있는 템플릿 함수
template <typename T>
void print_vector(const std::vector<T>& vec) {
    for (const auto& element : vec) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
}

void print_string_vector(const std::vector<std::string>& vec, const std::string& title = "") {
    if (!title.empty()) {
        std::cout << title << std::endl;
    }

    if (vec.empty()) {
        std::cout << "[]" << std::endl;
        return;
    }

    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << "\"" << vec[i] << "\"";
        // 마지막 원소가 아니면 쉼표와 공백을 추가
        if (i < vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

void print_port_schedule(const PortSchedule& schedule, const std::string& title = "Port Schedule") {
    std::cout << "--- " << title << " (Total " << schedule.size() << " moves) ---" << std::endl;

    // 스케줄이 비어있는 경우 메시지 출력 후 종료
    if (schedule.empty()) {
        std::cout << "(Schedule is empty)" << std::endl;
        std::cout << "-----------------------------------------" << std::endl;
        return;
    }

    // 인덱스 기반 for문을 사용하여 "Move 1", "Move 2" ... 와 같이 번호를 붙여줍니다.
    for (size_t i = 0; i < schedule.size(); ++i) {
        const auto& move = schedule[i]; // 현재 Move에 대한 참조
        const Path& path = move.first;   // Path에 대한 참조 (pair의 첫 번째 요소)
        int car_id = move.second;        // Car ID (pair의 두 번째 요소)

        // Move 번호와 차량 ID 출력
        std::cout << "Move " << i + 1 << ": { Car ID: " << car_id << ", Path: ";

        // Path (vector<int>)를 출력하는 내부 루프
        if (path.empty()) {
            std::cout << "(Empty Path)";
        } else {
            for (size_t j = 0; j < path.size(); ++j) {
                std::cout << path[j];
                // 마지막 요소가 아닐 경우에만 화살표 출력
                if (j < path.size() - 1) {
                    std::cout << " -> ";
                }
            }
        }
        std::cout << " }" << std::endl;
    }
    std::cout << "-----------------------------------------" << std::endl;
}

void print_loading_result(const LoadingResult& result) {
    std::cout << "================ Loading Result ================" << std::endl;

    // 1. routes (std::optional<PortSchedule>) 출력
    // .has_value()로 optional 객체 안에 값이 있는지 확인합니다.
    if (result.routes.has_value()) {
        // 값이 있다면, .value()로 실제 PortSchedule 객체를 얻어와서
        // 기존에 만든 printPortSchedule 함수를 호출합니다.
        print_port_schedule(result.routes.value(), "Generated Routes");
    } else {
        // 값이 없다면 (std::nullopt), 로딩이 실패했거나 경로가 없음을 알립니다.
        std::cout << "--- Generated Routes ---" << std::endl;
        std::cout << "(Routes not available / Loading failed)" << std::endl;
        std::cout << "--------------------------" << std::endl;
    }

    std::cout << std::endl; // 가독성을 위한 줄바꿈

    // 2. final_occupation (std::vector<int>) 출력
    std::cout << "--- Final Occupation State (Node: Car ID) ---" << std::endl;
    if (result.final_occupation.empty()) {
        std::cout << "(Occupation data is empty)" << std::endl;
    } else {
        // 각 노드(인덱스)에 어떤 차량이 주차되었는지 보기 쉽게 출력
        for (size_t i = 0; i < result.final_occupation.size(); ++i) {
            std::cout << "  Node " << i << ": " << result.final_occupation[i];
            // 5개씩 출력하고 줄바꿈하여 가독성 향상
            if ((i + 1) % 5 == 0) {
                std::cout << std::endl;
            } else {
                std::cout << "\t";
            }
        }
        std::cout << std::endl;
    }

    std::cout << "==============================================" << std::endl << std::endl;
}

// min_blocking_path 함수 추가
BlockingPathResult min_blocking_path(
    const AdjacencyList& adj_list,
    const std::vector<int>& node_allocations,
    int target,
    int gate = 0
) {
    const int INF_BLOCKS = 1000000000;
    const int INF_DIST = 1000000000;
    using CostPair = std::pair<int, int>; // (blocks, dist)

    int n_nodes = adj_list.size();
    std::vector<CostPair> dist(n_nodes, {INF_BLOCKS, INF_DIST});
    std::vector<int> prev(n_nodes, -1);

    dist[gate] = {0, 0};

    // Priority queue: (blocks, dist, node)
    using State = std::tuple<int, int, int>;
    std::priority_queue<State, std::vector<State>, std::greater<State>> pq;
    pq.push({0, 0, gate});

    while (!pq.empty()) {
        auto [b, d, u] = pq.top();
        pq.pop();

        if (std::make_pair(b, d) != dist[u]) continue;
        if (u == target) break;

        for (int v : adj_list[u]) {
            bool occupied = (node_allocations[v] != -1) && (v != target);
            int nb = b + (occupied ? 1 : 0);
            int nd = d + 1;

            if (std::make_pair(nb, nd) < dist[v]) {
                dist[v] = {nb, nd};
                prev[v] = u;
                pq.push({nb, nd, v});
            }
        }
    }

    // 경로 복원
    Path path;
    int cur = target;
    if (dist[target].first == INF_BLOCKS) {
        return {{}, {}};  // 경로 없음
    }

    while (cur != -1) {
        path.push_back(cur);
        cur = prev[cur];
    }
    std::reverse(path.begin(), path.end());

    // 차단 노드 찾기
    std::vector<int> blocking;
    for (size_t i = 1; i < path.size() - 1; ++i) {
        if (node_allocations[path[i]] != -1) {
            blocking.push_back(path[i]);
        }
    }

    return {path, blocking};
}

// BFS 구현
std::pair<std::vector<int>, std::vector<int>> bfs(
    const AdjacencyList &graph,
    const std::vector<int> &occ
) {
    int n = graph.size();
    std::vector<int> distances(n, -1);
    std::vector<int> reachable_nodes;
    std::queue<int> q;

    if (occ[0] == -1) {
        q.push(0);
        distances[0] = 0;
        reachable_nodes.push_back(0);
    }

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v : graph[u]) {
            if (distances[v] == -1 && occ[v] == -1) {
                distances[v] = distances[u] + 1;
                q.push(v);
                reachable_nodes.push_back(v);
            }
        }
    }

    return {reachable_nodes, distances};
}


// Dijkstra 구현
std::pair<std::vector<int>, std::vector<int>> dijkstra(const AdjacencyList& graph, const std::vector<int>& occ) {
    int n = graph.size();
    std::vector<int> dist(n, INF);
    std::vector<int> prev(n, -1);
    dist[0] = 0;

    using State = std::pair<int, int>;
    std::priority_queue<State, std::vector<State>, std::greater<State>> pq;
    pq.push({0, 0});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();

        if (d > dist[u]) continue;

        for (int v : graph[u]) {
            // int weight = (occ[v] != -1) ? 100000 : 1; // 점유된 노드는 높은 가중치
            int weight = 1;
            if (occ[v] != -1) continue;
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                prev[v] = u;
                pq.push({dist[v], v});
            }
        }
    }
    return {dist, prev};
}

// Path Backtracking 구현
Path path_backtracking(const std::vector<int>& prev, int start, int end) {
    Path path;
    for (int at = end; at != -1; at = prev[at]) {
        path.push_back(at);
    }
    std::reverse(path.begin(), path.end());
    if (path.empty() || path[0] != start) return {};
    return path;
}

std::vector<int> get_available_nodes(const int N, const std::vector<int>& occupation) {
    std::vector<int> available;
    for (int i = 0; i < N; ++i) {
        if (occupation[i] == -1) {
            available.push_back(i);
        }
    }
    return available;
}

FeasibilityResult check_feasibility(const int N, const Edge &E, const DemandInfo &K, const int P, const int F, const double LB, const SolutionType& solution) {
    // --- 1. 초기화 ---
    // 간선 정보를 빠른 조회를 위해 unordered_set으로 변환
    std::unordered_set<std::pair<int, int>, PairHash> edge_set;
    for (const auto& edge : E) {
        edge_set.insert(edge);
    }

    // Python: node_allocations = np.ones(N, dtype=int) * -1
    std::vector<int> node_allocations(N, -1);

    // Python: supposedly_loaded_demands_after_ports
    std::map<int, std::map<int, int>> supposedly_loaded_demands_after_ports;
    for (int p = 0; p < P; ++p) {
        for (size_t k = 0; k < K.size(); ++k) {
            if (K[k].first.first <= p && p < K[k].first.second) {
                supposedly_loaded_demands_after_ports[p][k] = K[k].second;
            }
        }
    }

    double obj = 0.0;
    std::vector<std::string> infeasibility;

    // --- 2. 유효성 검사 ---
    // C++에서는 Solution 타입 자체가 구조를 강제하므로 Python의 초반 구조 검사는 생략 가능.

    for (int p = 0; p < P; ++p) {
        if (solution.find(p) == solution.end()) {
            infeasibility.push_back("The solution does not contain route list for port " + std::to_string(p));
            continue; // 해당 포트 검사 중단
        }

        const auto& route_list = solution.at(p);

        for (const auto& move : route_list) {
            const auto& route = move.first;
            int k = move.second;

            // 경로 기본 유효성 검사
            if (route.size() <= 1) {
                infeasibility.push_back("The length of a route is less than 2");
            }
            if (!route.empty() && (*std::min_element(route.begin(), route.end()) < 0 || *std::max_element(route.begin(), route.end()) >= N)) {
                infeasibility.push_back("A route has invalid node index");
            }

            // 중복 노드 검사 (Python: len(route) != len(set(route)))
            std::unordered_set<int> unique_nodes(route.begin(), route.end());
            if (route.size() != unique_nodes.size()) {
                infeasibility.push_back("A route has a duplicated node index, i.e., the route should be simple.");
            }

            // 간선 유효성 및 상태 업데이트
            for (size_t i = 0; i < route.size() - 1; ++i) {
                int u = route[i];
                int v = route[i + 1];
                if (edge_set.find({u, v}) == edge_set.end() && edge_set.find({v, u}) == edge_set.end()) {
                    infeasibility.push_back("A route contains an invalid edge (" + std::to_string(u) + "," + std::to_string(v) + ")");
                }
            }

            if (route.empty()) continue; // 빈 경로는 더 이상 처리하지 않음

            // 경로 종류에 따른 처리 (로딩, 하역, 재배치)
            if (route.front() == 0) { // Loading
                int loading_node = route.back();
                if (node_allocations[loading_node] != -1) {
                    infeasibility.push_back("The loading node " + std::to_string(loading_node) + " is already occupied by demand " + std::to_string(node_allocations[loading_node]));
                }
                for (size_t i = 0; i < route.size() - 1; ++i) { // 경로 중간 노드 확인
                    if (node_allocations[route[i]] != -1) {
                        infeasibility.push_back("The loading route is blocked by node " + std::to_string(route[i]));
                    }
                }
                node_allocations[loading_node] = k;

            } else if (route.back() == 0) { // Unloading
                int unloading_node = route.front();
                if (node_allocations[unloading_node] == -1) {
                    infeasibility.push_back("The unloading node " + std::to_string(unloading_node) + " is not occupied by any demand");
                }
                for (size_t i = 1; i < route.size(); ++i) { // 경로 중간 노드 확인
                     if (node_allocations[route[i]] != -1) {
                        infeasibility.push_back("The unloading route is blocked by node " + std::to_string(route[i]));
                    }
                }
                node_allocations[unloading_node] = -1;

            } else { // Rehandling
                int unloading_node = route.front();
                int loading_node = route.back();
                if (node_allocations[unloading_node] == -1) {
                     infeasibility.push_back("The rehandling source node " + std::to_string(unloading_node) + " is not occupied");
                }
                if (node_allocations[loading_node] != -1) {
                     infeasibility.push_back("The rehandling destination node " + std::to_string(loading_node) + " is already occupied");
                }
                 for (size_t i = 1; i < route.size() - 1; ++i) {
                     if (node_allocations[route[i]] != -1) {
                        infeasibility.push_back("The rehandling route is blocked by node " + std::to_string(route[i]));
                    }
                }
                node_allocations[loading_node] = k;
                node_allocations[unloading_node] = -1;
            }
            obj += F + route.size() - 1;
        }

        // --- 3. 수요 만족도 검사 ---
        // Python: current_loading_status = Counter(node_allocations[node_allocations>=0])
        std::map<int, int> current_loading_status;
        for (int demand_k : node_allocations) {
            if (demand_k != -1) {
                current_loading_status[demand_k]++;
            }
        }

        const auto& supposed_status = supposedly_loaded_demands_after_ports[p];
        if (current_loading_status != supposed_status) {
            // 더 자세한 에러 메시지 생성
            for(const auto& supposed_pair : supposed_status) {
                auto it = current_loading_status.find(supposed_pair.first);

                if (it == current_loading_status.end()) {
                    // current_loading_status에 해당 수요가 없는 경우
                    infeasibility.push_back("Demand " + std::to_string(supposed_pair.first) + " is not loaded at port " + std::to_string(p));
                } else {
                    // 수요는 있지만 수량이 다른 경우
                    if (it->second != supposed_pair.second) {
                        infeasibility.push_back("Demand " + std::to_string(supposed_pair.first) + " quantity mismatch at port " + std::to_string(p));
                    }
                }
            }
        }
    }


    // --- 4. 최종 결과 반환 ---
    if (infeasibility.empty()) {
        obj = obj - LB;
        return {true, obj, std::nullopt, solution};
    } else {
        return {false, std::nullopt, infeasibility, solution};
    }

    }

std::vector<int> bfs_path_free_between(
    int N,
    const std::vector<std::vector<int>>& adj,
    const std::vector<int>& occ,
    int src,
    int dst
) {
    // 사전 조건: 출발지와 목적지가 같거나, 목적지가 비어있지 않으면 경로 없음
    if (src == dst || occ[dst] != -1) {
        return {}; // 빈 벡터 반환 (Python의 None에 해당)
    }

    std::vector<bool> seen(N, false);
    std::vector<int> prev(N, -1);
    std::queue<int> q;

    q.push(src);
    seen[src] = true;

    bool found = false;

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v : adj[u]) {
            // 제약 조건: 중간 노드(v가 dst가 아닌 경우)는 반드시 비어 있어야 함
            if (v != dst && occ[v] != -1) {
                continue;
            }

            if (!seen[v]) {
                seen[v] = true;
                prev[v] = u;

                if (v == dst) {
                    found = true;
                    break; // 목적지를 찾았으므로 이웃 탐색 중단
                }
                q.push(v);
            }
        }
        if (found) {
            break; // 외부 루프도 중단
        }
    }

    // 경로 복원
    if (found) {
        std::vector<int> path;
        int current = dst;
        while (current != -1) {
            path.push_back(current);
            current = prev[current];
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    return {}; // 목적지에 도달하지 못했으면 빈 벡터 반환
}

std::vector<int> _try_random_side_move(
    int N,
    const std::vector<std::vector<int>>& adj,
    const std::vector<int>& path_sel,
    const std::vector<int>& occ,
    int idx_bn,
    std::mt19937& rng
) {
    int bn = path_sel[idx_bn];

    // 금지 구역(ban) 구성: 게이트(0), 주 이동 경로, 경로 상의 다른 블로커들
    std::unordered_set<int> ban;
    for (int node : path_sel) {
        ban.insert(node);
    }
    ban.insert(0); // 게이트 제외

    // Python 코드의 이 부분은 이미 path_sel 전체가 ban에 포함되어 중복 로직이지만,
    // 원본을 충실히 따르기 위해 남겨둡니다.
    for (size_t i = 1; i < path_sel.size() - 1; ++i) {
        if (occ[path_sel[i]] != -1) {
            ban.insert(path_sel[i]);
        }
    }

    // 후보군 수집: 비어있고 금지 구역이 아닌 노드
    std::vector<int> raw_pool;
    for (int u = 0; u < N; ++u) {
        if (occ[u] == -1 && ban.find(u) == ban.end()) {
            raw_pool.push_back(u);
        }
    }

    if (raw_pool.empty()) {
        return {}; // 빈 벡터 반환으로 실패를 알림
    }

    // 속도-품질 절충을 위한 소표본 추출
    const int EMPTY_CAP = 64;
    std::shuffle(raw_pool.begin(), raw_pool.end(), rng);
    if (raw_pool.size() > EMPTY_CAP) {
        raw_pool.resize(EMPTY_CAP);
    }

    // 1) 실제로 bn -> u 가 빈칸만으로 이동 가능한 u만 필터링
    std::vector<int> reachables;
    std::unordered_map<int, std::vector<int>> path_cache;

    for (int u : raw_pool) {
        std::vector<int> path_bn_u = bfs_path_free_between(N, adj, occ, bn, u);
        if (!path_bn_u.empty()) {
            reachables.push_back(u);
            path_cache[u] = path_bn_u;
        }
    }

    if (reachables.empty()) {
        return {};
    }

    // 2) reachability-safe 필터 적용 (게이트 도달성 악화 방지)
    // 원본 코드에서 주석 처리되어 있었으므로 C++에서도 동일하게 처리
    // Path safe_cands = _reachability_safe_candidates(reachables, occ, occ[bn]);
    // if (safe_cands.empty()) {
    //     return {};
    // }

    // 3) 최종 후보 선택 (무작위)
    std::uniform_int_distribution<> distrib(0, reachables.size() - 1);
    int selected_node = reachables[distrib(rng)];

    // 캐시된 경로를 반환하거나, 만약 없다면 다시 계산 (안전장치)
    auto it = path_cache.find(selected_node);
    if (it != path_cache.end()) {
        return it->second;
    } else {
        return bfs_path_free_between(N, adj, occ, bn, selected_node);
    }
}

LexiDijkstraResult _lexi_dijkstra_all(
    int N,
    const std::vector<std::vector<int>>& adj,
    const std::vector<int>& occ,
    int gate = 0
) {
    // 비용(차단수, 거리)을 저장할 튜플 정의
    using Cost = std::pair<int, int>;
    // 우선순위 큐에 저장될 요소: {비용, 노드 인덱스}
    // C++의 pair는 첫 번째 요소부터 비교하므로 비용을 맨 앞에 둡니다.
    using PQ_Element = std::pair<Cost, int>;

    const int INF_VAL = 1e9;
    const Cost INF = {INF_VAL, INF_VAL};

    std::vector<Cost> dist(N, INF);
    std::vector<int> prev(N, -1);

    // 최소 힙으로 동작하는 우선순위 큐 선언
    std::priority_queue<PQ_Element, std::vector<PQ_Element>, std::greater<PQ_Element>> pq;

    // 시작 노드 초기화
    dist[gate] = {0, 0};
    pq.push({{0, 0}, gate});

    while (!pq.empty()) {
        // 우선순위 큐에서 가장 비용이 낮은 노드를 꺼냄
        auto [current_cost, u] = pq.top();
        pq.pop();

        // 더 낮은 비용으로 이미 처리된 노드라면 건너뜀
        if (current_cost > dist[u]) {
            continue;
        }

        // 모든 이웃 노드 v에 대해 반복
        for (int v : adj[u]) {
            // 이웃 노드 v를 경유할 때의 새로운 비용 계산
            int new_blocks = current_cost.first + (occ[v] != -1 ? 1 : 0);
            int new_dist = current_cost.second + 1;
            Cost new_cost = {new_blocks, new_dist};

            // 기존 비용보다 새로운 비용이 더 낮으면 갱신
            // std::pair는 사전식으로 비교되므로 Python의 튜플 비교와 동일
            if (new_cost < dist[v]) {
                dist[v] = new_cost;
                prev[v] = u;
                pq.push({new_cost, v});
            }
        }
    }

    return {dist, prev};
}

#endif //HELPER_H