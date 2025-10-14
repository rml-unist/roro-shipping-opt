#ifndef RUN_H
#define RUN_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <utility>
#include <random>
#include <chrono>
#include <cmath>
#include <vector>
#include <functional>
#include <set>
#include <numeric>
#include <queue>
#include <stack>
#include <algorithm>

#include "helper.h"
#include "heuristic.h"


// _apply_routes_to_occupancy 헬퍼 함수
std::vector<int> apply_routes_to_occupancy(
    const PortSchedule& routes,
    std::vector<int> occ,
    int N
) {
    for (const auto& [path, k] : routes) {
        if (path.empty()) continue;

        if (path.front() == 0 && path.back() != 0) {
            // Loading
            occ[path.back()] = k;
        } else if (path.back() == 0 && path.front() != 0) {
            // Unloading
            occ[path.front()] = -1;
        } else if (path.front() != 0 && path.back() != 0) {
            // Rehandling (intra-hold move)
            occ[path.front()] = -1;
            occ[path.back()] = k;
        }
    }
    return occ;
}

std::optional<SolutionType> create_neighbor_from_port(
    const int P,
    const int N,
    const DemandInfo& K,
    const AdjacencyList& adj_list,
    const ShortestDistances& shortest_distances,
    const ShortestPaths& shortest_paths,
    const std::vector<double>& node_centrality,
    Heatmap& route_heatmaps,
    const SolutionType& base_solution,
    bool use_SA = true  // SA 모드 활성화 여부
) {
    std::random_device rd;
    std::mt19937 gen(rd());

    // 1. 전략적 포트 선택
    std::vector<double> port_weights;
    for (int p = 0; p < P - 1; ++p) {
        double complexity = 0;
        auto it = base_solution.find(p);
        if (it != base_solution.end()) {
            const auto& routes = it->second;
            complexity = routes.size();
            for (const auto& [path, _] : routes) {
                complexity += path.size();
            }
        }
        port_weights.push_back(complexity);
    }

    int start_port;
    double total_weight = std::accumulate(port_weights.begin(), port_weights.end(), 0.0);

    if (total_weight > 0) {
        // 복잡도에 비례한 확률로 포트 선택
        std::discrete_distribution<> port_dist(port_weights.begin(), port_weights.end());
        start_port = port_dist(gen);
    } else {
        // 균등 확률로 선택
        std::uniform_int_distribution<> port_dist(0, P - 2);
        start_port = port_dist(gen);
    }

    // 2. 새 솔루션 초기화 (base_solution 복사)
    SolutionType new_sol = base_solution;

    // 3. start_port 직전까지의 점유 상태 복원
    std::vector<int> occ(N, -1);
    for (int t = 0; t < start_port; ++t) {
        auto it = new_sol.find(t);
        if (it != new_sol.end()) {
            occ = apply_routes_to_occupancy(it->second, occ, N);
        }
    }

    // 4. start_port에서 언로딩 수행
    auto unload_result = _unloading_heuristic_updated(K, adj_list, start_port, occ);
    PortSchedule unload_routes = unload_result.routes;
    std::vector<int> rehandling = unload_result.rehandling_demands;
    occ = unload_result.final_occupation;

    // 5. start_port에서 로딩 수행 (SA=true)
    LoadingResultWithHeatmap load_result;
    if (use_SA) {
        load_result = _loading_heuristic_updated(
            N, P, K, adj_list, shortest_distances,
            node_centrality, route_heatmaps,
            start_port, occ, rehandling,
            true, &gen  // SA 모드 활성화
        );
    } else {
        load_result = _loading_heuristic_updated(
            N, P, K, adj_list, shortest_distances,
            node_centrality, route_heatmaps,
            start_port, occ, rehandling,
            false, nullptr  // SA 모드 비활성화
        );
    }

    if (!load_result.routes.has_value()) {
        return std::nullopt;  // 로딩 실패
    }

    PortSchedule load_routes = load_result.routes.value();
    occ = load_result.final_occupation;
    route_heatmaps = load_result.updated_heatmaps;

    // 6. start_port의 경로 업데이트
    PortSchedule combined_routes = unload_routes;
    combined_routes.insert(combined_routes.end(), load_routes.begin(), load_routes.end());
    new_sol[start_port] = combined_routes;

    // 7. start_port+1부터 P-1까지 순차적으로 재시뮬레이션
    for (int t = start_port + 1; t < P; ++t) {
        // 언로딩
        auto unload_res = _unloading_heuristic_updated(K, adj_list, t, occ);
        PortSchedule unload_routes_t = unload_res.routes;
        std::vector<int> rehandling_t = unload_res.rehandling_demands;
        occ = unload_res.final_occupation;

        // 로딩 (마지막 포트가 아닌 경우만)
        if (t < P - 1) {
            auto load_res = _loading_heuristic_updated(
                N, P, K, adj_list, shortest_distances,
                node_centrality, route_heatmaps,
                t, occ, rehandling_t,
                false, nullptr  // 후속 포트는 SA 모드 비활성화
            );

            if (!load_res.routes.has_value()) {
                return std::nullopt;  // 로딩 실패
            }

            PortSchedule load_routes_t = load_res.routes.value();
            occ = load_res.final_occupation;
            route_heatmaps = load_res.updated_heatmaps;

            // 경로 결합
            PortSchedule combined_t = unload_routes_t;
            combined_t.insert(combined_t.end(), load_routes_t.begin(), load_routes_t.end());
            new_sol[t] = combined_t;
        } else {
            // 마지막 포트는 언로딩만
            new_sol[t] = unload_routes_t;
        }
    }

    return new_sol;
}

// 선택적: 노드 중심성 계산 함수 (NetworkX의 betweenness_centrality 등을 C++로 구현)
void calculate_node_centrality(
    const AdjacencyList& adj_list,
    std::vector<double>& node_centrality
) {
    int N = adj_list.size();

    // 간단한 degree centrality 계산 (실제로는 더 복잡한 메트릭 필요)
    for (int i = 0; i < N; ++i) {
        node_centrality[i] = static_cast<double>(adj_list[i].size()) / (N - 1);
    }

    // 게이트(노드 0)로부터의 접근성 가중치 추가
    auto [distances, _] = dijkstra(adj_list, std::vector<int>(N, -1));
    for (int i = 0; i < N; ++i) {
        if (distances[i] != INF) {
            double accessibility = 1.0 / (1.0 + distances[i]);
            node_centrality[i] *= (1.0 + 0.2 * accessibility);
        }
    }
}

// 선택적: 히트맵 재구성 함수
void rebuild_heatmaps(
    const SolutionType& solution,
    const DemandInfo& K,
    int P,
    Heatmap& route_heatmaps
) {
    route_heatmaps.clear();

    for (int p = 0; p < P - 1; ++p) {
        auto it = solution.find(p);
        if (it == solution.end()) continue;

        for (const auto& [path, k] : it->second) {
            if (!path.empty() && path.front() == 0) {  // Loading
                int d_target = K[k].first.second;
                if (d_target > p + 1) {
                    for (size_t i = 0; i < path.size() - 1; ++i) {
                        route_heatmaps[d_target][path[i]]++;
                    }
                }
            }
        }
    }
}


// 기존 run.h에 추가할 함수들

// 노드의 매력도 계산 함수
std::vector<double> calculate_attractiveness(
    const std::vector<int>& candidate_nodes,
    int victim_car_k,
    int current_port,
    const DemandInfo& K,
    const ShortestDistances& shortest_distances,
    const std::vector<double>& node_centrality,
    const Heatmap& route_heatmaps
) {
    std::vector<double> scores;
    int dest_port = K[victim_car_k].first.second;

    for (int node : candidate_nodes) {
        double score = 1.0;

        // 깊이 기반 점수 (더 깊은 노드 선호)
        double depth_score = shortest_distances[node] / 10.0;
        score *= (1.0 + depth_score);

        // 중심성 기반 점수
        if (!node_centrality.empty()) {
            score *= (1.0 + node_centrality[node] * 0.5);
        }

        // 히트맵 기반 페널티 (해당 노드가 미래에 자주 사용되면 피함)
        auto it = route_heatmaps.find(dest_port);
        if (it != route_heatmaps.end()) {
            auto node_it = it->second.find(node);
            if (node_it != it->second.end()) {
                score *= std::exp(-0.1 * node_it->second);
            }
        }

        scores.push_back(score);
    }

    return scores;
}

// ============= Articulation Points 관련 구조체 및 함수 =============

// Articulation Impact 결과 저장 구조체
struct ArticulationImpact {
    std::set<int> articulation_points;
    std::vector<double> cut_impact;

    ArticulationImpact(int N) : cut_impact(N, 0.0) {}
};

// Articulation Points를 찾기 위한 헬퍼 클래스
class ArticulationFinder {
private:
    const AdjacencyList& adj_list;
    int N;
    std::vector<bool> visited;
    std::vector<int> disc;
    std::vector<int> low;
    std::vector<int> parent;
    std::set<int> articulation_points;
    int timer;

    void dfs(int u) {
        int children = 0;
        visited[u] = true;
        disc[u] = low[u] = ++timer;

        for (int v : adj_list[u]) {
            if (!visited[v]) {
                children++;
                parent[v] = u;
                dfs(v);

                low[u] = std::min(low[u], low[v]);

                // 루트 노드이고 자식이 2개 이상
                if (parent[u] == -1 && children > 1) {
                    articulation_points.insert(u);
                }

                // 루트가 아니고 low[v] >= disc[u]
                if (parent[u] != -1 && low[v] >= disc[u]) {
                    articulation_points.insert(u);
                }
            }
            else if (v != parent[u]) {
                low[u] = std::min(low[u], disc[v]);
            }
        }
    }

public:
    ArticulationFinder(const AdjacencyList& adj, int n)
        : adj_list(adj), N(n), visited(n, false),
          disc(n, -1), low(n, -1), parent(n, -1), timer(0) {}

    std::set<int> find() {
        articulation_points.clear();

        // 모든 연결 컴포넌트에 대해 DFS 수행
        for (int i = 0; i < N; i++) {
            if (!visited[i]) {
                dfs(i);
            }
        }

        return articulation_points;
    }
};

// 특정 노드를 제외한 상태에서 시작 노드로부터 도달 가능한 노드 수 계산
int count_reachable_excluding_node(
    const AdjacencyList& adj_list,
    int start,
    int N,
    int excluded_node
) {
    if (start == excluded_node) return 0;

    std::vector<bool> visited(N, false);
    std::queue<int> q;
    q.push(start);
    visited[start] = true;
    int count = 1;

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v : adj_list[u]) {
            if (!visited[v] && v != excluded_node) {
                visited[v] = true;
                q.push(v);
                count++;
            }
        }
    }

    return count;
}

// Articulation impact 계산 함수
ArticulationImpact precompute_articulation_impact(
    const AdjacencyList& adj_list,
    int N
) {
    ArticulationImpact result(N);

    // 1. Articulation points 찾기
    try {
        ArticulationFinder finder(adj_list, N);
        result.articulation_points = finder.find();
    } catch (...) {
        // 실패 시 빈 set 유지
        return result;
    }

    // articulation points가 없거나 노드 0이 연결되지 않은 경우
    if (result.articulation_points.empty() ||
        (adj_list.size() > 0 && adj_list[0].empty())) {
        return result;
    }

    // 2. 각 articulation point의 영향도 계산
    for (int v : result.articulation_points) {
        if (v == 0) {
            // 게이트(노드 0)는 제외
            continue;
        }

        // v를 제거한 상태에서 노드 0에서 도달 가능한 노드 수
        int reach_count = count_reachable_excluding_node(adj_list, 0, N, v);

        // v 제거로 0에서 잃는 노드 수 (자기 자신 제외)
        int lost = std::max(0, (N - 1) - reach_count);
        result.cut_impact[v] = static_cast<double>(lost);
    }

    // 3. 정규화 (0~1 범위로)
    double max_impact = *std::max_element(
        result.cut_impact.begin(),
        result.cut_impact.end()
    );

    if (max_impact > 0) {
        for (double& impact : result.cut_impact) {
            impact /= max_impact;
        }
    }

    return result;
}

// ============= 기존 함수 수정: calculate_node_centrality =============
// 이 함수를 수정하여 articulation impact도 고려하도록 변경

void calculate_node_centrality_with_articulation(
    const AdjacencyList& adj_list,
    std::vector<double>& node_centrality,
    const ArticulationImpact& articulation_impact
) {
    int N = adj_list.size();

    // 기존 degree centrality 계산
    for (int i = 0; i < N; ++i) {
        node_centrality[i] = static_cast<double>(adj_list[i].size()) / (N - 1);
    }

    // 게이트로부터의 접근성 가중치
    auto [distances, _] = dijkstra(adj_list, std::vector<int>(N, -1));
    for (int i = 0; i < N; ++i) {
        if (distances[i] != INF) {
            double accessibility = 1.0 / (1.0 + distances[i]);
            node_centrality[i] *= (1.0 + 0.2 * accessibility);
        }
    }

    // Articulation impact 가중치 추가
    for (int i = 0; i < N; ++i) {
        if (articulation_impact.cut_impact[i] > 0) {
            // articulation point는 중요도가 높으므로 가중치 증가
            node_centrality[i] *= (1.0 + 0.3 * articulation_impact.cut_impact[i]);
        }
    }
}

// ============= calculate_attractiveness 함수 수정 =============
// articulation impact를 고려한 매력도 계산

std::vector<double> calculate_attractiveness_with_articulation(
    const std::vector<int>& candidate_nodes,
    int victim_car_k,
    int current_port,
    const DemandInfo& K,
    const ShortestDistances& shortest_distances,
    const std::vector<double>& node_centrality,
    const Heatmap& route_heatmaps,
    const ArticulationImpact& articulation_impact
) {
    std::vector<double> scores;
    int dest_port = K[victim_car_k].first.second;

    for (int node : candidate_nodes) {
        double score = 1.0;

        // 깊이 기반 점수 (더 깊은 노드 선호)
        double depth_score = shortest_distances[node] / 10.0;
        score *= (1.0 + depth_score);

        // 중심성 기반 점수
        if (!node_centrality.empty()) {
            score *= (1.0 + node_centrality[node] * 0.5);
        }

        // Articulation point 페널티 (중요한 연결점은 피함)
        if (articulation_impact.articulation_points.count(node) > 0) {
            score *= (1.0 - 0.3 * articulation_impact.cut_impact[node]);
        }

        // 히트맵 기반 페널티
        auto it = route_heatmaps.find(dest_port);
        if (it != route_heatmaps.end()) {
            auto node_it = it->second.find(node);
            if (node_it != it->second.end()) {
                score *= std::exp(-0.1 * node_it->second);
            }
        }

        scores.push_back(score);
    }

    return scores;
}

// ============= 개선된 run2 함수 =============
// Articulation impact를 활용한 버전

std::optional<SolutionType> run2_with_articulation(
    const int P,
    const int N,
    const DemandInfo& K,
    const AdjacencyList& adj_list,
    const ShortestDistances& shortest_distances,
    const ShortestPaths& shortest_paths,
    const SolutionType& solution
) {
    std::random_device rd;
    std::mt19937 gen(rd());

    // Articulation impact 계산
    ArticulationImpact articulation_impact = precompute_articulation_impact(adj_list, N);

    // node_centrality 계산 (articulation impact 포함)
    std::vector<double> node_centrality(N, 0.0);
    calculate_node_centrality_with_articulation(adj_list, node_centrality, articulation_impact);

    Heatmap route_heatmaps;
    rebuild_heatmaps(solution, K, P, route_heatmaps);

    // 유효한 포트 찾기 (로딩 작업이 있는 포트)
    std::vector<int> valid_ports;
    for (int p = 0; p < P - 1; ++p) {
        auto it = solution.find(p);
        if (it != solution.end()) {
            bool has_long_term_loading = false;
            for (const auto& [path, k] : it->second) {
                if (!path.empty() && path.front() == 0 && K[k].first.second > p + 1) {
                    has_long_term_loading = true;
                    break;
                }
            }
            if (has_long_term_loading) {
                valid_ports.push_back(p);
            }
        }
    }

    if (valid_ports.empty()) {
        return std::nullopt;
    }

    // 유효한 포트 중에서 선택
    std::uniform_int_distribution<> port_dist(0, valid_ports.size() - 1);
    int p = valid_ports[port_dist(gen)];

    SolutionType neighbor_sol = solution;

    // p 포트의 경로 가져오기
    auto it = neighbor_sol.find(p);
    if (it == neighbor_sol.end()) return std::nullopt;

    PortSchedule original_routes_p = it->second;

    // 언로딩과 로딩 이동 분리
    PortSchedule unloading_moves;
    std::vector<std::pair<int, std::pair<Path, int>>> loading_moves_info;

    for (size_t i = 0; i < original_routes_p.size(); ++i) {
        const auto& [path, k] = original_routes_p[i];
        if (!path.empty()) {
            if (path.front() != 0) {
                unloading_moves.push_back({path, k});
            } else {
                loading_moves_info.push_back({static_cast<int>(i), {path, k}});
            }
        }
    }

    // 장기 로딩 이동 찾기
    std::vector<std::pair<int, std::pair<Path, int>>> long_term_loading_moves;
    for (const auto& [idx, move] : loading_moves_info) {
        int k = move.second;
        if (K[k].first.second > p + 1) {
            long_term_loading_moves.push_back({idx, move});
        }
    }

    if (long_term_loading_moves.empty()) return std::nullopt;

    // 변경할 차량 선택
    std::uniform_int_distribution<> move_dist(0, long_term_loading_moves.size() - 1);
    auto selected = long_term_loading_moves[move_dist(gen)];
    int move_idx_in_original = selected.first;
    int victim_car_k = selected.second.second;

    // 고정될 차량들 정보 추출
    struct CarInfo {
        int k;
        int node;
    };
    std::vector<CarInfo> fixed_cars_info;

    for (const auto& [idx, move] : loading_moves_info) {
        if (idx != move_idx_in_original) {
            fixed_cars_info.push_back({move.second, move.first.back()});
        }
    }

    // 상태 복원
    std::vector<int> occ_after_unloading(N, -1);
    for (int t = 0; t < p; ++t) {
        auto port_it = neighbor_sol.find(t);
        if (port_it != neighbor_sol.end()) {
            occ_after_unloading = apply_routes_to_occupancy(
                port_it->second, occ_after_unloading, N
            );
        }
    }

    // p 포트의 언로딩만 적용
    for (const auto& [path, dem_id] : unloading_moves) {
        if (!path.empty()) {
            if (path.back() == 0 && path.front() != 0) {
                occ_after_unloading[path.front()] = -1;
            } else if (path.front() != 0 && path.back() != 0) {
                occ_after_unloading[path.front()] = -1;
                occ_after_unloading[path.back()] = dem_id;
            }
        }
    }

    // 초기 배치 노드들
    std::set<int> initial_placements_p;
    for (const auto& [_, move] : loading_moves_info) {
        if (!move.first.empty()) {
            initial_placements_p.insert(move.first.back());
        }
    }

    // BFS로 도달 가능한 노드 찾기
    auto [reachable_nodes, _] = bfs(adj_list, occ_after_unloading);

    // 후보 노드 선택
    std::vector<int> candidate_new_nodes;
    for (int n : reachable_nodes) {
        if (n != 0 && initial_placements_p.find(n) == initial_placements_p.end()) {
            candidate_new_nodes.push_back(n);
        }
    }

    if (candidate_new_nodes.empty()) return std::nullopt;

    // 새 노드 선택 (articulation impact 고려)
    std::vector<double> scores = calculate_attractiveness_with_articulation(
        candidate_new_nodes, victim_car_k, p, K,
        shortest_distances, node_centrality, route_heatmaps,
        articulation_impact
    );

    int new_node;
    double total_score = std::accumulate(scores.begin(), scores.end(), 0.0);
    if (total_score > 0) {
        std::discrete_distribution<> node_dist(scores.begin(), scores.end());
        new_node = candidate_new_nodes[node_dist(gen)];
    } else {
        std::uniform_int_distribution<> uniform_dist(0, candidate_new_nodes.size() - 1);
        new_node = candidate_new_nodes[uniform_dist(gen)];
    }

    // 최종 로딩 계획 (깊이 순 정렬)
    std::vector<CarInfo> final_loading_plan = fixed_cars_info;
    final_loading_plan.push_back({victim_car_k, new_node});

    std::sort(final_loading_plan.begin(), final_loading_plan.end(),
        [&shortest_distances](const CarInfo& a, const CarInfo& b) {
            return shortest_distances[a.node] > shortest_distances[b.node];
        });

    // 순차적 로딩
    PortSchedule repaired_loading_routes;
    std::vector<int> temp_occ = occ_after_unloading;
    auto [distances, prev_nodes] = dijkstra(adj_list, temp_occ);

    for (const auto& item : final_loading_plan) {
        int k = item.k;
        int target_node = item.node;
        temp_occ[target_node] = k;

        Path path = path_backtracking(prev_nodes, 0, target_node);
        repaired_loading_routes.push_back({path, k});
    }

    // p 포트 업데이트
    PortSchedule combined_routes = unloading_moves;
    combined_routes.insert(combined_routes.end(),
                          repaired_loading_routes.begin(),
                          repaired_loading_routes.end());
    neighbor_sol[p] = combined_routes;

    // 4. 미래 포트 시뮬레이션
    for (int t = p + 1; t < P; ++t) {
        // 언로딩
        auto unload_result = _unloading_heuristic_updated(K, adj_list, t, temp_occ);
        PortSchedule unload_routes = unload_result.routes;
        std::vector<int> rehandling = unload_result.rehandling_demands;
        temp_occ = unload_result.final_occupation;

        // 로딩 (마지막 포트가 아닌 경우)
        if (t < P - 1) {
            auto load_result = _loading_heuristic_updated(
                N, P, K, adj_list, shortest_distances,
                node_centrality, route_heatmaps,
                t, temp_occ, rehandling,
                false, nullptr
            );

            if (!load_result.routes.has_value()) {
                return std::nullopt;
            }

            PortSchedule load_routes = load_result.routes.value();
            temp_occ = load_result.final_occupation;
            route_heatmaps = load_result.updated_heatmaps;

            PortSchedule combined = unload_routes;
            combined.insert(combined.end(), load_routes.begin(), load_routes.end());
            neighbor_sol[t] = combined;
        } else {
            neighbor_sol[t] = unload_routes;
        }
    }

    return neighbor_sol;
}


// run3: create_neighbor_from_unload의 C++ 구현
std::optional<SolutionType> run3(
    const int P,
    const int N,
    const DemandInfo& K,
    const AdjacencyList& adj_list,
    const ShortestDistances& shortest_distances,
    const ShortestPaths& shortest_paths,
    const SolutionType& base_solution
) {
    std::random_device rd;
    std::mt19937 gen(rd());

    // node_centrality와 route_heatmaps 초기화
    std::vector<double> node_centrality(N, 0.0);
    calculate_node_centrality(adj_list, node_centrality);

    Heatmap route_heatmaps;
    rebuild_heatmaps(base_solution, K, P, route_heatmaps);

    // 1. 랜덤하게 시작 포트 선택
    std::uniform_int_distribution<> port_dist(0, P - 2);
    int start_port = port_dist(gen);

    // 디버깅용 (필요시)
    // std::cout << "Start port for run3: " << start_port << std::endl;

    // 2. new_sol은 base_solution을 복사
    SolutionType new_sol = base_solution;

    // 3. start_port 직전까지의 점유 상태 복원
    std::vector<int> occ(N, -1);
    for (int t = 0; t < start_port; ++t) {
        auto it = new_sol.find(t);
        if (it != new_sol.end()) {
            occ = apply_routes_to_occupancy(it->second, occ, N);
        }
    }

    // 4. start_port에서 언로딩을 SA 모드로 새로 계산
    auto unload_result = _unloading_heuristic_with_SA(
        N, start_port, K, adj_list, occ, true
    );
    PortSchedule unload_routes = unload_result.routes;
    std::vector<int> rehandling = unload_result.rehandling_demands;
    occ = unload_result.final_occupation;

    // 5. start_port 로딩을 SA=True로 새로 구성
    auto load_result = _loading_heuristic_updated(
        N, P, K, adj_list, shortest_distances,
        node_centrality, route_heatmaps,
        start_port, occ, rehandling,
        true, &gen  // SA 모드 활성화
    );

    if (!load_result.routes.has_value()) {
        return std::nullopt;  // 로딩 실패
    }

    PortSchedule load_routes = load_result.routes.value();
    occ = load_result.final_occupation;
    route_heatmaps = load_result.updated_heatmaps;

    // start_port의 경로 업데이트
    PortSchedule combined_routes = unload_routes;
    combined_routes.insert(combined_routes.end(), load_routes.begin(), load_routes.end());
    new_sol[start_port] = combined_routes;

    // 6. start_port+1부터 P-1까지 순차 재시뮬레이션
    for (int t = start_port + 1; t < P; ++t) {
        // 언로딩 (SA 모드 비활성화)
        auto unload_res = _unloading_heuristic_with_SA(
            N, t, K, adj_list, occ, false
        );
        PortSchedule unload_routes_t = unload_res.routes;
        std::vector<int> rehandling_t = unload_res.rehandling_demands;
        occ = unload_res.final_occupation;

        // 로딩 (마지막 포트가 아닌 경우)
        if (t < P - 1) {
            auto load_res = _loading_heuristic_updated(
                N, P, K, adj_list, shortest_distances,
                node_centrality, route_heatmaps,
                t, occ, rehandling_t,
                false, nullptr  // 후속 포트는 SA 모드 비활성화
            );

            if (!load_res.routes.has_value()) {
                return std::nullopt;  // 로딩 실패
            }

            PortSchedule load_routes_t = load_res.routes.value();
            occ = load_res.final_occupation;
            route_heatmaps = load_res.updated_heatmaps;

            // 경로 결합
            PortSchedule combined_t = unload_routes_t;
            combined_t.insert(combined_t.end(), load_routes_t.begin(), load_routes_t.end());
            new_sol[t] = combined_t;
        } else {
            // 마지막 포트는 언로딩만
            new_sol[t] = unload_routes_t;
        }
    }

    return new_sol;
}


// change_loading_node 함수의 C++ 구현
std::optional<SolutionType> run2(
    const int P,
    const int N,
    const DemandInfo& K,
    const AdjacencyList& adj_list,
    const ShortestDistances& shortest_distances,
    const ShortestPaths& shortest_paths,
    const SolutionType& solution
) {
    std::random_device rd;
    std::mt19937 gen(rd());

    // node_centrality와 route_heatmaps 초기화 (실제로는 외부에서 관리해야 함)
    std::vector<double> node_centrality(N, 0.0);
    // 실제로는 노드 중심성을 계산해야 함
    calculate_node_centrality(adj_list, node_centrality);

    Heatmap route_heatmaps;
    // 기존 solution에서 히트맵 재구성이 필요할 수 있음
    rebuild_heatmaps(solution, K, P, route_heatmaps);

    // 1. 랜덤 포트 선택
    std::uniform_int_distribution<> port_dist(0, P - 2);
    int p = port_dist(gen);

    SolutionType neighbor_sol = solution;

    // p 포트의 경로 가져오기
    auto it = neighbor_sol.find(p);
    if (it == neighbor_sol.end()) return std::nullopt;

    PortSchedule original_routes_p = it->second;

    // 언로딩과 로딩 이동 분리
    PortSchedule unloading_moves;
    std::vector<std::pair<int, std::pair<Path, int>>> loading_moves_info;

    for (int i = 0; i < original_routes_p.size(); ++i) {
        const auto& [path, k] = original_routes_p[i];
        if (!path.empty()) {
            if (path.front() != 0) {
                unloading_moves.push_back({path, k});
            } else {
                loading_moves_info.push_back({i, {path, k}});
            }
        }
    }

    // 장기 로딩 이동 찾기 (목적지가 p+1보다 뒤)
    std::vector<std::pair<int, std::pair<Path, int>>> long_term_loading_moves;
    for (const auto& [idx, move] : loading_moves_info) {
        int k = move.second;
        if (K[k].first.second > p + 1) {
            long_term_loading_moves.push_back({idx, move});
        }
    }

    if (long_term_loading_moves.empty()) return std::nullopt;

    // 변경할 차량 선택
    std::uniform_int_distribution<> move_dist(0, long_term_loading_moves.size() - 1);
    auto selected = long_term_loading_moves[move_dist(gen)];
    int move_idx_in_original = selected.first;
    Path original_path = selected.second.first;
    int victim_car_k = selected.second.second;

    // 고정될 차량들 정보 추출
    struct CarInfo {
        int k;
        int node;
    };
    std::vector<CarInfo> fixed_cars_info;

    for (const auto& [idx, move] : loading_moves_info) {
        if (idx != move_idx_in_original) {
            fixed_cars_info.push_back({move.second, move.first.back()});
        }
    }

    // 2. 상태 복원 (p항구 하역 완료 시점)
    std::vector<int> occ_after_unloading(N, -1);

    // p 이전 포트들의 상태 적용
    for (int t = 0; t < p; ++t) {
        auto port_it = neighbor_sol.find(t);
        if (port_it != neighbor_sol.end()) {
            for (const auto& [path, dem_id] : port_it->second) {
                if (!path.empty()) {
                    if (path.front() == 0) {  // Loading
                        occ_after_unloading[path.back()] = dem_id;
                    } else if (path.back() == 0) {  // Unloading
                        occ_after_unloading[path.front()] = -1;
                    }
                }
            }
        }
    }

    // p 포트의 언로딩 적용
    for (const auto& [path, dem_id] : unloading_moves) {
        if (!path.empty() && path.back() == 0) {
            occ_after_unloading[path.front()] = -1;
        }
    }

    // 초기 배치 노드들
    std::set<int> initial_placements_p;
    for (const auto& [_, move] : loading_moves_info) {
        initial_placements_p.insert(move.first.back());
    }

    // BFS로 도달 가능한 노드 찾기
    auto [reachable_nodes, _] = bfs(adj_list, occ_after_unloading);

    // 후보 노드 선택
    std::vector<int> candidate_new_nodes;
    for (int n : reachable_nodes) {
        if (n != 0 && initial_placements_p.find(n) == initial_placements_p.end()) {
            candidate_new_nodes.push_back(n);
        }
    }

    if (candidate_new_nodes.empty()) return std::nullopt;

    // 새 노드 선택 (매력도 기반)
    std::vector<double> scores = calculate_attractiveness(
        candidate_new_nodes, victim_car_k, p, K,
        shortest_distances, node_centrality, route_heatmaps
    );

    int new_node;
    double total_score = std::accumulate(scores.begin(), scores.end(), 0.0);
    if (total_score > 0) {
        std::discrete_distribution<> node_dist(scores.begin(), scores.end());
        new_node = candidate_new_nodes[node_dist(gen)];
    } else {
        std::uniform_int_distribution<> uniform_dist(0, candidate_new_nodes.size() - 1);
        new_node = candidate_new_nodes[uniform_dist(gen)];
    }

    // 최종 로딩 계획 (깊이 순 정렬)
    std::vector<CarInfo> final_loading_plan = fixed_cars_info;
    final_loading_plan.push_back({victim_car_k, new_node});

    std::sort(final_loading_plan.begin(), final_loading_plan.end(),
        [&shortest_distances](const CarInfo& a, const CarInfo& b) {
            return shortest_distances[a.node] > shortest_distances[b.node];
        });

    // 순차적 로딩
    PortSchedule repaired_loading_routes;
    std::vector<int> temp_occ = occ_after_unloading;
    auto [distances, prev_nodes] = dijkstra(adj_list, temp_occ);

    for (const auto& item : final_loading_plan) {
        int k = item.k;
        int target_node = item.node;
        temp_occ[target_node] = k;

        Path path = path_backtracking(prev_nodes, 0, target_node);
        repaired_loading_routes.push_back({path, k});
    }

    // p 포트 업데이트
    PortSchedule combined_routes = unloading_moves;
    combined_routes.insert(combined_routes.end(),
                          repaired_loading_routes.begin(),
                          repaired_loading_routes.end());
    neighbor_sol[p] = combined_routes;

    // 4. 미래 포트 시뮬레이션
    for (int t = p + 1; t < P; ++t) {
        // 언로딩
        auto unload_result = _unloading_heuristic_updated(K, adj_list, t, temp_occ);
        PortSchedule unload_routes = unload_result.routes;
        std::vector<int> rehandling = unload_result.rehandling_demands;
        temp_occ = unload_result.final_occupation;

        // 로딩 (마지막 포트가 아닌 경우)
        if (t < P - 1) {
            auto load_result = _loading_heuristic_updated(
                N, P, K, adj_list, shortest_distances,
                node_centrality, route_heatmaps,
                t, temp_occ, rehandling,
                false, nullptr
            );

            if (!load_result.routes.has_value()) {
                return std::nullopt;
            }

            PortSchedule load_routes = load_result.routes.value();
            temp_occ = load_result.final_occupation;
            route_heatmaps = load_result.updated_heatmaps;

            PortSchedule combined = unload_routes;
            combined.insert(combined.end(), load_routes.begin(), load_routes.end());
            neighbor_sol[t] = combined;
        } else {
            neighbor_sol[t] = unload_routes;
        }
    }

    return neighbor_sol;
}


// 기존 run 함수를 대체하는 새로운 버전
std::optional<SolutionType> run(
    const int P,
    const int N,
    const DemandInfo& K,
    const AdjacencyList& adj_list,
    const ShortestDistances& shortest_distances,
    const ShortestPaths& shortest_paths,
    const SolutionType& solution
) {
    // node_centrality와 route_heatmaps 초기화 (실제로는 외부에서 관리해야 함)
    std::vector<double> node_centrality(N, 0.0);
    // 실제로는 노드 중심성을 계산해야 함
    // calculate_node_centrality(adj_list, node_centrality);

    Heatmap route_heatmaps;
    // 기존 solution에서 히트맵 재구성이 필요할 수 있음
    // rebuild_heatmaps(solution, K, P, route_heatmaps);

    return create_neighbor_from_port(
        P, N, K, adj_list, shortest_distances, shortest_paths,
        node_centrality, route_heatmaps, solution, true
    );
}

#endif