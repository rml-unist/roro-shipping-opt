#ifndef HEURISTIC_H
#define HEURISTIC_H

#include <optional>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <tuple>
#include <map>
#include <limits>
#include <random>
#include <unordered_map>

#include "helper.h"

std::vector<double> _calculate_attractiveness(const int P, const DemandInfo K, const ShortestDistances shortest_distances, const std::vector<double>& node_centrality, const Heatmap& route_heatmaps, const AdjacencyList& adj_list, std::vector<int> &nodes, int car_k, int p) {
    std::vector<double> scores;
    scores.reserve(nodes.size());

    int d_target = K[car_k].first.second;

    // remoteness_ratio 계산
    double remoteness_ratio = 0.0;
    if ((P - 1) - (p + 1) > 0) {
        remoteness_ratio = static_cast<double>(d_target - (p + 1)) /
                          static_cast<double>((P - 1) - (p + 1));
    }

    // max_d 계산
    double max_d = 1.0;
    for (int dist : shortest_distances) {
        if (dist != std::numeric_limits<int>::max() && dist > max_d) {
            max_d = static_cast<double>(dist);
        }
    }

    double target_depth = max_d * remoteness_ratio;

    // 가중치
    double alpha = 1.0;  // 깊이 적합도
    double beta = 1.0;   // 중심성 페널티

    for (int node : nodes) {
        double depth_fit = 1.0 / (std::abs(shortest_distances[node] - target_depth) + 1.0);
        double cent_pen = node_centrality[node];

        // 히트맵 보너스 (현재 주석 처리된 부분)
        // double spine_bonus = 0.0;
        // if (route_heatmaps.find(d_target) != route_heatmaps.end()) {
        //     const auto& hm = route_heatmaps.at(d_target);
        //     for (int nb : adj_list[node]) {
        //         if (hm.find(nb) != hm.end()) {
        //             spine_bonus += hm.at(nb);
        //         }
        //     }
        //     spine_bonus = std::tanh(spine_bonus);
        // }

        double score = alpha * depth_fit - beta * cent_pen;
        scores.push_back(score);
    }

    return scores;
}

std::vector<int> r1_candidates_same_dest(
    const AdjacencyList& adj_list,
    const std::vector<int>& occupation,
    const DemandInfo& K,
    const std::vector<int>& reachable_nodes,
    int dest
) {
    std::unordered_set<int> reachable_set(reachable_nodes.begin(), reachable_nodes.end());
    std::vector<int> seeds;
    for (int n = 0; n < occupation.size(); ++n) {
        int k = occupation[n];
        if (k != -1 && K[k].first.second == dest) {
            seeds.push_back(n);
        }
    }

    if (seeds.empty()) return {};

    std::unordered_set<int> cand_set;
    for (int u : seeds) {
        if(!adj_list[u].empty()) {
            for (int v : adj_list[u]) {
                if (v != 0 && occupation[v] == -1 && std::find(adj_list[u].begin(), adj_list[u].end(), v) != adj_list[u].end()) {
                    cand_set.insert(v);
                }
            }
        }
    }
    return std::vector<int>(cand_set.begin(), cand_set.end());
}

// Python의 _reachability_safe_candidates에 해당
std::vector<int> reachability_safe_candidates(
    const AdjacencyList& adj_list,
    const std::vector<int>& cands,
    const std::vector<int>& occupation,
    int k_to_place
) {
    auto [before_nodes_vec, _] = bfs(adj_list, occupation);
    if (before_nodes_vec.empty()) return {};
    std::unordered_set<int> before_set(before_nodes_vec.begin(), before_nodes_vec.end());

    std::vector<int> safe_candidates;
    for (int n : cands) {
        if (before_set.find(n) == before_set.end()) continue;

        std::vector<int> temp_occupation = occupation;
        temp_occupation[n] = k_to_place;
        auto [after_nodes_vec, __] = bfs(adj_list, temp_occupation);
        if (after_nodes_vec.empty()) continue;

        std::unordered_set<int> after_set(after_nodes_vec.begin(), after_nodes_vec.end());

        bool is_safe = true;
        for (int before_node : before_set) {
            if (before_node == n) continue;
            if (after_set.find(before_node) == after_set.end()) {
                is_safe = false;
                break;
            }
        }

        if (is_safe) {
            safe_candidates.push_back(n);
        }
    }
    return safe_candidates;
}

LoadingResultWithHeatmap _loading_heuristic_updated(
    const int N,
    const int P,
    const DemandInfo& K,
    const AdjacencyList& adj_list,
    const ShortestDistances& shortest_distances,
    const std::vector<double>& node_centrality,
    Heatmap& route_heatmaps,  // 입출력 파라미터
    int p,
    const std::vector<int>& initial_occupation,
    const std::vector<int>& rehandling_demands,
    bool SA = false,
    std::mt19937* rng = nullptr  // SA용 랜덤 생성기
) {
    // 1. 선적 수요 집계
    std::map<int, int> k_load;
    for (size_t i = 0; i < K.size(); ++i) {
        if (K[i].first.first == p) {
            k_load[i] = K[i].second;
        }
    }
    for (int k : rehandling_demands) {
        k_load[k]++;
    }

    PortSchedule route_list;
    std::vector<int> last_rehandling_demands;
    std::vector<int> occupation = initial_occupation;

    int total_loading_demands = 0;
    for (const auto& pair : k_load) {
        total_loading_demands += pair.second;
    }

    // 2. 공간 확보 (리핸들링)
    auto [reachable_nodes, reachable_distances] = bfs(adj_list, occupation);
    auto available_nodes = get_available_nodes(N, occupation);

    if (available_nodes.size() < total_loading_demands) {
        return {std::nullopt, initial_occupation, route_heatmaps};
    }

    // 공간 부족 시 리핸들링
    if (reachable_nodes.size() < total_loading_demands) {
        std::unordered_set<int> reachable_set(reachable_nodes.begin(), reachable_nodes.end());
        std::vector<int> available_but_not_reachable;

        for (int n : available_nodes) {
            if (reachable_set.find(n) == reachable_set.end()) {
                available_but_not_reachable.push_back(n);
            }
        }

        while (reachable_nodes.size() < total_loading_demands && !available_but_not_reachable.empty()) {
            int n = available_but_not_reachable[0];
            available_but_not_reachable.erase(available_but_not_reachable.begin());

            auto [path, blocks] = min_blocking_path(adj_list, occupation, n, 0);

            // 차단 해제
            for (size_t i = 0; i < path.size() - 1; ++i) {
                if (occupation[path[i]] != -1) {
                    int k_block = occupation[path[i]];
                    last_rehandling_demands.push_back(k_block);

                    // 역방향 경로 생성
                    Path cut;
                    for (int x : path) {
                        cut.push_back(x);
                        if (x == path[i]) break;
                    }
                    std::reverse(cut.begin(), cut.end());
                    route_list.emplace_back(cut, k_block);
                    occupation[path[i]] = -1;
                    total_loading_demands++;
                }
            }

            auto [new_reachable, _] = bfs(adj_list, occupation);
            reachable_nodes = new_reachable;
        }
    }

    // 리핸들링 수요 병합
    for (int k : last_rehandling_demands) {
        k_load[k]++;
    }

    // 3. 화물 배치
    if (total_loading_demands > 0) {
        // 수요 분리
        std::vector<int> all_demands_to_load;
        for (const auto& pair : k_load) {
            for (int i = 0; i < pair.second; ++i) {
                all_demands_to_load.push_back(pair.first);
            }
        }

        std::vector<int> long_term_vehicles;
        std::vector<int> next_port_vehicles;

        for (int k : all_demands_to_load) {
            if (K[k].first.second > p + 1) {
                long_term_vehicles.push_back(k);
            } else {
                next_port_vehicles.push_back(k);
            }
        }

        // 장기 화물 정렬 (목적지가 먼 순서)
        std::sort(long_term_vehicles.begin(), long_term_vehicles.end(),
                  [&](int a, int b) { return K[a].first.second > K[b].first.second; });

        int reserve_cnt = next_port_vehicles.size();

        // 장기 화물 배치
        for (int k : long_term_vehicles) {
            int d_target = K[k].first.second;

            auto [current_reachable, _] = bfs(adj_list, occupation);
            std::vector<int> avail_now;
            for (int n : current_reachable) {
                if (n != 0 && occupation[n] == -1) {
                    avail_now.push_back(n);
                }
            }

            // 예약 노드 설정
            std::sort(avail_now.begin(), avail_now.end(),
                     [&](int a, int b) { return shortest_distances[a] < shortest_distances[b]; });

            std::unordered_set<int> reserved_nodes;
            for (size_t i = 0; i < std::min(avail_now.size(), static_cast<size_t>(reserve_cnt)); ++i) {
                reserved_nodes.insert(avail_now[i]);
            }

            // R1 후보
            auto r1_cands_all = r1_candidates_same_dest(adj_list, occupation, K, current_reachable, d_target);
            std::vector<int> r1_cands;
            for (int n : r1_cands_all) {
                if (reserved_nodes.find(n) == reserved_nodes.end()) {
                    r1_cands.push_back(n);
                }
            }

            auto r1_safe = reachability_safe_candidates(adj_list, r1_cands, occupation, k);

            int n_sel = -1;

            if (!r1_safe.empty()) {
                // R1에서 가장 깊은 노드 선택
                n_sel = *std::max_element(r1_safe.begin(), r1_safe.end(),
                                         [&](int a, int b) { return shortest_distances[a] < shortest_distances[b]; });
            } else {
                // R2 전략
                std::vector<int> all_cands;
                for (int n : current_reachable) {
                    if (n != 0 && occupation[n] == -1 && reserved_nodes.find(n) == reserved_nodes.end()) {
                        all_cands.push_back(n);
                    }
                }

                if (all_cands.empty()) continue;

                if (SA && rng) {
                    // SA 모드: 랜덤 선택
                    auto safe_cands = reachability_safe_candidates(adj_list, all_cands, occupation, k);
                    if (!safe_cands.empty()) {
                        std::uniform_int_distribution<> dis(0, safe_cands.size() - 1);
                        n_sel = safe_cands[dis(*rng)];
                    } else if (!all_cands.empty()) {
                        std::uniform_int_distribution<> dis(0, all_cands.size() - 1);
                        n_sel = all_cands[dis(*rng)];
                    }
                } else {
                    // 일반 모드
                    auto safe_cands = reachability_safe_candidates(adj_list, all_cands, occupation, k);
                    std::vector<int> cands_to_score = safe_cands.empty() ? all_cands : safe_cands;

                    if (cands_to_score.empty()) continue;

                    // 첫 차량 여부 확인
                    bool is_first_vehicle = true;
                    if (route_heatmaps.find(d_target) != route_heatmaps.end()) {
                        is_first_vehicle = route_heatmaps[d_target].empty();
                    }

                    if (is_first_vehicle) {
                        // Attractiveness 점수로 선택
                        auto att_scores = _calculate_attractiveness(P, K, shortest_distances,
                                                                   node_centrality, route_heatmaps,
                                                                   adj_list, cands_to_score, k, p);

                        auto max_it = std::max_element(att_scores.begin(), att_scores.end());
                        int max_idx = std::distance(att_scores.begin(), max_it);
                        n_sel = cands_to_score[max_idx];
                    } else {
                        // 히트맵 점수로 선택
                        int best_cand_node = -1;
                        double max_score = -std::numeric_limits<double>::infinity();

                        for (int n_cand : cands_to_score) {
                            auto [_, prev_nodes] = dijkstra(adj_list, occupation);
                            Path path_cand = path_backtracking(prev_nodes, 0, n_cand);

                            if (path_cand.empty()) continue;

                            double current_score = 0.0;

                            // 같은 목적지 히트맵 점수
                            for (size_t i = 0; i < path_cand.size() - 1; ++i) {
                                int node_on_path = path_cand[i];
                                if (route_heatmaps[d_target].find(node_on_path) != route_heatmaps[d_target].end()) {
                                    current_score += route_heatmaps[d_target][node_on_path];
                                }

                                // 다른 목적지 페널티
                                for (const auto& [other_dest, heatmap] : route_heatmaps) {
                                    if (other_dest != d_target) {
                                        if (heatmap.find(node_on_path) != heatmap.end()) {
                                            current_score -= heatmap.at(node_on_path) * 0.5;
                                        }
                                    }
                                }
                            }

                            // 깊이 보너스
                            current_score += shortest_distances[n_cand] * 0.1;

                            if (current_score > max_score) {
                                max_score = current_score;
                                best_cand_node = n_cand;
                            }
                        }

                        n_sel = (best_cand_node != -1) ? best_cand_node : cands_to_score.back();
                    }
                }
            }

            // 경로 계산 및 배치
            if (n_sel != -1) {
                auto [_, prev_nodes] = dijkstra(adj_list, occupation);
                Path path = path_backtracking(prev_nodes, 0, n_sel);

                occupation[n_sel] = k;
                route_list.emplace_back(path, k);

                // 히트맵 업데이트
                if (d_target > p + 1) {
                    for (size_t i = 0; i < path.size() - 1; ++i) {
                        route_heatmaps[d_target][path[i]]++;
                    }
                }
            }
        }

        // 단기 화물 배치
        auto [final_reachable, _] = bfs(adj_list, occupation);
        std::vector<int> loading_nodes;

        for (int n : final_reachable) {
            if (n != 0 && occupation[n] == -1) {
                loading_nodes.push_back(n);
                if (loading_nodes.size() >= next_port_vehicles.size()) break;
            }
        }

        std::reverse(loading_nodes.begin(), loading_nodes.end());

        auto [__, prev_nodes] = dijkstra(adj_list, occupation);

        for (size_t i = 0; i < std::min(next_port_vehicles.size(), loading_nodes.size()); ++i) {
            int k = next_port_vehicles[i];
            int n = loading_nodes[i];

            occupation[n] = k;
            Path path = path_backtracking(prev_nodes, 0, n);
            route_list.emplace_back(path, k);
        }
    }

    return {std::optional<PortSchedule>{route_list}, occupation, route_heatmaps};
}


LoadingResult _loading_heuristic(
    const int N,
    const DemandInfo K,
    const AdjacencyList adj_list,
    const ShortestDistances shortest_distances, // <<< ADDED for placement logic
    int p,
    const std::vector<int>& initial_occupation,
    const std::vector<int>& rehandling_demands
    // max_future_cars, strategic_tunnel_nodes 는 Python 로직에서 사용되지 않으므로 제거하거나 무시
) {
    // --- 1, 2, 3 단계는 기존 코드와 거의 동일 ---
    // --- 1. 선적 수요 집계 ---
    std::map<int, int> k_load;
    for (int i = 0; i < K.size(); ++i) {
        if (K[i].first.first == p) {
            k_load[i] = K[i].second;
        }
    }
    for (int k : rehandling_demands) {
        k_load[k]++;
    }

    // --- 2. 공간 확보를 위한 준비 ---
    PortSchedule route_list;
    std::vector<int> last_rehandling_demands;
    std::vector<int> occupation = initial_occupation;

    int total_loading_demands = 0;
    for(const auto& pair : k_load) total_loading_demands += pair.second;

    if (total_loading_demands == 0) {
        return { std::optional<PortSchedule>{PortSchedule{}}, initial_occupation };
    }

    // --- 3. 공간 확보 (재배치) ---
    // (이 부분은 기존 C++ 코드와 동일하게 유지합니다. 로직이 유사하고 견고합니다.)
    auto [reachable_nodes_vec, reachable_node_distances] = bfs(adj_list, occupation);
    if (reachable_nodes_vec.size() < total_loading_demands) {
        auto available_nodes = get_available_nodes(N, occupation);
        std::vector<int> available_but_not_reachable;
        std::unordered_set<int> reachable_set(reachable_nodes_vec.begin(), reachable_nodes_vec.end());
        for(int n : available_nodes) {
            if(reachable_set.find(n) == reachable_set.end()) {
                available_but_not_reachable.push_back(n);
            }
        }
    }

    // <<< MODIFIED >>>
    // --- 4. 화물 분류 및 배치 전략 ---
    if (total_loading_demands > 0) {
        // 단기/장기 화물 분류
        std::vector<int> all_demands_to_load;
        for(const auto& pair : k_load) {
            for(int i=0; i<pair.second; ++i) all_demands_to_load.push_back(pair.first);
        }

        std::vector<int> long_term_cars;
        std::vector<int> next_port_cars;
        for(int k : all_demands_to_load) {
            if (K[k].first.second > p + 1) {
                long_term_cars.push_back(k);
            } else { // == p + 1
                next_port_cars.push_back(k);
            }
        }
        // 장기 화물은 목적지가 먼 순으로 정렬
        std::sort(long_term_cars.begin(), long_term_cars.end(), [&](int a, int b){
            return K[a].first.second > K[b].first.second;
        });

        // --- 장기 화물 배치 (R1/R2 전략 적용) ---
        for (int car_k : long_term_cars) {
            int d_target = K[car_k].first.second;

            // 최신 상태의 접근 가능 노드 계산
            auto [current_reachable, _] = bfs(adj_list, occupation);
            if(current_reachable.empty()) continue;

            // ✅ 단기 화물을 위한 공간 예약
            std::vector<int> avail_now;
            for(int n : current_reachable) {
                if (n != 0 && occupation[n] == -1) {
                    avail_now.push_back(n);
                }
            }
            std::sort(avail_now.begin(), avail_now.end(), [&](int a, int b){
                return shortest_distances[a] < shortest_distances[b];
            });

            std::unordered_set<int> reserved_nodes;
            for(size_t i = 0; i < std::min(avail_now.size(), next_port_cars.size()); ++i) {
                reserved_nodes.insert(avail_now[i]);
            }

            int n_sel = -1; // 최종 선택될 노드

            // --- R1: 같은 목적지 클러스터 주변 탐색 ---
            auto r1_cands_all = r1_candidates_same_dest(adj_list, occupation, K, current_reachable, d_target);
            std::vector<int> r1_cands; // 예약 공간 제외된 후보
            for(int n : r1_cands_all) {
                if(reserved_nodes.find(n) == reserved_nodes.end()) {
                    r1_cands.push_back(n);
                }
            }

            // 안전성 검사
            auto r1_safe = reachability_safe_candidates(adj_list, r1_cands, occupation, car_k);

            if (!r1_safe.empty()) {
                // R1 후보 중 가장 깊은 곳 선택
                n_sel = *std::max_element(r1_safe.begin(), r1_safe.end(), [&](int a, int b){
                    return shortest_distances[a] < shortest_distances[b];
                });
            } else {
                // --- R2: 전체 후보 중 탐색 ---
                std::vector<int> all_cands;
                for(int n : current_reachable) { // avail_now를 사용해도 무방
                    if(n != 0 && occupation[n] == -1 && reserved_nodes.find(n) == reserved_nodes.end()) {
                        all_cands.push_back(n);
                    }
                }
                if (all_cands.empty()) continue;
                // R2 후보 중 가장 깊은 곳 선택 (BFS 결과는 보통 거리순 정렬되어 있으므로 마지막 원소)
                n_sel = all_cands.back();
            }

            // 경로 계산 및 배치
            if (n_sel != -1) {
                auto [__, prev_nodes] = dijkstra(adj_list, occupation);
                Path path = path_backtracking(prev_nodes, 0, n_sel);
                occupation[n_sel] = car_k;
                route_list.emplace_back(path, car_k);
            }
        }

        // --- 단기 화물 배치 ---
        auto [final_reachable, ___] = bfs(adj_list, occupation);
        std::vector<int> available_for_short_term;
        for(int n : final_reachable) {
            if (n != 0 && occupation[n] == -1) {
                available_for_short_term.push_back(n);
            }
        }

        // 게이트에서 가장 가까운 노드들부터 선택
        size_t zone_size = std::min(next_port_cars.size(), available_for_short_term.size());
        std::vector<int> loading_nodes(available_for_short_term.begin(), available_for_short_term.begin() + zone_size);

        // 블로킹 방지를 위해 깊은 곳(리스트의 뒤쪽)부터 배치하도록 역순 정렬
        std::reverse(loading_nodes.begin(), loading_nodes.end());

        // 경로 일괄 계산을 위해 dijkstra 한 번만 호출
        auto [_____, prev_nodes] = dijkstra(adj_list, occupation);

        for (size_t i = 0; i < std::min(next_port_cars.size(), loading_nodes.size()); ++i) {
            int car_k = next_port_cars[i];
            int spot_n = loading_nodes[i];

            Path path = path_backtracking(prev_nodes, 0, spot_n);
            occupation[spot_n] = car_k;
            route_list.emplace_back(path, car_k);
        }
    }

    // --- 5. 결과 반환 ---
    return { std::optional<PortSchedule>{route_list}, occupation };
}

UnloadingResult _unloading_heuristic(
    const DemandInfo K,
    const ShortestPaths shortest_paths,
    int time_step,
    const std::vector<int>& initial_occupation
//     const std::optional<std::vector<int>>& fixed_demands
) {
    // --- 1. 초기화 ---
    PortSchedule routes;
    std::vector<int> rehandling_demands;
    std::vector<int> occ = initial_occupation; // 상태 변경을 위한 내부 복사본

    // --- 2. 하역 대상 식별 ---
    std::vector<std::pair<int, int>> unloading_tasks; // {demand_k_idx, node}
//    if (!fixed_demands) { // fixed_demands가 지정되지 않은 경우
    for (int node = 0; node < occ.size(); ++node) {
        int demand_k_idx = occ[node];
        if (demand_k_idx != -1 && K[demand_k_idx].first.second == time_step) {
            unloading_tasks.emplace_back(demand_k_idx, node);
        }
    }
//        }
//    } else { // fixed_demands가 지정된 경우
//        for (int demand_k_idx : fixed_demands.value()) {
//            // Python의 np.where(occ == demand_k_idx)와 동일
//            for (int loc = 0; loc < occ.size(); ++loc) {
//                if (occ[loc] == demand_k_idx) {
//                    unloading_tasks.emplace_back(demand_k_idx, loc);
//                }
//            }
//        }
//    }

    // --- 3. 작업 우선순위 결정 ---
    // 게이트에서 가까운 순서로 정렬
    std::sort(unloading_tasks.begin(), unloading_tasks.end(), [&](const auto& a, const auto& b) {
        bool a_has_path = !shortest_paths[a.second].empty();
        bool b_has_path = !shortest_paths[b.second].empty();
        if (!a_has_path) return false; // a 경로 없으면 우선순위 최하
        if (!b_has_path) return true;  // b 경로 없으면 우선순위 최하
        // task.second가 노드 위치
        return shortest_paths[a.second][0].size() < shortest_paths[b.second][0].size();
    });

    // --- 4. 정렬된 작업 실행 ---
    for (const auto& task : unloading_tasks) {
        int demand_id = task.first;
        int node = task.second;

        // 중요: 이전 재배치 작업으로 인해 이미 차가 옮겨졌을 수 있으므로 확인
        if (occ[node] != demand_id) {
            continue;
        }

        if (shortest_paths[node].empty()) continue; // 경로 없으면 처리 불가

        // --- 최적 경로 탐색 (방해물이 가장 적은 경로) ---
        using PathOption = std::tuple<int, std::vector<int>, Path>; // blocker_count, blocker_nodes, path
        std::vector<PathOption> path_options;

        for (const auto& path : shortest_paths[node]) {
            std::vector<int> blocking_nodes;
            // path[:-1]와 동일: 마지막 노드(게이트)는 제외하고 경로 상의 노드들 확인
            for (size_t i = 0; i < path.size() - 1; ++i) {
                int n_on_path = path[i];
                if (occ[n_on_path] != -1) {
                    blocking_nodes.push_back(n_on_path);
                }
            }
            path_options.emplace_back(blocking_nodes.size(), blocking_nodes, path);
        }

        // 방해물 수가 가장 적은 경로 선택
        auto min_it = std::min_element(path_options.begin(), path_options.end(), [](const auto& a, const auto& b){
            return std::get<0>(a) < std::get<0>(b);
        });

        auto [blocker_count, blocking_nodes, best_path] = *min_it;

        // --- 방해물 재배치 ---
        // 방해물들도 게이트와 가까운 순서로 처리
        std::sort(blocking_nodes.begin(), blocking_nodes.end(), [&](int a, int b) {
            bool a_has_path = !shortest_paths[a].empty();
            bool b_has_path = !shortest_paths[b].empty();
            if (!a_has_path) return false;
            if (!b_has_path) return true;
            return shortest_paths[a][0].size() < shortest_paths[b][0].size();
        });

        for (int blocking_node : blocking_nodes) {
            // best_path에서 blocking_node까지의 경로 부분(segment)을 찾음
            auto it = std::find(best_path.begin(), best_path.end(), blocking_node);
            Path path_segment(best_path.begin(), it + 1);

            int rehandling_demand_id = occ[blocking_node];

            // 경로를 뒤집어 (노드 -> 게이트) 재배치 경로 생성
            std::reverse(path_segment.begin(), path_segment.end());
            routes.emplace_back(path_segment, rehandling_demand_id);

            // 이번에 내릴 화물이 아니었다면, 나중에 다시 실어야 할 목록에 추가
            if (K[rehandling_demand_id].first.second != time_step) {
                rehandling_demands.push_back(rehandling_demand_id);
            }
            occ[blocking_node] = -1; // 점유 상태 업데이트
        }

        // --- 최종 하역 ---
        Path final_path = best_path;
        std::reverse(final_path.begin(), final_path.end());
        routes.emplace_back(final_path, demand_id);
        occ[node] = -1; // 점유 상태 업데이트
    }

    // --- 5. 결과 반환 ---
    return { routes, rehandling_demands, occ };
}


UnloadingResult _unloading_heuristic_with_SA(
    int N,
    int p, // 현재 포트(시간 단계)
    const DemandInfo& K,
    const AdjacencyList& adj_list,
    const std::vector<int>& initial_occupation,
    bool SA = false
) {
    // --- 1. 초기화 ---
    PortSchedule route_list;
    std::vector<int> rehandling_demands;
    std::vector<int> node_allocations = initial_occupation; // 상태 변경을 위한 내부 복사본

    // C++ 랜덤 엔진 초기화
    std::random_device rd;
    std::mt19937 rng(rd());

    // 하역 대상들의 위치 수집
    std::vector<std::pair<int, int>> pending; // {demand_k_idx, node}
    for (int k = 0; k < K.size(); ++k) {
        if (K[k].first.second == p) { // 이번 포트 하역 대상
            for (int n = 0; n < N; ++n) {
                if (node_allocations[n] == k) {
                    pending.emplace_back(k, n);
                }
            }
        }
    }

    // SA 모드에서의 사이드 이동 시도 설정
    const double SA_TRY_RATIO = SA ? 0.5 : 0.0; // 블로커의 50% 정도를 랜덤 시도
    const int SA_TRY_CAP = SA ? 2 : 0;          // 경로별 최대 2개까지만 시도

    // --- 2. 메인 루프 ---
    while (!pending.empty()) {
        // 현재 노드 상태에서 최적 경로 재계산
        auto [dist, prev] = _lexi_dijkstra_all(N, adj_list, node_allocations, 0);

        std::vector<std::tuple<int, int, Path>> zero_block; // k, n, path
        // k, n, path, blocks, B(blocker_nodes)
        std::vector<std::tuple<int, int, Path, int, std::vector<int>>> cand;

        // list(pending)과 동일한 효과를 위해 복사본 사용
        auto pending_copy = pending;
        for (const auto& task : pending_copy) {
            int k = task.first;
            int n = task.second;

            auto [b, d] = dist[n];
            if (b >= 1e9) continue;

            // 경로 복원 (0->n)
            Path path;
            int cur = n;
            while (cur != -1) {
                path.push_back(cur);
                if (prev[cur] == cur) break; // 무한 루프 방지
                cur = prev[cur];
            }
            std::reverse(path.begin(), path.end());

            if (path.empty() || path[0] != 0) continue;

            int blocks = b - (node_allocations[n] != -1 ? 1 : 0);

            if (blocks == 0) {
                zero_block.emplace_back(k, n, path);
            } else {
                std::vector<int> B;
                // path[1:-1]
                for (size_t i = 1; i < path.size() - 1; ++i) {
                    if (node_allocations[path[i]] != -1) {
                        B.push_back(path[i]);
                    }
                }
                cand.emplace_back(k, n, path, blocks, B);
            }
        }

        // --- 3. 0-블록 후보 처리 ---
        if (!zero_block.empty()) {
            for (const auto& item : zero_block) {
                auto [k_sel, n_sel, path_sel] = item;
                if (node_allocations[n_sel] != k_sel) continue;

                Path reversed_path = path_sel;
                std::reverse(reversed_path.begin(), reversed_path.end());
                route_list.emplace_back(reversed_path, k_sel);
                node_allocations[n_sel] = -1;

                pending.erase(std::remove_if(pending.begin(), pending.end(),
                    [k_sel, n_sel](const auto& p){ return p.first == k_sel && p.second == n_sel; }), pending.end());
            }
            continue; // 메인 루프 재시작
        }

        if (cand.empty()) {
            break; // 처리할 후보가 더 이상 없음
        }

        // --- 4. 가장 쉬운 후보 선택 (차단수, 경로길이) ---
        std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) {
            if (std::get<3>(a) != std::get<3>(b)) {
                return std::get<3>(a) < std::get<3>(b); // blocks
            }
            return std::get<2>(a).size() < std::get<2>(b).size(); // len(path)
        });

        auto [k_sel, n_sel, path_sel, blocks_sel, B_sel] = cand[0];

        // --- 5. 블로커 처리 ---
        std::map<int, int> pos;
        for (size_t i = 0; i < path_sel.size(); ++i) {
            pos[path_sel[i]] = i;
        }
        std::vector<int> B_work = B_sel;

        std::vector<int> try_set;
        if (SA && !B_work.empty()) {
            std::shuffle(B_work.begin(), B_work.end(), rng);
            int k_try = std::min(SA_TRY_CAP, std::max(1, static_cast<int>(B_work.size() * SA_TRY_RATIO)));
            try_set.assign(B_work.begin(), B_work.begin() + k_try);
        }

        for (int bn : B_work) {
            if (node_allocations[bn] == -1) continue; // 이미 처리됨

            auto it = pos.find(bn);
            if (it == pos.end()) continue;
            int idx_bn = it->second;

            int k_block = node_allocations[bn];

            // 블로커가 이번 포트 하역 대상이면 그냥 하역
            if (K[k_block].first.second == p) {
                Path cut(path_sel.begin(), path_sel.begin() + idx_bn + 1);
                std::reverse(cut.begin(), cut.end());
                route_list.emplace_back(cut, k_block);
                node_allocations[bn] = -1;

                pending.erase(std::remove_if(pending.begin(), pending.end(),
                    [k_block, bn](const auto& p){ return p.first == k_block && p.second == bn; }), pending.end());
                continue;
            }

            bool moved = false;
            if (std::find(try_set.begin(), try_set.end(), bn) != try_set.end()) {
                // 프리픽스 옆자리로 0-블록 사이드 이동 시도
                Path side_path = _try_random_side_move(N, adj_list, path_sel, node_allocations, idx_bn, rng);
                if (!side_path.empty()) {
                    route_list.emplace_back(side_path, k_block);
                    node_allocations[bn] = -1;
                    node_allocations[side_path.back()] = k_block; // 새 위치로 이동
                    moved = true;
                }
            }

            if (!moved) {
                // 폴백: 임시하역 (bn -> 0)
                Path cut(path_sel.begin(), path_sel.begin() + idx_bn + 1);
                std::reverse(cut.begin(), cut.end());
                route_list.emplace_back(cut, k_block);

                bool is_unload_target_at_p = false;
                for (size_t k_idx = 0; k_idx < K.size(); ++k_idx) {
                    if (K[k_idx].first.second == p && k_idx == k_block) {
                        is_unload_target_at_p = true;
                        break;
                    }
                }
                if (!is_unload_target_at_p) {
                   rehandling_demands.push_back(k_block);
                }
                node_allocations[bn] = -1;
            }
        }

        // --- 6. 본체 하역 ---
        if (node_allocations[n_sel] == k_sel) {
            Path reversed_path = path_sel;
            std::reverse(reversed_path.begin(), reversed_path.end());
            route_list.emplace_back(reversed_path, k_sel);
            node_allocations[n_sel] = -1;

            pending.erase(std::remove_if(pending.begin(), pending.end(),
                [k_sel, n_sel](const auto& p){ return p.first == k_sel && p.second == n_sel; }), pending.end());
        }
    }

    return { route_list, rehandling_demands, node_allocations };
}


int _calculate_future_next_port_peak_load(
    const DemandInfo K,
    int origin,
    int destination
) {
    int max_next_port_load = 0;

    for (int current_p = origin; current_p < destination; ++current_p) {
        int current_next_port_load = 0;
        for (const auto &demand : K) {
            if (demand.first.first == current_p && demand.first.second == current_p + 1) {
                current_next_port_load += demand.second;
            }
        }

        max_next_port_load = std::max(max_next_port_load, current_next_port_load);
    }

    return max_next_port_load;
}

// Unloading heuristic은 min_blocking_path를 사용하도록 수정
UnloadingResult _unloading_heuristic_updated(
    const DemandInfo& K,
    const AdjacencyList& adj_list,
    int p,
    const std::vector<int>& initial_occupation
) {
    PortSchedule route_list;
    std::vector<int> rehandling_demands;
    std::vector<int> node_allocations = initial_occupation;

    // 언로딩 대상 수요
    std::map<int, int> K_unload;
    for (size_t idx = 0; idx < K.size(); ++idx) {
        if (K[idx].first.second == p) {
            K_unload[idx] = K[idx].second;
        }
    }

    for (const auto& [k, r] : K_unload) {
        std::vector<int> unloading_nodes;
        for (int n = 0; n < node_allocations.size(); ++n) {
            if (node_allocations[n] == k) {
                unloading_nodes.push_back(n);
            }
        }
        std::sort(unloading_nodes.begin(), unloading_nodes.end());

        for (int n : unloading_nodes) {
            if (node_allocations[n] == -1) continue;

            auto [path_0_to_n, blocking_nodes] = min_blocking_path(adj_list, node_allocations, n, 0);

            if (blocking_nodes.empty()) {
                // 차단 없음 - 바로 언로딩
                Path reversed_path = path_0_to_n;
                std::reverse(reversed_path.begin(), reversed_path.end());
                route_list.emplace_back(reversed_path, k);
                node_allocations[n] = -1;
            } else {
                // 차단 노드 처리
                for (int bn : blocking_nodes) {
                    // bn까지의 부분 경로 추출
                    Path cut;
                    for (int x : path_0_to_n) {
                        cut.push_back(x);
                        if (x == bn) break;
                    }
                    std::reverse(cut.begin(), cut.end());
                    route_list.emplace_back(cut, node_allocations[bn]);

                    // 리핸들링 수요 확인
                    if (K_unload.find(node_allocations[bn]) == K_unload.end()) {
                        rehandling_demands.push_back(node_allocations[bn]);
                    }
                    node_allocations[bn] = -1;
                }

                // 본체 하역
                Path reversed_path = path_0_to_n;
                std::reverse(reversed_path.begin(), reversed_path.end());
                route_list.emplace_back(reversed_path, k);
                node_allocations[n] = -1;
            }
        }
    }

    return {route_list, rehandling_demands, node_allocations};
}

#endif // HEURISTIC_H