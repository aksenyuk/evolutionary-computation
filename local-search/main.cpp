#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <functional>
#include <chrono>
#include <random>
#include <set>



struct Node {
    double x;
    double y;
    double cost;
};



std::vector<Node> readCSV(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<Node> nodes;
    std::string line;
    while (std::getline(file, line)) {
        Node node;
        size_t pos = 0;

        pos = line.find(";");
        node.x = std::stod(line.substr(0, pos));
        line.erase(0, pos + 1);

        pos = line.find(";");
        node.y = std::stod(line.substr(0, pos));
        line.erase(0, pos + 1);

        node.cost = std::stod(line);

        nodes.push_back(node);
    }
    file.close();
    return nodes;
}



double eucDistance(const Node &a, const Node &b) {
    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
}



std::vector<std::vector<double>> computeDistanceMatrix(const std::vector<Node> &nodes) {
    int n = nodes.size();
    std::vector<std::vector<double>> matrix(n, std::vector<double>(n));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = eucDistance(nodes[i], nodes[j]) + nodes[j].cost;
        }
    }
    return matrix;
}



using AlgoFunc = std::function<std::pair<std::vector<int>, double>(
    const std::vector<std::vector<double>> &,
    const std::vector<Node> &,
    int
)>;



struct Metrics {
    double avgTime;
    double minTime;
    double maxTime;
    double avgCost;
    double minCost;
    double maxCost;
    std::vector<int> bestSolution;
};



void writeMetricsToCSV(
    const std::vector<std::vector<Metrics>> &allMetrics, 
    const std::vector<std::string> &files, 
    const std::vector<std::string> &algoNames
) {
    std::ofstream timeFile("timeMetrics.csv");
    std::ofstream costFile("costMetrics.csv");
    std::ofstream solutionFile("bestSolution.csv");

    timeFile << "Algorithm;";
    costFile << "Algorithm;";
    solutionFile << "Algorithm;";
    for (const auto &filename : files) {
        timeFile << filename << ";";
        costFile << filename << ";";
        solutionFile << filename << ";";
    }
    timeFile << "\n";
    costFile << "\n";
    solutionFile << "\n";

    for (size_t j = 0; j < allMetrics.size(); ++j) {
        const auto &algoMetrics = allMetrics[j];

        timeFile << algoNames[j] << ";";
        costFile << algoNames[j] << ";";
        solutionFile << algoNames[j] << ";";

        for (size_t i = 0; i < files.size(); ++i) {
            timeFile << algoMetrics[i].avgTime << " (" << algoMetrics[i].minTime << ", " << algoMetrics[i].maxTime << ")";
            costFile << algoMetrics[i].avgCost << " (" << algoMetrics[i].minCost << ", " << algoMetrics[i].maxCost << ")";

            solutionFile << "\"";
            for (const auto &node : algoMetrics[i].bestSolution) {
                solutionFile << node;
                if (&node != &algoMetrics[i].bestSolution.back()) {
                    solutionFile << ",";
                }
}
            solutionFile << "\"";

            if (i != files.size() - 1) {
                timeFile << "; ";
                costFile << "; ";
                solutionFile << "; ";
            }
        }

        timeFile << "\n";
        costFile << "\n";
        solutionFile << "\n";
    }

    timeFile.close();
    costFile.close();
    solutionFile.close();
}



std::pair<std::vector<int>, double> randomSearch(
    const std::vector<std::vector<double>> &matrix,
    const std::vector<Node> &nodes,
    int startIndex
) {
    int n = nodes.size();
    std::vector<int> path = {startIndex};
    std::vector<int> remainingNodes;

    for (int i = 0; i < n; ++i) {
        if (i != startIndex) {
            remainingNodes.push_back(i);
        }
    }

    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

    while (path.size() < n / 2) {
        int randIndex = rng() % remainingNodes.size();
        int nextNode = remainingNodes[randIndex];
        path.push_back(nextNode);
        remainingNodes.erase(remainingNodes.begin() + randIndex);
    }

    double totalCost = 0.0;
    for (size_t i = 0; i < path.size() - 1; ++i) {
        totalCost += matrix[path[i]][path[i+1]];
    }
    totalCost += matrix[path.back()][startIndex];

    return {path, totalCost};
}



std::pair<std::vector<int>, double> greedyCycle(
    const std::vector<std::vector<double>> &matrix,
    const std::vector<Node> &nodes,
    int current_node_index = -1
) {
    int n = matrix.size();
    
    if (current_node_index == -1) {
        std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
        current_node_index = rng() % n;
    }

    std::vector<int> to_visit(n);
    std::iota(to_visit.begin(), to_visit.end(), 0); 
    to_visit.erase(std::remove(to_visit.begin(), to_visit.end(), current_node_index), to_visit.end());

    std::sort(to_visit.begin(), to_visit.end(),
        [&matrix, &current_node_index](int a, int b) {
            return matrix[current_node_index][a] < matrix[current_node_index][b];
        }
    );

    std::vector<int> solution = {current_node_index};

    while (solution.size() < n / 2) {
        int closest_neighbor = -1;
        double closest_neighbor_distance = std::numeric_limits<double>::max();
        size_t closest_neighbor_position;

        for (const int &neighbor : to_visit) {
            double neighbor_distance;
            size_t candidate_position;

            if (solution.size() == 1) {
                neighbor_distance = matrix[current_node_index][neighbor] + matrix[neighbor][current_node_index];
                candidate_position = 1;
            } else {
                double min_dist = std::numeric_limits<double>::max();
                for (size_t i = 1; i < solution.size(); ++i) {
                    double dist = matrix[solution[i-1]][neighbor] + matrix[neighbor][solution[i]] - matrix[solution[i-1]][solution[i]];
                    if (dist < min_dist) {
                        min_dist = dist;
                        candidate_position = i;
                    }
                }
                neighbor_distance = min_dist;
            }

            if (neighbor_distance < closest_neighbor_distance) {
                closest_neighbor = neighbor;
                closest_neighbor_distance = neighbor_distance;
                closest_neighbor_position = candidate_position;
            }
        }

        solution.insert(solution.begin() + closest_neighbor_position, closest_neighbor);
        to_visit.erase(std::remove(to_visit.begin(), to_visit.end(), closest_neighbor), to_visit.end());
    }

    double total_cost = 0.0;
    for (size_t i = 0; i < solution.size() - 1; ++i) {
        total_cost += matrix[solution[i]][solution[i+1]];
    }
    total_cost += matrix[solution.back()][solution[0]];

    return {solution, total_cost};
}




int main() {
    std::vector<std::string> files = {"./data/TSPA.csv", "./data/TSPB.csv", "./data/TSPC.csv", "./data/TSPD.csv"};
    std::vector<AlgoFunc> algorithms = {randomSearch, greedyCycle};

    std::vector<std::vector<Metrics>> allMetrics;

    for (const auto &algo : algorithms) {
        std::vector<Metrics> algoMetrics;
        
        for (const auto &filename : files) {
            Metrics metrics = {
                .avgTime = 0.0,
                .minTime = std::numeric_limits<double>::max(),
                .maxTime = std::numeric_limits<double>::min(),
                .avgCost = 0.0,
                .minCost = std::numeric_limits<double>::max(),
                .maxCost = std::numeric_limits<double>::min(),
            };

            auto nodes = readCSV(filename);
            auto matrix = computeDistanceMatrix(nodes);

            int runs = nodes.size();
            for (int i = 0; i < runs; ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                auto [solution, cost] = algo(matrix, nodes, i);
                auto end = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> diff = end - start;
                metrics.avgTime += diff.count();
                metrics.maxTime = std::max(metrics.maxTime, diff.count());
                metrics.minTime = std::min(metrics.minTime, diff.count());

                metrics.avgCost += cost;
                metrics.maxCost = std::max(metrics.maxCost, cost);
                metrics.minCost = std::min(metrics.minCost, cost);

                if (metrics.bestSolution.empty() || cost < metrics.minCost) {
                    metrics.bestSolution = solution;
                }
            }

            metrics.avgTime /= runs;
            metrics.avgCost /= runs;
            algoMetrics.push_back(metrics);
        }

        allMetrics.push_back(algoMetrics);
    }

    std::vector<std::string> algoNames = {"Random Search", "Greedy Cycle"};
    writeMetricsToCSV(allMetrics, files, algoNames);

    return 0;
}
