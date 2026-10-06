#include <iostream>
#include <cstdint>
#include <random>
#include <array>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <string>

static std::random_device RD;
static std::mt19937 GEN(RD());

// number of battlefields
const size_t N = 3;
// number of soldiers
const size_t S = 5;
// total strategies
uint16_t nCr(int n, int r) {
    if (r < 0 || r > n) return 0;
    if (r == 0) return 1;
    return nCr(n, r-1) * (n-r+1) / r;
}
const size_t TOTAL_STRATEGIES = nCr(N + S - 1, N-1);

class CrmBlottoAgent {
 private:
    std::vector<std::vector<int8_t>> utility_matrix;

    // for calculating current strategy
    std::vector<double> regretSum = std::vector<double>(TOTAL_STRATEGIES, 0);

    // for calculating current strategy
    std::vector<double> currentStrategy = std::vector<double>(TOTAL_STRATEGIES, 1.0 / TOTAL_STRATEGIES);

    // for calculating convergent strategy
    std::vector<double> accumulatedStrategy = std::vector<double>(TOTAL_STRATEGIES, 0);

 public:
    std::vector<std::string> strategy_IDs;

    CrmBlottoAgent() {
        permuteStrategies();
        computeUtilities();
        updateStrategy();
    }

    // calculate all strategies and map to ids
    void permuteStrategies() {
        strategy_IDs.reserve(TOTAL_STRATEGIES);
        std::array<uint16_t, N+S-1> perm_array = {0};
        std::fill_n(perm_array.begin(), N-1, 1);  // 1 = battlefield separator
        size_t strategy_count = 0;
        for (size_t _ = 0; _ < TOTAL_STRATEGIES; _++) {
            std::next_permutation(perm_array.begin(), perm_array.end());
            std::string strategy_ID = "";
            uint16_t last = 0;
            for (size_t i = 0; i <= perm_array.size(); i++) {
                if ((i == perm_array.size()) || (perm_array[i] == 1)) {
                    strategy_ID += static_cast<char>('a' + last);
                    last = 0;
                    continue;
                }
                last++;
            }
            strategy_IDs.push_back(strategy_ID);
        }
    }

    // precompute utility matrix
    void computeUtilities() {
        for (size_t me = 0; me < strategy_IDs.size(); me++) {
            utility_matrix.push_back(std::vector<int8_t>(strategy_IDs.size(), 0));
            for (size_t opp = 0; opp < strategy_IDs.size(); opp++) {
                std::string my_strat = strategy_IDs[me];
                std::string opp_strat = strategy_IDs[opp];
                int utility = 0;
                for (size_t b = 0; b < my_strat.size(); b++) {
                    if (my_strat[b] > opp_strat[b])
                        utility++;
                    else if (my_strat[b] < opp_strat[b])
                        utility--;
                }
                utility_matrix[me][opp] = static_cast<int8_t>(std::min(std::max(-1, utility), 1));
            }
        }
    }
    void updateStrategy() {
        // Calculate positive regret
        double totalPositiveRegret = std::transform_reduce(regretSum.begin(), regretSum.end(), 0.,
            std::plus<>{}, [](double v) { return std::max(v, 0.); });

        // Default uniform strategy if total regret is non-positive
        if (totalPositiveRegret <= 0) {
            for (size_t i = 0; i < currentStrategy.size(); i++)
                currentStrategy[i] = 1.0 / currentStrategy.size();
            return;
        }

        // Calculate normalized positive regret strategy
        for (size_t i = 0; i < currentStrategy.size(); i++) {
            currentStrategy[i] = std::max(regretSum[i], 0.) / totalPositiveRegret;
        }
    }

    std::vector<double> getAverageStrategy() {
        double normalizer = 0;
        for (size_t i = 0; i < accumulatedStrategy.size(); i++) {
            normalizer += accumulatedStrategy[i];
        }
        std::vector<double> avgStrategy = std::vector<double>(accumulatedStrategy.size(), 0);
        for (size_t i = 0; i < accumulatedStrategy.size(); i++) {
            avgStrategy[i] = accumulatedStrategy[i] / normalizer;
        }
        return avgStrategy;
    }

    uint16_t getAction() {
        // randomly choose the action
        double r = static_cast<double>(GEN()) / static_cast<double>(GEN.max());
        double boundary = 0;
        for (size_t i = 0; i < currentStrategy.size(); i++) {
            boundary += currentStrategy[i];
            if (r < boundary)
                return static_cast<uint16_t>(i);
        }
        return static_cast<uint16_t>(currentStrategy.size() - 1);
    }

    std::vector<double> getRegretSum() {
        return regretSum;
    }

    std::vector<double> getStrategy() {
        return currentStrategy;
    }

    void accumulation(uint16_t my_action, uint16_t opp_action) {
        // accumulate strategy
        for (size_t i = 0; i < accumulatedStrategy.size(); i++)
            accumulatedStrategy[i] += currentStrategy[i];

        // accumulate regret
        int8_t utility = utility_matrix[my_action][opp_action];
        for (int a = 0; a < TOTAL_STRATEGIES; a++) {
            regretSum[a] += utility_matrix[a][opp_action] - utility;
        }
        updateStrategy();
    }
};

const int ITERATIONS = 1000000;
const int OUTPUT_ITERS = 100000;

int main() {
    printf("Start training\n");
    CrmBlottoAgent crmAgentA;
    CrmBlottoAgent crmAgentB;

    for (auto s : crmAgentA.strategy_IDs)
        printf("%s ", s.c_str());
    printf("\n");

    for (int i = 0; i < ITERATIONS; i++) {
        uint16_t actionA = crmAgentA.getAction();
        uint16_t actionB = crmAgentB.getAction();
        crmAgentA.accumulation(actionA, actionB);
        crmAgentB.accumulation(actionB, actionA);

        if ((i+1) % OUTPUT_ITERS == 0 || (i+1) == ITERATIONS) {
            printf("Iteration %d | CRM Average Strategy \nAgent A: [", i+1);
            for (int s = 0; s < crmAgentA.strategy_IDs.size(); s++)
                if (crmAgentA.getAverageStrategy()[s] > 1E-4)
                    printf("%s: %.4f, ", crmAgentA.strategy_IDs[s].c_str(), crmAgentA.getAverageStrategy()[s]);
            printf("] | \nAgent B: [");
            for (int s = 0; s < crmAgentB.strategy_IDs.size(); s++)
                if (crmAgentB.getAverageStrategy()[s] > 1E-4)
                    printf("%s: %.4f, ", crmAgentB.strategy_IDs[s].c_str(), crmAgentB.getAverageStrategy()[s]);
            printf("]\n");
        }
    }
    return 0;
}
