#include <iostream>
#include <cstdint>
#include <random>
#include <array>
#include <algorithm>
#include <cstdio>
#include <cmath>

static std::random_device RD;
static std::mt19937 GEN(RD());

// number of battlefields
const size_t N = 4;
// number of soldiers
const size_t S = 10;
// total strategies
const size_t TOTAL_STRATEGIES = nCr(N + S, S);
uint16_t nCr(int n, int r) {
    if (r < 0 || r > n) return 0;
    if (r == 0) return 1;
    return nCr(n, r-1) * (n-r+1) / r;
}

class CrmRpsAgent {
 private:
    // for calculating current strategy
    std::array<double, N*S> regretSum = {0};

    // for calculating current strategy
    std::array<double, N*S> currentStrategy = {0};

    // for calculating convergent strategy
    std::array<double, N*S> accumulatedStrategy = {0};

 public:
    CrmRpsAgent() {
        updateStrategy();
    }

    // converts each decision array of size N to a unique ID
    uint32_t decisionToID(uint8_t* decision) {
        uint32_t id = 0;
        for (int i = 0; i < N; i++) {
            id += decision[i];
            id *= S;
        }
        return id;
    }
    // converts each decision ID back to a decision array of size N
    void IDToDecision(uint32_t id, uint8_t* decision) {
        for (int i = 0; i < N; i++) {
            decision[N-i-1] = id % S;
            id /= S;
        }
    }

    std::array<double, N*S> updateStrategy() {
        // Calculate positive regret
        std::array<double, static_cast<size_t>(N*S)> regrets;
        double totalPositiveRegret = std::transform_reduce(regretSum.begin(), regretSum.end(), 0,
            std::plus<>{}, [](double v) { return std::max(v, 0.); });

        // Default uniform strategy if total regret is non-positive
        if (totalPositiveRegret <= 0) {
            for (size_t i = 0; i < currentStrategy.size(); i++) {
                currentStrategy[i] = 1.0 / currentStrategy.size();
            }
        } else {
        // Calculate normalized positive regret strategy
            for (size_t i = 0; i < currentStrategy.size(); i++) {
                currentStrategy[i] = std::max(regretSum[i], 0.) / totalPositiveRegret;
            }
        }

        return currentStrategy;
    }

    std::array<double, N*S> getAverageStrategy() {
        double normalizer = 0;
        for (size_t i = 0; i < accumulatedStrategy.size(); i++) {
            normalizer += accumulatedStrategy[i];
        }
        std::array<double, N*S> avgStrategy = {0};
        for (size_t i = 0; i < accumulatedStrategy.size(); i++) {
            avgStrategy[i] = accumulatedStrategy[i] / normalizer;
        }
        return avgStrategy;
    }

    uint8_t getAction() {
        // randomly choose the action
        double r = static_cast<double>(GEN()) / static_cast<double>(GEN.max());
        double boundary = 0;
        for (size_t i = 0; i < currentStrategy.size(); i++) {
            boundary += currentStrategy[i];
            if (r < boundary)
                return static_cast<uint8_t>(i);
        }
        return static_cast<uint8_t>(currentStrategy.size() - 1);
    }

    std::array<double, N*S> getRegretSum() {
        return regretSum;
    }

    std::array<double, N*S> getStrategy() {
        return currentStrategy;
    }

    void accumulation(int8_t my_action, int8_t opp_action) {
        // accumulate strategy
        for (size_t i = 0; i < accumulatedStrategy.size(); i++)
            accumulatedStrategy[i] += currentStrategy[i];

        // accumulate regret
        int8_t utility = utility_matrix[my_action][opp_action];
        for (int a = 0; a < 3; a++) {
            regretSum[a] += utility_matrix[a][opp_action] - utility;
        }

        updateStrategy();
    }
};

const int ITERATIONS = 100000;
const int OUTPUT_ITERS = 10000;

int main() {
    printf("Start training\n");
    CrmRpsAgent crmAgentA;
    CrmRpsAgent crmAgentB;

    for (int i = 0; i < ITERATIONS; i++) {
        uint8_t actionA = crmAgentA.getAction();
        uint8_t actionB = crmAgentB.getAction();
        crmAgentA.accumulation(actionA, actionB);
        crmAgentB.accumulation(actionB, actionA);

        if ((i+1) % OUTPUT_ITERS == 0 || (i+1) == ITERATIONS) {
            printf("Iteration %d | CRM Average Strategy Agent A: [%.5f, %.5f, %.5f] | Agent B: [%.5f, %.5f, %.5f] | Current strategy: [%.2f, %.2f, %.2f] | Regret: [%.1f, %.1f, %.1f] | \n",
                   i+1,
                   crmAgentA.getAverageStrategy()[ROCK],
                   crmAgentA.getAverageStrategy()[PAPER],
                   crmAgentA.getAverageStrategy()[SCISSORS],
                   crmAgentB.getAverageStrategy()[ROCK],
                   crmAgentB.getAverageStrategy()[PAPER],
                   crmAgentB.getAverageStrategy()[SCISSORS],
                   crmAgentA.getStrategy()[ROCK],
                   crmAgentA.getStrategy()[PAPER],
                   crmAgentA.getStrategy()[SCISSORS],
                   crmAgentA.getRegretSum()[ROCK],
                   crmAgentA.getRegretSum()[PAPER],
                   crmAgentA.getRegretSum()[SCISSORS]);
        }
    }
    return 0;
}
