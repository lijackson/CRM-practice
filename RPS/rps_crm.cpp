#include <iostream>
#include <cstdint>
#include <random>
#include <array>
#include <algorithm>
#include <cstdio>
#include <cmath>

static int ROCK = 0;
static int PAPER = 1;
static int SCISSORS = 2;

static std::random_device RD;
static std::mt19937 GEN(RD());

class RockBiasedRpsAgent {
 private:
    double strategy[3] = {0.4, 0.3, 0.3};

 public:
    uint8_t getAction() {
        double r = static_cast<double>(GEN()) / static_cast<double>(GEN.max());
        if (r < strategy[ROCK])
            return ROCK;
        if (r < strategy[ROCK] + strategy[PAPER])
            return PAPER;
        return SCISSORS;
    }
};

class CrmRpsAgent {
 private:
    // Game score for matrix[my_action][opp_action]
    const int8_t utility_matrix[3][3] = {{0, -1, 1},
                                        {1, 0, -1},
                                        {-1, 1, 0}};

    // for calculating current strategy
    std::array<double, 3> regretSum = {0, 0, 0};

    // for calculating current strategy
    std::array<double, 3> currentStrategy = {0, 0, 0};

    // for calculating convergent strategy
    std::array<double, 3> accumulatedStrategy = {0, 0, 0};

 public:
    CrmRpsAgent() {
        updateStrategy();
    }

    std::array<double, 3> updateStrategy() {
        // Calculate positive regret
        double rockRegret = std::fmax(0, regretSum[ROCK]);
        double paperRegret = std::fmax(0, regretSum[PAPER]);
        double scissorsRegret = std::fmax(0, regretSum[SCISSORS]);
        double totalRegret = rockRegret + paperRegret + scissorsRegret;

        // Default uniform strategy if total regret is non-positive
        if (totalRegret <= 0)
            return std::array<double, 3>{1.0 / 3, 1.0 / 3, 1.0 / 3};

        // Calculate normalized positive regret strategy
        currentStrategy[ROCK] = rockRegret / totalRegret;
        currentStrategy[PAPER] = paperRegret / totalRegret;
        currentStrategy[SCISSORS] = scissorsRegret / totalRegret;
        return currentStrategy;
    }

    std::array<double, 3> getAverageStrategy() {
        double normalizer = accumulatedStrategy[ROCK] + accumulatedStrategy[PAPER] + accumulatedStrategy[SCISSORS];
        return {accumulatedStrategy[ROCK] / normalizer,
                accumulatedStrategy[PAPER] / normalizer,
                accumulatedStrategy[SCISSORS] / normalizer};
    }

    uint8_t getAction() {
        // randomly choose the action
        double r = static_cast<double>(GEN()) / static_cast<double>(GEN.max());
        if (r < currentStrategy[ROCK])
            return ROCK;
        if (r < currentStrategy[ROCK] + currentStrategy[PAPER])
            return PAPER;
        return SCISSORS;
    }

    std::array<double, 3> getRegretSum() {
        return regretSum;
    }

    std::array<double, 3> getStrategy() {
        return currentStrategy;
    }

    void accumulation(int8_t my_action, int8_t opp_action) {
        // accumulate strategy
        accumulatedStrategy[ROCK] += currentStrategy[ROCK];
        accumulatedStrategy[PAPER] += currentStrategy[PAPER];
        accumulatedStrategy[SCISSORS] += currentStrategy[SCISSORS];

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
