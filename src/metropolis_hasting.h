#ifndef METROPOLIS_HASTING_H
#define METROPOLIS_HASTING_H

#include <cmath>
#include <vector>
#include <iostream>
#include "types.h"
#include "blockmodel.h"
#include "output_functions.h"

/* Cooling schedules */
double exponential_schedule(unsigned int t, float_vec_t cooling_schedule_kwargs) noexcept;

double linear_schedule(unsigned int t, float_vec_t cooling_schedule_kwargs) noexcept;

double logarithmic_schedule(unsigned int t, float_vec_t cooling_schedule_kwargs) noexcept;

double constant_schedule(unsigned int t, float_vec_t cooling_schedule_kwargs) noexcept;

class metropolis_hasting {
protected:
    std::uniform_real_distribution<> random_real;

public:
    // Ctor
    metropolis_hasting() : random_real(0, 1) { ; }
    double accu_r_ = 0.;  // for Tiago Peixoto's trick
    double entropy_min_ = 0.;
    double entropy_max_ = 0.;  // for automatic detection to stop the algorithm after T successive MCMC sweeps occurred

    // Virtual methods
    virtual std::vector<mcmc_state_t> sample_proposal_distribution(
            blockmodel_t &blockmodel,
            std::mt19937 &engine
    ) noexcept { return std::vector<mcmc_state_t>(); }  // bogus virtual implementation

    // TODO: check -- why passing moves as a reference doesn't make the program run faster?
    virtual double transition_ratio(
            const blockmodel_t &blockmodel,
            const float_mat_t &p,
            const std::vector<mcmc_state_t> moves
    ) noexcept { return 0; } // bogus virtual implementation

    // Common methods
    bool step(blockmodel_t &blockmodel,
              const float_mat_t &p,
              double temperature,
              std::mt19937 &engine) noexcept;

    bool step_for_estimate(blockmodel_t &blockmodel,
                           const float_mat_t &p,
                           std::mt19937 &engine) noexcept;

    double marginalize(blockmodel_t &blockmodel,
                       uint_mat_t &marginal_distribution,
                       const float_mat_t &p,
                       unsigned int burn_in_time,
                       unsigned int sampling_frequency,
                       unsigned int num_samples,
                       std::mt19937 &engine) noexcept;

    double anneal(blockmodel_t &blockmodel,
                  const float_mat_t &p,
                  double (*cooling_schedule)(unsigned int, float_vec_t),
                  float_vec_t cooling_schedule_kwargs,
                  unsigned int duration,
                  unsigned int steps_await,
                  std::mt19937 &engine) noexcept;

    double estimate(blockmodel_t &blockmodel,
                    uint_mat_t &marginal_distribution,
                    const float_mat_t &p,
                    unsigned int burn_in_time,
                    unsigned int sampling_frequency,
                    unsigned int num_samples,
                    std::mt19937 &engine) noexcept;
};

/* Inherited classes with specific definitions */
class mh_naive : public metropolis_hasting {
public:
    std::vector<mcmc_state_t> sample_proposal_distribution(blockmodel_t &blockmodel, std::mt19937 &engine) noexcept override;

    double transition_ratio(const blockmodel_t &blockmodel,
                            const float_mat_t &p,
                            const std::vector<mcmc_state_t> moves) noexcept override;
};

class mh_tiago : public metropolis_hasting {
public:
    std::vector<mcmc_state_t> sample_proposal_distribution(blockmodel_t &blockmodel, std::mt19937 &engine) noexcept override;

    double transition_ratio(const blockmodel_t &blockmodel,
                            const float_mat_t &p,
                            const std::vector<mcmc_state_t> moves) noexcept override;
};

class mh_riolo : public metropolis_hasting {
public:
    std::vector<mcmc_state_t> sample_proposal_distribution(blockmodel_t &blockmodel,
                                                          std::mt19937 &engine) noexcept override;

    double transition_ratio(const blockmodel_t &blockmodel,
                            const float_mat_t &p,
                            const std::vector<mcmc_state_t> moves) noexcept override;
};

class mh_riolo_uni1: public metropolis_hasting {
public:
    std::vector<mcmc_state_t> sample_proposal_distribution(blockmodel_t &blockmodel,
                                                           std::mt19937 &engine) noexcept override;

    double transition_ratio(const blockmodel_t &blockmodel,
                            const float_mat_t &p,
                            const std::vector<mcmc_state_t> moves) noexcept override;
};

class mh_riolo_uni2: public metropolis_hasting {
public:
    std::vector<mcmc_state_t> sample_proposal_distribution(blockmodel_t &blockmodel,
                                                           std::mt19937 &engine) noexcept override;

    double transition_ratio(const blockmodel_t &blockmodel,
                            const float_mat_t &p,
                            const std::vector<mcmc_state_t> moves) noexcept override;
};

#endif // METROPOLIS_HASTING_H
