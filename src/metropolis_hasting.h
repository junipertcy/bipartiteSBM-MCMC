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

double abrupt_cool_schedule(unsigned int t, float_vec_t cooling_schedule_kwargs) noexcept;

class metropolis_hasting {
protected:
    std::uniform_real_distribution<> random_real;

public:
    // Ctor
    metropolis_hasting() : random_real(0, 1) { ; }
    bool is_last_state_rejected_ = true;  // for `estimate` mode
    double log_idl_ = 0.;  // for `estimate` mode
    double cand_log_idl_ = 0.;  // for `estimate` mode
    double accu_r_ = 0.;  // for Tiago Peixoto's smart MCMC
    double entropy_min_ = 0.;
    double entropy_max_ = 0.;  // for automatic detection to stop the algorithm after T successive MCMC sweeps occurred

    // Virtual methods
    virtual std::vector<mcmc_state_t> sample_proposal_distribution(
            blockmodel_t &blockmodel,
            std::mt19937 &engine
    ) noexcept { return std::vector<mcmc_state_t>(); }  // bogus virtual implementation

    virtual double transition_ratio(
            const blockmodel_t &blockmodel,
            const std::vector<mcmc_state_t> &moves
    ) noexcept { return 0; } // bogus virtual implementation

    virtual double transition_ratio_est(
            blockmodel_t &blockmodel,
            std::vector<mcmc_state_t> &moves
    ) noexcept { return 0; } // bogus virtual implementation

    // Common methods
    bool step(blockmodel_t &blockmodel,
              double temperature,
              std::mt19937 &engine) noexcept;

    bool step_for_estimate(blockmodel_t &blockmodel,
                           std::mt19937 &engine) noexcept;

    double marginalize(blockmodel_t &blockmodel,
                       uint_mat_t &marginal_distribution,
                       unsigned int burn_in_time,
                       unsigned int sampling_frequency,
                       unsigned int num_samples,
                       std::mt19937 &engine) noexcept;

    double anneal(blockmodel_t &blockmodel,
                  double (*cooling_schedule)(unsigned int, float_vec_t),
                  float_vec_t cooling_schedule_kwargs,
                  unsigned int duration,
                  unsigned int steps_await,
                  std::mt19937 &engine) noexcept;

    double estimate(blockmodel_t &blockmodel,
                    unsigned int sampling_frequency,
                    unsigned int num_samples,
                    std::mt19937 &engine) noexcept;
};

/* Inherited classes with specific definitions */
class mh_tiago : public metropolis_hasting {
public:
    std::vector<mcmc_state_t> sample_proposal_distribution(blockmodel_t &blockmodel, std::mt19937 &engine) noexcept override;

    double transition_ratio(const blockmodel_t &blockmodel,
                            const std::vector<mcmc_state_t> &moves) noexcept override;
};

class mh_riolo : public metropolis_hasting {
public:
    std::vector<mcmc_state_t> sample_proposal_distribution(blockmodel_t &blockmodel,
                                                          std::mt19937 &engine) noexcept override;

    double transition_ratio_est(blockmodel_t &blockmodel, std::vector<mcmc_state_t> &moves) noexcept override;
};

class mh_riolo_uni: public metropolis_hasting {
public:

    std::vector<mcmc_state_t> sample_proposal_distribution(blockmodel_t &blockmodel,
                                                           std::mt19937 &engine) noexcept override;

    double transition_ratio_est(blockmodel_t &blockmodel, std::vector<mcmc_state_t> &moves) noexcept override;
};

#endif // METROPOLIS_HASTING_H
