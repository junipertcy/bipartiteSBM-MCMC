#ifndef METROPOLIS_HASTING_HH
#define METROPOLIS_HASTING_HH

#include <cmath>
#include <vector>
#include <iostream>
#include "types.hh"
#include "blockmodel.hh"
#include "output_functions.hh"
#include "support/cache.hh"

/* Cooling schedules */
double exponential_schedule(size_t t, float_vec_t cooling_schedule_kwargs) noexcept;

double linear_schedule(size_t t, float_vec_t cooling_schedule_kwargs) noexcept;

double logarithmic_schedule(size_t t, float_vec_t cooling_schedule_kwargs) noexcept;

double constant_schedule(size_t t, float_vec_t cooling_schedule_kwargs) noexcept;

double abrupt_cool_schedule(size_t t, float_vec_t cooling_schedule_kwargs) noexcept;

class metropolis_hasting {

protected:
    std::uniform_real_distribution<> random_real;
    double cand_log_idl_ = 0.;  // for `estimate` mode
    double log_idl_ = 0.;  // for `estimate` mode
    bool is_last_state_rejected_ = true;  // for `estimate` mode
    double entropy_min_ = 0.;
    double entropy_max_ = 0.;  // for automatic detection to stop the algorithm after T successive MCMC sweeps occurred
    double accu_r_ = 0.;  // for Tiago Peixoto's smart MCMC

private:
    /// for marginalize
    uint_vec_t memberships_;
    std::vector<mcmc_state_t> moves_;
    std::vector<mcmc_state_t> states_;

    bool step_for_estimate(blockmodel_t &blockmodel,
                           std::mt19937 &engine) noexcept;

public:
    // Ctor
    metropolis_hasting() : random_real(0, 1) {
        ;
    }

    // Virtual methods
    virtual std::vector<mcmc_state_t> sample_proposal_distribution(
            blockmodel_t&& blockmodel,
            std::mt19937&& engine
    ) const noexcept { return std::vector<mcmc_state_t>(1); }  // bogus virtual implementation

    virtual double transition_ratio_est(
            blockmodel_t &blockmodel,
            std::vector<mcmc_state_t> &moves
    ) noexcept { return 0; } // bogus virtual implementation

    // Common methods
    inline bool step(blockmodel_t&& blockmodel,
              double temperature,
              std::mt19937 &engine) noexcept;

    inline const double transition_ratio(blockmodel_t&& blockmodel,
                                         const std::vector<mcmc_state_t>& moves) noexcept;

    double marginalize(blockmodel_t&& blockmodel,
                       uint_mat_t &marginal_distribution,
                       size_t burn_in_time,
                       size_t sampling_frequency,
                       size_t num_samples,
                       std::mt19937 &engine) noexcept;

    double anneal(blockmodel_t &blockmodel,
                  double (*cooling_schedule)(size_t, float_vec_t),
                  float_vec_t cooling_schedule_kwargs,
                  size_t duration,
                  size_t steps_await,
                  std::mt19937 &engine) noexcept;

    double estimate(blockmodel_t &blockmodel,
                    size_t sampling_frequency,
                    size_t num_samples,
                    std::mt19937 &engine) noexcept;

private:
    size_t v_{0};
    size_t r_{0};
    size_t s_{0};
    size_t B_{0};

    // TODO: how do we initiate values for these vectors? (or, should we?)
    const int_vec_t* ki;
    const uint_mat_t* m0;
    const uint_vec_t* padded_m0;

    std::vector<unsigned int>::const_iterator citer_m0_r;
    std::vector<unsigned int>::const_iterator citer_m0_s;
    std::vector<unsigned int>::const_iterator citer_padded_m0;

};

/* Inherited classes with specific definitions */
class mh_tiago : public metropolis_hasting {
public:
    std::vector<mcmc_state_t>
    sample_proposal_distribution(blockmodel_t&& blockmodel, std::mt19937&& engine) const noexcept override;

//    double transition_ratio(const blockmodel_t &blockmodel,
//                            const std::vector<mcmc_state_t> &moves) noexcept override;

};

class mh_riolo : public metropolis_hasting {
public:
    std::vector<mcmc_state_t> sample_proposal_distribution(blockmodel_t&& blockmodel,
                                                           std::mt19937&& engine) const noexcept override;

    double transition_ratio_est(blockmodel_t &blockmodel, std::vector<mcmc_state_t> &moves) noexcept override;
};

class mh_riolo_uni : public metropolis_hasting {
public:

    std::vector<mcmc_state_t> sample_proposal_distribution(blockmodel_t&& blockmodel,
                                                           std::mt19937&& engine) const noexcept override;

    double transition_ratio_est(blockmodel_t &blockmodel, std::vector<mcmc_state_t> &moves) noexcept override;
};

#endif // METROPOLIS_HASTING_H
