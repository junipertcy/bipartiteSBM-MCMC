#include "metropolis_hasting.hh"
#include "support/cache.hh"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Non class methods
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Implemented from
// http://www.fys.ku.dk/~andresen/BAhome/ownpapers/permanents/annealSched.pdf
double exponential_schedule(size_t t, float_vec_t cooling_schedule_kwargs) noexcept {
    // kwargs is the speed of the exponential cooling.
    return cooling_schedule_kwargs[0] * std::pow(cooling_schedule_kwargs[1], t);
}

double linear_schedule(size_t t, float_vec_t cooling_schedule_kwargs) noexcept {
    // kwargs are the initial temperature and a rate of linear cooling.
    return cooling_schedule_kwargs[0] - cooling_schedule_kwargs[1] * t;
}

double logarithmic_schedule(size_t t, float_vec_t cooling_schedule_kwargs) noexcept {
    // kwargs are the rate of linear cooling and a delay (typically 1).
    return cooling_schedule_kwargs[0] / safelog_fast(t + cooling_schedule_kwargs[1]);
}

double constant_schedule(size_t t, float_vec_t cooling_schedule_kwargs) noexcept {
    // kwargs are the rate of linear cooling and a delay (typically 1).
    return cooling_schedule_kwargs[0];
}

double abrupt_cool_schedule(size_t t, float_vec_t cooling_schedule_kwargs) noexcept {
    // Does there exist a way to improve the cooling? Maybe make it an exponential decay one against an abrupt cooling?
    if (t < cooling_schedule_kwargs[0]) {
        return 1.;
    } else {
        return 0.;
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// metropolis_hasting class
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
inline bool metropolis_hasting::step(blockmodel_t&& blockmodel,
                              double temperature,
                              std::mt19937& engine) noexcept {
    moves_ = sample_proposal_distribution(std::move(blockmodel), std::move(engine));
    double a = 0.;
    double exp_minus_diff_entropy = transition_ratio(std::move(blockmodel), moves_);
    if (temperature == 0.) {
        if (exp_minus_diff_entropy >= 1.) {
            a = 1.;
        }
    } else {
        a = std::pow(exp_minus_diff_entropy , 1. / temperature) * accu_r_;
    }
    if (random_real(engine) < a) {
        blockmodel.apply_mcmc_moves(std::move(moves_));
        return true;
    }
    return false;
}

bool metropolis_hasting::step_for_estimate(blockmodel_t &blockmodel,
                                           std::mt19937& engine) noexcept {
    states_ = sample_proposal_distribution(std::move(blockmodel), std::move(engine));
    double a = transition_ratio_est(blockmodel, states_);
    if (random_real(engine) < a) {
        if (blockmodel.get_is_bipartite()) {
            blockmodel.apply_mcmc_states(states_);
        } else {
            blockmodel.apply_mcmc_states_u(states_);
        }
        is_last_state_rejected_ = false;
        return true;
    }
    is_last_state_rejected_ = true;
    return false;
}

double metropolis_hasting::marginalize(blockmodel_t&& blockmodel,
                                       uint_mat_t &marginal_distribution,
                                       size_t burn_in_time,
                                       size_t sampling_frequency,
                                       size_t num_samples,
                                       std::mt19937 &engine) noexcept {

    size_t accepted_steps = 0;
    // Burn-in period
    for (size_t t = 0; t < burn_in_time; ++t) {
        step(std::move(blockmodel), 1.0, engine);
    }
    // Sampling
    memberships_.clear();
    memberships_.resize(blockmodel.get_g(), 0);
    for (size_t t = 0; t < sampling_frequency * num_samples; ++t) {
        if (t % sampling_frequency == 0) {
            // Sample the blockmodel
            memberships_ = *blockmodel.get_memberships();
#if OUTPUT_HISTORY == 1 // compile time output
            output_vec<uint_vec_t>(memberships_, std::cout);
#endif
            const size_t n = blockmodel.get_N();
            for (size_t i = 0; i < n; ++i) {
                marginal_distribution[i][memberships_[i]] += 1;
            }
        }
        if (step(std::move(blockmodel), 1.0, engine)) {
            ++accepted_steps;
        }
    }
    return (double) accepted_steps / ((double) sampling_frequency * num_samples);
}

double metropolis_hasting::anneal(
        blockmodel_t &blockmodel,
        double (*cooling_schedule)(size_t, float_vec_t),
        float_vec_t cooling_schedule_kwargs,
        size_t duration,
        size_t steps_await,
        std::mt19937 &engine) noexcept {

    size_t accepted_steps = 0;
    size_t u = 0;
    entropy_min_ = 1000000;
    entropy_max_ = 0;
    for (size_t t = 0; t < duration; ++t) {
#if OUTPUT_HISTORY == 1  // compile time output
        output_vec<uint_vec_t>(*blockmodel.get_memberships(), std::cout);
#endif
        double _entropy_max = entropy_max_;
        double _entropy_min = entropy_min_;

        if (step(std::move(blockmodel), cooling_schedule(t, cooling_schedule_kwargs), engine)) {
            ++accepted_steps;
        }
        // TODO: check the effect of `epsilon` from the code block here
        if (_entropy_max == entropy_max_ && _entropy_min == entropy_min_) {
            u += 1;
        } else {
            u = 0;
        }
        if (u == steps_await) {
            std::clog << "algorithm stops after: " << t << " steps. \n";
            t = duration;  // TODO: check -- if acceptance rate is even meaningful in annealing mode?
            return double(accepted_steps) / double(t);
        }
    }
    return double(accepted_steps) / double(duration);
}

double metropolis_hasting::estimate(blockmodel_t &blockmodel,
                                    size_t sampling_frequency,
                                    size_t num_samples,
                                    std::mt19937 &engine) noexcept {
    size_t accepted_steps = 0;
    size_t t_1000 = sampling_frequency * num_samples - 1000;  // stdout the last 1000 steps

    if (blockmodel.get_is_bipartite()) {
        log_idl_ = blockmodel.get_int_data_likelihood_from_mb_bi(*blockmodel.get_memberships(), false);
    } else {
        log_idl_ = blockmodel.get_int_data_likelihood_from_mb_uni(*blockmodel.get_memberships(), false);
    }

    // Sampling
    memberships_.clear();
    memberships_.resize(blockmodel.get_KA() + blockmodel.get_KB(), 0);
    for (size_t t = 0; t < sampling_frequency * num_samples; ++t) {
        if (t % sampling_frequency == 0) {
            // Sample the blockmodel
            if (t >= t_1000) {
#if OUTPUT_HISTORY == 1 // compile time output
                if (blockmodel.get_is_bipartite()) {
                    std::cout << t << "," << blockmodel.get_KA() << "," << blockmodel.get_KB() << "," << blockmodel.compute_log_posterior_from_mb_bi(*blockmodel.get_memberships());
                    memberships_ = *blockmodel.get_memberships();
                    for (auto const &i: memberships_) std::cout << "," << i;
                    std::cout << "\n";
                  //  std::cout << t << "," << blockmodel.get_KA() << "," << blockmodel.get_KB() << "," << blockmodel.get_log_posterior_from_mb(*blockmodel.get_memberships()) << "\n";
                } else {
                    std::cout << t << "," << blockmodel.get_K() << "," << blockmodel.get_log_posterior_from_mb_uni(*blockmodel.get_memberships());
                    memberships_ = *blockmodel.get_memberships();
                    for (auto const &i: memberships_) std::cout << "," << i;
                    std::cout << "\n";
                }
                //output_vec<uint_vec_t>(memberships, std::cout)
#endif
            }
        }
        if (step_for_estimate(blockmodel, engine)) {
            ++accepted_steps;
        }
    }
    return (double) accepted_steps / ((double) sampling_frequency * num_samples);
}


inline const double metropolis_hasting::transition_ratio(blockmodel_t&& blockmodel,
                                     const std::vector<mcmc_state_t> &moves) noexcept {
    v_ = moves[0].vertex;
    r_ = moves[0].source;
    s_ = moves[0].target;
    double epsilon = blockmodel.get_epsilon();

    size_t KA = blockmodel.get_KA();
    size_t KB = blockmodel.get_KA();

    ki = blockmodel.get_k(v_);
    int deg = blockmodel.get_degree(v_);

    m0 = blockmodel.get_m();
    padded_m0 = blockmodel.get_m_r();

    citer_m0_s = m0->at(s_).begin();

    citer_padded_m0 = (*padded_m0).begin();
    citer_m0_r = m0->at(r_).begin();

    size_t INT_padded_m0r = padded_m0->at(r_);
    size_t INT_padded_m1r = INT_padded_m0r - deg;

    size_t INT_padded_m0s = padded_m0->at(s_);
    size_t INT_padded_m1s = INT_padded_m0s + deg;

    double accu0 = 0.;
    double accu1 = 0.;
    double entropy0 = 0.;
    double entropy1 = 0.;

    if (r_ < KA) {  // Ka-case
        B_ = KA;
    } else {  // Kb-case
        B_ = KB;
    }
    auto criterion = (r_ < KA) ? [](size_t a, size_t k) { return a >= k; } : [](size_t a, size_t k) { return a < k; };
    for (auto const& _k: *ki ){
        size_t index = &_k - &ki->at(0);
        if (criterion(index, KA)) {
            accu0 += _k * (*citer_m0_s + epsilon) / (*citer_padded_m0 + epsilon * B_);
            accu1 += _k * ((*citer_m0_r - _k) + epsilon) / (*citer_padded_m0 + epsilon * B_);  // last term, originally `*citer_padded_m1`.
            entropy0 -= *citer_m0_r * (safelog_fast(*citer_m0_r) - safelog_fast(INT_padded_m0r) - safelog_fast(*citer_padded_m0));
            entropy0 -= *citer_m0_s * (safelog_fast(*citer_m0_s) - safelog_fast(INT_padded_m0s) - safelog_fast(*citer_padded_m0));
            entropy1 -= (*citer_m0_r - _k) * (safelog_fast((*citer_m0_r - _k)) - safelog_fast(INT_padded_m1r) - safelog_fast(*citer_padded_m0));
            entropy1 -= (*citer_m0_s + _k) * (safelog_fast((*citer_m0_s + _k)) - safelog_fast(INT_padded_m1s) - safelog_fast(*citer_padded_m0));
        }
        ++citer_m0_s;
        ++citer_padded_m0;
        ++citer_m0_r;
    }

    accu_r_ = accu1 / accu0;
    if (entropy0 >= entropy_max_) {
        entropy_max_ = entropy0;
    }
    if (entropy0 <= entropy_min_) {
        entropy_min_ = entropy0;
    }

    return std::exp(-(entropy1 - entropy0));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// virtual functions implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/* Implementation for the single vertex change (SBM) */
std::vector<mcmc_state_t> mh_tiago::sample_proposal_distribution(blockmodel_t&& blockmodel,
                                                                 std::mt19937&& engine) const noexcept {
    return blockmodel.single_vertex_change_tiago(engine);
}


std::vector<mcmc_state_t> mh_riolo_uni::sample_proposal_distribution(blockmodel_t&& blockmodel, std::mt19937&& engine) const noexcept {
    return blockmodel.mcmc_state_change_riolo_uni(engine);
}

double mh_riolo_uni::transition_ratio_est(blockmodel_t &blockmodel, std::vector<mcmc_state_t> &states) noexcept {
    double log_idl_0 = log_idl_;
    if (!is_last_state_rejected_) {  // candidate state accepted
        log_idl_0 = cand_log_idl_;
        blockmodel.sync_internal_states_est();

        log_idl_ = log_idl_0;
    }
    double log_idl_1 = blockmodel.get_int_data_likelihood_from_mb_uni(states[0].memberships, true);
    cand_log_idl_ = log_idl_1;
    return std::exp(+log_idl_1 - log_idl_0);
}

std::vector<mcmc_state_t>
mh_riolo::sample_proposal_distribution(blockmodel_t&& blockmodel, std::mt19937&& engine) const noexcept {
    return blockmodel.mcmc_state_change_riolo(engine);
}

double mh_riolo::transition_ratio_est(blockmodel_t &blockmodel, std::vector<mcmc_state_t> &states) noexcept {
    double log_idl_0 = log_idl_;
    if (!is_last_state_rejected_) {  // candidate state accepted
        log_idl_0 = cand_log_idl_;
        blockmodel.sync_internal_states_est();
        log_idl_ = log_idl_0;
    }
    double log_idl_1 = blockmodel.get_int_data_likelihood_from_mb_bi(states[0].memberships, true);
    cand_log_idl_ = log_idl_1;
    return std::exp(+log_idl_1 - log_idl_0);
}

