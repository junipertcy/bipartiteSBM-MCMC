#include "metropolis_hasting.h"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Non class methods
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Implemented from
// http://www.fys.ku.dk/~andresen/BAhome/ownpapers/permanents/annealSched.pdf
double exponential_schedule(unsigned int t, float_vec_t cooling_schedule_kwargs) noexcept {
    // kwargs is the speed of the exponential cooling.
    return cooling_schedule_kwargs[0] * std::pow(cooling_schedule_kwargs[1], t);
}

double linear_schedule(unsigned int t, float_vec_t cooling_schedule_kwargs) noexcept {
    // kwargs are the initial temperature and a rate of linear cooling.
    return cooling_schedule_kwargs[0] - cooling_schedule_kwargs[1] * t;
}

double logarithmic_schedule(unsigned int t, float_vec_t cooling_schedule_kwargs) noexcept {
    // kwargs are the rate of linear cooling and a delay (typically 1).
    return cooling_schedule_kwargs[0] / std::log(t + cooling_schedule_kwargs[1]);
}

double constant_schedule(unsigned int t, float_vec_t cooling_schedule_kwargs) noexcept {
    // kwargs are the rate of linear cooling and a delay (typically 1).
    return cooling_schedule_kwargs[0];
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// metropolis_hasting class
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
bool metropolis_hasting::step(blockmodel_t &blockmodel,
                              double temperature,
                              std::mt19937 &engine) noexcept {
    std::vector<mcmc_state_t> moves = sample_proposal_distribution(blockmodel, engine);
    double a = std::pow(transition_ratio(blockmodel, moves), 1 / temperature) * accu_r_;
    if (random_real(engine) < a) {
        blockmodel.apply_mcmc_moves(moves);
        return true;
    }
    return false;
}

bool metropolis_hasting::step_for_estimate(blockmodel_t &blockmodel,
                                           std::mt19937 &engine) noexcept {
    std::vector<mcmc_state_t> states = sample_proposal_distribution(blockmodel, engine);
    double a = transition_ratio_est(blockmodel, states);
    if (random_real(engine) < a) {
        if (blockmodel.get_is_bipartite()) {
            blockmodel.apply_mcmc_states(states);
        } else {
            blockmodel.apply_mcmc_states_u(states);
        }
        is_last_state_rejected_ = false;
        return true;
    }
    is_last_state_rejected_ = true;
    return false;
}

double metropolis_hasting::marginalize(blockmodel_t &blockmodel,
                                       uint_mat_t &marginal_distribution,
                                       unsigned int burn_in_time,
                                       unsigned int sampling_frequency,
                                       unsigned int num_samples,
                                       std::mt19937 &engine) noexcept {
    unsigned int accepted_steps = 0;
    // Burn-in period
    for (unsigned int t = 0; t < burn_in_time; ++t) {
        step(blockmodel, 1.0, engine);
    }
    // Sampling
    for (unsigned int t = 0; t < sampling_frequency * num_samples; ++t) {
        if (t % sampling_frequency == 0) {
            // Sample the blockmodel
            uint_vec_t memberships = blockmodel.get_memberships();
#if OUTPUT_HISTORY == 1 // compile time output
            output_vec<uint_vec_t>(memberships, std::cout);
#endif
            for (unsigned int i = 0; i < blockmodel.get_N(); ++i) {
                marginal_distribution[i][memberships[i]] += 1;
            }
        }
        if (step(blockmodel, 1.0, engine)) {
            ++accepted_steps;
        }
    }
    return (double) accepted_steps / ((double) sampling_frequency * num_samples);
}

double metropolis_hasting::anneal(
        blockmodel_t &blockmodel,
        double (*cooling_schedule)(unsigned int, float_vec_t),
        float_vec_t cooling_schedule_kwargs,
        unsigned int duration,
        unsigned int steps_await,
        std::mt19937 &engine) noexcept {

    unsigned int accepted_steps = 0;
    unsigned int u = 0;
    entropy_min_ = 1000000;
    entropy_max_ = 0;
    for (unsigned int t = 0; t < duration; ++t) {
#if OUTPUT_HISTORY == 1  // compile time output
        output_vec<uint_vec_t>(blockmodel.get_memberships(), std::cout);
#endif
        double _entropy_max = entropy_max_;
        double _entropy_min = entropy_min_;

        if (step(blockmodel, cooling_schedule(t, cooling_schedule_kwargs), engine)) {
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
            t = duration;  // TODO: check -- if acceptance rate even meaningful in annealing mode?
            return double(accepted_steps) / double(t);
        }
    }
    return double(accepted_steps) / double(duration);
}

double metropolis_hasting::estimate(blockmodel_t &blockmodel,
                                    unsigned int sampling_frequency,
                                    unsigned int num_samples,
                                    std::mt19937 &engine) noexcept {
    unsigned int accepted_steps = 0;
    unsigned int t_1000 = sampling_frequency * num_samples - 1000;  // stdout the last 1000 steps

    if (blockmodel.get_is_bipartite()) {
        log_idl_ = blockmodel.get_int_data_likelihood_from_mb_bi(blockmodel.get_memberships(), false);
    } else {
        log_idl_ = blockmodel.get_int_data_likelihood_from_mb_uni(blockmodel.get_memberships(), false);
    }

    // Sampling
    for (unsigned int t = 0; t < sampling_frequency * num_samples; ++t) {
        if (t % sampling_frequency == 0) {
            // Sample the blockmodel
            if (t >= t_1000) {
#if OUTPUT_HISTORY == 1 // compile time output
                if (blockmodel.get_is_bipartite()) {
                    std::cout << t << "," << blockmodel.get_KA() << "," << blockmodel.get_KB() << "," << blockmodel.compute_log_posterior_from_mb_bi(blockmodel.get_memberships());
                    uint_vec_t mb_ = blockmodel.get_memberships();
                    for (auto const &i: mb_) std::cout << "," << i;
                    std::cout << "\n";
                  //  std::cout << t << "," << blockmodel.get_KA() << "," << blockmodel.get_KB() << "," << blockmodel.get_log_posterior_from_mb(blockmodel.get_memberships()) << "\n";
                } else {
                    std::cout << t << "," << blockmodel.get_K() << "," << blockmodel.get_log_posterior_from_mb_uni(blockmodel.get_memberships());
                    uint_vec_t mb_ = blockmodel.get_memberships();
                    for (auto const &i: mb_) std::cout << "," << i;
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// virtual functions implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/* Implementation for the single vertex change (SBM) */
std::vector<mcmc_state_t> mh_tiago::sample_proposal_distribution(blockmodel_t &blockmodel,
                                                                 std::mt19937 &engine) noexcept {
    return blockmodel.single_vertex_change_tiago(engine);
}

double mh_tiago::transition_ratio(const blockmodel_t &blockmodel,
                                  const std::vector<mcmc_state_t> &moves) noexcept {
    unsigned int v = moves[0].vertex;
    unsigned int r = moves[0].source;
    unsigned int s = moves[0].target;
    double epsilon = blockmodel.get_epsilon();

    int_vec_t ki = blockmodel.get_k(v);
    int_vec_t deg = blockmodel.get_degree();
    int_vec_t n = blockmodel.get_size_vector();

    uint_mat_t m0 = blockmodel.get_m();
    uint_vec_t m0_r = blockmodel.get_m_r();

    uint_mat_t m1 = m0;
    uint_vec_t m1_r = m0_r;

    m1_r[r] -= deg[v];
    m1_r[s] += deg[v];

    for (unsigned int i = 0; i < ki.size(); ++i) {
        if (ki[i] != 0) {
            m1[r][i] -= ki[i];
            m1[s][i] += ki[i];

            m1[i][r] = m1[r][i];
            m1[i][s] = m1[s][i];
        }
    }
    double accu0_ = 0;
    double accu1_ = 0;
    int B_;
    if (r < blockmodel.get_KA()) {
        B_ = blockmodel.get_KA();
        // If it is type-A nodes to move, then on K_A label possibilities are allowed for the node to change
    } else {
        B_ = blockmodel.get_KB();
    }
    for (unsigned int t_ = 0; t_ < n.size(); ++t_) {
        accu0_ += (double) ki[t_] * ((double) m0[t_][s] + epsilon) / ((double) m0_r[t_] + epsilon * B_);
        accu1_ += (double) ki[t_] * ((double) m1[t_][r] + epsilon) / ((double) m1_r[t_] + epsilon * B_);
    }

    double entropy0 = 0.;
    double entropy1 = 0.;
    for (unsigned int i = 0; i < ki.size(); ++i) {  // TODO: re-write it
            if (m0_r[r] * m0_r[i] * m0[r][i] != 0) {
                entropy0 -= 1. / 1. * (double) m0[r][i] *
                            std::log((double) m0[r][i] / (double) m0_r[r] / (double) m0_r[i]);
            }
            if (m0_r[s] * m0_r[i] * m0[s][i] != 0) {
                entropy0 -= 1. / 1. * (double) m0[s][i] *
                            std::log((double) m0[s][i] / (double) m0_r[s] / (double) m0_r[i]);
            }
            if (m1_r[r] * m1_r[i] * m1[r][i] != 0) {
                entropy1 -= 1. / 1. * (double) m1[r][i] *
                            std::log((double) m1[r][i] / (double) m1_r[r] / (double) m1_r[i]);
            }
            if (m1_r[s] * m1_r[i] * m1[s][i] != 0) {
                entropy1 -= 1. / 1. * (double) m1[s][i] *
                            std::log((double) m1[s][i] / (double) m1_r[s] / (double) m1_r[i]);
            }
    }

    accu_r_ = accu1_ / accu0_;
    if (entropy0 >= entropy_max_) {
        entropy_max_ = entropy0;
    }
    if (entropy0 <= entropy_min_) {
        entropy_min_ = entropy0;
    }
    double a = std::exp(- (entropy1 - entropy0));
    return a;
}

std::vector<mcmc_state_t> mh_riolo_uni::sample_proposal_distribution(blockmodel_t &blockmodel, std::mt19937 &engine) noexcept {
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

std::vector<mcmc_state_t> mh_riolo::sample_proposal_distribution(blockmodel_t &blockmodel, std::mt19937 &engine) noexcept {
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

