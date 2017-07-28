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
                              const float_mat_t &p,
                              double temperature,
                              std::mt19937 &engine) noexcept {
    // TODO: this function makes the program unable to compile in -O3 mode
    std::vector<mcmc_state_t> moves = sample_proposal_distribution(blockmodel, engine);
    double a = std::pow(transition_ratio(blockmodel, p, moves) * std::pow(accu_r_, temperature), 1 / temperature);
    double b = std::pow(transition_ratio(blockmodel, p, moves), 1 / temperature);
    if (random_real(engine) < a) {
        blockmodel.apply_mcmc_moves(moves);
        return true;
    }
    return false;
}

bool metropolis_hasting::step_for_estimate(blockmodel_t &blockmodel,
                                           const float_mat_t &p,
                                           std::mt19937 &engine) noexcept {
    std::vector<mcmc_state_t> states = sample_proposal_distribution(blockmodel, engine);
    double a = transition_ratio(blockmodel, p, states);

    if (random_real(engine) < a) {
        if (blockmodel.get_is_bipartite()) {
            blockmodel.apply_mcmc_states(states);
        } else {
            blockmodel.apply_mcmc_states_u(states);
        }
        return true;
    }

    return false;
}

// TODO: this function is deprecated.
bool metropolis_hasting::step_for_estimate_heat_bath(blockmodel_t &blockmodel,
                                                     const float_mat_t &p,
                                                     std::mt19937 &engine) noexcept {

    unsigned int nsize_A_ = blockmodel.get_nsize_A();
    unsigned int nsize_B_ = blockmodel.get_nsize_B();
    unsigned int total_q = nsize_A_ + nsize_B_ + 2;
    double rnd = random_real(engine);
    double checkpoint_1 = double(nsize_A_ + nsize_B_) / double(total_q);
    double checkpoint_2 = double(nsize_A_ + nsize_B_ + 1) / double(total_q);
    if (rnd < checkpoint_1) {
        std::vector<mcmc_state_t> moves = sample_proposal_distribution(blockmodel, engine);
        blockmodel.apply_mcmc_moves(moves);
        return true;
    } else if (rnd >= checkpoint_1 && rnd < checkpoint_2) {
        // move_KA
        return blockmodel.change_KA(engine);
    } else if (rnd >= checkpoint_2) {
        // move_KB
        return blockmodel.change_KB(engine);
    } else {
        std::clog << "Sanity check: this should not happen! \n";
        return false;
    }
}

double metropolis_hasting::marginalize(blockmodel_t &blockmodel,
                                       uint_mat_t &marginal_distribution,
                                       const float_mat_t &p,
                                       unsigned int burn_in_time,
                                       unsigned int sampling_frequency,
                                       unsigned int num_samples,
                                       std::mt19937 &engine) noexcept {
    unsigned int accepted_steps = 0;
    // Burn-in period
    for (unsigned int t = 0; t < burn_in_time; ++t) {
        step(blockmodel, p, 1.0, engine);
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
        if (step(blockmodel, p, 1.0, engine)) {
            ++accepted_steps;
        }
    }
    return (double) accepted_steps / ((double) sampling_frequency * num_samples);
}

double metropolis_hasting::anneal(
        blockmodel_t &blockmodel,
        const float_mat_t &p,
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
        double entropy_max__ = entropy_max_;
        double entropy_min__ = entropy_min_;

        if (step(blockmodel, p, cooling_schedule(t, cooling_schedule_kwargs), engine)) {
            ++accepted_steps;
        };
        if (entropy_max__ == entropy_max_ && entropy_min__ == entropy_min_) {
            u += 1;
        } else {
            u = 0;
        }
        if (u == steps_await) {
            std::clog << "algorithm stops after: " << t << " steps. \n";
            t = duration;
            std::clog << "the acceptance rate is: " << double(accepted_steps) / double(t) << "\n";
//          output_vec<uint_vec_t>(memberships_entropy_min_, std::clog);
        }
    }
}

double metropolis_hasting::estimate(blockmodel_t &blockmodel,
                                    uint_mat_t &marginal_distribution,
                                    const float_mat_t &p,
                                    unsigned int burn_in_time,
                                    unsigned int sampling_frequency,
                                    unsigned int num_samples,
                                    std::mt19937 &engine) noexcept {
    unsigned int accepted_steps = 0;
    // Sampling
    for (unsigned int t = 0; t < sampling_frequency * num_samples; ++t) {
        if (t % sampling_frequency == 0) {
            // Sample the blockmodel
            if (t > 1) { // was: 100000
#if OUTPUT_HISTORY == 1 // compile time output
                if (blockmodel.get_is_bipartite()) {
                    std::cout << t << "," << blockmodel.get_KA() << "," << blockmodel.get_KB() << "," << blockmodel.get_log_posterior_from_mb(blockmodel.get_memberships()) << "\n";
                } else {
                    std::cout << t << "," << blockmodel.get_K() << "," << blockmodel.get_log_posterior_from_mb(blockmodel.get_memberships()) << "\n";
                }
                //output_vec<uint_vec_t>(memberships, std::cout)
#endif
            }
        }
        if (step_for_estimate(blockmodel, p, engine)) {

            ++accepted_steps;
        }
    }
    return (double) accepted_steps / ((double) sampling_frequency * num_samples);
}



// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// virtual functions implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/* Implementation for the single vertex change (SBM) */
std::vector<mcmc_state_t> mh_naive::sample_proposal_distribution(blockmodel_t &blockmodel,
                                                                 std::mt19937 &engine) noexcept {
    return blockmodel.single_vertex_change_naive(engine);
}

double mh_naive::transition_ratio(const blockmodel_t &blockmodel,
                                  const float_mat_t &p,
                                  const std::vector<mcmc_state_t> moves) noexcept {
    // NOTE: p is not used
    unsigned int v = moves[0].vertex;
    unsigned int r = moves[0].source;
    unsigned int s = moves[0].target;
    int_vec_t ki = blockmodel.get_k(v);
    int_vec_t n = blockmodel.get_size_vector();

    unsigned int num_edges = blockmodel.get_num_edges();
    double entropy_from_degree_correction = blockmodel.get_entropy_from_degree_correction();

    uint_mat_t m0 = blockmodel.get_m();
    uint_vec_t m0_r = blockmodel.get_m_r();

    uint_mat_t m1 = m0;
    uint_vec_t m1_r = m0_r;

    unsigned int m_r_ = 0;

    if (r == s) {
        m1 = m0;
    } else {
        for (unsigned int l = 0; l < n.size(); ++l) {
            if (l != r && l != s) {
                m1[r][l] = m0[r][l] - ki[l];
                m1[l][r] = m1[r][l];

                m1[s][l] = m0[s][l] + ki[l];
                m1[l][s] = m1[s][l];
            }
        }
        m1[r][s] = m0[r][s] - ki[s] + ki[r];
        m1[s][r] = m1[r][s];
    }


    for (unsigned int l = 0; l < n.size(); ++l) {
        if (l != r && l != s) {
            m1[r][l] -= ki[l];
            m1[l][r] = m1[r][l];

            m1[s][l] += ki[l];
            m1[l][s] = m1[s][l];
        }
    }

    for (unsigned int r = 0; r < n.size(); ++r) {
        m_r_ = 0;
        for (unsigned int s = 0; s < n.size(); ++s) {
            m_r_ += m1[r][s];
        }
        m1_r[r] = m_r_;
    }

    double entropy0 = -(double) num_edges - entropy_from_degree_correction;
    double entropy1 = entropy0;

    for (unsigned int r_ = 0; r_ < n.size(); ++r_) {
        for (unsigned int s_ = 0; s_ < n.size(); ++s_) {
            if (m0_r[r_] * m0_r[s_] * m0[r_][s_] != 0) {
                entropy0 -= 1. / 2. * (double) m0[r_][s_] *
                            std::log((double) m0[r_][s_] / (double) m0_r[r_] / (double) m0_r[s_]);
            }
            if (m1_r[r_] * m1_r[s_] * m1[r_][s_] != 0) {
                entropy1 -= 1. / 2. * (double) m1[r_][s_] *
                            std::log((double) m1[r_][s_] / (double) m1_r[r_] / (double) m1_r[s_]);
            }

        }
    }
    accu_r_ = 1.;
    if (entropy0 >= entropy_max_) {
        entropy_max_ = entropy0;
    }
    if (entropy0 <= entropy_min_) {
        entropy_min_ = entropy0;
    }

    double a = std::exp(+entropy0 - entropy1);
    return a;
}


/* Implementation for the single vertex change (PPM) */
std::vector<mcmc_state_t> mh_tiago::sample_proposal_distribution(blockmodel_t &blockmodel,
                                                                 std::mt19937 &engine) noexcept {
    return blockmodel.single_vertex_change_tiago(engine);
}

double mh_tiago::transition_ratio(const blockmodel_t &blockmodel,
                                  const float_mat_t &p,
                                  const std::vector<mcmc_state_t> moves) noexcept {
    // NOTE: p is not used
    unsigned int v = moves[0].vertex;
    unsigned int r = moves[0].source;
    unsigned int s = moves[0].target;
    double epsilon = blockmodel.get_epsilon();

    int_vec_t ki = blockmodel.get_k(v);
    int_vec_t n = blockmodel.get_size_vector();

    unsigned int num_edges = blockmodel.get_num_edges();
    double entropy_from_degree_correction = blockmodel.get_entropy_from_degree_correction();

    uint_mat_t m0 = blockmodel.get_m();
    uint_vec_t m0_r = blockmodel.get_m_r();

    uint_mat_t m1 = m0;
    uint_vec_t m1_r = m0_r;

    unsigned int m_r_ = 0;

    if (r == s) {
        m1 = m0;
    } else {
        for (unsigned int l = 0; l < n.size(); ++l) {
            if (l != r && l != s) {
                m1[r][l] = m0[r][l] - ki[l];
                m1[l][r] = m1[r][l];

                m1[s][l] = m0[s][l] + ki[l];
                m1[l][s] = m1[s][l];
            }
        }
        m1[r][s] = m0[r][s] - ki[s] + ki[r];
        m1[s][r] = m1[r][s];
    }


    for (unsigned int l = 0; l < n.size(); ++l) {
        if (l != r && l != s) {
            m1[r][l] -= ki[l];
            m1[l][r] = m1[r][l];

            m1[s][l] += ki[l];
            m1[l][s] = m1[s][l];
        }
    }

    for (unsigned int r = 0; r < n.size(); ++r) {
        m_r_ = 0;
        for (unsigned int s = 0; s < n.size(); ++s) {
            m_r_ += m1[r][s];
        }
        m1_r[r] = m_r_;
    }

    double entropy0 = -(double) num_edges - entropy_from_degree_correction;
    double entropy1 = entropy0;

    for (unsigned int r_ = 0; r_ < n.size(); ++r_) {
        for (unsigned int s_ = 0; s_ < n.size(); ++s_) {
            if (m0_r[r_] * m0_r[s_] * m0[r_][s_] != 0) {
                entropy0 -= 1. / 2. * (double) m0[r_][s_] *
                            std::log((double) m0[r_][s_] / (double) m0_r[r_] / (double) m0_r[s_]);
            }
            if (m1_r[r_] * m1_r[s_] * m1[r_][s_] != 0) {
                entropy1 -= 1. / 2. * (double) m1[r_][s_] *
                            std::log((double) m1[r_][s_] / (double) m1_r[r_] / (double) m1_r[s_]);
            }

        }
    }


//     Tiago Peixoto's trick
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

    accu_r_ = accu1_ / accu0_;
    if (entropy0 >= entropy_max_) {
        entropy_max_ = entropy0;
    }
    if (entropy0 <= entropy_min_) {
        entropy_min_ = entropy0;
    }
    double a = std::exp(+entropy0 - entropy1);
    return a;
}

std::vector<mcmc_state_t> mh_heat_bath::sample_proposal_distribution(blockmodel_t &blockmodel,
                                                                     std::mt19937 &engine) noexcept {
    return blockmodel.single_vertex_change_heat_bath(engine);
}

double mh_heat_bath::transition_ratio(const blockmodel_t &blockmodel,
                                      const float_mat_t &p,
                                      const std::vector<mcmc_state_t> moves) noexcept {
    // heat-bath jumps are rejection-free
    accu_r_ = 1.;
    return 1.;
}

std::vector<mcmc_state_t> mh_riolo_uni1::sample_proposal_distribution(blockmodel_t &blockmodel, std::mt19937 &engine) noexcept {
    return blockmodel.mcmc_state_change_riolo_uni1(engine);
}

std::vector<mcmc_state_t> mh_riolo_uni2::sample_proposal_distribution(blockmodel_t &blockmodel, std::mt19937 &engine) noexcept {
    return blockmodel.mcmc_state_change_riolo_uni2(engine);
}

double mh_riolo_uni1::transition_ratio(const blockmodel_t &blockmodel,
                                  const float_mat_t &p,
                                  const std::vector<mcmc_state_t> states) noexcept {

    uint_mat_t m_rs_0_ = blockmodel.get_m();
    uint_vec_t m_r_0_ = blockmodel.get_m_r();
    int_vec_t n_r_0_ = blockmodel.get_size_vector();
    unsigned int num_edges = blockmodel.get_num_edges();
    int_vec_t k_r_0_ = blockmodel.get_k_r_from_mb(blockmodel.get_memberships());
    double p_0_ = 2. * num_edges / (double) blockmodel.get_N() / (double) blockmodel.get_N();

    double log_idl_0 = 0.;

    for (auto r = 0; r < n_r_0_.size(); ++r) {
        log_idl_0 += k_r_0_[r] * std::log(n_r_0_[r]) + blockmodel.get_log_factorial(n_r_0_[r] - 1) -
                     blockmodel.get_log_factorial(n_r_0_[r] + k_r_0_[r] - 1);
        log_idl_0 += blockmodel.get_log_factorial(m_rs_0_[r][r]) -
                     (m_rs_0_[r][r] + 1.) * std::log(0.5 * p_0_ * n_r_0_[r] * n_r_0_[r] + 1);
        for (auto s = 0; s < r; ++s) {
            log_idl_0 += blockmodel.get_log_factorial(m_rs_0_[r][s]) -
                         (m_rs_0_[r][s] + 1.) * std::log(p_0_ * n_r_0_[r] * n_r_0_[s] + 1);
        }

    }
    double log_idl_1 = blockmodel.get_int_data_likelihood_from_mb(states[0].memberships);

    accu_r_ = 1.;

    double a = std::exp(+log_idl_1 - log_idl_0);
    return a;


}

double mh_riolo_uni2::transition_ratio(const blockmodel_t &blockmodel,
                                    const float_mat_t &p,
                                    const std::vector<mcmc_state_t> states) noexcept {

    uint_mat_t m_rs_0_ = blockmodel.get_m();
    uint_vec_t m_r_0_ = blockmodel.get_m_r();
    int_vec_t n_r_0_ = blockmodel.get_size_vector();
    unsigned int num_edges = blockmodel.get_num_edges();
    int_vec_t k_r_0_ = blockmodel.get_k_r_from_mb(blockmodel.get_memberships());
    double p_0_ = 2. * num_edges / (double) blockmodel.get_N() / (double) blockmodel.get_N();

    double log_idl_0 = 0.;

    for (auto r = 0; r < n_r_0_.size(); ++r) {
        log_idl_0 += k_r_0_[r] * std::log(n_r_0_[r]) + blockmodel.get_log_factorial(n_r_0_[r] - 1) -
                     blockmodel.get_log_factorial(n_r_0_[r] + k_r_0_[r] - 1);
        log_idl_0 += blockmodel.get_log_factorial(m_rs_0_[r][r]) -
                     (m_rs_0_[r][r] + 1.) * std::log(0.5 * p_0_ * n_r_0_[r] * n_r_0_[r] + 1);
        for (auto s = 0; s < r; ++s) {
            log_idl_0 += blockmodel.get_log_factorial(m_rs_0_[r][s]) -
                         (m_rs_0_[r][s] + 1.) * std::log(p_0_ * n_r_0_[r] * n_r_0_[s] + 1);
        }

    }
    double log_idl_1 = blockmodel.get_int_data_likelihood_from_mb(states[0].memberships);

    accu_r_ = 1.;

    double a = std::exp(+log_idl_1 - log_idl_0);
    return a;


}

std::vector<mcmc_state_t> mh_riolo::sample_proposal_distribution(blockmodel_t &blockmodel, std::mt19937 &engine) noexcept {
    return blockmodel.mcmc_state_change_riolo(engine);
}

double mh_riolo::transition_ratio(const blockmodel_t &blockmodel,
                                  const float_mat_t &p,
                                  const std::vector<mcmc_state_t> states) noexcept {


    uint_mat_t m_rs_0_ = blockmodel.get_m();
    uint_vec_t m_r_0_ = blockmodel.get_m_r();
    int_vec_t n_r_0_ = blockmodel.get_size_vector();
    unsigned int num_edges = blockmodel.get_num_edges();
    int_vec_t k_r_0_ = blockmodel.get_k_r_from_mb(blockmodel.get_memberships());

    double p_0_ = 2. * num_edges / (double) blockmodel.get_N() / (double) blockmodel.get_N();

    double log_idl_0 = 0.;

    for (auto r = 0; r < n_r_0_.size(); ++r) {
        log_idl_0 += k_r_0_[r] * std::log(n_r_0_[r]) + blockmodel.get_log_factorial(n_r_0_[r] - 1) -
                     blockmodel.get_log_factorial(n_r_0_[r] + k_r_0_[r] - 1);
//        std::clog << log_idl_0 << "\n";
//        std::clog << k_r_0_[r] << " , " << n_r_0_[r] << " , " << blockmodel.get_log_factorial(n_r_0_[r] - 1) << "\n";
        log_idl_0 += blockmodel.get_log_factorial(m_rs_0_[r][r]) -
                     (m_rs_0_[r][r] + 1.) * std::log(0.5 * p_0_ * n_r_0_[r] * n_r_0_[r] + 1);
        for (auto s = 0; s < r; ++s) {
            log_idl_0 += blockmodel.get_log_factorial(m_rs_0_[r][s]) -
                         (m_rs_0_[r][s] + 1.) * std::log(p_0_ * n_r_0_[r] * n_r_0_[s] + 1);
        }

    }

    double log_idl_1 = blockmodel.get_int_data_likelihood_from_mb(states[0].memberships);
//    double entropy_from_degree_correction = blockmodel.get_entropy_from_degree_correction();
//    entropy_from_degree_correction = 0.;
//    double entropy0 = -(double) num_edges - entropy_from_degree_correction;
//
//    double entropy1 = entropy0;
//    for (unsigned int r_ = 0; r_ < n.size(); ++r_) {
//        for (unsigned int s_ = 0; s_ < n.size(); ++s_) {
//            if (m0_r[r_] * m0_r[s_] * m0[r_][s_] != 0) {
//                entropy0 -= 1. / 2. * (double) m0[r_][s_] *
//                            std::log((double) m0[r_][s_] / (double) m0_r[r_] / (double) m0_r[s_]);
//            }
//        }
//    }
//
//
//    uint_mat_t m1 = blockmodel.get_m_from_membership(states[0].memberships);
////    std::clog << "-- still okay here --\n";
//    uint_vec_t m1_r = blockmodel.get_m_r_from_m(m1);
//    for (unsigned int r_ = 0; r_ < m1_r.size(); ++r_) {
//        for (unsigned int s_ = 0; s_ < m1_r.size(); ++s_) {
//            if (m1_r[r_] * m1_r[s_] * m1[r_][s_] != 0) {
//                entropy1 -= 1. / 2. * (double) m1[r_][s_] *
//                            std::log((double) m1[r_][s_] / (double) m1_r[r_] / (double) m1_r[s_]);
//            }
//        }
//    }

//    std::clog << "-- still okay here 2 --\n";

    accu_r_ = 1.;
//    if (entropy0 >= entropy_max_) {
//        entropy_max_ = entropy0;
//    }
//
//    if (entropy0 <= entropy_min_) {
//        entropy_min_ = entropy0;
//    }
//    std::clog << "log_idl_1 = " << log_idl_1 << "; log_idl_0 = " << log_idl_0 << "\n";
    double a = std::exp(+log_idl_1 - log_idl_0);
//    std::clog << "hasting's ratio is: " << a << "\n";
    return a;


}

