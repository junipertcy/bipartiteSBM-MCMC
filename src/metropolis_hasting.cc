#include "metropolis_hasting.hh"
#include "support/cache.hh"
#include "support/int_part.hh"

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
inline bool metropolis_hasting::step(blockmodel_t& blockmodel, size_t vtx, double temperature,
        std::mt19937& engine) noexcept {
    moves_ = sample_proposal_distribution(blockmodel, vtx, engine);
    double a{0.};
    double dS = transition_ratio(blockmodel, moves_);
    if (temperature == 0.) {
        if (dS < 0) {
            return blockmodel.apply_mcmc_moves(moves_, dS);
        } else {
            return false;
        }
    } else {
        a = - 1. / temperature * dS + std::log(accu_r_);
        if (a > 0.) {
            return blockmodel.apply_mcmc_moves(moves_, dS);
        } else if (random_real(engine) < std::exp(a)) {
            return blockmodel.apply_mcmc_moves(moves_, dS);
        }
    }
    return false;
}

double metropolis_hasting::anneal(
        blockmodel_t &blockmodel,
        double (*cooling_schedule)(size_t, float_vec_t),
        const float_vec_t& cooling_schedule_kwargs,
        size_t duration,
        size_t steps_await,
        std::mt19937 &engine) noexcept {
    size_t num_nodes = blockmodel.get_memberships()->size();
    size_t accepted_steps = 0;
    size_t u = 0;

    entropy_min_ = std::numeric_limits<double>::infinity();
    auto all_sweeps = size_t(duration / num_nodes);
    double temperature{1};

    uint_vec_t& vlist = blockmodel.get_vlist();
    for (size_t sweep = 0; sweep < all_sweeps; ++sweep) {
        std::shuffle(vlist.begin(), vlist.end(), engine);

        size_t current_step = num_nodes * sweep;
        for (size_t vi = 0; vi < vlist.size(); ++vi) {
            temperature = cooling_schedule(current_step + vi, cooling_schedule_kwargs);
            if (step(blockmodel, vlist[vi], temperature, engine)) {
                ++accepted_steps;
                if (blockmodel.get_entropy() < entropy_min_) {  // TODO: this can be improved
                    entropy_min_ = blockmodel.get_entropy();
                    u = 0;
                }
            }
            if (temperature < 1.) {
                ++u;
            }
        }
        if (u >= steps_await) {
            std::clog << "algorithm stops after: " << sweep << " sweeps. \n";
            blockmodel.summary();
            return double(accepted_steps) / double((sweep + 1) * num_nodes);
        }
    }
    return double(accepted_steps) / double(duration);  // TODO: check these numbers
}

inline const double metropolis_hasting::transition_ratio(const blockmodel_t& blockmodel,
                                     const std::vector<mcmc_move_t> &moves) noexcept {
    v_ = moves[0].vertex;
    r_ = moves[0].source;
    s_ = moves[0].target;

    if (r_ == s_) {
        accu_r_ = 1.;
        return 0.;
    }
    double accu0 = 0.;
    double accu1 = 0.;
    double entropy0 = 0.;
    double entropy1 = 0.;

    size_t KA = blockmodel.get_KA();
    size_t KB = blockmodel.get_KB();
    double K = KA + KB;
    if ((r_ < KA && s_ >= KA) || (r_ >= KA && s_ < KA)) {
        return std::numeric_limits<double>::infinity();
    }

    double epsilon = blockmodel.get_epsilon();
    ki = blockmodel.get_k(v_);
    int deg = blockmodel.get_degree(v_);
    m0 = blockmodel.get_m();
    padded_m0 = blockmodel.get_m_r();
    n_r = blockmodel.get_n_r();
    int INT_n_r_r = n_r->at(r_);

    int INT_n_r_s = n_r->at(s_);
    eta_rk = blockmodel.get_eta_rk_();
    int INT_eta_rk_r_deg = eta_rk->at(r_)[deg];
    int INT_eta_rk_s_deg = eta_rk->at(s_)[deg];

    citer_m0_s = m0->at(s_).begin();

    citer_padded_m0 = (*padded_m0).begin();
    citer_m0_r = m0->at(r_).begin();

    int INT_padded_m0r = padded_m0->at(r_);
    int INT_padded_m1r = INT_padded_m0r - deg;

    int INT_padded_m0s = padded_m0->at(s_);
    int INT_padded_m1s = INT_padded_m0s + deg;

    auto criterion = (r_ < KA) ? [](size_t a, size_t k) { return a >= k; } : [](size_t a, size_t k) { return a < k; };
    for (auto const& _k: *ki ){
        size_t index = &_k - &ki->at(0);
        if (criterion(index, KA) && _k != 0) {
            accu0 += _k * (*citer_m0_s + epsilon) / (*citer_padded_m0 + epsilon * K) / deg;
            accu1 += _k * (*citer_m0_r - _k + epsilon) / (*citer_padded_m0 + epsilon * K) / deg;
            entropy0 -= lgamma_fast(*citer_m0_r + 1);
            entropy0 -= lgamma_fast(*citer_m0_s + 1);
            entropy1 -= lgamma_fast(*citer_m0_r - _k + 1);
            entropy1 -= lgamma_fast(*citer_m0_s + _k + 1);
        }
        ++citer_m0_s;
        ++citer_padded_m0;
        ++citer_m0_r;
    }
    entropy0 -= -lgamma_fast(INT_padded_m0r + 1);
    entropy0 -= -lgamma_fast(INT_padded_m0s + 1);

    entropy1 -= -lgamma_fast(INT_padded_m1r + 1);
    entropy1 -= -lgamma_fast(INT_padded_m1s + 1);

    // entropy from degree distribution
    // Note: we do not need entropy from partition (cancels with the denominator of the entropy from degree dist)
    // as well as entropy from edge counts (no change).
    entropy0 += -lgamma_fast(INT_eta_rk_r_deg + 1);
    entropy0 += -lgamma_fast(INT_eta_rk_s_deg + 1);

    entropy1 += -lgamma_fast(INT_eta_rk_r_deg - 1 + 1);
    entropy1 += -lgamma_fast(INT_eta_rk_s_deg + 1 + 1);

    entropy0 += log_q(INT_padded_m0r, INT_n_r_r);
    entropy0 += log_q(INT_padded_m0s, INT_n_r_s);

    entropy1 += log_q(INT_padded_m1r, INT_n_r_r - 1);
    entropy1 += log_q(INT_padded_m1s, INT_n_r_s + 1);

    if (deg == 0) {
        accu_r_ = 1;
    } else {
        accu_r_ = accu1 / accu0;
    }

    return entropy1 - entropy0;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// virtual functions implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/* Implementation for the single vertex change (SBM) */
inline std::vector<mcmc_move_t> mh_tiago::sample_proposal_distribution(blockmodel_t& blockmodel,
                                                                 size_t vtx,
                                                                 std::mt19937& engine) const noexcept {
    return blockmodel.single_vertex_change(engine, vtx);
}
