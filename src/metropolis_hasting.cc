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
inline bool metropolis_hasting::step(blockmodel_t& blockmodel, size_t vtx, double temperature,
        std::mt19937& engine) noexcept {

    moves_ = sample_proposal_distribution(std::move(blockmodel), vtx, engine);
    double a = 0.;
    double exp_minus_diff_entropy = transition_ratio(blockmodel, moves_);
    if (temperature == 0.) {
        if (exp_minus_diff_entropy >= 1.) {
            a = 1.;
        }
    } else {
        a = std::pow(exp_minus_diff_entropy , 1. / temperature) * accu_r_;
    }
    if (random_real(engine) < a) {
        return blockmodel.apply_mcmc_moves(moves_);
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

    entropy_min_ = 1000000;
    entropy_max_ = 0;

    auto all_sweeps = size_t(duration / num_nodes);
    double _entropy_max = entropy_max_;
    double _entropy_min = entropy_min_;
    std::vector< std::vector<size_t> >& adj_list = blockmodel.get_adj_list();

    for (size_t sweep = 0; sweep < all_sweeps; ++sweep) {
        uint_vec_t& vlist = blockmodel.get_vlist();
        std::shuffle(vlist.begin(), vlist.end(), engine);
        size_t current_step = num_nodes * sweep;
        for (size_t vi = 0; vi < vlist.size(); ++vi) {
            if (adj_list[vlist[vi]].empty()) {
                continue;
            }

            if (step(blockmodel, vlist[vi], cooling_schedule(current_step + vi, cooling_schedule_kwargs), engine)) {
                ++accepted_steps;
                // TODO: check the effect of `epsilon` from the code block here
                if (_entropy_max == entropy_max_ && _entropy_min == entropy_min_) {
                    u += 1;
                } else {
                    u = 0;
                }
            }
        }
        if (u >= steps_await) {
            std::clog << "algorithm stops after: " << sweep << " sweeps. \n";
            sweep = all_sweeps;  // TODO: check -- if acceptance rate is even meaningful in annealing mode?
            return double(accepted_steps) / double(sweep * num_nodes);
        }
    }

    return double(accepted_steps) / double(duration);
}

inline const double metropolis_hasting::transition_ratio(const blockmodel_t& blockmodel,
                                     const std::vector<mcmc_state_t> &moves) noexcept {
    v_ = moves[0].vertex;
    r_ = moves[0].source;
    s_ = moves[0].target;

    if (r_ == s_) {
        accu_r_ = 1.;
        return 1.;
    }

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
            entropy1 -= (*citer_m0_r - _k) * (safelog_fast(*citer_m0_r - _k) - safelog_fast(INT_padded_m1r) - safelog_fast(*citer_padded_m0));
            entropy1 -= (*citer_m0_s + _k) * (safelog_fast(*citer_m0_s + _k) - safelog_fast(INT_padded_m1s) - safelog_fast(*citer_padded_m0));
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
                                                                 size_t vtx,
                                                                 std::mt19937& engine) const noexcept {

    return blockmodel.single_vertex_change_tiago(engine, vtx);
}

