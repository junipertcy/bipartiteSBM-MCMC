#include <iostream>
#include <queue>

#include "blockmodel.hh"
#include "graph_utilities.hh"  // for the is_disjoint function
#include "output_functions.hh"

#include "support/cache.hh"
#include "support/int_part.hh"
#include "support/util.hh"

using namespace std;

/** Default constructor */
blockmodel_t::blockmodel_t(const uint_vec_t &memberships, uint_vec_t types, size_t g, size_t KA,
                           size_t KB, double epsilon, const adj_list_t *adj_list_ptr) :
        adj_list_ptr_(adj_list_ptr),
        types_(std::move(types)) {
    KA_ = KA;
    KB_ = KB;
    K_ = KA + KB;
    epsilon_ = epsilon;
    memberships_ = memberships;
    n_r_.resize(g, 0);
    deg_.resize(memberships.size(), 0);
    vlist_.resize(memberships.size(), 0);
    blist_.resize(memberships.size(), 0);
    num_edges_ = 0;
    entropy_from_degree_correction_ = 0.;

    for (size_t j = 0; j < memberships.size(); ++j) {
        if (types_[j] == 0) {
            na_ += 1;
        } else if (types_[j] == 1) {
            nb_ += 1;
        }

        ++n_r_[memberships[j]];

        for (auto nb = adj_list_ptr_->at(j).begin(); nb != adj_list_ptr_->at(j).end(); ++nb) {
            ++deg_[j];
            ++num_edges_;
        }
        vlist_[j] = j;
    }
    num_edges_ /= 2;
    max_degree_ = *std::max_element(deg_.begin(), deg_.end());

    // initiate caches
    init_cache(num_edges_);
    init_q_cache(1000);

    double deg_factorial = 0;
    for (size_t node = 0; node < memberships.size(); ++node) {
        deg_factorial = 0;

        for (size_t deg = 1; deg <= deg_[node]; ++deg) {
            deg_factorial += safelog_fast(deg);
        }
        entropy_from_degree_correction_ += deg_factorial;
    }

    // Note that Tiago's MCMC proposal jumps has to randomly access elements in an adjacency list
    // Here, we define an vectorized data structure to make such data access O(1) [else it'll be O(n)].
    adj_list_.resize(adj_list_ptr_->size());
    for (size_t i = 0; i < adj_list_ptr_->size(); ++i) {
        adj_list_[i].resize(adj_list_ptr_->at(i).size(), 0);
    }
    for (size_t node = 0; node < memberships_.size(); ++node) {
        size_t idx = 0;
        for (auto nb = adj_list_ptr_->at(node).begin(); nb != adj_list_ptr_->at(node).end(); ++nb) {
            adj_list_[node][idx] = size_t(*nb);
            ++idx;
        }
    }
}

const int_vec_t *blockmodel_t::get_k(size_t vertex) const noexcept { return &k_[vertex]; }

const int blockmodel_t::get_degree(size_t vertex) const noexcept { return deg_.at(vertex); }

const int blockmodel_t::get_num_edges() const noexcept { return num_edges_; }

const int blockmodel_t::get_na() const noexcept { return na_; }

const int blockmodel_t::get_nb() const noexcept { return nb_; }

const uint_vec_t *blockmodel_t::get_memberships() const noexcept { return &memberships_; }

double blockmodel_t::get_epsilon() const noexcept { return epsilon_; }

double blockmodel_t::get_entropy() const noexcept { return entropy_; }

const int_mat_t *blockmodel_t::get_m() const noexcept { return &m_; }

const int_vec_t *blockmodel_t::get_m_r() const noexcept { return &m_r_; }

const uint_mat_t *blockmodel_t::get_eta_rk_() const noexcept { return &eta_rk_; }

const int_vec_t *blockmodel_t::get_n_r() const noexcept { return &n_r_; }

inline size_t blockmodel_t::get_g() const noexcept { return K_; }

size_t blockmodel_t::get_KA() const noexcept { return KA_; }

size_t blockmodel_t::get_KB() const noexcept { return KB_; }

uint_vec_t& blockmodel_t::get_vlist() noexcept { return vlist_; }

void blockmodel_t::agg_merge(std::mt19937 &engine, int diff_a, int diff_b, int nm) noexcept {
    std::vector<std::set<size_t>> accepted_set_vec;
//    accepted_queue.resize(diff_a + diff_b);

    typedef std::pair<double, int> pi;
    std::priority_queue<pi, vector<pi>, greater<> > q;

    compute_b_adj_list();

    blist_.resize(K_, 0);
    std::iota(blist_.begin(), blist_.end(), 0);

    std::vector<block_move_t> moves;
    moves.resize(nm * blist_.size());

    for (auto const &v: blist_) {
        size_t index = &v - &blist_.at(0);
        for (size_t i_ = 0; i_ < nm; ++i_) {
            moves[index * nm + i_] = single_block_change(engine, v);
            double dS = compute_dS(moves[index * nm + i_]);
            int idx = index * nm + i_;
            q.push(std::make_pair(dS, idx));
        }
    }
    std::set<std::string> set_str;
    std::set<size_t> set_e;

    size_t a{0};
    size_t b{0};
    block_move_t mv;
    std::string identifier_a;
    std::string identifier_b;
    std::set<size_t> _set;

    while (diff_a + diff_b != 0 && !q.empty()) {
        mv = moves[q.top().second];
        _set.clear();
        _set.insert(mv.target);
        _set.insert(mv.source);
        if (mv.source < KA_ && diff_a != 0) {
            identifier_a = std::to_string(mv.source) + "->" + std::to_string(mv.target);
            if (set_str.count(identifier_a) == 0 && !(set_e.count(mv.source) > 0 && set_e.count(mv.target) > 0)) {
                diff_a -= 1;
                a += 1;
                if (set_e.count(mv.source) == 0 && set_e.count(mv.target) == 0) {
                    accepted_set_vec.push_back(_set);
                } else {
                    for (auto& _s: accepted_set_vec) {
                        if (_s.count(mv.target) > 0 || _s.count(mv.source) > 0) {
                            _s.insert({mv.source, mv.target});
                            break;
                        }
                    }
                }
                set_str.insert(identifier_a);
                set_e.insert(mv.source);
                set_e.insert(mv.target);
            }
        } else if (mv.source >= KA_ && diff_b != 0) {
            identifier_b = std::to_string(mv.source) + "->" + std::to_string(mv.target);
            if (set_str.count(identifier_b) == 0 && !(set_e.count(mv.source) > 0 && set_e.count(mv.target) > 0)) {
                diff_b -= 1;
                b += 1;
                if (set_e.count(mv.source) == 0 && set_e.count(mv.target) == 0) {
                    accepted_set_vec.push_back(_set);
                } else {
                    for (auto& _s: accepted_set_vec) {
                        if (_s.count(mv.target) > 0 || _s.count(mv.source) > 0) {
                            _s.insert({mv.source, mv.target});
                            break;
                        }
                    }
                }

                set_str.insert(identifier_b);
                set_e.insert(mv.source);
                set_e.insert(mv.target);
            }
        }
        q.pop();
    }
    if (q.empty()) {
        std::cerr << "queue running out \n";
    }
    apply_block_moves(set_e, accepted_set_vec);
}

inline void blockmodel_t::compute_b_adj_list() noexcept {
    b_adj_list_.resize(K_);
    for (size_t i = 0; i < K_; ++i) {
        b_adj_list_[i].resize(std::count_if(m_[i].begin(), m_[i].end(), [](int i) { return i > 0; }), 0);
    }
    for (size_t node = 0; node < K_; ++node) {
        size_t idx = 0;
        for (size_t mb = 0; mb < K_; ++mb) {
            if (m_[node][mb] > 0) {
                b_adj_list_[node][idx] = mb;
                ++idx;
            }
        }
    }
}

double blockmodel_t::compute_dS(mcmc_move_t move) noexcept {
    size_t v_ = move.vertex;
    size_t r_ = move.source;
    size_t s_ = move.target;

    if (r_ == s_) {
        return std::numeric_limits<double>::infinity();
    }
    double entropy0 = 0.;
    double entropy1 = 0.;

    int_vec_t ki = k_[v_];
    int deg = deg_.at(v_);

    auto citer_padded_m0 = m_r_.begin();
    auto citer_m0_r = m_.at(r_).begin();
    auto citer_m0_s = m_.at(s_).begin();

    int INT_padded_m0r = m_r_.at(r_);
    int INT_padded_m1r = INT_padded_m0r - deg;

    int INT_padded_m0s = m_r_.at(s_);
    int INT_padded_m1s = INT_padded_m0s + deg;

    auto criterion = (r_ < KA_) ? [](size_t a, size_t k) { return a >= k; } : [](size_t a, size_t k) { return a < k; };
    for (auto const &_k: ki) {
        size_t index = &_k - &ki.at(0);
        if (criterion(index, KA_) && _k != 0) {
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

    return entropy1 - entropy0;
}

double blockmodel_t::compute_dS(block_move_t move) noexcept {
    size_t r_ = move.source;
    size_t s_ = move.target;

    if (r_ == s_ || (r_ < KA_ && s_ >= KA_) || (r_ >= KA_ && s_ < KA_)) {
        return std::numeric_limits<double>::infinity();
    }

    double entropy0 = 0.;
    double entropy1 = 0.;

    auto citer_padded_m0 = m_r_.begin();
    auto citer_m0_r = m_.at(r_).begin();
    auto citer_m0_s = m_.at(s_).begin();

    int INT_padded_m0r = m_r_.at(r_);


    int INT_padded_m0s = m_r_.at(s_);
    int INT_padded_m1 = INT_padded_m0r + INT_padded_m0s;

    auto criterion = (r_ < KA_) ? [](size_t a, size_t k) { return a >= k; } : [](size_t a, size_t k) { return a < k; };

    for (auto const &_m_r: m_r_) {
        size_t index = &_m_r - &m_r_.at(0);
        if (criterion(index, KA_) && _m_r != 0) {
            entropy0 -= lgamma_fast(*citer_m0_r + 1);
            entropy0 -= lgamma_fast(*citer_m0_s + 1);
            entropy1 -= lgamma_fast(*citer_m0_s + *citer_m0_r + 1);
        }
        ++citer_m0_s;
        ++citer_padded_m0;
        ++citer_m0_r;
    }
    entropy0 -= -lgamma_fast(INT_padded_m0r + 1);
    entropy0 -= -lgamma_fast(INT_padded_m0s + 1);
    entropy1 -= -lgamma_fast(INT_padded_m1 + 1);

    return entropy1 - entropy0;
}

std::vector<std::vector<size_t> > &blockmodel_t::get_adj_list() noexcept { return adj_list_; }

bool blockmodel_t::apply_mcmc_moves(std::vector<mcmc_move_t> &moves, double dS) noexcept {
    for (auto const &mv: moves) {
        __source__ = mv.source;
        __target__ = mv.target;
        __vertex__ = mv.vertex;

        --n_r_[__source__];
        if (n_r_[__source__] == 0) {  // No move that makes an empty group will be allowed
            ++n_r_[__source__];
            return false;
        }
        ++n_r_[__target__];

        --eta_rk_[__source__][deg_[__vertex__]];
        ++eta_rk_[__target__][deg_[__vertex__]];

        ki_ = get_k(__vertex__);
        size_t ki_size = ki_->size();
        for (size_t i = 0; i < ki_size; ++i) {
            int ki_at_i = ki_->at(i);
            if (ki_at_i != 0) {
                m_[__source__][i] -= ki_at_i;
                m_[__target__][i] += ki_at_i;
                m_[i][__source__] = m_[__source__][i];
                m_[i][__target__] = m_[__target__][i];
            }
        }
        m_r_[__source__] -= deg_[__vertex__];
        m_r_[__target__] += deg_[__vertex__];

        // Change block degrees and block sizes
        for (auto const &neighbour: adj_list_ptr_->at(__vertex__)) {
            --k_[neighbour][__source__];
            ++k_[neighbour][__target__];
        }

        // Set new memberships
        memberships_[__vertex__] = unsigned(int(__target__));

        entropy_ += dS;
    }
    return true;
}

bool blockmodel_t::apply_block_moves(std::set<size_t>& impacted, std::vector<std::set<size_t>>& accepted) noexcept {
    std::map<int, int> n2o_map;
    for (size_t i = 0; i < memberships_.size(); ++i) {
        n2o_map[i] = -1;
    }

    for (auto& mb: memberships_) {
        if (impacted.count(mb) > 0) {
            for (auto const& a_: accepted) {
                if (a_.count(mb) > 0) {
                    mb = *a_.begin();
                }
            }
        }
    }

    KA_ = 0;
    KB_ = 0;
    size_t n{0};
    for (auto& mb: memberships_) {
        size_t index = &mb - &memberships_.at(0);
        if (n2o_map[mb] == -1) {
            n2o_map[mb] = n;
            n++;
        }
        mb = n2o_map[mb];
        if (index < na_) {
            if (mb > KA_) {
                KA_ = mb;
            }
        } else {
            if (mb > KB_) {
                KB_ = mb;
            }
        }
    }
    KB_ -= KA_;
    KA_ += 1;
    K_ =  KA_ + KB_;

    if (n != K_) {
        std::cerr << "inconsistency 1;\n";
        std::clog << "KA_: " << KA_ << "; KB_: " << KB_ << "; n: " << n << "; na_: " << na_ << "; nb_: " << nb_ << "\n";
        exit(0);
    }

    compute_n_r();
    compute_k();
    compute_m();
    compute_m_r();
    compute_eta_rk();
    return true;
}

std::vector<mcmc_move_t> blockmodel_t::single_vertex_change(std::mt19937 &engine, size_t vtx) noexcept {
    if ((types_[vtx] == 0 && KA_ == 1) || (types_[vtx] == 1 && KB_ == 1)) {
        __target__ = memberships_[vtx];
    } else if (adj_list_[vtx].empty()) {
        __target__ = size_t(random_real(engine) * K_);
    } else {
        which_to_move_ = size_t(random_real(engine) * adj_list_[vtx].size());
        vertex_j_ = adj_list_[vtx][which_to_move_];
        proposal_t_ = memberships_[vertex_j_];
        R_t_ = epsilon_ * K_ / (m_r_[proposal_t_] + epsilon_ * K_);

        if (random_real(engine) < R_t_) {
            __target__ = size_t(random_real(engine) * K_);
        } else {
            std::discrete_distribution<size_t> d(m_[proposal_t_].begin(), m_[proposal_t_].end());
            __target__ = d(gen);
        }
    }
    __source__ = memberships_[vtx];
    moves_[0].source = __source__;
    moves_[0].target = __target__;
    moves_[0].vertex = vtx;

    return moves_;
}

block_move_t& blockmodel_t::single_block_change(std::mt19937 &engine, size_t src) noexcept {
    if ((src < KA_ && KA_ == 1) || (src >= KA_ && KB_ == 1)) {
        bmove_.source = src;
        bmove_.target = src;
        return bmove_;
    }
    if (b_adj_list_[src].empty()) {
        __target__ = size_t(random_real(engine) * K_);
    } else {
        which_to_move_ = size_t(random_real(engine) * b_adj_list_[src].size());
        proposal_t_ = b_adj_list_[src][which_to_move_];

        R_t_ = epsilon_ * K_ / (m_r_[proposal_t_] + epsilon_ * K_);

        if (random_real(engine) < R_t_) {
            __target__ = size_t(random_real(engine) * K_);
        } else {
            std::discrete_distribution<size_t> d(m_[proposal_t_].begin(), m_[proposal_t_].end());
            __target__ = d(gen);
        }
    }
    if (src > __target__) {
        bmove_.source = src;
        bmove_.target = __target__;
    } else {
        bmove_.source = __target__;
        bmove_.target = src;
    }


    return bmove_;
}


void blockmodel_t::shuffle_bisbm(std::mt19937 &engine, size_t NA, size_t NB) noexcept {
    std::shuffle(&memberships_[0], &memberships_[NA], engine);
    std::shuffle(&memberships_[NA], &memberships_[NA + NB], engine);
    compute_k();
    compute_m();
    compute_m_r();
    compute_eta_rk();
}

void blockmodel_t::init_bisbm() noexcept {
    compute_k();
    compute_m();
    compute_m_r();
    compute_eta_rk();
}

// The following 4 functions should only be executed once.
inline void blockmodel_t::compute_k() noexcept {
    k_.clear();
    k_.resize(adj_list_ptr_->size());
    for (size_t i = 0; i < adj_list_ptr_->size(); ++i) {
        k_[i].resize(this->n_r_.size(), 0);
        for (auto nb = adj_list_ptr_->at(i).begin(); nb != adj_list_ptr_->at(i).end(); ++nb) {
            ++k_[i][memberships_[*nb]];
        }
    }
}

inline void blockmodel_t::compute_m() noexcept {
    m_.clear();
    m_.resize(get_g());
    for (auto i = 0; i < get_g(); ++i) {
        m_[i].resize(get_g(), 0);
    }
    for (size_t vertex = 0; vertex < adj_list_ptr_->size(); ++vertex) {
        __vertex__ = memberships_[vertex];
        for (auto const &nb: adj_list_ptr_->at(vertex)) {
            ++m_[__vertex__][memberships_[nb]];
        }
    }
}

inline void blockmodel_t::compute_m_r() noexcept {
    m_r_.clear();
    m_r_.resize(get_g(), 0);
    size_t _m_r = 0;
    for (size_t r = 0; r < get_g(); ++r) {
        _m_r = 0;
        for (size_t s = 0; s < get_g(); ++s) {
            _m_r += m_[r][s];
        }
        m_r_[r] = unsigned(int(_m_r));
    }
}

inline void blockmodel_t::compute_eta_rk() noexcept {
    eta_rk_.clear();
    eta_rk_.resize(get_g());
    for (size_t idx = 0; idx < get_g(); ++idx) {
        eta_rk_[idx].resize(max_degree_ + 1, 0);
    }
    for (size_t j = 0; j < memberships_.size(); ++j) {
        ++eta_rk_[memberships_[j]][deg_[j]];
    }
}

inline void blockmodel_t::compute_n_r() noexcept {
    n_r_.clear();
    n_r_.resize(get_g(), 0);
    for (auto const &mb: memberships_) {
        ++n_r_[mb];
    }
}

void blockmodel_t::summary() noexcept {
    std::clog << "(Ka, Kb) = (" << KA_ << ", " << KB_ << ") \n";
    std::clog << "entropy: " << entropy() << "\n";
}

double blockmodel_t::entropy() noexcept {
    double ent{0};
    for (auto const &k: deg_) {
        ent -= lgamma_fast(k + 1);
    }
    for (auto const &r: m_) {
        size_t index = &r - &m_[0];
        for (auto const &s: r) {
            size_t index_s = &s - &r[0];
            if (index_s > index) {
                ent -= lgamma_fast(s + 1);
            }
        }
        for (auto const &eta: eta_rk_[index]) {
            ent -= lgamma_fast(eta + 1);
        }
        ent += lgamma_fast(m_r_[index] + 1);
        ent += log_q(m_r_[index], n_r_[index]);
    }

    ent += lbinom_fast(KA_ * KB_ + num_edges_ - 1, num_edges_);
    ent += lbinom(na_ - 1, KA_ - 1);
    ent += lbinom(nb_ - 1, KB_ - 1);
    ent += safelog_fast(na_ * nb_);
    ent += lgamma_fast(na_ + 1);
    ent += lgamma_fast(nb_ + 1);
    return ent;

}