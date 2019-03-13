#include <iostream>
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

uint_vec_t &blockmodel_t::get_vlist() noexcept { return vlist_; }

std::vector<std::vector<size_t> > &blockmodel_t::get_adj_list() noexcept { return adj_list_; };


bool blockmodel_t::apply_mcmc_moves(std::vector<mcmc_state_t> &moves, double dS) noexcept {
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

std::vector<mcmc_state_t> blockmodel_t::single_vertex_change(std::mt19937 &engine, size_t vtx) noexcept {
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