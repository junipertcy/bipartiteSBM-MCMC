#include <iostream>
#include "blockmodel.hh"
#include "graph_utilities.hh"  // for the is_disjoint function
#include "output_functions.hh"

#include "support/cache.hh"

using namespace std;

const size_t compute_total_num_groups_from_mb(uint_vec_t &mb) noexcept {
    size_t cand_K_ = 0;
    for (auto const &_mb: mb) {
        if (_mb > cand_K_) cand_K_ = _mb;
    }
    size_t cand_n = cand_K_ + 1;
    return cand_n;
}

/** Default constructor */
blockmodel_t::blockmodel_t(uint_vec_t& memberships, const uint_vec_t& types, size_t g, size_t KA,
                           size_t KB, double epsilon, const adj_list_t* adj_list_ptr, bool is_bipartite) :
        random_block_(0, g - 1), random_node_(0, (*adj_list_ptr).size() - 1), adj_list_ptr_(adj_list_ptr), types_(types) {
    is_bipartite_ = is_bipartite;
    K_ = KA + KB;
    KA_ = KA;
    KB_ = KB;
    epsilon_ = epsilon;
    memberships_ = memberships;
    n_.resize(g, 0);
    deg_.resize(memberships.size(), 0);
    num_edges_ = 0;
    entropy_from_degree_correction_ = 0.;

    for (size_t j = 0; j < memberships.size(); ++j) {
        if (types_[j] == 0) {
            nsize_A_ += 1;
        } else if (types_[j] == 1) {
            nsize_B_ += 1;
        }

        ++n_[memberships[j]];

        for (auto nb = adj_list_ptr_->at(j).begin(); nb != adj_list_ptr_->at(j).end(); ++nb) {
            ++deg_[j];
            ++num_edges_;
        }
    }
    num_edges_ /= 2;
    init_cache(num_edges_);

    double deg_factorial = 0;
    for (size_t node = 0; node < memberships.size(); ++node) {
        deg_factorial = 0;

        for (size_t deg = 1; deg <= deg_[node]; ++deg) {
            deg_factorial += safelog_fast(deg);
        }
        entropy_from_degree_correction_ += deg_factorial;
    }
    compute_k();
    compute_m();
    compute_m_r();

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


std::vector<mcmc_state_t> blockmodel_t::mcmc_state_change_riolo_uni(std::mt19937 &engine) noexcept {
    std::vector<mcmc_state_t> states(1);
    bool cond = true;
    while (cond) {
        states[0].memberships.resize(memberships_.size(), 0);
        for (auto i = 0; i < memberships_.size(); ++i) {
            states[0].memberships[i] = memberships_[i];
        }
        // decide whether to update type-a nodes or type-b nodes
        auto num_nodes = (double) states[0].memberships.size();
        if (random_real(engine) < 1. / (num_nodes - 1)) {  // type-II move
            auto r = size_t(random_real(engine) * (K_ + 1));
            auto s = r;
            while (s == r) {
                s = unsigned(int(random_real(engine) * (K_ + 1)));
            }
            if (r != K_) {  // re-labeling, else no re-labeling is necessary!
                for (auto &n_: states[0].memberships) {
                    if (n_ == r) n_ = K_;
                }
            }
            size_t counter = 0;
            size_t rnd_node_in_label_s;
            bool last_node_in_s = false;
            if (s != K_) {
                rnd_node_in_label_s = (unsigned) (int) (random_real(engine) * n_[s]);
                if (n_[s] == 1) {
                    last_node_in_s = true;
                }
            } else {
                rnd_node_in_label_s = (unsigned) (int) (random_real(engine) * n_[r]);
                if (n_[r] == 1) {
                    states[0].memberships = memberships_;
                    return states;
                }
            }
            for (size_t i = 0; i < memberships_.size(); ++i) {
                if (states[0].memberships[i] == s) {
                    if (counter == rnd_node_in_label_s) {
                        states[0].memberships[i] = r;
                    }
                    counter++;
                }
            }
            // check if we have removed the last remaining node of label s;
            if (last_node_in_s) {
                for (auto &n_: states[0].memberships) {
                    if (n_ == K_) n_ = s;
                }
            }
        } else {  // type-I move
            if (K_ == 1) {
                // we do nothing
                states[0].memberships = memberships_;
                return states;
            } else {
                auto r = unsigned(int(random_real(engine) * K_));
                size_t s = r;
                while (s == r) {
                    s = unsigned(int(random_real(engine) * K_));
                }
                size_t counter = 0;
                auto rnd_node_in_label_r = (unsigned) (int) (random_real(engine) * n_[r]);
                for (size_t i = 0; i < memberships_.size(); ++i) {
                    if (memberships_[i] == r) {
                        if (counter == rnd_node_in_label_r) {
                            states[0].memberships[i] = s;
                        }
                        counter++;
                    }
                }
                // check if we have removed the last remaining node of label r;
                if (n_[r] == 1) {
                    for (size_t i = 0; i < memberships_.size(); ++i) {
                        if (states[0].memberships[i] == K_ - 1) {
                            states[0].memberships[i] = r;
                        }
                    }
                }
            }
        }
        cond = !cond;
    }
    return states;
}

std::vector<mcmc_state_t> blockmodel_t::mcmc_state_change_riolo(std::mt19937 &engine) noexcept {
    // This function returns a move, which will be accepted or rejected by Hasting's rule
    // if it is updated AND it increases K_a or K_b, then we re-label the nodes outside of this function
    std::vector<mcmc_state_t> states(1);

    states[0].memberships.resize(memberships_.size(), 0);
    for (auto i = 0; i < memberships_.size(); ++i) {
        states[0].memberships[i] = memberships_[i];
    }
    // decide whether to update type-a nodes or type-b nodes
    auto num_nodes = (double) states[0].memberships.size();
    auto num_nodes_a = (double) nsize_A_;
    auto num_nodes_b = (double) nsize_B_;

//    double prior_a = get_log_single_type_prior(states[0].memberships, 1);
//    double prior_b = get_log_single_type_prior(states[0].memberships, 2);
//    double beta_prob = 1. / (1. + std::exp(prior_a - prior_b));
//    beta_prob = 0.9;
    double beta_prob_old = num_nodes_a / num_nodes;
    if (random_real(engine) < beta_prob_old) {  // move type-a nodes
        if (random_real(engine) < 1. / (num_nodes_a - 1.)) {  // type-II move for type-a nodes
            auto r = unsigned(int(random_real(engine) * (KA_ + 1)));  // 0 or 1; if r = 1, s = 0
            size_t s = r;
            while (s == r) {
                s = unsigned(int(random_real(engine) * (KA_ + 1)));
            }

            if (r != KA_) {  // re-labeling, else no re-labeling is necessary!
                for (auto node = 0; node < states[0].memberships.size(); ++node) {
                    if (types_[node] == 0 && states[0].memberships[node] == r) {
                        states[0].memberships[node] = KA_;
                    } else if (types_[node] == 1) {
                        ++states[0].memberships[node];
                    }
                }
            } else {
                for (auto node = 0; node < states[0].memberships.size(); ++node) {  // if r = 1, s = 0;
                    if (types_[node] == 1) {
                        ++states[0].memberships[node];
                    }
                }
            }
            // The code here deals with special cases: i.e. when nodes to be moved in group-s is the last node
            // in the group
            size_t counter = 0;
            size_t rnd_node_in_label_s;
            bool last_node_in_s = false;

            if (s != KA_) {
                rnd_node_in_label_s = (unsigned) (int) (random_real(engine) * n_[s]);
                if (n_[s] == 1) {
                    last_node_in_s = true;
                }
            } else {
                rnd_node_in_label_s = (unsigned) (int) (random_real(engine) * n_[r]);
                if (n_[r] == 1) {
                    states[0].memberships = memberships_;
                    return states;
                }
            }

            for (size_t i = 0; i < memberships_.size(); ++i) {
                if (types_[i] == 0 && states[0].memberships[i] == s) {
                    if (counter == rnd_node_in_label_s) {
                        states[0].memberships[i] = r;
                    }
                    counter++;
                }
            }

            // check if we have removed the last remaining node of label s;
            if (last_node_in_s) {
                for (auto node = 0; node < states[0].memberships.size(); ++node) {
                    if (types_[node] == 0 && states[0].memberships[node] == KA_) {
                        states[0].memberships[node] = s;
                    } else if (types_[node] == 1) {
                        --states[0].memberships[node];
                    }
                }
            }
        } else {  // type-I move for type-a nodes
            if (KA_ == 1) {
                // we do nothing
                states[0].memberships = memberships_;
            } else {
                auto r = unsigned(int(random_real(engine) * KA_));
                size_t s = r;
                while (s == r) {
                    s = unsigned(int(random_real(engine) * KA_));
                }

                size_t counter = 0;
                auto rnd_node_in_label_r = (unsigned) (int) (random_real(engine) * n_[r]);

                for (size_t i = 0; i < memberships_.size(); ++i) {
                    if (types_[i] == 0 && memberships_[i] == r) {
                        if (counter == rnd_node_in_label_r) {
                            states[0].memberships[i] = s;
                        }
                        counter++;
                    }
                }

                // check if we have removed the last remaining node of label r;
                if (n_[r] == 1) {
                    for (size_t i = 0; i < memberships_.size(); ++i) {
                        if (types_[i] == 0 && states[0].memberships[i] == KA_ - 1) {
                            states[0].memberships[i] = r;
                        } else if (types_[i] == 1) {
                            --states[0].memberships[i];
                        }
                    }
                }
            }
        }
    } else {
        if (random_real(engine) < 1. / (num_nodes_b - 1)) {  // type-II move for type-b nodes
            auto r = unsigned(int(random_real(engine) * (KB_ + 1)));
            size_t s = r;
            while (s == r) {
                s = unsigned(int(random_real(engine) * (KB_ + 1)));
            }

            if (r != KB_) {  // re-labeling, else no re-labeling is necessary!
                for (auto node = 0; node < states[0].memberships.size(); ++node) {
                    if (types_[node] == 1 && states[0].memberships[node] == r + KA_) {
                        states[0].memberships[node] = KB_ + KA_;
                    } else if (types_[node] == 0) {
                        // do nothing;
                    }
                }
            }
            // Now, move a node in s to empty group r;
            size_t counter = 0;
            size_t rnd_node_in_label_s;
            bool last_node_in_s = false;

            if (s != KB_) {
                rnd_node_in_label_s = (unsigned) (int) (random_real(engine) * n_[s + KA_]);
                if (n_[s + KA_] == 1) {
                    last_node_in_s = true;
                }
            } else {
                rnd_node_in_label_s = (unsigned) (int) (random_real(engine) * n_[r + KA_]);
                if (n_[r + KA_] == 1) { // if s == KB_ + 1, and n[s] = 1, we move the node back to r: zero-sum game.
                    states[0].memberships = memberships_;
                    return states;
                }
            }

            for (size_t i = 0; i < states[0].memberships.size(); ++i) {
                if (types_[i] == 1 && states[0].memberships[i] == s + KA_) {
                    if (counter == rnd_node_in_label_s) {
                        states[0].memberships[i] = r + KA_;
                    }
                    counter++;
                }
            }

            // check if we have removed the last remaining node of label s;
            if (last_node_in_s) {
                for (auto node = 0; node < states[0].memberships.size(); ++node) {
                    if (types_[node] == 1 && states[0].memberships[node] == KB_ + KA_) {
                        states[0].memberships[node] = s + KA_;
                    } else if (types_[node] == 0) {
                        // do nothing;
                    }
                }
            }
        } else {  // type-I move for type-b nodes
            if (KB_ == 1) {
                states[0].memberships = memberships_;  // we do nothing
            } else {
                auto r = unsigned(int(random_real(engine) * KB_));
                size_t s = r;
                while (s == r) {
                    s = unsigned(int(random_real(engine) * KB_));
                }

                size_t counter = 0;
                auto rnd_node_in_label_r = (unsigned) (int) (random_real(engine) * n_[r + KA_]);
                for (size_t i = 0; i < states[0].memberships.size(); ++i) {
                    if (types_[i] == 1 && memberships_[i] == r + KA_) {
                        if (counter == rnd_node_in_label_r) {
                            states[0].memberships[i] = s + KA_;
                        }
                        counter++;
                    }
                }

                // check if we have removed the last remaining node of label r;
                if (n_[r + KA_] == 1) {
                    for (size_t i = 0; i < memberships_.size(); ++i) {
                        if (types_[i] == 1 && states[0].memberships[i] == KA_ + KB_ - 1) {
                            states[0].memberships[i] = r + KA_;
                        } else if (types_[i] == 0) {
                            // do nothing;
                        }
                    }
                }
            }
        }
    }
    return states;
}


const int_vec_t* blockmodel_t::get_k(size_t vertex) const noexcept { return &k_[vertex]; }

const int_vec_t* blockmodel_t::get_size_vector() const noexcept { return &n_; }

const int_vec_t* blockmodel_t::get_degree() const noexcept { return &deg_; }

const int blockmodel_t::get_degree(size_t vertex) const noexcept {return deg_.at(vertex);}

const uint_vec_t* blockmodel_t::get_memberships() const noexcept { return &memberships_; }

double blockmodel_t::get_epsilon() const noexcept { return epsilon_; }

const uint_mat_t* blockmodel_t::get_m() const noexcept { return &m_; }

const uint_vec_t* blockmodel_t::get_m_r() const noexcept { return &m_r_; }


// TODO: move it to the template?
inline const void blockmodel_t::compute_m_from_mb(uint_vec_t &mb, bool proposal) noexcept {
    // Note that in Riolo's setting, we have to compare two jump choices of different sizes;
    // For the newly proposed system with matrix m, we have to calculate its size every time here;
    size_t max_n = compute_total_num_groups_from_mb(mb);

    if (proposal) {
        cand_m_.assign(max_n, uint_vec_t(max_n, 0));
        for (size_t vertex = 0; vertex < adj_list_ptr_->size(); ++vertex) {
            __vertex__ = mb[vertex];
            for (auto const& nb: adj_list_ptr_->at(vertex)) {
                ++cand_m_[__vertex__][mb[nb]];
            }
        }
    } else {
        m_.assign(max_n, uint_vec_t(max_n, 0));
        for (size_t vertex = 0; vertex < adj_list_ptr_->size(); ++vertex) {
            __vertex__ = mb[vertex];
            for (auto const& nb: adj_list_ptr_->at(vertex)){
                ++m_[__vertex__][mb[nb]];
            }
        }
    }
}

const void blockmodel_t::compute_n_r_from_mb(uint_vec_t &mb, bool proposal) noexcept {
    // if proposal == true, modify cand_*_ contents; otherwise, modify *_ contents.
    size_t max_n = compute_total_num_groups_from_mb(mb);
    if (proposal) {
        cand_n_r_.assign(max_n, 0);
        for (auto const &_mb: mb) {
            ++cand_n_r_[_mb];
        }
    } else {
        n_r_.assign(max_n, 0);
        for (auto const &_mb: mb) {
            ++n_r_[_mb];
        }
    }
}

bool blockmodel_t::get_is_bipartite() const noexcept { return is_bipartite_; }

size_t blockmodel_t::get_N() const noexcept { return memberships_.size(); }

size_t blockmodel_t::get_g() const noexcept { return n_.size(); }

size_t blockmodel_t::get_K() const noexcept { return K_; }

size_t blockmodel_t::get_KA() const noexcept { return KA_; }

size_t blockmodel_t::get_nsize_A() const noexcept { return nsize_A_; }

size_t blockmodel_t::get_KB() const noexcept { return KB_; }

size_t blockmodel_t::get_nsize_B() const noexcept { return nsize_B_; }

void blockmodel_t::apply_mcmc_moves(std::vector<mcmc_state_t>&& moves) noexcept {
    for (auto const &mv: moves) {
        __source__ = mv.source;
        __target__ = mv.target;
        __vertex__ = mv.vertex;

        ki_ = *get_k(__vertex__);
        for (size_t i = 0; i < ki_.size(); ++i) {
            if (ki_[i] != 0) {
                m_[__source__][i] -= ki_[i];
                m_[__target__][i] += ki_[i];
                m_[i][__source__] = m_[__source__][i];
                m_[i][__target__] = m_[__target__][i];
            }
        }
        m_r_[__source__] -= deg_[__vertex__];
        m_r_[__target__] += deg_[__vertex__];
        // Change block degrees and block sizes
        for (auto const& neighbour: adj_list_ptr_->at(__vertex__)) {
            --k_[neighbour][__source__];
            ++k_[neighbour][__target__];
        }

        --n_[__source__];
        ++n_[__target__];
        // Set new memberships
        memberships_[__vertex__] = unsigned(int(__target__));
    }
}

void blockmodel_t::apply_mcmc_states_u(std::vector<mcmc_state_t> states) noexcept {
    // Key things to do here:
    // 1. update memberships_, n_;
    // 2. update KA_, KB_;
    for (size_t j = 0; j < states.size(); ++j) {
        for (size_t i = 0; i < memberships_.size(); ++i) {
            memberships_[i] = states[0].memberships[i];
        }
        size_t max_K_ = 0;
        for (auto const &mb_: memberships_) {
            if (mb_ > max_K_) max_K_ = mb_;  // key point; keep membership compact;
        }
        K_ = max_K_ + 1;
        n_.resize(K_, 0);

        for (auto &j_: n_) j_ = 0;
        for (auto const &j_: memberships_) ++n_[j_];
    }
}

void blockmodel_t::apply_mcmc_states(std::vector<mcmc_state_t> states) noexcept {
    // Key things to do here:
    // 1. update memberships_, n_;
    // 2. update KA_, KB_;
    for (size_t j = 0; j < states.size(); ++j) {
        for (size_t i = 0; i < memberships_.size(); ++i) {
            memberships_[i] = states[0].memberships[i];
        }
        size_t max_KA_ = 0;
        size_t max_KB_ = 0;

        for (auto i = 0; i < memberships_.size(); ++i) {
            if (types_[i] == 0) {
                if (memberships_[i] > max_KA_) max_KA_ = memberships_[i]; // key point; keep membership compact;
            } else {
                if (memberships_[i] > max_KB_) max_KB_ = memberships_[i];
            }
        }
        KA_ = max_KA_ + 1;
        KB_ = max_KB_ + 1 - KA_;
        K_ = KA_ + KB_;
        n_.resize(KA_ + KB_, 0);
        for (auto &j_: n_) j_ = 0;
        for (auto const &k: memberships_) ++n_[k];
    }
}

void blockmodel_t::shuffle_bisbm(std::mt19937 &engine, size_t NA, size_t NB) noexcept {
    std::shuffle(&memberships_[0], &memberships_[NA], engine);
    std::shuffle(&memberships_[NA], &memberships_[NA + NB], engine);
    compute_k();
    compute_m();
    compute_m_r();
}

inline void blockmodel_t::compute_k() noexcept {
    // In principle, this function only executes once.
    k_.clear();
    k_.resize(adj_list_ptr_->size());
    for (size_t i = 0; i < adj_list_ptr_->size(); ++i) {
        k_[i].resize(this->n_.size(), 0);
        for (auto nb = adj_list_ptr_->at(i).begin(); nb != adj_list_ptr_->at(i).end(); ++nb) {
            ++k_[i][memberships_[*nb]];
        }
    }
}

inline void blockmodel_t::compute_m() noexcept {
    // In principle, this function only executes once.
    m_.clear();
    m_.resize(get_g());
    for (auto i = 0; i < get_g(); ++i) {
        m_[i].resize(get_g(), 0);
    }
    for (size_t vertex = 0; vertex < adj_list_ptr_->size(); ++vertex) {
        __vertex__ = memberships_[vertex];
        for (auto const& nb: adj_list_ptr_->at(vertex)) {
            ++m_[__vertex__][memberships_[nb]];
        }
    }

}

inline void blockmodel_t::compute_m_r() noexcept {
    // In principle, this function only executes once.
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

inline const void blockmodel_t::compute_k_r_from_mb(uint_vec_t &mb, bool proposal) noexcept {
    size_t max_n = compute_total_num_groups_from_mb(mb);
    if (proposal) {
        cand_k_r_.assign(max_n, 0);
        for (auto i = 0; i < mb.size(); ++i) {
            cand_k_r_[mb[i]] += deg_[i];
        }
    } else {
        k_r_.assign(max_n, 0);
        for (auto i = 0; i < mb.size(); ++i) {
            k_r_[mb[i]] += deg_[i];
        }
    }
}

inline double blockmodel_t::get_log_factorial(int number) const noexcept {
    double log_factorial = lgamma_fast(number + 1.);
    return log_factorial;
}

double blockmodel_t::get_int_data_likelihood_from_mb_uni(uint_vec_t mb, bool proposal) noexcept {
    compute_m_from_mb(mb, proposal);
    compute_n_r_from_mb(mb, proposal);
    compute_k_r_from_mb(mb, proposal);

    double p_ = 2. * num_edges_ / (double) mb.size() / (double) mb.size();
    double log_idl = 0.;

    if (proposal) {  // TODO: use pointers!
        for (auto r = 0; r < cand_n_r_.size(); ++r) {
            log_idl +=
                    cand_k_r_[r] * safelog_fast(cand_n_r_[r]) + get_log_factorial(cand_n_r_[r] - 1) - get_log_factorial(cand_n_r_[r] + cand_k_r_[r] - 1);
            log_idl += get_log_factorial(cand_m_[r][r]) - (cand_m_[r][r] + 1.) * safelog_fast(0.5 * p_ * cand_n_r_[r] * cand_n_r_[r] + 1);
            for (auto s = 0; s < r; ++s) {
                log_idl += get_log_factorial(cand_m_[r][s]) - (cand_m_[r][s] + 1.) * safelog_fast(p_ * cand_n_r_[r] * cand_n_r_[s] + 1);
            }
        }
    } else {
        for (auto r = 0; r < n_r_.size(); ++r) {
            log_idl +=
                    k_r_[r] * safelog_fast(n_r_[r]) + get_log_factorial(n_r_[r] - 1) - get_log_factorial(n_r_[r] + k_r_[r] - 1);
            log_idl += get_log_factorial(m_[r][r]) - (m_[r][r] + 1.) * safelog_fast(0.5 * p_ * n_r_[r] * n_r_[r] + 1);
            for (auto s = 0; s < r; ++s) {
                log_idl += get_log_factorial(m_[r][s]) - (m_[r][s] + 1.) * safelog_fast(p_ * n_r_[r] * n_r_[s] + 1);
            }
        }
    }
    return log_idl;
}

double blockmodel_t::get_int_data_likelihood_from_mb_bi(uint_vec_t mb, bool proposal) noexcept {
    compute_m_from_mb(mb, proposal);
    compute_n_r_from_mb(mb, proposal);
    compute_k_r_from_mb(mb, proposal);

    double p_ = 1. * num_edges_ / (double) get_nsize_A() / (double) get_nsize_B();
    double log_idl = 0.;

    if (proposal) {  // TODO: use pointers!
        for (auto r = 0; r < cand_n_r_.size(); ++r) {
            log_idl +=
                    cand_k_r_[r] * safelog_fast(cand_n_r_[r]) + get_log_factorial(cand_n_r_[r] - 1) -
                    get_log_factorial(cand_n_r_[r] + cand_k_r_[r] - 1);
            for (auto s = 0; s < r; ++s) {
                log_idl += get_log_factorial(cand_m_[r][s]) - (cand_m_[r][s] + 1.) * safelog_fast(p_ * cand_n_r_[r] * cand_n_r_[s] + 1);
            }
        }
    } else {
        for (auto r = 0; r < n_r_.size(); ++r) {
            log_idl +=
                    k_r_[r] * safelog_fast(n_r_[r]) + get_log_factorial(n_r_[r] - 1) -
                    get_log_factorial(n_r_[r] + k_r_[r] - 1);
            for (auto s = 0; s < r; ++s) {
                log_idl += get_log_factorial(m_[r][s]) - (m_[r][s] + 1.) * safelog_fast(p_ * n_r_[r] * n_r_[s] + 1);
            }
        }
    }
    return log_idl;
}

double blockmodel_t::get_log_posterior_from_mb_uni(uint_vec_t mb) noexcept {
    compute_m_from_mb(mb, false);
    compute_n_r_from_mb(mb, false);
    compute_k_r_from_mb(mb, false);

    double p_ = 2. * num_edges_ / (double) mb.size() / (double) mb.size();
    double log_posterior = 0.;

    for (auto r = 0; r < n_r_.size(); ++r) {
        log_posterior += get_log_factorial(n_r_[r]);
        log_posterior +=
                k_r_[r] * safelog_fast(n_r_[r]) + get_log_factorial(n_r_[r] - 1) -
                get_log_factorial(n_r_[r] + k_r_[r] - 1);
        log_posterior +=
                get_log_factorial(m_[r][r]) - (m_[r][r] + 1.) * safelog_fast(0.5 * p_ * n_r_[r] * n_r_[r] + 1);
        for (auto s = 0; s < r; ++s) {
            log_posterior +=
                    get_log_factorial(m_[r][s]) - (m_[r][s] + 1.) * safelog_fast(p_ * n_r_[r] * n_r_[s] + 1);
        }
    }
    log_posterior -= K_ * safelog_fast(memberships_.size() - 2.);
    return log_posterior;
}

double blockmodel_t::compute_log_posterior_from_mb_bi(uint_vec_t mb) noexcept {
    compute_m_from_mb(mb, false);
    compute_n_r_from_mb(mb, false);
    compute_k_r_from_mb(mb, false);

    double p_ = 1. * num_edges_ / (double) get_nsize_A() / (double) get_nsize_B();
    double log_posterior = 0.;

    for (auto r = 0; r < n_r_.size(); ++r) {
        log_posterior += get_log_factorial(n_r_[r]);
        log_posterior +=
                k_r_[r] * safelog_fast(n_r_[r]) + get_log_factorial(n_r_[r] - 1) -
                get_log_factorial(n_r_[r] + k_r_[r] - 1);
        for (auto s = 0; s < r; ++s) {
            log_posterior +=
                    get_log_factorial(m_[r][s]) - (m_[r][s] + 1.) * safelog_fast(p_ * n_r_[r] * n_r_[s] + 1);
        }
    }
    log_posterior -= KA_ * safelog_fast(nsize_A_ - 2.) + KB_ * safelog_fast(nsize_B_ - 2.);
    return log_posterior;
}

void blockmodel_t::sync_internal_states_est() noexcept {
    // TODO: (check) is it any faster if clear() is called?
    m_ = cand_m_;
    n_r_ = cand_n_r_;
    k_r_ = cand_k_r_;
}

uint_vec_t blockmodel_t::get_types() const noexcept { return types_; }

size_t blockmodel_t::get_num_edges() const noexcept { return num_edges_; }

double blockmodel_t::get_entropy_from_degree_correction() const noexcept { return entropy_from_degree_correction_; }

double blockmodel_t::get_log_single_type_prior(uint_vec_t mb, size_t type) noexcept {
    // This implements the single-type prior, P(g, K)
    // if type == 1, we output the prior probability for type-a nodes, P(g, K_a)
    // else if type == 2, we output P(g, K_b);
    compute_n_r_from_mb(mb, false);
    double log_prior_probability = 0.;
    if (type == 1) {
        for (auto r = 0; r < KA_; ++r) {
            log_prior_probability += get_log_factorial(n_r_[r]);
        }
        log_prior_probability -= KA_ * safelog_fast(nsize_A_ - 2.);

    } else if (type == 2) {
        for (auto r = KA_; r < KA_ + KB_; ++r) {
            log_prior_probability += get_log_factorial(n_r_[r]);
        }
        log_prior_probability -= KB_ * safelog_fast(nsize_B_ - 2.);
    }
    return log_prior_probability;
}

double blockmodel_t::compute_entropy_from_m_mr(uint_mat_t m, uint_vec_t m_r) const noexcept {
    double entropy = -(double) num_edges_ - entropy_from_degree_correction_;
    for (unsigned r_ = 0; r_ < n_.size(); ++r_) {
        for (unsigned s_ = 0; s_ < n_.size(); ++s_) {
            if (m_r[r_] * m_r[s_] * m[r_][s_] != 0) {
                entropy -= 1. / 2. * (double) m[r_][s_] *
                           safelog_fast((double) m[r_][s_] / (double) m_r[r_] / (double) m_r[s_]);
            }
        }
    }
    return entropy;
}

