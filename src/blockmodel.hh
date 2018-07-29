#ifndef BLOCKMODEL_HH
#define BLOCKMODEL_HH


#include <random>
#include <utility>
#include <algorithm> // std::shuffle
#include <vector>
#include "types.hh"

const size_t compute_total_num_groups_from_mb(uint_vec_t &mb) noexcept;


class blockmodel_t {
protected:
    std::uniform_real_distribution<> random_real;
    std::random_device rd;
    std::mt19937 gen{rd()};

public:
    /** Default constructor */
    blockmodel_t(uint_vec_t& memberships, const uint_vec_t &types, size_t g, size_t KA,
                 size_t KB, double epsilon, const adj_list_t* adj_list_ptr, bool is_bipartite);

    std::vector<mcmc_state_t> mcmc_state_change_riolo_uni(std::mt19937 &engine) noexcept;

    std::vector<mcmc_state_t> mcmc_state_change_riolo(std::mt19937 &engine) noexcept;

    template<class RNG>
    inline auto single_vertex_change_tiago(RNG&& engine) noexcept {
        R_t_ = 0.;
        proposal_membership_ = 0;
        __source__ = 0;
        __target__ = 0;
        while (__source__ == __target__) {
            K = 0;
            while (K == 0) {
                __vertex__ = random_node_(engine);   // TODO: it's a hot fix
                while (adj_list_[__vertex__].empty()) {
                    __vertex__ = random_node_(engine);
                }
                if (KA_ == 1 && KB_ == 1) {
                    __source__ = memberships_[__vertex__];
                    __target__ = __source__;
                    return moves;
                }
                if (types_[__vertex__] == 0) {
                    K = KA_;
                    if (K == 1) {
                        __vertex__ += nsize_A_;
                        K = KB_;
                    }
                } else {
                    K = KB_;
                    if (K == 1) {
                        __vertex__ -= nsize_A_;
                        K = KA_;
                    }
                }
            }
            __source__ = memberships_[__vertex__];

            // Here, instead of naively move to adjacent blocks, we follow Tiago Peixoto's approach (PRE 89, 012804 [2014])
            which_to_move_ = size_t(random_real(engine) * adj_list_[__vertex__].size());
            vertex_j_ = adj_list_[__vertex__][which_to_move_];
            proposal_t_ = memberships_[vertex_j_];

            if (types_[__vertex__] == 0) {
                proposal_membership_ = size_t(random_real(engine) * KA_);
                R_t_ = epsilon_ * (KA_) / (m_r_[proposal_t_] + epsilon_ * (KA_));
            } else {
                proposal_membership_ = size_t(random_real(engine) * KB_) + size_t(KA_);
                R_t_ = epsilon_ * (KB_) / (m_r_[proposal_t_] + epsilon_ * (KB_));
            }

            if (random_real(engine) < R_t_) {
                __target__ = proposal_membership_;
            } else {
                std::discrete_distribution<size_t> d(m_[proposal_t_].begin(), m_[proposal_t_].end());
                __target__ = d(gen);
            }
        }

        moves[0].source = __source__;
        moves[0].target = __target__;
        moves[0].vertex = __vertex__;
        return moves;
    }


    const int_vec_t* get_k(size_t vertex) const noexcept;

    const int_vec_t* get_size_vector() const noexcept;

    const int_vec_t* get_degree() const noexcept;

    const int get_degree(size_t vertex) const noexcept;

    const uint_vec_t* get_memberships() const noexcept;

    const uint_mat_t* get_m() const noexcept;  /* Optimizing this function (pass by ref) is extremely important!! (why?) */

    const uint_vec_t* get_m_r() const noexcept;

    bool get_is_bipartite() const noexcept;

    size_t get_N() const noexcept;

    size_t get_g() const noexcept;

    double get_epsilon() const noexcept;

    size_t get_K() const noexcept;

    size_t get_KA() const noexcept;

    size_t get_KB() const noexcept;

    double get_log_factorial(int number) const noexcept;

    size_t get_nsize_A() const noexcept;

    size_t get_nsize_B() const noexcept;

    bool apply_mcmc_moves(std::vector<mcmc_state_t>&& moves) noexcept;

    void apply_mcmc_states_u(std::vector<mcmc_state_t> states) noexcept;  // for uni-partite SBM

    void apply_mcmc_states(std::vector<mcmc_state_t> states) noexcept;

    void shuffle_bisbm(std::mt19937 &engine, size_t NA, size_t NB) noexcept;

    double get_int_data_likelihood_from_mb_uni(uint_vec_t mb, bool proposal) noexcept;

    double get_int_data_likelihood_from_mb_bi(uint_vec_t mb, bool proposal) noexcept;

    double get_log_posterior_from_mb_uni(uint_vec_t mb) noexcept;

    double compute_log_posterior_from_mb_bi(uint_vec_t mb) noexcept;

    uint_vec_t get_types() const noexcept;

    double get_log_single_type_prior(uint_vec_t mb, size_t type) noexcept;

    double compute_entropy_from_m_mr(uint_mat_t m, uint_vec_t m_r) const noexcept;

    size_t get_num_edges() const noexcept;

    double get_entropy_from_degree_correction() const noexcept;

    void sync_internal_states_est() noexcept;

private:
    /// State variable
    bool is_bipartite_ = true;
    size_t K_ = 0;
    size_t KA_ = 0;
    size_t nsize_A_ = 0;
    size_t KB_ = 0;
    size_t nsize_B_ = 0;
    double epsilon_ = 0.;
    const adj_list_t * const adj_list_ptr_;

    int_mat_t k_;
    int_vec_t n_;

    int_vec_t deg_;
    std::vector< std::vector<size_t> > adj_list_;
    size_t num_edges_ = 0;

    uint_vec_t memberships_;
    const uint_vec_t types_;

    double entropy_from_degree_correction_ = 0.;

    uint_mat_t m_;
    uint_vec_t m_r_;

    /// used for `estimate` mode
    uint_mat_t cand_m_;
    int_vec_t n_r_;
    int_vec_t cand_n_r_;
    int_vec_t k_r_;
    int_vec_t cand_k_r_;
    size_t which_to_move_;

    /// in apply_mcmc_moves
    const int_vec_t* ki_;

    /// for single_vertex_change_tiago
    double R_t_;
    size_t vertex_j_;
    size_t proposal_t_;
    size_t proposal_membership_;
    size_t K;

    /// for single_vertex_change_tiago and apply_mcmc_moves
    size_t __vertex__ = 0;
    size_t __source__ = 0;
    size_t __target__ = 0;

    std::vector<mcmc_state_t> moves = std::vector<mcmc_state_t>(1);

    /// Internal distribution. Generator must be passed as a service
    std::uniform_int_distribution<size_t> random_block_;
    std::uniform_int_distribution<size_t> random_node_;

    /// Private methods
    /* Compute stuff from scratch. */
    void compute_k() noexcept;
    void compute_m() noexcept;  // Note: get_m and compute_m are different.
    void compute_m_r() noexcept;

    /* Compute stuff from scratch; used for `estimate` mode */

    const void compute_m_from_mb(uint_vec_t &mb, bool proposal) noexcept;
    const void compute_n_r_from_mb(uint_vec_t &mb, bool proposal) noexcept;
    const void compute_k_r_from_mb(uint_vec_t &mb, bool proposal) noexcept;

};

#endif // BLOCKMODEL_H