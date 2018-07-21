#ifndef BLOCKMODEL_H
#define BLOCKMODEL_H


#include <random>
#include <utility>
#include <algorithm> // std::shuffle
#include <vector>
#include "types.hh"

const unsigned int compute_total_num_groups_from_mb(uint_vec_t &mb) noexcept;


class blockmodel_t {
protected:
    std::uniform_real_distribution<> random_real;
    std::random_device rd;
    std::mt19937 gen{rd()};

public:
    // Ctor
    blockmodel_t(adj_list_t *adj_list_ptr, const uint_vec_t &types) : random_real(0, 1), adj_list_ptr_(adj_list_ptr), types_(types) { ; }

    blockmodel_t(const uint_vec_t &memberships, const uint_vec_t &types, unsigned int g, unsigned int KA,
                 unsigned int KB, double epsilon, unsigned int N, const adj_list_t * const adj_list_ptr, bool is_bipartite);

    std::vector<mcmc_state_t> mcmc_state_change_riolo_uni(std::mt19937 &engine) noexcept;

    std::vector<mcmc_state_t> mcmc_state_change_riolo(std::mt19937 &engine) noexcept;

    const std::vector<mcmc_state_t> single_vertex_change_tiago(std::mt19937 &engine) noexcept;

    const int_vec_t* get_k(unsigned int vertex) const noexcept;

    const int_vec_t* get_size_vector() const noexcept;

    const int_vec_t* get_degree() const noexcept;

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

    unsigned int get_nsize_A() const noexcept;

    unsigned int get_nsize_B() const noexcept;

    void apply_mcmc_moves(std::vector<mcmc_state_t> &moves) noexcept;

    void apply_mcmc_states_u(std::vector<mcmc_state_t> states) noexcept;  // for uni-partite SBM

    void apply_mcmc_states(std::vector<mcmc_state_t> states) noexcept;

    void shuffle_bisbm(std::mt19937 &engine, unsigned int NA, unsigned int NB) noexcept;

    double get_int_data_likelihood_from_mb_uni(uint_vec_t mb, bool proposal) noexcept;

    double get_int_data_likelihood_from_mb_bi(uint_vec_t mb, bool proposal) noexcept;

    double get_log_posterior_from_mb_uni(uint_vec_t mb) noexcept;

    double compute_log_posterior_from_mb_bi(uint_vec_t mb) noexcept;

    uint_vec_t get_types() const noexcept;

    double get_log_single_type_prior(uint_vec_t mb, unsigned int type) noexcept;

    double compute_entropy_from_m_mr(uint_mat_t m, uint_vec_t m_r) const noexcept;

    unsigned int get_num_edges() const noexcept;

    double get_entropy_from_degree_correction() const noexcept;

    void sync_internal_states_est() noexcept;

private:
    /// State variable
    bool is_bipartite_ = true;
    unsigned int K_ = 0;
    unsigned int KA_ = 0;
    unsigned int nsize_A_ = 0;
    unsigned int KB_ = 0;
    unsigned int nsize_B_ = 0;
    double epsilon_ = 0.;
    const adj_list_t * const adj_list_ptr_;

    int_mat_t k_;
    int_vec_t n_;

    int_vec_t deg_;
    uint_vec_t memberships_;
    const uint_vec_t types_;
    unsigned int num_edges_ = 0;
    double entropy_from_degree_correction_ = 0.;
    uint_mat_t adj_list_;
    uint_mat_t m_;
    uint_vec_t m_r_;

    /// used for `estimate` mode
    uint_mat_t cand_m_;
    int_vec_t n_r_;
    int_vec_t cand_n_r_;
    int_vec_t k_r_;
    int_vec_t cand_k_r_;
    int which_to_move_;

    /// in apply_mcmc_moves
    int_vec_t ki_;

    /// for single_vertex_change_tiago
    double R_t_;
    unsigned int vertex_j_;
    unsigned int proposal_t_;
    int proposal_membership_;
    unsigned int K;

    /// for single_vertex_change_tiago and apply_mcmc_moves
    unsigned int __vertex__ = 0;
    unsigned int __source__ = 0;
    unsigned int __target__ = 0;

    std::vector<mcmc_state_t> moves = std::vector<mcmc_state_t>(1);

    /// Internal distribution. Generator must be passed as a service
    std::uniform_int_distribution<> random_block_;
    std::uniform_int_distribution<> random_node_;

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