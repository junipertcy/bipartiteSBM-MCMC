#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <set>
#include <utility>

using edge_t = std::pair<unsigned int, unsigned int>;
using edge_list_t = std::vector<edge_t>;
using neighbourhood_t = std::set<unsigned int>;
using adj_list_t = std::vector<neighbourhood_t>;


using uint_vec_t = std::vector<unsigned int>;
using int_vec_t = std::vector<int>;
using float_vec_t = std::vector<float>;
using uint_mat_t = std::vector< std::vector<unsigned int> >;
using int_mat_t = std::vector< std::vector<int> >;
using float_mat_t = std::vector< std::vector<float> >;

using mcmc_move_t = struct mcmc_move_t
{
    unsigned int vertex;
    unsigned int source;
    unsigned int target;
};


using mcmc_state_t = struct mcmc_state_t : mcmc_move_t
{
    uint_vec_t memberships;
};


#endif // TYPES_H