#ifndef TYPES_HH
#define TYPES_HH

#include <vector>
#include <map>
#include <utility>

using edge_t = std::pair<size_t, size_t>;
using edge_list_t = std::vector<edge_t>;
using neighbourhood_t = std::vector<size_t>;
using adj_list_t = std::vector<neighbourhood_t>;


using uint_vec_t = std::vector<unsigned int>;
using int_vec_t = std::vector<int>;
using float_vec_t = std::vector<float>;
using uint_mat_t = std::vector< std::vector<unsigned int> >;
using int_mat_t = std::vector< std::vector<int> >;
using float_mat_t = std::vector< std::vector<float> >;
using int_map_vec_t = std::vector< std::map<int, int> >;

using pi = std::pair<double, int>;

using mcmc_move_t = struct mcmc_move_t
{
    size_t vertex;
    size_t source;
    size_t target;
};

using block_move_t = struct block_move_t
{
    size_t source;
    size_t target;
};

#endif // TYPES_H
