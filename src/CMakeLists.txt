add_executable(
        mcmc
        mcmc_main.cc metropolis_hasting.cc output_functions.cc graph_utilities.cc blockmodel.cc
        support/cache.cc)
add_executable(
        mcmc_history
        mcmc_main.cc metropolis_hasting.cc output_functions.cc graph_utilities.cc blockmodel.cc
        support/cache.cc)

set_target_properties(mcmc PROPERTIES COMPILE_DEFINITIONS "OUTPUT_HISTORY=0")
set_target_properties(mcmc_history PROPERTIES COMPILE_DEFINITIONS "OUTPUT_HISTORY=1")

if (Boost_FOUND)
    target_link_libraries(mcmc ${Boost_LIBRARIES})
    target_link_libraries(mcmc_history ${Boost_LIBRARIES})
endif (Boost_FOUND)
