#include "physics.hpp"

#include <mace/mh1.hpp>

namespace potential_test {
std::unique_ptr<lapis::Calculator> make_model(lapis::NeighborBackend backend) {
    lapis::Options options;
    options.backend = backend;
    return std::make_unique<mace::MH1>(TEST_MODEL_PATH, options);
}
} // namespace potential_test
