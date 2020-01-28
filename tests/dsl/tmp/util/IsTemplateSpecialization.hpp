#include <HighPerMeshes/dsl/meta_programming/util/IsTemplateSpecialization.hpp>

#include <tuple>
#include <vector>

static_assert(HPM::IsTemplateSpecialization<std::tuple<int, int>, std::tuple>);

static_assert(not HPM::IsTemplateSpecialization<std::vector<int>, std::tuple>);