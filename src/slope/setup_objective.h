#pragma once

#include "objectives/objective.h"
#include <memory>

namespace slope {

std::unique_ptr<Objective>
setupObjective(const std::string family);

} // namespace slope
