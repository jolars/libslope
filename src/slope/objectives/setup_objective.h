#pragma once

#include "objective.h"
#include <memory>

namespace slope {

/**
 * @brief Sets up the objective function based on the given family.
 * @param family The family of the objective function.
 * @return A unique pointer to the objective function.
 */
std::unique_ptr<Objective>
setupObjective(const std::string family);

}
