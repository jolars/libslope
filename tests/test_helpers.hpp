#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <cmath>

template<typename T>
std::vector<double>
asStdVec(const T& x)
{
  std::vector<double> vec(x.data(), x.data() + x.size());

  return vec;
}

template<typename Range>
struct VectorApproxEqualMatcher : Catch::Matchers::MatcherGenericBase
{
  VectorApproxEqualMatcher(const Range& range, const double eps)
    : range{ range }
    , eps{ eps }
  {
  }

  template<typename OtherRange>
  bool match(OtherRange const& other) const
  {
    size_t n = range.size();
    if (n != other.size()) {
      return false;
    }

    for (size_t i = 0; i < range.size(); ++i) {
      // Check if either value is NaN
      if (std::isnan(range[i]) || std::isnan(other[i])) {
        // If one is NaN and the other isn't, they're not equal
        if (!std::isnan(range[i]) || !std::isnan(other[i])) {
          return false;
        }
        // If both are NaN, continue to next element
        continue;
      }
      // Normal comparison for non-NaN values
      if (std::abs(range[i] - other[i]) > eps) {
        return false;
      }
    }

    return true;
  }

  std::string describe() const override
  {
    return "Approximately equal to: " + Catch::rangeToString(asStdVec(range));
  }

private:
  const Range& range;
  const double eps;
};

template<typename Range>
auto
VectorApproxEqual(const Range& range, const double eps = 1e-8)
  -> VectorApproxEqualMatcher<Range>
{
  return VectorApproxEqualMatcher<Range>{ range, eps };
}
