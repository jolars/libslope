#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

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

    if (range.size() != other.size()) {
      return false;
    }

    for (int i = 0; i < range.size(); ++i) {
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
VectorApproxEqual(const Range& range, const double eps = 1e-6)
  -> VectorApproxEqualMatcher<Range>
{
  return VectorApproxEqualMatcher<Range>{ range, eps };
}
