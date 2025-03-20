#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <slope/logger.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Helper function to add a bunch of warnings from multiple threads
void
add_warnings_in_parallel(int num_threads, int warnings_per_thread)
{
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
  {
    int thread_id = 0;
#ifdef _OPENMP
    thread_id = omp_get_thread_num();
#endif

    for (int i = 0; i < warnings_per_thread; i++) {
      slope::WarningLogger::addWarning(slope::WarningCode::GENERIC_WARNING,
                                       "Warning from thread " +
                                         std::to_string(thread_id) +
                                         ", iteration " + std::to_string(i));
    }
  }
}

TEST_CASE("WarningLogger basic functionality", "[logger]")
{
  // Make sure we start with a clean state
  slope::WarningLogger::clearWarnings();

  SECTION("No warnings initially")
  {
    REQUIRE_FALSE(slope::WarningLogger::hasWarnings());
    REQUIRE(slope::WarningLogger::getWarnings().empty());
  }

  SECTION("Add and retrieve a single warning")
  {
    slope::WarningLogger::addWarning(slope::WarningCode::GENERIC_WARNING,
                                     "Test message");
    REQUIRE(slope::WarningLogger::hasWarnings());
    auto warnings = slope::WarningLogger::getWarnings();
    REQUIRE(warnings.size() == 1);
    REQUIRE(warnings[0].code == slope::WarningCode::GENERIC_WARNING);
    REQUIRE(warnings[0].message == "Test message");
  }

  SECTION("Warning codes are deduplicated")
  {
    slope::WarningLogger::addWarning(slope::WarningCode::DEPRECATED_FEATURE,
                                     "First message");
    slope::WarningLogger::addWarning(slope::WarningCode::DEPRECATED_FEATURE,
                                     "Second message");

    auto warnings = slope::WarningLogger::getWarnings();
    REQUIRE(warnings.size() == 2);
    // Both messages should be present, with the second one being added after
    // the first
    REQUIRE(warnings[0].code == slope::WarningCode::DEPRECATED_FEATURE);
    REQUIRE(warnings[0].message == "First message");
    REQUIRE(warnings[1].code == slope::WarningCode::DEPRECATED_FEATURE);
    REQUIRE(warnings[1].message == "Second message");
  }

  SECTION("Multiple warning types")
  {
    slope::WarningLogger::addWarning(slope::WarningCode::GENERIC_WARNING,
                                     "Generic");
    slope::WarningLogger::addWarning(slope::WarningCode::DEPRECATED_FEATURE,
                                     "Deprecated");
    slope::WarningLogger::addWarning(slope::WarningCode::MAXIT_REACHED,
                                     "Max iterations");
    slope::WarningLogger::addWarning(slope::WarningCode::LINE_SEARCH_FAILED,
                                     "Line search");

    auto warnings = slope::WarningLogger::getWarnings();
    REQUIRE(warnings.size() == 4);

    // Collect all the messages to check they all exist
    std::vector<std::string> messages;
    for (const auto& warning : warnings) {
      messages.push_back(warning.message);
    }

    REQUIRE(std::find(messages.begin(), messages.end(), "Generic") !=
            messages.end());
    REQUIRE(std::find(messages.begin(), messages.end(), "Deprecated") !=
            messages.end());
    REQUIRE(std::find(messages.begin(), messages.end(), "Max iterations") !=
            messages.end());
    REQUIRE(std::find(messages.begin(), messages.end(), "Line search") !=
            messages.end());
  }

  SECTION("Clear warnings")
  {
    slope::WarningLogger::addWarning(slope::WarningCode::GENERIC_WARNING,
                                     "Test message");
    REQUIRE(slope::WarningLogger::hasWarnings());

    slope::WarningLogger::clearWarnings();
    REQUIRE_FALSE(slope::WarningLogger::hasWarnings());
    REQUIRE(slope::WarningLogger::getWarnings().empty());
  }
}

TEST_CASE("WarningLogger with multiple threads", "[logger][parallel]")
{
  slope::WarningLogger::clearWarnings();

  SECTION("Warnings from multiple threads")
  {
    size_t num_threads = 4;
    size_t warnings_per_thread = 3;

    add_warnings_in_parallel(num_threads, warnings_per_thread);

    // We should have num_threads * warnings_per_thread warnings
    auto warnings = slope::WarningLogger::getWarnings();
    REQUIRE(warnings.size() == num_threads * warnings_per_thread);
  }

  SECTION("Different warning codes per thread")
  {
#ifdef _OPENMP
#pragma omp parallel num_threads(4)
#endif
    {
      int thread_id = 0;
#ifdef _OPENMP
      thread_id = omp_get_thread_num();
#endif

      // Use different warning codes based on thread ID
      slope::WarningCode code;
      switch (thread_id % 4) {
        case 0:
          code = slope::WarningCode::GENERIC_WARNING;
          break;
        case 1:
          code = slope::WarningCode::DEPRECATED_FEATURE;
          break;
        case 2:
          code = slope::WarningCode::MAXIT_REACHED;
          break;
        default:
          code = slope::WarningCode::LINE_SEARCH_FAILED;
          break;
      }

      slope::WarningLogger::addWarning(code,
                                       "Thread " + std::to_string(thread_id));
    }

    // We should have up to 4 different warnings
    auto warnings = slope::WarningLogger::getWarnings();

#ifdef _OPENMP
    size_t expected_warnings = std::min(4, omp_get_max_threads());
#else
    size_t expected_warnings = 1;
#endif

    REQUIRE(warnings.size() == expected_warnings);
  }

  SECTION("Get warnings by thread ID")
  {
#ifdef _OPENMP
#pragma omp parallel num_threads(4)
#endif
    {
      int thread_id = 0;
#ifdef _OPENMP
      thread_id = omp_get_thread_num();
#endif

      slope::WarningLogger::addWarning(slope::WarningCode::GENERIC_WARNING,
                                       "Thread specific " +
                                         std::to_string(thread_id));
    }

#ifdef _OPENMP
    // Check for thread-specific warnings
    for (int i = 0; i < std::min(4, omp_get_max_threads()); i++) {
      auto thread_warnings = slope::WarningLogger::getThreadWarnings(i);
      REQUIRE(thread_warnings.size() == 1);
      REQUIRE(thread_warnings[0].code == slope::WarningCode::GENERIC_WARNING);
      REQUIRE(thread_warnings[0].message ==
              "Thread specific " + std::to_string(i));
    }
#endif
  }

  SECTION("Warning code to string conversion")
  {
    REQUIRE(slope::warningCodeToString(slope::WarningCode::GENERIC_WARNING) ==
            "GENERIC_WARNING");
    REQUIRE(slope::warningCodeToString(
              slope::WarningCode::DEPRECATED_FEATURE) == "DEPRECATED_FEATURE");
    REQUIRE(slope::warningCodeToString(slope::WarningCode::MAXIT_REACHED) ==
            "MAXIT_REACHED");
    REQUIRE(slope::warningCodeToString(
              slope::WarningCode::LINE_SEARCH_FAILED) == "LINE_SEARCH_FAILED");
  }
}
