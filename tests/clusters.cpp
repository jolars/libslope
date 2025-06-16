#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <slope/clusters.h>

TEST_CASE("Clusters", "[clusters]")
{
  using Catch::Matchers::Equals;
  using Catch::Matchers::UnorderedEquals;
  using ivec = std::vector<int>;
  using vec = std::vector<double>;

  SECTION("Initialization with example vector")
  {
    Eigen::VectorXd beta(7);
    beta << 2, -1, 1, 0, 0, 3, 5;

    slope::Clusters clusters(beta);

    // Check number of clusters - should exclude zeros
    REQUIRE(clusters.size() == 4);

    // Check cluster coefficients (unique absolute values in descending order)
    // No zeros included
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 5, 3, 2, 1 }));

    // Check indices for each cluster
    // Cluster 0 (coeff 5) should contain index 6
    REQUIRE(clusters.cluster_size(0) == 1);
    REQUIRE(*(clusters.cbegin(0)) == 6);

    // Cluster 1 (coeff 3) should contain index 5
    REQUIRE(clusters.cluster_size(1) == 1);
    REQUIRE(*(clusters.cbegin(1)) == 5);

    // Cluster 2 (coeff 2) should contain index 0
    REQUIRE(clusters.cluster_size(2) == 1);
    REQUIRE(*(clusters.cbegin(2)) == 0);

    // Cluster 3 (coeff 1) should contain indices 1, 2
    REQUIRE(clusters.cluster_size(3) == 2);
    ivec cluster3_indices(clusters.cbegin(3), clusters.cend(3));
    REQUIRE_THAT(cluster3_indices, UnorderedEquals(ivec{ 1, 2 }));

    // Zero indices (3, 4) should not be in any cluster

    auto all_clusters = clusters.getClusters();
    REQUIRE(all_clusters.size() == 4);

    REQUIRE_THAT(all_clusters[0], Equals(ivec{ 6 }));
    REQUIRE_THAT(all_clusters[1], Equals(ivec{ 5 }));
    REQUIRE_THAT(all_clusters[2], Equals(ivec{ 0 }));
    REQUIRE_THAT(all_clusters[3], UnorderedEquals(ivec{ 1, 2 }));
  }

  SECTION("Update single coefficient")
  {
    Eigen::VectorXd beta(7);
    beta << 2, -1, 1, 0, 0, 3, 5;

    slope::Clusters clusters(beta);

    // Update coefficient at index 2 (value 2) to value 4
    // This should move it between clusters with values 5 and 3
    clusters.update(2, 1, 4);

    // Should now have 5 clusters with coefficients [5, 4, 3, 1]
    REQUIRE(clusters.size() == 4);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 5, 4, 3, 1 }));

    // Check each cluster's contents
    auto all_clusters = clusters.getClusters();

    REQUIRE_THAT(all_clusters[0], Equals(ivec{ 6 })); // coeff 5
    REQUIRE_THAT(all_clusters[1], Equals(ivec{ 0 })); // coeff 4 (updated)
    REQUIRE_THAT(all_clusters[2], Equals(ivec{ 5 })); // coeff 3
    REQUIRE_THAT(all_clusters[3], UnorderedEquals(ivec{ 1, 2 })); // coeff 1
  }

  SECTION("Update coefficient causing cluster merge")
  {
    // Initial vector: [2, -1, 1, 0, 0, 3, 5]
    Eigen::VectorXd beta(7);
    beta << 2, -1, 1, 0, 0, 3, 5;

    slope::Clusters clusters(beta);

    // Update coefficient at index 2 (value 2) to value 3
    // This should merge it with the cluster containing index 5
    clusters.update(2, 1, 3);

    // Should now have 4 clusters with coefficients [5, 3, 1]
    REQUIRE(clusters.size() == 3);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 5, 3, 1 }));

    // Check each cluster's contents
    auto all_clusters = clusters.getClusters();
    for (auto& cluster : all_clusters) {
      if (cluster.size() > 1) {
        std::sort(cluster.begin(), cluster.end());
      }
    }

    REQUIRE_THAT(all_clusters[0], Equals(ivec{ 6 }));    // coeff 5
    REQUIRE_THAT(all_clusters[1], Equals(ivec{ 0, 5 })); // coeff 3 (merged)
    REQUIRE_THAT(all_clusters[2], Equals(ivec{ 1, 2 })); // coeff 1
  }

  SECTION("No change update - coefficient remains the same")
  {
    Eigen::VectorXd beta(7);
    beta << 2, -1, 1, 0, 0, 3, 5;

    slope::Clusters clusters(beta);
    auto original_clusters = clusters.getClusters();

    // Update with same value - should be no-op
    clusters.update(2, 2, 2);

    REQUIRE(clusters.size() == 4);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 5, 3, 2, 1 }));
    REQUIRE_THAT(clusters.getClusters(), Equals(original_clusters));
  }

  SECTION("Reordering from beginning to end")
  {
    Eigen::VectorXd beta(5);
    beta << 5, 4, 3, 2, 1;

    slope::Clusters clusters(beta);

    // Update first coefficient to smallest value
    clusters.update(0, 4, 0.5);

    REQUIRE(clusters.size() == 5);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 4, 3, 2, 1, 0.5 }));

    auto all_clusters = clusters.getClusters();
    REQUIRE_THAT(all_clusters[4],
                 Equals(ivec{ 0 })); // coeff 0.5 (moved from beginning to end)
  }

  SECTION("All zeros vector")
  {
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(5);

    slope::Clusters clusters(beta);

    // With zero clusters removed, there should be no clusters
    REQUIRE(clusters.size() == 0);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{}));

    auto all_clusters = clusters.getClusters();
    REQUIRE(all_clusters.size() == 0);
  }

  SECTION("All identical non-zero values")
  {
    Eigen::VectorXd beta(5);
    beta << 3, 3, 3, 3, 3;

    slope::Clusters clusters(beta);

    REQUIRE(clusters.size() == 1);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 3 }));

    auto all_clusters = clusters.getClusters();
    REQUIRE(all_clusters.size() == 1);

    // Check all indices are in the single cluster
    REQUIRE_THAT(all_clusters[0], UnorderedEquals(ivec{ 0, 1, 2, 3, 4 }));
  }

  SECTION("Multiple sequential updates")
  {
    Eigen::VectorXd beta(5);
    beta << 5, 4, 3, 2, 1;

    slope::Clusters clusters(beta);

    // Initial clusters: [0], [1], [2], [3], [4]
    //                    5    4    3    2    1
    REQUIRE(clusters.size() == 5);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 5, 4, 3, 2, 1 }));

    // First update - change coefficient of cluster 0 (value 5) to 3
    clusters.update(0, 2, 3); // Move cluster 0 to position 2, with value 3

    // After 1st update: [1], [0,2], [3], [4]
    //                    4     3     2    1
    REQUIRE(clusters.size() == 4);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 4, 3, 2, 1 }));

    // Second update - change coefficient of cluster 0 (now value 4) to 3
    clusters.update(0, 1, 3); // Move cluster 0 to position 1, with value 3

    // After 2nd update: [0,2,1], [3], [4]
    //                      3      2    1
    REQUIRE(clusters.size() == 3);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 3, 2, 1 }));

    // Third update - change coefficient of cluster 3 (value 1) to 6
    clusters.update(2, 0, 6); // Move cluster 3 to position 0, with value 6

    // Final expected: [4], [0,1,2], [3]
    REQUIRE(clusters.size() == 3);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 6, 3, 2 }));

    auto all_clusters = clusters.getClusters();
    for (auto& cluster : all_clusters) {
      std::sort(cluster.begin(), cluster.end());
    }

    REQUIRE_THAT(all_clusters[0], Equals(ivec{ 4 })); // coeff 6
    REQUIRE_THAT(all_clusters[1],
                 UnorderedEquals(ivec{ 0, 1, 2 })); // coeff 3 (merged cluster)
    REQUIRE_THAT(all_clusters[2], Equals(ivec{ 3 })); // coeff 1
  }

  SECTION("Negative coefficients handling")
  {
    Eigen::VectorXd beta(5);
    beta << -5, 4, -3, 2, -1;

    slope::Clusters clusters(beta);

    // Check that absolute values are used for clustering
    REQUIRE(clusters.size() == 5);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 5, 4, 3, 2, 1 }));

    auto all_clusters = clusters.getClusters();
    REQUIRE_THAT(all_clusters[0], Equals(ivec{ 0 })); // abs(-5) = 5
    REQUIRE_THAT(all_clusters[1], Equals(ivec{ 1 })); // abs(4) = 4
    REQUIRE_THAT(all_clusters[2], Equals(ivec{ 2 })); // abs(-3) = 3
  }

  SECTION("Update that splits a cluster using full update")
  {
    Eigen::VectorXd beta(4);
    beta << 5, 5, 3, 1;

    slope::Clusters clusters(beta);

    // Initially we should have 3 clusters
    REQUIRE(clusters.size() == 3);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 5, 3, 1 }));

    // Create an updated beta vector where one of the 5s is changed to 4
    Eigen::VectorXd updated_beta(4);
    updated_beta << 4, 5, 3, 1;

    // Perform a full update
    clusters.update(updated_beta);

    // Now we should have 4 clusters
    REQUIRE(clusters.size() == 4);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 5, 4, 3, 1 }));

    auto all_clusters = clusters.getClusters();
    REQUIRE_THAT(all_clusters[0], Equals(ivec{ 1 })); // Value 5
    REQUIRE_THAT(all_clusters[1], Equals(ivec{ 0 })); // Value 4
  }

  SECTION("Single element vector")
  {
    Eigen::VectorXd beta(1);
    beta << 3;

    slope::Clusters clusters(beta);

    REQUIRE(clusters.size() == 1);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 3 }));

    auto all_clusters = clusters.getClusters();
    REQUIRE(all_clusters.size() == 1);
    REQUIRE_THAT(all_clusters[0], Equals(ivec{ 0 }));

    // Update to zero
    clusters.update(0, 0, 0);
    REQUIRE(clusters.size() == 0);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{}));
  }

  SECTION("Debug update method for infinite loop")
  {
    // Create a simple vector with distinct values in different clusters
    Eigen::VectorXd beta(6);
    beta << 1.0, 3.0, 2.0, -2.0, 3.0, -1.0;

    slope::Clusters clusters(beta);

    // Verify initial state
    REQUIRE(clusters.size() == 3);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 3.0, 2.0, 1.0 }));

    // Get initial clusters for comparison
    auto initial_clusters = clusters.getClusters();
    REQUIRE_THAT(initial_clusters[0], UnorderedEquals(ivec{ 1, 4 }));
    REQUIRE_THAT(initial_clusters[1], UnorderedEquals(ivec{ 2, 3 }));
    REQUIRE_THAT(initial_clusters[2], UnorderedEquals(ivec{ 0, 5 }));

    // First update - move from first cluster to second
    INFO("About to perform first update");
    clusters.update(0, 1, 2.0);

    // Verify state after first update
    INFO("After first update");
    REQUIRE(clusters.size() == 2);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 2.0, 1.0 }));

    auto clusters_after_first = clusters.getClusters();
    REQUIRE_THAT(clusters_after_first[0], UnorderedEquals(ivec{ 1, 2, 3, 4 }));
    REQUIRE_THAT(clusters_after_first[1], UnorderedEquals(ivec{ 0, 5 }));

    // Second update - move from second cluster to first
    INFO("About to perform second update");
    clusters.update(1, 0, 2.0);

    // Verify final state
    INFO("After second update");
    REQUIRE(clusters.size() == 1);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 2.0 }));

    auto final_clusters = clusters.getClusters();
    REQUIRE_THAT(final_clusters[0], UnorderedEquals(ivec{ 0, 1, 2, 3, 4, 5 }));
  }

  SECTION("Update coefficient to zero")
  {
    Eigen::VectorXd beta(7);
    beta << 2, -1, 1, 0, 0, 3, 5;

    slope::Clusters clusters(beta);

    // Initial state - 4 clusters (excluding zeros)
    REQUIRE(clusters.size() == 4);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 5, 3, 2, 1 }));

    // Update coefficient at index 2 (value 2) to 0
    clusters.update(2, 2, 0);

    // Should now have 3 clusters with coefficients [5, 3, 1]
    REQUIRE(clusters.size() == 3);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 5, 3, 1 }));

    // Cluster with value 2 should be removed
    auto all_clusters = clusters.getClusters();
    REQUIRE_THAT(all_clusters[0], Equals(ivec{ 6 }));             // coeff 5
    REQUIRE_THAT(all_clusters[1], Equals(ivec{ 5 }));             // coeff 3
    REQUIRE_THAT(all_clusters[2], UnorderedEquals(ivec{ 1, 2 })); // coeff 1
  }

  SECTION("Merge to end of indices")
  {
    Eigen::VectorXd beta(6);
    beta << 5, 10, 9, -5, 3, -5;

    slope::Clusters clusters(beta);

    int old_index = 2;
    int new_index = 3;

    clusters.update(old_index, new_index, 3);

    auto all_clusters = clusters.getClusters();

    REQUIRE_THAT(all_clusters[2], UnorderedEquals(ivec{ 0, 3, 4, 5 }));
  }
}

TEST_CASE("Pattern matrix", "[clusters][pattern]")
{
  SECTION("Standard pattern matrix")
  {
    Eigen::VectorXd beta(7);
    beta << 1.0, 3.0, 2.0, 3.0, -1.0, 0.0, 0.0;

    auto patt = slope::patternMatrix(beta);

    // Check dimensions
    REQUIRE(patt.rows() == beta.size());
    REQUIRE(patt.cols() == 3);
    REQUIRE(patt.nonZeros() == beta.size() - 2);

    // Check contents
    // Should be:
    // 0  0  1
    // 1  0  0
    // 0  1  0
    // 1  0  0
    // 0  0 -1
    // 0  0  0
    // 0  0  0
    REQUIRE(patt.coeff(0, 0) == 0);
    REQUIRE(patt.coeff(0, 2) == 1);
    REQUIRE(patt.coeff(1, 0) == 1);
    REQUIRE(patt.coeff(2, 1) == 1);
    REQUIRE(patt.coeff(3, 0) == 1);
    REQUIRE(patt.coeff(3, 2) == 0);
    REQUIRE(patt.coeff(4, 2) == -1);
    REQUIRE(patt.coeff(5, 0) == 0);
    REQUIRE(patt.coeff(5, 1) == 0);
    REQUIRE(patt.coeff(5, 2) == 0);
    REQUIRE(patt.coeff(6, 0) == 0);
    REQUIRE(patt.coeff(6, 1) == 0);

    auto patt2 = slope::patternMatrix(beta);

    Eigen::MatrixXi patt1_mat = patt;
    Eigen::MatrixXi patt2_mat = patt2;

    REQUIRE_THAT(patt2_mat.reshaped(),
                 VectorApproxEqual(patt1_mat.reshaped(), 1e-4));
  }

  SECTION("No zeros")
  {
    Eigen::VectorXd beta(5);
    beta << 1.0, 3.0, 2.0, 3.0, -1.0;

    auto patt = slope::patternMatrix(beta);

    // Check dimensions
    REQUIRE(patt.rows() == beta.size());
    REQUIRE(patt.cols() == 3);
    REQUIRE(patt.nonZeros() == beta.size());

    // Check contents
    // Should be:
    // 0  0  1
    // 1  0  0
    // 0  1  0
    // 1  0  0
    // 0  0 -1
    REQUIRE(patt.coeff(0, 0) == 0);
    REQUIRE(patt.coeff(0, 2) == 1);
    REQUIRE(patt.coeff(1, 0) == 1);
    REQUIRE(patt.coeff(2, 1) == 1);
    REQUIRE(patt.coeff(3, 0) == 1);
    REQUIRE(patt.coeff(3, 2) == 0);
    REQUIRE(patt.coeff(4, 2) == -1);

    auto patt2 = slope::patternMatrix(beta);

    Eigen::MatrixXi patt1_mat = patt;
    Eigen::MatrixXi patt2_mat = patt2;

    REQUIRE_THAT(patt2_mat.reshaped(),
                 VectorApproxEqual(patt1_mat.reshaped(), 1e-4));
  }

  SECTION("Degenerative all zeros case")
  {
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(5);

    auto patt = slope::patternMatrix(beta);

    REQUIRE(patt.rows() == beta.size());
    REQUIRE(patt.cols() == 0); // No columns for an all-zero vector
    REQUIRE(patt.nonZeros() == 0);
  }
}
