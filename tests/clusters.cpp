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

    // Check number of clusters
    REQUIRE(clusters.n_clusters() == 5);

    // Check cluster coefficients (unique absolute values in descending order)
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 5, 3, 2, 1, 0 }));

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

    // Cluster 4 (coeff 0) should contain indices 3, 4
    REQUIRE(clusters.cluster_size(4) == 2);
    ivec cluster4_indices(clusters.cbegin(4), clusters.cend(4));
    REQUIRE_THAT(cluster4_indices, UnorderedEquals(ivec{ 3, 4 }));

    auto all_clusters = clusters.getClusters();
    REQUIRE(all_clusters.size() == 5);

    REQUIRE_THAT(all_clusters[0], Equals(ivec{ 6 }));
    REQUIRE_THAT(all_clusters[1], Equals(ivec{ 5 }));
    REQUIRE_THAT(all_clusters[2], Equals(ivec{ 0 }));
    REQUIRE_THAT(all_clusters[3], UnorderedEquals(ivec{ 1, 2 }));
    REQUIRE_THAT(all_clusters[4], UnorderedEquals(ivec{ 3, 4 }));
  }

  SECTION("Update single coefficient")
  {
    Eigen::VectorXd beta(7);
    beta << 2, -1, 1, 0, 0, 3, 5;

    slope::Clusters clusters(beta);

    // Update coefficient at index 0 (value 2) to value 4
    // This should move it between clusters with values 5 and 3
    clusters.update(2, 1, 4);

    // Should now have 5 clusters with coefficients [5, 4, 3, 1, 0]
    REQUIRE(clusters.n_clusters() == 5);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 5, 4, 3, 1, 0 }));

    // Check each cluster's contents
    auto all_clusters = clusters.getClusters();

    REQUIRE_THAT(all_clusters[0], Equals(ivec{ 6 })); // coeff 5
    REQUIRE_THAT(all_clusters[1], Equals(ivec{ 0 })); // coeff 4 (updated)
    REQUIRE_THAT(all_clusters[2], Equals(ivec{ 5 })); // coeff 3
    REQUIRE_THAT(all_clusters[3], UnorderedEquals(ivec{ 1, 2 })); // coeff 1
    REQUIRE_THAT(all_clusters[4], UnorderedEquals(ivec{ 3, 4 })); // coeff 0
  }

  SECTION("Update coefficient causing cluster merge")
  {
    // Initial vector: [2, -1, 1, 0, 0, 3, 5]
    Eigen::VectorXd beta(7);
    beta << 2, -1, 1, 0, 0, 3, 5;

    slope::Clusters clusters(beta);

    // Update coefficient at index 0 (value 2) to value 3
    // This should merge it with the cluster containing index 5
    clusters.update(2, 1, 3);

    // Should now have 4 clusters with coefficients [5, 3, 1, 0]
    REQUIRE(clusters.n_clusters() == 4);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 5, 3, 1, 0 }));

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
    REQUIRE_THAT(all_clusters[3], Equals(ivec{ 3, 4 })); // coeff 0
  }

  SECTION("No change update - coefficient remains the same")
  {
    Eigen::VectorXd beta(7);
    beta << 2, -1, 1, 0, 0, 3, 5;

    slope::Clusters clusters(beta);
    auto original_clusters = clusters.getClusters();

    // Update with same value - should be no-op
    clusters.update(2, 2, 2);

    REQUIRE(clusters.n_clusters() == 5);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 5, 3, 2, 1, 0 }));
    REQUIRE_THAT(clusters.getClusters(), Equals(original_clusters));
  }

  SECTION("Reordering from beginning to end")
  {
    Eigen::VectorXd beta(5);
    beta << 5, 4, 3, 2, 1;

    slope::Clusters clusters(beta);

    // Update first coefficient to smallest value
    clusters.update(0, 4, 0.5);

    REQUIRE(clusters.n_clusters() == 5);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 4, 3, 2, 1, 0.5 }));

    auto all_clusters = clusters.getClusters();
    REQUIRE_THAT(all_clusters[4],
                 Equals(ivec{ 0 })); // coeff 0.5 (moved from beginning to end)
  }

  SECTION("All zeros vector")
  {
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(5);

    slope::Clusters clusters(beta);

    REQUIRE(clusters.n_clusters() == 1);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 0 }));

    auto all_clusters = clusters.getClusters();
    REQUIRE(all_clusters.size() == 1);

    // Check all indices are in the single cluster
    std::sort(all_clusters[0].begin(), all_clusters[0].end());
    REQUIRE_THAT(all_clusters[0], Equals(ivec{ 0, 1, 2, 3, 4 }));
  }

  SECTION("All identical non-zero values")
  {
    Eigen::VectorXd beta(5);
    beta << 3, 3, 3, 3, 3;

    slope::Clusters clusters(beta);

    REQUIRE(clusters.n_clusters() == 1);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 3 }));

    auto all_clusters = clusters.getClusters();
    REQUIRE(all_clusters.size() == 1);

    // Check all indices are in the single cluster
    REQUIRE_THAT(all_clusters[0], UnorderedEquals(ivec{ 0, 1, 2, 3, 4 }));
  }

  SECTION("Multiple sequential updates - debugging")
  {
    Eigen::VectorXd beta(5);
    beta << 5, 4, 3, 2, 1;
    slope::Clusters clusters(beta);

    // Check initial state
    REQUIRE(clusters.n_clusters() == 5);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 5, 4, 3, 2, 1 }));

    // First update
    clusters.update(0, 2, 3);

    // Check state after first update
    INFO("After first update");
    REQUIRE(clusters.n_clusters() == 4);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 4, 3, 2, 1 }));

    // Second update
    clusters.update(0, 1, 3);

    // Check state after second update
    INFO("After second update");
    REQUIRE(clusters.n_clusters() == 3);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 3, 2, 1 }));

    // Third update
    clusters.update(1, 0, 6);

    // Final state
    INFO("After third update");
    REQUIRE(clusters.n_clusters() == 3);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 6, 3, 1 }));

    auto all_clusters = clusters.getClusters();

    REQUIRE_THAT(all_clusters[0], Equals(ivec{ 3 }));                // coeff 6
    REQUIRE_THAT(all_clusters[1], UnorderedEquals(ivec{ 0, 1, 2 })); // coeff 3
    REQUIRE_THAT(all_clusters[2], Equals(ivec{ 4 }));                // coeff 1
  }

  SECTION("Multiple sequential updates")
  {
    Eigen::VectorXd beta(5);
    beta << 5, 4, 3, 2, 1;

    slope::Clusters clusters(beta);

    // Initial clusters: [0], [1], [2], [3], [4]
    //                    5    4    3    2    1
    REQUIRE(clusters.n_clusters() == 5);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 5, 4, 3, 2, 1 }));

    // First update - change coefficient of cluster 0 (value 5) to 3
    clusters.update(0, 2, 3); // Move cluster 0 to position 2, with value 3

    // After 1st update: [1], [0,2], [3], [4]
    //                    4     3     2    1
    REQUIRE(clusters.n_clusters() == 4);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 4, 3, 2, 1 }));

    // Second update - change coefficient of cluster 0 (now value 4) to 3
    clusters.update(0, 1, 3); // Move cluster 0 to position 1, with value 3

    // After 2nd update: [0,2,1], [3], [4]
    //                      3      2    1
    REQUIRE(clusters.n_clusters() == 3);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 3, 2, 1 }));

    // Third update - change coefficient of cluster 3 (value 1) to 6
    clusters.update(2, 0, 6); // Move cluster 3 to position 0, with value 6

    // Final expected: [4], [0,1,2], [3]
    REQUIRE(clusters.n_clusters() == 3);
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
    REQUIRE(clusters.n_clusters() == 5);
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
    REQUIRE(clusters.n_clusters() == 3);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 5, 3, 1 }));

    // Create an updated beta vector where one of the 5s is changed to 4
    Eigen::VectorXd updated_beta(4);
    updated_beta << 4, 5, 3, 1;

    // Perform a full update
    clusters.update(updated_beta);

    // Now we should have 4 clusters
    REQUIRE(clusters.n_clusters() == 4);
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

    REQUIRE(clusters.n_clusters() == 1);
    REQUIRE_THAT(clusters.coeffs(), Equals(vec{ 3 }));

    auto all_clusters = clusters.getClusters();
    REQUIRE(all_clusters.size() == 1);
    REQUIRE_THAT(all_clusters[0], Equals(ivec{ 0 }));
  }
}
