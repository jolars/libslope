#include <list>
#include <unordered_set>

class Cluster
{
public:
  Cluster(const double coef,
          const std::unordered_set<int>& indices,
          const int pointer)
    : coef(coef)
    , indices(indices)
    , pointer(pointer)
  {
  }

  double getCoeff() const { return coef; }
  const std::unordered_set<int>& getIndices() const { return indices; }
  int getPointer() const { return pointer; }

private:
  double coef;
  std::unordered_set<int> indices;
  int pointer;
};

void
updateClusters(Cluster& cluster_old, Cluster& cluster_new)
{
  double c_old = cluster_old.getCoeff();
  double c_new = cluster_new.getCoeff();

  if (c_new != c_old) {
    if (c_new == coeff(new_index)) {
      old_cluster.merge(new_cluster);
      merge(old_index, new_index);
    } else {
      setCoeff(old_index, c_new);
      if (old_index != new_index) {
        reorder(old_index, new_index);
      }
    }
  }
}

void
updateClusters(std::list<Cluster>& clusters,
               std::list<Cluster>::iterator it_old,
               std::list<Cluster>::iterator it_new)
{
  double c_old = (*it_old).getCoeff();
  double c_new = (*it_new).getCoeff();

  if (c_new != c_old) {
    if (c_new == coeff(new_index)) {
      // old_cluster.merge(new_cluster);
      // merge(old_index, new_index);
    } else {
      // setCoeff(old_index, c_new);
      // if (old_index != new_index) {
      //   reorder(old_index, new_index);
      // }
    }
  }
}

// std::list<Widget>::iterator it = /* some iterator */;
// while (/* some condition */) {
//     // Check if we need to move the current element
//     if (/* some other condition */) {
//         // Determine the direction to move
//         if (/* moving up */ && it != widgets.begin()) {
//             auto prev = std::prev(it);
//             std::swap(*it, *prev);
//             it = prev;
//         } else if (/* moving down */ && std::next(it) != widgets.end()) {
//             auto next = std::next(it);
//             std::swap(*it, *next);
//             it = next;
//         }
//     }
//     // Move the iterator in the desired direction
//     if (/* moving up */ && it != widgets.begin()) {
//         --it;
//     } else if (/* moving down */ && it != widgets.end()) {
//         ++it;
//     }
// }
