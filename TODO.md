# TODO

## Hybrid coordinate-descent performance

- [x] Add a focused benchmark for singleton sparse coordinate derivatives,
  which account for about 44% of the optimized RCV1 profile inclusively.
  Replace full residual and weight copies with a single traversal of the
  feature column that accumulates the weighted first- and second-order terms.
- [ ] Cache response-wise weight sums while the IRLS weights remain fixed, and
  benchmark caching singleton Hessians during a coordinate-descent phase.
  Adopt the latter only if its runtime benefit justifies the extra state.
- [ ] Aggregate centered residual offsets by response during cluster updates
  instead of applying one dense adjustment per cluster member. Investigate
  reusing the sparse cluster workspace to avoid traversing the feature columns
  again when updating the residual.
- [ ] Benchmark maintaining response-wise weighted residual sums incrementally
  to eliminate repeated dense reductions in singleton and multi-feature
  cluster derivatives. Consider lazy centered residual offsets only if this
  simpler approach leaves a material bottleneck.
- [ ] Reprofile the RCV1 path after optimizing coordinate derivatives and
  residual updates. Optimize the linear predictor, currently about 4% of the
  profile, only if it remains material.
- [ ] Remove the unused coordinate-descent work vector and reuse temporary sign
  storage where practical.

## Dual points and screening

- [ ] Evaluate conditionally optimizing the multinomial intercept before scaling
  the generalized residual. Compare dual-gap tightness and runtime with the
  current domain-preserving anchor construction, and retain the anchor as a
  fallback when a finite intercept solution is unavailable.
- [x] Strengthen direct multinomial dual-point tests with multiple non-reference
  classes, classwise zero-sum checks, simplex-boundary cases, absent
  classes, SLOPE dual-norm scaling, and nonnegative end-to-end gaps.
- [ ] Explore dual extrapolation or another tighter candidate construction that
  preserves the intercept equality, each loss's conjugate domain, and the
  sorted \(\ell\_1\) dual-norm constraint. Benchmark certificate quality
  before adopting the added complexity.
- [ ] Define and document solver behavior when an unpenalized intercept has no
  finite minimizer, including single-class binomial responses, all-zero
  Poisson responses, and absent multinomial classes. Keep dual-certificate
  behavior distinct from primal-solution attainment.
- [ ] Investigate an intercept-aware Gap Safe screening rule using projected or
  centered feature norms. Establish the required loss-specific curvature
  bounds first; canonical Poisson requires a local-curvature argument rather
  than the classical globally strongly concave dual bound.
