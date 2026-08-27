# TODO

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
