# Changelog

## [3.0.0](https://github.com/jolars/libslope/compare/v2.9.0...v3.0.0) (2025-06-16)

### ⚠ BREAKING CHANGES

* remove `patternMatrix()` from `Clusters`

### Features

* remove `patternMatrix()` from `Clusters` ([8fb22d3](https://github.com/jolars/libslope/commit/8fb22d3577db969545f1c21cf5644ceef558318c))

### Bug Fixes

* fix cluster updating ([205ce89](https://github.com/jolars/libslope/commit/205ce89cdf22c7f15a87fbd6ae3609c5e2c1811b))

### Performance Improvements

* shrink `sorted` to fit actual size ([9536c24](https://github.com/jolars/libslope/commit/9536c2464854e6fa682b90636a3d6dc31db01714))

## [2.9.0](https://github.com/jolars/libslope/compare/v2.8.3...v2.9.0) (2025-06-16)

### Features

* add randomized coordinate descent method ([ec33a86](https://github.com/jolars/libslope/commit/ec33a8637ec71cc3e6cdda7f6846e3a719f223df))

## [2.8.3](https://github.com/jolars/libslope/compare/v2.8.2...v2.8.3) (2025-06-12)

### Bug Fixes

* fix path relaxation ([0238e28](https://github.com/jolars/libslope/commit/0238e28d8fd0292e46eabaa6a4ce58c56586befb))

## [2.8.2](https://github.com/jolars/libslope/compare/v2.8.1...v2.8.2) (2025-06-10)

### Bug Fixes

* correctly catch violations in `kktCheck()` ([5dc5f69](https://github.com/jolars/libslope/commit/5dc5f69540ec0ada37a20618da8e43e3c9547194))
* correctly return total iterations ([8f48277](https://github.com/jolars/libslope/commit/8f482774196b189826f85fc2d347a8f8fe3a5b37))
* fix `move_elements` when `from < to` and size > 1 ([6f58fff](https://github.com/jolars/libslope/commit/6f58fff06d12f258d684ced7906a33468a845be0))

## [2.8.1](https://github.com/jolars/libslope/compare/v2.8.0...v2.8.1) (2025-06-10)

### Bug Fixes

* don't set `alpha` to zero when relaxing the fit ([bf59237](https://github.com/jolars/libslope/commit/bf59237c58b6a887c990020f51fadc319f2650ec))

## [2.8.0](https://github.com/jolars/libslope/compare/v2.7.0...v2.8.0) (2025-06-06)

### Features

* restart iteration count if there are KKT violations ([64f032f](https://github.com/jolars/libslope/commit/64f032f315c62c72cd9b346f61537674beb95dc3))

## [2.7.0](https://github.com/jolars/libslope/compare/v2.6.1...v2.7.0) (2025-05-30)

### Features

* don't screen if no regularization ([6346802](https://github.com/jolars/libslope/commit/63468029603be5dc70a91c26cecfefc85f8645b1))

## [2.6.1](https://github.com/jolars/libslope/compare/v2.6.0...v2.6.1) (2025-05-30)

### Bug Fixes

* safeguard dividing by too large numbers in `dualNorm` ([dd62081](https://github.com/jolars/libslope/commit/dd620813628aaa1b57900a952e09e8f40f9fb1c8))

## [2.6.0](https://github.com/jolars/libslope/compare/v2.5.1...v2.6.0) (2025-05-26)

### Features

* add julia bindings ([ea9d6bf](https://github.com/jolars/libslope/commit/ea9d6bf25fea6c3afa08a4adcd5bacaf72607321))

## [2.5.1](https://github.com/jolars/libslope/compare/v2.5.0...v2.5.1) (2025-05-22)

### Bug Fixes

* actually set `best_alpha_ind` ([5989870](https://github.com/jolars/libslope/commit/5989870dbfbf80e70558135773bb39d07b0df614))

## [2.5.0](https://github.com/jolars/libslope/compare/v2.4.1...v2.5.0) (2025-05-21)

### Features

* support `Eigen::Map` in `crossValidate()` ([bb0b8da](https://github.com/jolars/libslope/commit/bb0b8daf56d7cbbc72413b8ad5b7cb43f902725d))

## [2.4.1](https://github.com/jolars/libslope/compare/v2.4.0...v2.4.1) (2025-05-19)

### Bug Fixes

* use correct dual formulation for logistic loss ([d89cca0](https://github.com/jolars/libslope/commit/d89cca00e1ef9e96cc78d2ccc96490b9b0ce028f))

## [2.4.0](https://github.com/jolars/libslope/compare/v2.3.0...v2.4.0) (2025-05-19)

### Features

* add `copy_x` argument to `CvConfig` ([49e0d46](https://github.com/jolars/libslope/commit/49e0d46a6ef012795ccdb34a131045f30370648c))

## [2.3.0](https://github.com/jolars/libslope/compare/v2.2.0...v2.3.0) (2025-05-19)

### Features

* allow passing a matrix view as x ([d5a1b6d](https://github.com/jolars/libslope/commit/d5a1b6d06ff7c4a0cbfd2ae0ab2680fed2f07e6c))

## [2.2.0](https://github.com/jolars/libslope/compare/v2.1.0...v2.2.0) (2025-05-19)

### Features

* add template instantiation for sparse matrix map ([5c8c364](https://github.com/jolars/libslope/commit/5c8c36447523e799fac3ab0dc8445aae89a299de))

## [2.1.0](https://github.com/jolars/libslope/compare/v2.0.2...v2.1.0) (2025-05-15)

### Features

* fix multinomial convergence issue ([#148](https://github.com/jolars/libslope/issues/148)) ([f963255](https://github.com/jolars/libslope/commit/f96325501bcc4187e05927612efb16a7ca3852d5)), closes [#147](https://github.com/jolars/libslope/issues/147)

## [2.0.2](https://github.com/jolars/libslope/compare/v2.0.1...v2.0.2) (2025-05-15)

### Bug Fixes

* use correct objective in hybrid failsafe step ([fae62d5](https://github.com/jolars/libslope/commit/fae62d500533f50d3dde11bfd39981f03ecf2fd0))

## [2.0.1](https://github.com/jolars/libslope/compare/v2.0.0...v2.0.1) (2025-05-14)

### Bug Fixes

* include penalty in descent check in hybrid method ([09d982d](https://github.com/jolars/libslope/commit/09d982de05ce5f8489f00df69a7b8043f2d9f107))

## [2.0.0](https://github.com/jolars/libslope/compare/v1.2.0...v2.0.0) (2025-05-13)

### ⚠ BREAKING CHANGES

* use non-redundant multinomial logistic formulation (#145)

### Features

* use non-redundant multinomial logistic formulation ([#145](https://github.com/jolars/libslope/issues/145)) ([b2dcdbb](https://github.com/jolars/libslope/commit/b2dcdbbd27b101d1abbc1cae6073eb7889a6a669))

## [1.2.0](https://github.com/jolars/libslope/compare/v1.1.1...v1.2.0) (2025-05-08)

### Features

* support multinomial logistic regression in hybrid solver ([#143](https://github.com/jolars/libslope/issues/143)) ([9f87daa](https://github.com/jolars/libslope/commit/9f87daa50fb876f7f7845e5d7f4c3ff471e3e1b0)), closes [#65](https://github.com/jolars/libslope/issues/65)

## [1.1.1](https://github.com/jolars/libslope/compare/v1.1.0...v1.1.1) (2025-05-08)

### Performance Improvements

* do not modify weights and response in quadratic loss ([9370122](https://github.com/jolars/libslope/commit/93701228ac5fbdb25dd9539cafb69bed8569cda3))

## [1.1.0](https://github.com/jolars/libslope/compare/v1.0.1...v1.1.0) (2025-05-07)

### Features

* **solver:** fall back to pgd method if no progress ([252b1c0](https://github.com/jolars/libslope/commit/252b1c0a20d805c0ffcd51f79f2d3dc4ad1e613c)), closes [#83](https://github.com/jolars/libslope/issues/83)

## [1.0.1](https://github.com/jolars/libslope/compare/v1.0.0...v1.0.1) (2025-04-29)

### Bug Fixes

* use correct argument name in error message ([fe77f45](https://github.com/jolars/libslope/commit/fe77f45bf62c130d994f8c7f63b193ce1bbb4eca))

## [1.0.0](https://github.com/jolars/libslope/compare/v0.32.0...v1.0.0) (2025-04-23)

### ⚠ BREAKING CHANGES

* increase default maximum number of iterations to 1e5

### Features

* increase default maximum number of iterations to 1e5 ([3882c93](https://github.com/jolars/libslope/commit/3882c93c0aed4261b29ef7938174f57e86c5a2ae))

## [0.32.0](https://github.com/jolars/libslope/compare/v0.31.0...v0.32.0) (2025-04-23)

### Features

* set defaults for theta1 and theta2 ([9902044](https://github.com/jolars/libslope/commit/9902044eb7ab778e0910270e32160e9efe168d52))

## [0.31.0](https://github.com/jolars/libslope/compare/v0.30.1...v0.31.0) (2025-04-16)


### ⚠ BREAKING CHANGES

* attempt a platform-indepenent shuffle method

### Features

* support features of type `Eigen::Map<>` ([24c41f0](https://github.com/jolars/libslope/commit/24c41f017ecbe1392a0ed9615b7ccff8d7c3154d))


### Bug Fixes

* add missing threads header ([6e306a7](https://github.com/jolars/libslope/commit/6e306a78db35139e6fed279c705585cbf4adc3e5))
* install license file ([265eac0](https://github.com/jolars/libslope/commit/265eac06dd400b80f22f1710fc5d4fbd39dda5e4))
* **windows:** don't set omp_set_max_active_levels on windows ([77b2c05](https://github.com/jolars/libslope/commit/77b2c05b80f7c9e9144870fc2da6e07292a28e29))
* **windows:** use signed int in openmp for loop ([4f39c08](https://github.com/jolars/libslope/commit/4f39c08b8ddeaf0b823e3a1e227c3fe11036abe5))
* **windows:** use signed integer in for loop for openmp ([b3fe760](https://github.com/jolars/libslope/commit/b3fe760e7333bab72eabe3fe4e2791ba077f0ca6))
* wrap openmp pragmas in if else guards ([1fdec59](https://github.com/jolars/libslope/commit/1fdec596abf1b1966a12ab1f14565705605e0ce6))


### Reverts

* "build: try to avoid eigen cmake checks" ([e44c900](https://github.com/jolars/libslope/commit/e44c9009bf657bc31afc6b1ad2d8131a13851dc6))
* "build(deps): provide eigen in external/" ([822bcd7](https://github.com/jolars/libslope/commit/822bcd7d7ec6c0f62fde895b3282fb753b13a57c))
* "fix(windows): don't set omp_set_max_active_levels on windows" ([0d1a4d2](https://github.com/jolars/libslope/commit/0d1a4d2172e9c0a573ea5f00a00a96c99995a943))
* "refactor!: attempt a platform-indepenent shuffle method" ([93b348c](https://github.com/jolars/libslope/commit/93b348c7609a0819ead3f36138274513dd38c16a))


### Code Refactoring

* attempt a platform-indepenent shuffle method ([8626db8](https://github.com/jolars/libslope/commit/8626db8fcd119e9fe3adaa2512852b795565a524))

## [0.30.1](https://github.com/jolars/libslope/compare/v0.30.0...v0.30.1) (2025-04-09)


### Bug Fixes

* remove slope path from included header ([ea404fe](https://github.com/jolars/libslope/commit/ea404fe5f54ed30bc848dfd7b07ad9a7fcaa3a7d))
* remove slope path from inclusions in `estimate_alpha.h` ([dcbf379](https://github.com/jolars/libslope/commit/dcbf37986934b0284219eb0a801426a7b9f1daf6))

## [0.30.0](https://github.com/jolars/libslope/compare/v0.29.0...v0.30.0) (2025-04-08)


### ⚠ BREAKING CHANGES

* remove `getCoefs(size_t)` method from `SlopePath`

### Features

* add `hasIntercept()` to `SlopeFit` class ([5d53ab2](https://github.com/jolars/libslope/commit/5d53ab24696f35962b4f13518384add89e34a045))
* add `relax()` method for paths ([7976455](https://github.com/jolars/libslope/commit/7976455c9f64fad9a55c0063d7932a988bb53b92))
* add `size()` method to `SlopePath` ([c2a8ba7](https://github.com/jolars/libslope/commit/c2a8ba7e05a7dc9bbfd9c5ecafe76cb7d9fde6b0))
* add mixing parameter `gamma` for relaxed fits ([#134](https://github.com/jolars/libslope/issues/134)) ([a2019ae](https://github.com/jolars/libslope/commit/a2019ae3bf566011a49d6c859a1aa3bfec947b3b)), closes [#133](https://github.com/jolars/libslope/issues/133)
* add optional warm starting to `relax()` ([a39207a](https://github.com/jolars/libslope/commit/a39207a7d23dc21269a4842892d72f04f8d140ef))
* add original scale arguments to coef/intercept getters ([5d53ab2](https://github.com/jolars/libslope/commit/5d53ab24696f35962b4f13518384add89e34a045))
* add relaxed SLOPE method ([#131](https://github.com/jolars/libslope/issues/131)) ([9d94d3d](https://github.com/jolars/libslope/commit/9d94d3dc23837438b17e32821f7c9e72e32b3701)), closes [#101](https://github.com/jolars/libslope/issues/101)
* correctly collect diagnostics in `relax()` ([c94ccb8](https://github.com/jolars/libslope/commit/c94ccb87e322c7c79db81f174dd4dc30265c6b1c))
* decrease default tolerance in relaxed solver ([d1b347d](https://github.com/jolars/libslope/commit/d1b347d09751d778c92b90a80e6c4aa1781d6283))
* enable cross-validation over relaxation parameter ([e8a1af4](https://github.com/jolars/libslope/commit/e8a1af492c742e299e8b9374735dfd6b5d88dad2))
* move `max_it_inner` to setter ([07d0c52](https://github.com/jolars/libslope/commit/07d0c526c770ecd7d2147535f5e3f4e64a8b5478))
* move `maxit_irls` to setter ([4f12077](https://github.com/jolars/libslope/commit/4f12077e412d8e9eb29cb69bf8d0430d0d9a6e58))
* move `tol` in `relax()` to setter ([2076056](https://github.com/jolars/libslope/commit/207605635f4bb61f659950a2d316c2e2ce0d2b64))
* reduce tolerance in relaxed method ([1d5afea](https://github.com/jolars/libslope/commit/1d5afeaba234366447eb1f2192c797f672c17d4d))
* remove `getCoefs(size_t)` method from `SlopePath` ([5d53ab2](https://github.com/jolars/libslope/commit/5d53ab24696f35962b4f13518384add89e34a045))


### Bug Fixes

* correctly reshape beta in `relax()` ([840354f](https://github.com/jolars/libslope/commit/840354f131a1d11d8793285e5a66e40520268baf))
* **relax:** initialize gradient to zero ([442eaee](https://github.com/jolars/libslope/commit/442eaee2623d6c33272b7ad8fdd225f528d85517))
* use correct scale for coefs in cross-validation ([5d53ab2](https://github.com/jolars/libslope/commit/5d53ab24696f35962b4f13518384add89e34a045))


### Performance Improvements

* use sparse representation when rescaling coefs ([208853a](https://github.com/jolars/libslope/commit/208853a8daa3f1e7d1816d0005a7de8a4e7a8cf2))

## [0.29.0](https://github.com/jolars/libslope/compare/v0.28.0...v0.29.0) (2025-04-03)


### ⚠ BREAKING CHANGES

* drop argument of `patternMatrix()` of `Clusters`

### Features

* drop argument of `patternMatrix()` of `Clusters` ([403f4fe](https://github.com/jolars/libslope/commit/403f4fedd93999771f9e944250f8e55ad21c00d0))


### Performance Improvements

* lazily compute zero clusters ([#128](https://github.com/jolars/libslope/issues/128)) ([b9f34db](https://github.com/jolars/libslope/commit/b9f34db9610f65e5c473234944728dbb0668d7ef))

## [0.28.0](https://github.com/jolars/libslope/compare/v0.27.0...v0.28.0) (2025-04-02)


### ⚠ BREAKING CHANGES

* remove `setReturnClusters()` from `Slope`

### Features

* add `patternMatrix()` + equiv method for `Clusters` ([3659336](https://github.com/jolars/libslope/commit/36593365f64035550343327dd67426644cbccde7))
* remove `setReturnClusters()` from `Slope` ([e1fe04b](https://github.com/jolars/libslope/commit/e1fe04b1754af1e9fa49952880797f517e48b630))


### Bug Fixes

* use correct estimate of standard deviation ([b1aed85](https://github.com/jolars/libslope/commit/b1aed8526881f2d5c4186000ab93a46f659acea1))


### Reverts

* "feat!: remove `setReturnClusters()` from `Slope`" ([4ca2e73](https://github.com/jolars/libslope/commit/4ca2e734dfbcc57c5722540386c487778ce1d359))

## [0.27.0](https://github.com/jolars/libslope/compare/v0.26.0...v0.27.0) (2025-03-19)


### ⚠ BREAKING CHANGES

* redesign logger

### Features

* degrade error from maxit in alpha est to warning ([1e2d44a](https://github.com/jolars/libslope/commit/1e2d44a118ca16b99193398a18dab1b8d0168e15))
* redesign logger ([40bb591](https://github.com/jolars/libslope/commit/40bb59139651bf8dcee2808e621bf4aef232389e))
* set `alpha_est_maxit` default to 1000 ([5f45da2](https://github.com/jolars/libslope/commit/5f45da2e1c10b53447cae6ac4581d0e7e37160ff))

## [0.26.0](https://github.com/jolars/libslope/compare/v0.25.0...v0.26.0) (2025-03-19)


### Features

* add a `Screening` class ([266cd42](https://github.com/jolars/libslope/commit/266cd422e31ab0f3ea3ca23100a9f7561de6f155))


### Performance Improvements

* conditionally parallelize gradient computation ([6716762](https://github.com/jolars/libslope/commit/6716762493033daab7bf44077d6fc8bcea6ce947))
* **hybrid:** use sparse collapsed cluster ([#124](https://github.com/jolars/libslope/issues/124)) ([7c3c8d9](https://github.com/jolars/libslope/commit/7c3c8d94b2722fe00d53ca3120e41647c4148053))
* parallelize linear predictor ([efa7e2d](https://github.com/jolars/libslope/commit/efa7e2d2acd593daee40e2f885de54ae5e183368))

## [0.25.0](https://github.com/jolars/libslope/compare/v0.24.0...v0.25.0) (2025-03-18)


### ⚠ BREAKING CHANGES

* remove solvers namespace

### Features

* remove solvers namespace ([6ddb7be](https://github.com/jolars/libslope/commit/6ddb7be8188e33ea93c3f1a9259545c3c94f09bc))


### Reverts

* "docs: document solvers namespace" ([a67de13](https://github.com/jolars/libslope/commit/a67de1344d94a90969b1e995af27c2116fb5da57))
* "feat!: wrap loss functions in `losses` namespace" ([143c83b](https://github.com/jolars/libslope/commit/143c83b5bc5e06697fbadea28fd0f6e8a87b0d91))

## [0.24.0](https://github.com/jolars/libslope/compare/v0.23.0...v0.24.0) (2025-03-17)


### ⚠ BREAKING CHANGES

* wrap loss functions in `losses` namespace

### Features

* wrap loss functions in `losses` namespace ([ef38bbf](https://github.com/jolars/libslope/commit/ef38bbfc9a90d634ef2d86f60df60fcbc72633a9))

## [0.23.0](https://github.com/jolars/libslope/compare/v0.22.0...v0.23.0) (2025-03-17)


### Features

* add `fitOls()`, a function for fitting OLS ([3c7408a](https://github.com/jolars/libslope/commit/3c7408a9565a64f83221eacc5cee1be49cfb3d6c))
* add alpha estimation procedure ([#118](https://github.com/jolars/libslope/issues/118)) ([b4f17af](https://github.com/jolars/libslope/commit/b4f17afa3d792d7e11e181d222a81f6a47292301))

## [0.22.0](https://github.com/jolars/libslope/compare/v0.21.0...v0.22.0) (2025-03-12)


### Features

* add `Clusters::getClusters()` ([c79dba7](https://github.com/jolars/libslope/commit/c79dba7356374009e3add363ee0444623fa2419b))
* add `size()` method to `Clusters` ([84aeaa1](https://github.com/jolars/libslope/commit/84aeaa19ec392c4e36ea78a2d5a50ae7b4d329bd))
* add a warnings logger ([#115](https://github.com/jolars/libslope/issues/115)) ([1c3e4b2](https://github.com/jolars/libslope/commit/1c3e4b239566794989764e532c762269173239c7)), closes [#108](https://github.com/jolars/libslope/issues/108)
* add repeated k-fold cv ([#106](https://github.com/jolars/libslope/issues/106)) ([32d526c](https://github.com/jolars/libslope/commit/32d526cf9e58fa17796d122ad4208b25ac321959))


### Bug Fixes

* **cv:** pick correct optimum in cross-validation ([0b85be7](https://github.com/jolars/libslope/commit/0b85be7212831edbaff0312c5b2684c5b02051c0))
* handle exceptions in openmp regions ([#110](https://github.com/jolars/libslope/issues/110)) ([54a7cf0](https://github.com/jolars/libslope/commit/54a7cf0cc12d2288b2463e7cee3927226430582e)), closes [#105](https://github.com/jolars/libslope/issues/105)


### Performance Improvements

* speed up Clusters::update() ([3d56e60](https://github.com/jolars/libslope/commit/3d56e60acde9936c564f0d5d6d678e4d54fecdd1))
* use lazy cumulative sums in threshold function ([1c4ce3b](https://github.com/jolars/libslope/commit/1c4ce3b41e30a0ca34176526600eb01f801eec53))

## [0.21.0](https://github.com/jolars/libslope/compare/v0.20.1...v0.21.0) (2025-03-05)


### ⚠ BREAKING CHANGES

* remove log-factorial term from poisson loss
* remove `nullDeviance()`
* correctly compute deviance

### Features

* add cross validation ([#103](https://github.com/jolars/libslope/issues/103)) ([da894c8](https://github.com/jolars/libslope/commit/da894c849bf3f25e795e46b5f0b1b0e07c0fc5cd)), closes [#100](https://github.com/jolars/libslope/issues/100)
* add predict method for loss functions ([#97](https://github.com/jolars/libslope/issues/97)) ([fac54f1](https://github.com/jolars/libslope/commit/fac54f1f563d98c83d9070d7634ce5c7d9775c1b)), closes [#70](https://github.com/jolars/libslope/issues/70)
* add prediction method to `SlopeFit` ([#99](https://github.com/jolars/libslope/issues/99)) ([f45b628](https://github.com/jolars/libslope/commit/f45b628c9c83a6de5c05df8766d4a19c8e4dd668))
* handle small or nonpositive input to poisson loss and link ([801ccd7](https://github.com/jolars/libslope/commit/801ccd7558c6ca164200bb001e48ff3dbac44ffd))
* remove `nullDeviance()` ([042c2b2](https://github.com/jolars/libslope/commit/042c2b2fd73d43e979e6cc828976d144c981b075))
* remove log-factorial term from poisson loss ([85660d7](https://github.com/jolars/libslope/commit/85660d7eabcf9fa56f256396700a7005a6acc718))


### Bug Fixes

* correctly compute deviance ([042c2b2](https://github.com/jolars/libslope/commit/042c2b2fd73d43e979e6cc828976d144c981b075))
* setup path correctly when path_length = 1 ([17e78a4](https://github.com/jolars/libslope/commit/17e78a49fb94a8bcb18c278ad47d3678bf0c42e4))
* **sparse:** fix bug with sparse and modify x ([78c83b1](https://github.com/jolars/libslope/commit/78c83b1238be721b88a6ce29f79889576a4826f1))

## [0.20.1](https://github.com/jolars/libslope/compare/v0.20.0...v0.20.1) (2025-02-28)


### Bug Fixes

* fix FISTA update ([492ee92](https://github.com/jolars/libslope/commit/492ee92b3878e81e79f215f09c662803893615a6))

## [0.20.0](https://github.com/jolars/libslope/compare/v0.19.0...v0.20.0) (2025-02-28)


### Features

* add `pause()` and `resume()` methods to timer ([67baf5f](https://github.com/jolars/libslope/commit/67baf5f706e9b514e19ffb78ebca3e5c29ac22ac))


### Bug Fixes

* compute true dual  when collecting diagnostics ([5adffda](https://github.com/jolars/libslope/commit/5adffda07ec3c6bb1db362a990f3c1cee486ce2a))

## [0.19.0](https://github.com/jolars/libslope/compare/v0.18.0...v0.19.0) (2025-02-27)


### Features

* check x_scales for zero-variance ([965c007](https://github.com/jolars/libslope/commit/965c007af29074463a664f218dd8a5a9cc2d5888))


### Bug Fixes

* move timer into path loop ([95cb6bd](https://github.com/jolars/libslope/commit/95cb6bdbd22de27789cac37d09015e5b2f1d58a4))
* return correct indices from `kktCheck()` ([c87496d](https://github.com/jolars/libslope/commit/c87496da2b186d7169417ac6841ba13d8bf2aa50))

## [0.18.0](https://github.com/jolars/libslope/compare/v0.17.3...v0.18.0) (2025-02-27)


### Features

* parallelize gradient computations ([#89](https://github.com/jolars/libslope/issues/89)) ([30aa617](https://github.com/jolars/libslope/commit/30aa617f316adb5f34597e38ea9823a84d9d49e2))


### Bug Fixes

* avoid repeating duals, primals, and time ([90dd5bf](https://github.com/jolars/libslope/commit/90dd5bf5c5893a1a246fac48e6663bdcc27acf54))
* handle `path_length = 1` case ([50e2a3e](https://github.com/jolars/libslope/commit/50e2a3e57d11b62d01582c700f1e56a8fceb004c))

## [0.17.3](https://github.com/jolars/libslope/compare/v0.17.2...v0.17.3) (2025-02-26)


### Bug Fixes

* **solvers:** use correct FISTA update ([1fa7bfa](https://github.com/jolars/libslope/commit/1fa7bfab020dc28247f72f460a3349ab7a500725))

## [0.17.2](https://github.com/jolars/libslope/compare/v0.17.1...v0.17.2) (2025-02-26)


### Performance Improvements

* start the fit with the intercept completely fit ([122b41e](https://github.com/jolars/libslope/commit/122b41ec73a65acfb1dc7b7999956e61e531fcbe))
* use `colwise()` in softmax ([f108172](https://github.com/jolars/libslope/commit/f108172afeed7a60152ed50c016b328d4b681e03))

## [0.17.1](https://github.com/jolars/libslope/compare/v0.17.0...v0.17.1) (2025-02-25)


### Bug Fixes

* allow oscar parameters that are 0 ([1627a1e](https://github.com/jolars/libslope/commit/1627a1ea700aa9a155892d45e2992437da9c2070))

## [0.17.0](https://github.com/jolars/libslope/compare/v0.16.0...v0.17.0) (2025-02-25)


### ⚠ BREAKING CHANGES

* rename `setHybridPgdFreq()` to `setHybridCdIterations()`
* rename `setmaxIt()` to `setMaxIterations()`
* rename loss function
* remove `setMaxItInner()`

### Features

* remove `setMaxItInner()` ([3538cd1](https://github.com/jolars/libslope/commit/3538cd1d64f3f438be33693d276a16aceeaa0495))
* rename `setHybridPgdFreq()` to `setHybridCdIterations()` ([3c391c1](https://github.com/jolars/libslope/commit/3c391c1ae98228a2c1e89808bcc17b651a991914))
* rename `setmaxIt()` to `setMaxIterations()` ([5d66dde](https://github.com/jolars/libslope/commit/5d66dde5d2612259a06fffdb093a594aca9dd38b))
* rename loss function ([7d5bfce](https://github.com/jolars/libslope/commit/7d5bfceb5ed6eed8c86f62d55fd07f4604962201))


### Bug Fixes

* add log(y!) to poisson primal and duals ([e4be976](https://github.com/jolars/libslope/commit/e4be9763a39876e74ed4b4d10a7067b17bdcfc44))

## [0.16.0](https://github.com/jolars/libslope/compare/v0.15.2...v0.16.0) (2025-02-24)


### ⚠ BREAKING CHANGES

* rename objective to loss
* rename `setPgdFreq()` to `setHybridPgdFreq()`

### Features

* rename `setPgdFreq()` to `setHybridPgdFreq()` ([e9bec62](https://github.com/jolars/libslope/commit/e9bec623f5db26eae7fd962e94803811f0d0be4c))
* rename objective to loss ([7baa508](https://github.com/jolars/libslope/commit/7baa508e42b27423f3f4f250709e38a47aac0fd7))


### Reverts

* "fix: fix intercept update in hybrid pgd method" ([8d7af13](https://github.com/jolars/libslope/commit/8d7af1317dad4a87bf8fb4955f932242b0bd5376))

## [0.15.2](https://github.com/jolars/libslope/compare/v0.15.1...v0.15.2) (2025-02-24)


### Bug Fixes

* fix intercept update in hybrid pgd method ([f91313a](https://github.com/jolars/libslope/commit/f91313a9e1b7d55adc249c01dbaf315aef4a6047))

## [0.15.1](https://github.com/jolars/libslope/compare/v0.15.0...v0.15.1) (2025-02-23)


### Bug Fixes

* correctly setup JIT scaling ([737c808](https://github.com/jolars/libslope/commit/737c808dcf832f37da23b161003735ba8e45eaf1))

## [0.15.0](https://github.com/jolars/libslope/compare/v0.14.0...v0.15.0) (2025-02-23)


### ⚠ BREAKING CHANGES

* make collecting diagnostics configurable

### Features

* make collecting diagnostics configurable ([3fd42e7](https://github.com/jolars/libslope/commit/3fd42e794cb5c97cfe4bca156b54af7a3567f952))


### Bug Fixes

* correctly handle centers without scales or vice versa ([db1e49e](https://github.com/jolars/libslope/commit/db1e49eed4a23a014a8b7b9bfe95ecda66069bfd))

## [0.14.0](https://github.com/jolars/libslope/compare/v0.13.0...v0.14.0) (2025-02-22)


### ⚠ BREAKING CHANGES

* change centering and scaling API

### Features

* change centering and scaling API ([ac3b6c6](https://github.com/jolars/libslope/commit/ac3b6c65527f4dc39afcde99d4bffcb96db8b8d2))
* support additional scaling and centering types ([71e502b](https://github.com/jolars/libslope/commit/71e502b32c716e606c702789134fb08986cf33bc))


### Performance Improvements

* simplify stddev computation ([9228aa4](https://github.com/jolars/libslope/commit/9228aa42905106276720d1bb6c1e6b9b9c51bb93))

## [0.13.0](https://github.com/jolars/libslope/compare/v0.12.1...v0.13.0) (2025-02-21)


### ⚠ BREAKING CHANGES

* rename standardize_jit to normalize_jit
* allow user-specified scales and centers ([#76](https://github.com/jolars/libslope/issues/76))

### Features

* allow user-specified scales and centers ([#76](https://github.com/jolars/libslope/issues/76)) ([bae335d](https://github.com/jolars/libslope/commit/bae335d043ac6bad06b8175aaea13dfb6232507c)), closes [#37](https://github.com/jolars/libslope/issues/37)


### Bug Fixes

* crop returned alpha path to length of path ([23f387d](https://github.com/jolars/libslope/commit/23f387d4f63dad3d8fd9ecc51bfc95764ee8f794))
* expose OSCAR parameters through `Slope::setOscarParameters()` ([a60e6f3](https://github.com/jolars/libslope/commit/a60e6f3a26fc471ac43339e1d8d1d48f66edc9c8))
* provide n to `lambdaSequence()` ([3058290](https://github.com/jolars/libslope/commit/305829097ceb2a7f5e6bdc160a4782809c1d4c11))
* rescale intercepts for multinomial case correctly ([f8cea24](https://github.com/jolars/libslope/commit/f8cea24aeca6409013e2f11be5eaea1d4521d527))


### Code Refactoring

* rename standardize_jit to normalize_jit ([f2ca28e](https://github.com/jolars/libslope/commit/f2ca28e74006334eee666bbf136237bdd642c46c))

## [0.12.1](https://github.com/jolars/libslope/compare/v0.12.0...v0.12.1) (2025-02-20)


### Bug Fixes

* correctly setup lambda path when using auto alpha ([a402ed2](https://github.com/jolars/libslope/commit/a402ed2976fc761b19fe8d452801c7589e882c1b))

## [0.12.0](https://github.com/jolars/libslope/compare/v0.11.0...v0.12.0) (2025-02-20)


### Features

* return passes from solver ([fd8b512](https://github.com/jolars/libslope/commit/fd8b512db3b5a360c8c864e73a0b650bda103c5f))
* return wall clock time from solver ([2b72b4b](https://github.com/jolars/libslope/commit/2b72b4be2d5888dad1114ec2b27350392fcca00f))


### Bug Fixes

* return alpha correctly from `Slope::fit()` ([fab20f1](https://github.com/jolars/libslope/commit/fab20f16d542c081b9c1554d25ad3ca31565f046))

## [0.11.0](https://github.com/jolars/libslope/compare/v0.10.0...v0.11.0) (2025-02-19)


### ⚠ BREAKING CHANGES

* no longer print anything
* Separate `fit()` and `path()` methods ([#71](https://github.com/jolars/libslope/issues/71))

### Features

* add FISTA solver ([#67](https://github.com/jolars/libslope/issues/67)) ([d7e510a](https://github.com/jolars/libslope/commit/d7e510a461387f7027b1957af638c6baabd3efd3)), closes [#63](https://github.com/jolars/libslope/issues/63)
* no longer print anything ([f90ff48](https://github.com/jolars/libslope/commit/f90ff486a112fd9e4dd4749a7970ce3577256841))
* Separate `fit()` and `path()` methods ([#71](https://github.com/jolars/libslope/issues/71)) ([9712a19](https://github.com/jolars/libslope/commit/9712a1920f0668135eafc89be889642db54672cd))


### Performance Improvements

* vectorize binomial weight updates ([15964d6](https://github.com/jolars/libslope/commit/15964d69fbb71719d8cbb7b8b88f3831fe3734aa))

## [0.10.0](https://github.com/jolars/libslope/compare/v0.9.0...v0.10.0) (2025-02-18)


### ⚠ BREAKING CHANGES

* remove unused standardize argument

### Features

* enable early stopping criteria for path ([#64](https://github.com/jolars/libslope/issues/64)) ([6b387f5](https://github.com/jolars/libslope/commit/6b387f5c4d404e6970594939d0016d027c228c10))
* remove unused standardize argument ([fc6d468](https://github.com/jolars/libslope/commit/fc6d468463a5c3e591525167bb23f56b88d257e9))


### Bug Fixes

* correctly honor `coef_sparsity` and `x_sparsity` ([d65665d](https://github.com/jolars/libslope/commit/d65665dbfac0415e39291ade870a0c3cf7ea2db1))
* use matrix for beta in `generateData()` ([b4f1fd5](https://github.com/jolars/libslope/commit/b4f1fd50b8cd94d78a274f64a8dbdec09e48f05f))

## [0.9.0](https://github.com/jolars/libslope/compare/v0.8.0...v0.9.0) (2025-02-18)


### ⚠ BREAKING CHANGES

* reparameterize the penalty ([#55](https://github.com/jolars/libslope/issues/55))

### Code Refactoring

* reparameterize the penalty ([#55](https://github.com/jolars/libslope/issues/55)) ([713b892](https://github.com/jolars/libslope/commit/713b8923c6ed63d1d89a9b9a36c759dac554505a))

## [0.8.0](https://github.com/jolars/libslope/compare/v0.7.0...v0.8.0) (2025-02-17)


### Features

* add strong screening rule ([#53](https://github.com/jolars/libslope/issues/53)) ([5d4dee3](https://github.com/jolars/libslope/commit/5d4dee32e5dc78098d04f863499d0baed1c76c6f)), closes [#33](https://github.com/jolars/libslope/issues/33)


### Bug Fixes

* correctly create gradient for standardize JIT case ([b38ec31](https://github.com/jolars/libslope/commit/b38ec318631cf43fb87e1d381d9971ee23645d49))

## [0.7.0](https://github.com/jolars/libslope/compare/v0.6.0...v0.7.0) (2025-02-13)


### ⚠ BREAKING CHANGES

* use dynamic dispatch for solver setup
* explicitly instantiate solver matrix type combos

### Features

* explicitly instantiate solver matrix type combos ([ba722aa](https://github.com/jolars/libslope/commit/ba722aa24c2669db671a2121cb0f8df51330746a))
* raise learning rate in hybrid pgd method if accept ([fd2fd6c](https://github.com/jolars/libslope/commit/fd2fd6cebe00049123d7d473c6a6e1455145381b))
* use dynamic dispatch for solver setup ([36e32e6](https://github.com/jolars/libslope/commit/36e32e6167262f1ed294e3a5612c2972d5e1a4c4))

## [0.6.0](https://github.com/jolars/libslope/compare/v0.5.0...v0.6.0) (2025-02-12)


### ⚠ BREAKING CHANGES

* prepare api for multivariate families

### Features

* add oscar and gaussian paths ([#41](https://github.com/jolars/libslope/issues/41)) ([c3b2ae9](https://github.com/jolars/libslope/commit/c3b2ae98d83c007078a1667f41265ec88e80a31a))
* add PGD solver ([#45](https://github.com/jolars/libslope/issues/45)) ([fb58c0a](https://github.com/jolars/libslope/commit/fb58c0a130d6806c2608ae2593b61270dad5d910)), closes [#44](https://github.com/jolars/libslope/issues/44)
* add support for multinomial model ([#46](https://github.com/jolars/libslope/issues/46)) ([8498205](https://github.com/jolars/libslope/commit/8498205ec2192d6dd424bff636cef065cd478a7e))
* modularize solvers ([#43](https://github.com/jolars/libslope/issues/43)) ([08cddfb](https://github.com/jolars/libslope/commit/08cddfb463e61b4e58a6f8389c453af3020731aa)), closes [#42](https://github.com/jolars/libslope/issues/42)
* validate input to setters for Slope ([7b5bf66](https://github.com/jolars/libslope/commit/7b5bf661f923ad75d14741bdb71aa1cbede996cd))


### Code Refactoring

* prepare api for multivariate families ([3d196c0](https://github.com/jolars/libslope/commit/3d196c0a4f886222c2f9c0f361093a7ff311a116))

## [0.5.0](https://github.com/jolars/libslope/compare/v0.4.0...v0.5.0) (2025-01-31)


### Features

* add support for Poisson regression ([3c6529e](https://github.com/jolars/libslope/commit/3c6529e9cef1e6b587051e3775556d29eb738e95)), closes [#34](https://github.com/jolars/libslope/issues/34)


### Bug Fixes

* fix dual computation for binomial objective ([ba2fc31](https://github.com/jolars/libslope/commit/ba2fc311988a9be3e90ffc08a84ef9ea812059eb))

## [0.4.0](https://github.com/jolars/libslope/compare/v0.3.3...v0.4.0) (2025-01-29)


### Features

* allow `fit()` to modify X when standardizing ([5f2c757](https://github.com/jolars/libslope/commit/5f2c757d9879e321483b50ad3f0c3d4c71e6faac))


### Bug Fixes

* remove pragma from cpp file ([d19addf](https://github.com/jolars/libslope/commit/d19addfa1ab887af19a535dc5aeefe0696896438))


### Reverts

* "docs: document gaussian class" ([2ec919a](https://github.com/jolars/libslope/commit/2ec919a431d89428828234e28f1955c80a6bfb1f))

## [0.3.3](https://github.com/jolars/libslope/compare/v0.3.2...v0.3.3) (2025-01-28)


### Bug Fixes

* clamp index to avoid out-of-bounds indexing in thresholder ([7a1dbfa](https://github.com/jolars/libslope/commit/7a1dbfaff2feb3140c00b411962bbd34785d9cf8))
* fix binomial model ([2250f60](https://github.com/jolars/libslope/commit/2250f607374e90cc679768dab2043edb4b17f6ff))

## [0.3.2](https://github.com/jolars/libslope/compare/v0.3.1...v0.3.2) (2024-02-07)


### Bug Fixes

* fix error in Welford's standardization algorithm ([3d70d06](https://github.com/jolars/libslope/commit/3d70d06787c13a7dc8e48dcec8b6d0df4c21c91a))

## [0.3.1](https://github.com/jolars/libslope/compare/v0.3.0...v0.3.1) (2024-02-06)


### Bug Fixes

* avoid infinite loop by bounding stopping criterion ([209414c](https://github.com/jolars/libslope/commit/209414c388423bf71d9680ec4f2ce00751c3b9c2))

## [0.3.0](https://github.com/jolars/libslope/compare/v0.2.0...v0.3.0) (2023-12-11)


### ⚠ BREAKING CHANGES

* change interface to class-based design

### Features

* change interface to class-based design ([7b17034](https://github.com/jolars/libslope/commit/7b17034763186127b313aa858a4c7073395c4b97))
* modify the default parameters ([13f8efc](https://github.com/jolars/libslope/commit/13f8efc1e5b28f88196e20e931b3303e72be8c34))


### Bug Fixes

* setup sl1_norm after creating sequence ([ba47b6d](https://github.com/jolars/libslope/commit/ba47b6db8d18d12999f4ede4b96c1902e311c9a9))

## [0.2.0](https://github.com/jolars/libslope/compare/v0.1.0...v0.2.0) (2023-12-07)


### ⚠ BREAKING CHANGES

* change library name to just slope
* take parameters as a parameter pack
* support generation of BH-type lambda sequence

### Features

* add installation rule to install library and headers ([1523a53](https://github.com/jolars/libslope/commit/1523a53d65a63907dc39f02b29c7cd54a95a58a5))
* change library name to just slope ([f150fa7](https://github.com/jolars/libslope/commit/f150fa79eba4a01f36df8a62cd4f1b7a01b05336))
* return alpha and lambda from slope() ([d3294ed](https://github.com/jolars/libslope/commit/d3294edc9ab1f1d4539750070426b6698c15c518))
* return total number of iterations from algorithm ([e13ab6b](https://github.com/jolars/libslope/commit/e13ab6b77739f242a784685ef4ed0d330c0a825e))
* support generation of BH-type lambda sequence ([4570756](https://github.com/jolars/libslope/commit/4570756db4ceb8e84b3edc01e9b8550b3e151703))
* take parameters as a parameter pack ([60d7bde](https://github.com/jolars/libslope/commit/60d7bde4baf6bfe44da87d70d2a055717660c16e))
* use duality gap as stopping criterion ([7decb6a](https://github.com/jolars/libslope/commit/7decb6a264ddc15b051732fd536b189c84229c3a))


### Bug Fixes

* add beta0 via array interface ([31eed71](https://github.com/jolars/libslope/commit/31eed71b458b489594d4e99311336b12cda48c58))
* add minus one in division for alpha path computation ([905e522](https://github.com/jolars/libslope/commit/905e522217193e4e868fdf7a6e79d4da3c64f2e5))
* use autogenerated versions ([f9e09ab](https://github.com/jolars/libslope/commit/f9e09ab4fcf33c62fe52ab6fec1525bc6bcfb576))


### Reverts

* "refactor(tests): use folder-specific paths for tests" ([a6fb793](https://github.com/jolars/libslope/commit/a6fb793253b1e46ec46bbf2244c69e14a9348d8b))

## 0.1.0 (2023-11-28)


### ⚠ BREAKING CHANGES

* reorder arguments in function

### Features

* add Eigen to external libraries and set cmake path ([464b01b](https://github.com/jolars/libslope/commit/464b01b7bd177524631ab4d4cb74eacb274aefa2))
* don't build a cli binary ([a5c369c](https://github.com/jolars/libslope/commit/a5c369c747fd0d9b8356ff52a0f76f8f8bcee352))
* make library public ([e6e3bd9](https://github.com/jolars/libslope/commit/e6e3bd9a9c10196d289f50bd5184a71622c693e7))
* **python:** add a python binding ([6f3ff01](https://github.com/jolars/libslope/commit/6f3ff01d2aec80700af3674624b6cf6f804684b4))
* reorder arguments in function ([7ccdeea](https://github.com/jolars/libslope/commit/7ccdeea01ed02018e4f7d47a0b21c278d8c0cae2))
* upload WIP implementation of SLOPE ([cbe8ab3](https://github.com/jolars/libslope/commit/cbe8ab363628ef87b9f17561f239b38128f36da2))


### Bug Fixes

* **build:** fix yaml setting ([3e46077](https://github.com/jolars/libslope/commit/3e46077a31de99c20285950d549b6309aadc48a4))
* **ci:** stick to unstable dev mode ([732c519](https://github.com/jolars/libslope/commit/732c5197fd6948bc2a599173a239b14caacd4fdb))
