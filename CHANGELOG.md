# Changelog

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
