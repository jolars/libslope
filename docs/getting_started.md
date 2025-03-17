# Getting Started {#getting_started}

First, we define our model. Let's use logistic regression, by setting the
loss to `"logistic"`.

```cpp
#include "slope.h"

Slope::Model model;

model.setLoss("logistic");
```

Next, we set the data matrix `x` and the response vector `y`. Here we use some
toy data.

```cpp
Eigen::MatrixXd x(3, 2);
Eigen::VectorXd y(3);

x << 1.1, 2.3, 0.2, 1.5, 0.5, 0.2;
y << 0, 1, 0;
```

Finally, we call the `path()` method to fit the full SLOPE path.

```cpp
auto res = model.path(x, y);
```

Now we can retrieve the coefficients by calling `res.getCoefs()`.
