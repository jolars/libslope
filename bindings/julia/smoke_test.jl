using Test

module LibSlope
using CxxWrap

const library = abspath(ARGS[1])

@wrapmodule(() -> library, :define_julia_module)

function __init__()
    @initcxx
end
end

predictions, n_columns =
    LibSlope.slope_predict(reshape([0.0, 1.0], 2, 1), 2, 1, "logistic")

@test predictions == [0.0, 1.0]
@test n_columns == 1
