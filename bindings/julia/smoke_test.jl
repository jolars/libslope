using Test
using SparseArrays
using CxxWrap: StdVector

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

function model_arguments(x, y, loss; alpha = [0.05], path_length = UInt(3))
    return (y, alpha, Float64[], Int64(size(x, 1)), Int64(size(x, 2)),
            Int64(1), true, loss, "bh", "mean", "sd", path_length, 1e-7,
            Int64(10000), 0.1, Int64(size(x, 2) + 1), 0.0, 1.0, 0.1,
            "cyclical", Int32(42))
end

function feature_arguments(x::Matrix{Float64})
    return (x,)
end

function feature_arguments(x::SparseMatrixCSC{Float64})
    return (Int64.(x.colptr), Int64.(x.rowval), nonzeros(x))
end

function fit_model(x, y, loss; weights = nothing)
    output = (Float64[], Int64[], Int64[], Float64[], Int64[], Float64[], Float64[])
    path_length = issparse(x) ? Int64(3) : UInt(3)
    arguments = (feature_arguments(x)...,
                 model_arguments(x, y, loss; path_length)..., output...)
    if weights === nothing
        fit = issparse(x) ? LibSlope.fit_slope_sparse : LibSlope.fit_slope_dense
        fit(arguments...)
    else
        fit = issparse(x) ? LibSlope.fit_slope_sparse_weighted : LibSlope.fit_slope_dense_weighted
        fit(arguments..., weights)
    end
    return output
end

function cv_model(x, y, loss; weights = nothing)
    metric = loss == "quadratic" ? "mse" : "deviance"
    arguments = (feature_arguments(x)..., model_arguments(x, y, loss; alpha = Float64[])...,
                 Int64(3), Int64(1), metric, StdVector([0.1]), StdVector([1.0]),
                 StdVector(collect(Int64, 1:size(x, 1))))
    if weights === nothing
        cv = issparse(x) ? LibSlope.cv_slope_sparse : LibSlope.cv_slope_dense
        return cv(arguments...)
    end
    cv = issparse(x) ? LibSlope.cv_slope_sparse_weighted : LibSlope.cv_slope_dense_weighted
    return cv(arguments..., weights)
end

@testset "Observation weights" begin
    x = [-1.2 0.0; -0.8 0.2; -0.5 -0.7; -0.2 0.0;
          0.0 1.3; 0.2 -0.5; 0.4 0.8; 0.6 0.0;
          0.8 -0.2; 1.0 0.9; 1.2 -0.4; 1.4 0.3]
    weights = Float64[1, 2, 1, 3, 0, 1, 2, 1, 1, 2, 1, 1]
    rows = vcat((fill(i, Int(weights[i])) for i in eachindex(weights))...)
    responses = (
        "quadratic" => vec(0.3 .+ x[:, 1] .- 0.4 .* x[:, 2]),
        "logistic" => Float64[0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0],
        "poisson" => Float64[0, 1, 2, 0, 3, 1, 2, 1, 4, 2, 1, 3],
        "multinomial" => Float64[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
    )

    for (loss, y) in responses
        @testset "$loss" begin
            weighted_fit = fit_model(x, y, loss; weights)
            replicated_fit = fit_model(x[rows, :], y[rows], loss)
            sparse_fit = fit_model(sparse(x), y, loss; weights)
            rescaled_fit = fit_model(x, y, loss; weights = 7 .* weights)
            uniform_fit = fit_model(x, y, loss; weights = ones(size(x, 1)))
            unweighted_fit = fit_model(x, y, loss)
            for i in eachindex(weighted_fit)
                @test weighted_fit[i] ≈ replicated_fit[i] atol=1e-5
                @test weighted_fit[i] ≈ sparse_fit[i] atol=1e-5
                @test weighted_fit[i] ≈ rescaled_fit[i] atol=1e-6
                @test uniform_fit[i] ≈ unweighted_fit[i] atol=1e-6
            end

            weighted_cv = cv_model(x, y, loss; weights)
            sparse_cv = cv_model(sparse(x), y, loss; weights)
            rescaled_cv = cv_model(x, y, loss; weights = 7 .* weights)
            uniform_cv = cv_model(x, y, loss; weights = ones(size(x, 1)))
            unweighted_cv = cv_model(x, y, loss)
            for i in eachindex(weighted_cv)
                @test weighted_cv[i] ≈ sparse_cv[i] atol=1e-5
                @test weighted_cv[i] ≈ rescaled_cv[i] atol=1e-5
                @test uniform_cv[i] ≈ unweighted_cv[i] atol=1e-6
            end
        end
    end

    @testset "Invalid weights" begin
        y = first(responses).second
        invalid_weights = (ones(size(x, 1) - 1), zeros(size(x, 1)),
                           fill(-1.0, size(x, 1)), fill(NaN, size(x, 1)),
                           fill(Inf, size(x, 1)))
        for features in (x, sparse(x)), invalid in invalid_weights
            @test_throws ErrorException fit_model(features, y, "quadratic"; weights = invalid)
            @test_throws ErrorException cv_model(features, y, "quadratic"; weights = invalid)
        end
    end
end
