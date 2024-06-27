module JudiLing

using DataFrames
using Random, Distributions
using SparseArrays, LinearAlgebra, Statistics, SuiteSparse
using BenchmarkTools
using DataStructures
using ProgressMeter
using CSV
using GZip
using Embeddings
using BSON: @save, @load
using Requires
using DataLoaders
using Distances

function __init__()
    @require PyCall="438e738f-606a-5dbb-bf0a-cddfbfd45ab0" begin
     using .PyCall
     @eval include("pickle.jl")
     @eval include("pyndl.jl")
 end
 @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" @eval include("deep_learning.jl")
 ProgressMeter.ijulia_behavior(:clear)
end

include("utils.jl")
include("wh.jl")
include("make_cue_matrix.jl")
include("make_semantic_matrix.jl")
include("cholesky.jl")
include("make_adjacency_matrix.jl")
include("make_yt_matrix.jl")
include("find_path.jl")
include("eval.jl")
include("output.jl")
include("preprocess.jl")
include("test_combo.jl")
include("display.jl")
include("input.jl")

end
