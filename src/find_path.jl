"""
Store paths' information built by `learn_paths` or `build_paths`
"""
struct Result_Path_Info_Struct
    ngrams_ind::Array
    num_tolerance::Int64
    support::Float64
end

"""
Store gold paths' information including indices and indices' support and total
support. It can be used to evaluate how low the threshold needs to be set in
order to find most of the correct paths or if set very low, all of the correct paths.
"""
struct Gold_Path_Info_Struct
    ngrams_ind::Array
    ngrams_ind_support::Array
    support::Float64
end

"""
Store threshold and tolerance proportional for each timestep.
"""
struct Threshold_Stat_Struct
    threshold::Float64
    threshold_prop::Array
    tolerance::Float64
    tolerance_prop::Array
    n_rows::Int64
    timestep::Int64
end

"""
A sequence finding algorithm using discrimination learning to predict, for a given
word, which n-grams are best supported for a given position in the sequence of n-grams.
"""
function learn_paths end

"""
The build_paths function constructs paths by only considering those n-grams that are
close to the target. It first takes the predicted c-hat vector and finds the
closest n neighbors in the C matrix. Then it selects all n-grams of these neighbors,
and constructs all valid paths with those n-grams. The path producing the best
correlation with the target semantic vector (through synthesis by analysis) is selected.
"""
function build_paths end

"""
    learn_paths(
        data_train::DataFrame,
        data_val::DataFrame,
        C_train::Union{Matrix, SparseMatrixCSC},
        S_val::Union{Matrix, SparseMatrixCSC},
        F_train,
        Chat_val::Union{Matrix, SparseMatrixCSC},
        A::SparseMatrixCSC,
        i2f::Dict,
        f2i::Dict;
        gold_ind::Union{Nothing, Vector} = nothing,
        Shat_val::Union{Nothing, Matrix} = nothing,
        check_gold_path::Bool = false,
        max_t::Int = 15,
        max_can::Int = 10,
        threshold::Float64 = 0.1,
        is_tolerant::Bool = false,
        tolerance::Float64 = (-1000.0),
        max_tolerance::Int = 3,
        grams::Int = 3,
        tokenized::Bool = false,
        sep_token::Union{Nothing, String} = nothing,
        keep_sep::Bool = false,
        target_col::Union{Symbol, String} = "Words",
        start_end_token::String = "#",
        issparse::Union{Symbol, Bool} = :auto,
        sparse_ratio::Float64 = 0.05,
        if_pca::Bool = false,
        pca_eval_M::Union{Nothing, Matrix} = nothing,
        activation::Union{Nothing, Function} = nothing,
        ignore_nan::Bool = true,
        check_threshold_stat::Bool = false,
        verbose::Bool = false
    )

A sequence finding algorithm using discrimination learning to predict, for a given
word, which n-grams are best supported for a given position in the sequence of n-grams.

# Obligatory Arguments
- `data::DataFrame`: the training dataset
- `data_val::DataFrame`: the validation dataset
- `C_train::Union{SparseMatrixCSC, Matrix}`: the C matrix for training dataset
- `S_val::Union{SparseMatrixCSC, Matrix}`: the S matrix for validation dataset
- `F_train::Union{SparseMatrixCSC, Matrix, Chain}`: the F matrix for training dataset, or a deep learning comprehension model trained on the training data
- `Chat_val::Union{SparseMatrixCSC, Matrix}`: the Chat matrix for validation dataset
- `A::SparseMatrixCSC`: the adjacency matrix
- `i2f::Dict`: the dictionary returning features given indices
- `f2i::Dict`: the dictionary returning indices given features

# Optional Arguments
- `gold_ind::Union{Nothing, Vector}=nothing`: gold paths' indices
- `Shat_val::Union{Nothing, Matrix}=nothing`: the Shat matrix for the validation dataset
- `check_gold_path::Bool=false`: if true, return a list of support values for the gold path; this information is returned as second output value
- `max_t::Int64=15`: maximum timestep
- `max_can::Int64=10`: maximum number of candidates to consider
- `threshold::Float64=0.1`:the value set for the support such that if the support of an n-gram is higher than this value, the n-gram will be taking into consideration
- `is_tolerant::Bool=false`: if true, select a specified number (given by `max_tolerance`) of n-grams whose supports are below threshold but above a second tolerance threshold to be added to the path
- `tolerance::Float64=(-1000.0)`: the value set for the second threshold (in tolerant mode) such that if the support for an n-gram is in between this value and the threshold and the max_tolerance number has not been reached, then allow this n-gram to be added to the path
- `max_tolerance::Int64=4`: maximum number of n-grams allowed in a path
- `grams::Int64=3`: the number n of grams that make up an n-gram
- `tokenized::Bool=false`: if true, the dataset target is tokenized
- `sep_token::Union{Nothing, String, Char}=nothing`: separator token
- `keep_sep::Bool=false`:if true, keep separators in cues
- `target_col::Union{String, :Symbol}=:Words`: the column name for target strings
- `start_end_token::Union{String, Char}="#"`: start and end token in boundary cues
- `issparse::Union{Symbol, Bool}=:auto`: control of whether output of Mt matrix is a dense matrix or a sparse matrix
- `sparse_ratio::Float64=0.05`: the ratio to decide whether a matrix is sparse
- `if_pca::Bool=false`: turn on to enable pca mode
- `pca_eval_M::Matrix=nothing`: pass original F for pca mode
- `activation::Function=nothing`: the activation function you want to pass
- `ignore_nan::Bool=true`: whether to ignore NaN when compare correlations, otherwise NaN will be selected as the max correlation value
- `check_threshold_stat::Bool=false`: if true, return a threshold and torlerance proportion for each timestep
- `verbose::Bool=false`: if true, more information is printed

# Examples
```julia
# basic usage without tokenization
res = JudiLing.learn_paths(
latin,
latin,
cue_obj.C,
S,
F,
Chat,
A,
cue_obj.i2f,
max_t=max_t,
max_can=10,
grams=3,
threshold=0.1,
tokenized=false,
keep_sep=false,
target_col=:Word,
verbose=true)

# basic usage with tokenization
res = JudiLing.learn_paths(
french,
french,
cue_obj.C,
S,
F,
Chat,
A,
cue_obj.i2f,
max_t=max_t,
max_can=10,
grams=3,
threshold=0.1,
tokenized=true,
sep_token="-",
keep_sep=true,
target_col=:Syllables,
verbose=true)

# basic usage for validation data
res_val = JudiLing.learn_paths(
latin_train,
latin_val,
cue_obj_train.C,
S_val,
F_train,
Chat_val,
A,
cue_obj_train.i2f,
max_t=max_t,
max_can=10,
grams=3,
threshold=0.1,
tokenized=false,
keep_sep=false,
target_col=:Word,
verbose=true)

# turn on tolerance mode
res_val = JudiLing.learn_paths(
...
threshold=0.1,
is_tolerant=true,
tolerance=-0.1,
max_tolerance=4,
...)

# turn on check gold paths mode
res_train, gpi_train = JudiLing.learn_paths(
...
gold_ind=cue_obj_train.gold_ind,
Shat_val=Shat_train,
check_gold_path=true,
...)

res_val, gpi_val = JudiLing.learn_paths(
...
gold_ind=cue_obj_val.gold_ind,
Shat_val=Shat_val,
check_gold_path=true,
...)

# control over sparsity
res_val = JudiLing.learn_paths(
...
issparse=:auto,
sparse_ratio=0.05,
...)

# pca mode
res_learn = JudiLing.learn_paths(
korean,
korean,
Array(Cpcat),
S,
F,
ChatPCA,
A,
cue_obj.i2f,
cue_obj.f2i,
check_gold_path=false,
gold_ind=cue_obj.gold_ind,
Shat_val=Shat,
max_t=max_t,
max_can=10,
grams=3,
threshold=0.1,
tokenized=true,
sep_token="_",
keep_sep=true,
target_col=:Verb_syll,
if_pca=true,
pca_eval_M=Fo,
verbose=true);

```
"""
function learn_paths(
    data_train::DataFrame,
    data_val::DataFrame,
    C_train::Union{Matrix, SparseMatrixCSC},
    S_val::Union{Matrix, SparseMatrixCSC},
    F_train,
    Chat_val::Union{Matrix, SparseMatrixCSC},
    A::SparseMatrixCSC,
    i2f::Dict,
    f2i::Dict;
    gold_ind::Union{Nothing, Vector} = nothing,
    Shat_val::Union{Nothing, Matrix} = nothing,
    check_gold_path::Bool = false,
    max_t::Int = 15,
    max_can::Int = 10,
    threshold::Float64 = 0.1,
    is_tolerant::Bool = false,
    tolerance::Float64 = (-1000.0),
    max_tolerance::Int = 3,
    grams::Int = 3,
    tokenized::Bool = false,
    sep_token::Union{Nothing, String} = nothing,
    keep_sep::Bool = false,
    target_col::Union{Symbol, String} = "Words",
    start_end_token::String = "#",
    issparse::Union{Symbol, Bool} = :auto,
    sparse_ratio::Float64 = 0.05,
    if_pca::Bool = false,
    pca_eval_M::Union{Nothing, Matrix} = nothing,
    activation::Union{Nothing, Function} = nothing,
    ignore_nan::Bool = true,
    check_threshold_stat::Bool = false,
    verbose::Bool = false
)

    # initialize queues for storing paths
    n_val = size(data_val, 1)
    # working_q = Array{Queue{Array{Int64,1}},1}(undef, n_val)
    working_q = Vector{Queue{Tuple{Vector{Int64},Int64}}}(undef, n_val)
    # res = Array{Array{Array{Int64}},1}(undef, n_val)
    res = Vector{Vector{Tuple{Vector{Int64},Int64}}}(undef, n_val)
    for j = 1:n_val
        res[j] = Tuple{Vector{Int64},Int64}[]
    end

    # store thr proportion for each timestep and each utterance
    thr_stat = zeros(Float64, (n_val, max_t))
    tlr_stat = zeros(Float64, (n_val, max_t))

    # # initialize gold_path_info supports
    if check_gold_path && !isnothing(gold_ind)
        gold_path_info_supports =
            Vector{Vector{Float64}}(undef, size(data_val, 1))
        for i = 1:n_val
            gi = gold_ind[i]
            gold_path_info_supports[i] = Vector{Float64}(undef, length(gi))
        end
    end

    verbose && println("Making fac C")
    fac_C_train = make_transform_fac(C_train)

    C_dim = size(Chat_val, 2)

    for i = 1:max_t
        verbose && println("="^10)
        verbose && println("Timestep $i")
        verbose &&
            i >= 2 &&
            println("average $(mean(length.(working_q))) of paths currently")

        verbose && println("Calculating Yt...")
        Yt_train = make_Yt_matrix(
            i,
            data_train,
            f2i,
            grams = grams,
            target_col = target_col,
            tokenized = tokenized,
            sep_token = sep_token,
            keep_sep = keep_sep,
            start_end_token = start_end_token
        )

        verbose && println("Calculating Mt...")
        Mt_train = make_transform_matrix(
            fac_C_train,
            C_train,
            Yt_train,
            output_format = issparse,
            sparse_ratio = sparse_ratio,
            verbose = verbose,
        )
        # Mt = sparse(Mt)
        # verbose && println("Sparsity of Mt: $(length(Mt.nzval)/Mt.m/Mt.n)")

        verbose && println("Calculating Ythat...")
        # Ythat_val = Matrix(Chat_val)*Matrix(Mt_train)
        Ythat_val = Chat_val * Mt_train
        if is_truly_sparse(Ythat_val, verbose = verbose)
            Ythat_val = sparse(Ythat_val)
        end

        # apply activation to Yt hat
        if !isnothing(activation)
            Ythat_val = activation.(Ythat_val)
        end
        # collect supports for gold path each timestep
        if check_gold_path && !isnothing(gold_ind)
            for j = 1:size(data_val, 1)
                gi = gold_ind[j]
                if i <= length(gi)
                    gold_path_info_supports[j][i] = Ythat_val[j, gi[i]]
                end
            end
        end

        verbose && println("Finding paths...")
        iter = 1:n_val
        if verbose
            pb = Progress(n_val)
        end

        for j in iter
            # collect all n-grams which has greater support than the threshold
            candidates_t = findall(x -> x > threshold, Ythat_val[j, :])
            candidates_t_tlr =
                findall(x -> x > tolerance && x <= threshold, Ythat_val[j, :])

            # calculate threshold stat
            if check_threshold_stat
                prop_candidates_t = length(candidates_t) / C_dim
                prop_candidates_t_tlr = length(candidates_t_tlr) / C_dim
                thr_stat[j, i] = prop_candidates_t
                tlr_stat[j, i] = prop_candidates_t_tlr
            end

            # for timestep 2 and after 2
            if isassigned(working_q, j)
                tmp_working_q = Queue{Tuple{Vector{Int64},Int64}}()
                while !isempty(working_q[j])
                    a = dequeue!(working_q[j]) ## a = [11] A[11,5] == 1 # candidates = [1, 5, 7]

                    for c in candidates_t ## c = 5 # a = [11, 1, 5, 7] # a = [11, 1] [11, 5] [11, 7]
                        # if a n-grams is attachable then append it
                        if isattachable(a[1], c, A)
                            a_copy = deepcopy(a[1])
                            push!(a_copy, c)
                            # if the path is complete then move it to result list
                            if iscomplete(
                                a_copy,
                                i2f,
                                tokenized = tokenized,
                                sep_token = sep_token,
                                start_end_token = start_end_token,
                            )
                                push!(res[j], (a_copy, a[2]))
                            else
                                # otherwise enqueue it to the next timestep
                                enqueue!(tmp_working_q, (a_copy, a[2]))
                            end
                        end
                    end

                    if is_tolerant && a[2] < max_tolerance
                        for c in candidates_t_tlr ## c = 5 # a = [11, 1, 5, 7] # a = [11, 1] [11, 5] [11, 7]
                            # if a n-grams is attachable then append it
                            if isattachable(a[1], c, A)
                                a_copy = deepcopy(a[1])
                                push!(a_copy, c)
                                # if the path is complete then move it to result list
                                if iscomplete(
                                    a_copy,
                                    i2f,
                                    tokenized = tokenized,
                                    sep_token = sep_token,
                                    start_end_token = start_end_token,
                                )
                                    push!(res[j], (a_copy, a[2] + 1))
                                else
                                    # otherwise enqueue it to the next timestep
                                    enqueue!(tmp_working_q, (a_copy, a[2] + 1))
                                end
                            end
                        end
                    end

                end

                # refresh queue for the next timestep
                working_q[j] = tmp_working_q
                # for timestep 1
            else
                working_q[j] = Queue{Tuple{Vector{Int64},Int64}}()
                for c in candidates_t
                    # check whether a n-gram is a start n-gram
                    if isstart(
                        c,
                        i2f,
                        tokenized = tokenized,
                        sep_token = sep_token,
                        start_end_token = start_end_token
                    )
                        a = Int64[]
                        push!(a, c)
                        # check whether this n-gram is both start and complete
                        if iscomplete(
                            a,
                            i2f,
                            tokenized = tokenized,
                            sep_token = sep_token,
                            start_end_token = start_end_token,
                        )
                            push!(res[j], (a, 0))
                        else
                            enqueue!(working_q[j], (a, 0))
                        end
                    end
                end

                if is_tolerant && 0 < max_tolerance
                    for c in candidates_t_tlr
                        # check whether a n-gram is a start n-gram
                        if isstart(
                            c,
                            i2f,
                            tokenized = tokenized,
                            sep_token = sep_token,
                            start_end_token = start_end_token,
                        )
                            a = Int64[]
                            push!(a, c)
                            # check whether this n-gram is both start and complete
                            if iscomplete(
                                a,
                                i2f,
                                tokenized = tokenized,
                                sep_token = sep_token,
                                start_end_token = start_end_token,
                            )
                                push!(res[j], (a, 1))
                            else
                                enqueue!(working_q[j], (a, 1))
                            end
                        end
                    end
                end
            end
            flush(stdout)
            if verbose
                ProgressMeter.next!(pb)
            end
        end
    end

    ts = Threshold_Stat_Struct(threshold, thr_stat, tolerance, tlr_stat, n_val, max_t)

    verbose && println("Evaluating paths...")
    res =
        eval_can(res, S_val, F_train, i2f, max_can, if_pca, pca_eval_M, ignore_nan, verbose)

    # initialize gpi
    if check_gold_path
        if isnothing(gold_ind)
            throw(ErrorException("gold_ind is nothing! Perhaps you forgot to pass gold_ind as an argument."))
        end

        if isnothing(Shat_val)
            throw(ErrorException("Shat_val is nothing! Perhaps you forgot to pass Shat_val as an argument."))
        end

        gpi =
            Vector{Gold_Path_Info_Struct}(undef, size(data_val, 1))

        # calculate all shat correlation with S
        Scors = [cor(Shat_val[i, :], S_val[i, :]) for i = 1:n_val]

        for i = 1:size(data_val, 1)
            gpi[i] = Gold_Path_Info_Struct(
                gold_ind[i],
                gold_path_info_supports[i],
                Scors[i],
            )
        end
    end

    if check_gold_path
        if check_threshold_stat
            return res, gpi, ts
        else
            return res, gpi
        end
    else
        if check_threshold_stat
            return res, ts
        else
            return res
        end
    end
end

"""
    learn_paths(
        data::DataFrame,
        cue_obj::Cue_Matrix_Struct,
        S_val::Union{SparseMatrixCSC, Matrix},
        F_train,
        Chat_val::Union{SparseMatrixCSC, Matrix};
        Shat_val::Union{Nothing, Matrix} = nothing,
        check_gold_path::Bool = false,
        threshold::Float64 = 0.1,
        is_tolerant::Bool = false,
        tolerance::Float64 = (-1000.0),
        max_tolerance::Int = 3,
        activation::Union{Nothing, Function} = nothing,
        ignore_nan::Bool = true,
        verbose::Bool = true)

A high-level wrapper function for `learn_paths` with much less control. It aims
for users who is very new to JudiLing and `learn_paths` function.

# Obligatory Arguments
- `data::DataFrame`: the training dataset
- `cue_obj::Cue_Matrix_Struct`: the C matrix object containing all information with C
- `S_val::Union{SparseMatrixCSC, Matrix}`: the S matrix for validation dataset
- `F_train::Union{SparseMatrixCSC, Matrix, Chain}`: either the F matrix for training dataset, or a deep learning comprehension model trained on the training set
- `Chat_val::Union{SparseMatrixCSC, Matrix}`: the Chat matrix for validation dataset

# Optional Arguments
- `Shat_val::Union{Nothing, Matrix}=nothing`: the Shat matrix for the validation dataset
- `check_gold_path::Bool=false`: if true, return a list of support values for the gold path; this information is returned as second output value
- `threshold::Float64=0.1`:the value set for the support such that if the support of an n-gram is higher than this value, the n-gram will be taking into consideration
- `is_tolerant::Bool=false`: if true, select a specified number (given by `max_tolerance`) of n-grams whose supports are below threshold but above a second tolerance threshold to be added to the path
- `tolerance::Float64=(-1000.0)`: the value set for the second threshold (in tolerant mode) such that if the support for an n-gram is in between this value and the threshold and the max_tolerance number has not been reached, then allow this n-gram to be added to the path
- `max_tolerance::Int64=4`: maximum number of n-grams allowed in a path
- `activation::Function=nothing`: the activation function you want to pass
- `ignore_nan::Bool=true`: whether to ignore NaN when compare correlations, otherwise NaN will be selected as the max correlation value
- `verbose::Bool=false`: if true, more information is printed

# Examples
```julia
res = learn_paths(latin, cue_obj, S, F, Chat)
```
"""
function learn_paths(
    data::DataFrame,
    cue_obj::Cue_Matrix_Struct,
    S_val::Union{SparseMatrixCSC, Matrix},
    F_train,
    Chat_val::Union{SparseMatrixCSC, Matrix};
    Shat_val::Union{Nothing, Matrix} = nothing,
    check_gold_path::Bool = false,
    threshold::Float64 = 0.1,
    is_tolerant::Bool = false,
    tolerance::Float64 = (-1000.0),
    max_tolerance::Int = 3,
    activation::Union{Nothing, Function} = nothing,
    ignore_nan::Bool = true,
    verbose::Bool = true)

    max_t = JudiLing.cal_max_timestep(data, cue_obj.target_col,
        tokenized=cue_obj.tokenized, sep_token=cue_obj.sep_token)

    learn_paths(
        data,
        data,
        cue_obj.C,
        S_val,
        F_train,
        Chat_val,
        cue_obj.A,
        cue_obj.i2f,
        cue_obj.f2i;
        gold_ind = cue_obj.gold_ind,
        Shat_val = Shat_val,
        check_gold_path = check_gold_path,
        max_t = max_t,
        max_can = 10,
        threshold = threshold,
        is_tolerant = is_tolerant,
        tolerance = tolerance,
        max_tolerance = max_tolerance,
        grams = cue_obj.grams,
        tokenized = cue_obj.tokenized,
        sep_token = cue_obj.sep_token,
        keep_sep = cue_obj.keep_sep,
        target_col = cue_obj.target_col,
        activation = activation,
        ignore_nan = ignore_nan,
        verbose = verbose,
    )
end

"""
    build_paths(
        data_val,
        C_train,
        S_val,
        F_train,
        Chat_val,
        A,
        i2f,
        C_train_ind;
        rC = nothing,
        max_t = 15,
        max_can = 10,
        n_neighbors = 10,
        grams = 3,
        tokenized = false,
        sep_token = nothing,
        target_col = :Words,
        start_end_token = "#",
        if_pca = false,
        pca_eval_M = nothing,
        ignore_nan = true,
        verbose = false,
    )

The build_paths function constructs paths by only considering those n-grams that are
close to the target. It first takes the predicted c-hat vector and finds the
closest n neighbors in the C matrix. Then it selects all n-grams of these neighbors,
and constructs all valid paths with those n-grams. The path producing the best
correlation with the target semantic vector (through synthesis by analysis) is selected.

# Obligatory Arguments
- `data::DataFrame`: the training dataset
- `data_val::DataFrame`: the validation dataset
- `C_train::SparseMatrixCSC`: the C matrix for the training dataset
- `S_val::Union{SparseMatrixCSC, Matrix}`: the S matrix for the validation dataset
- `F_train::Union{SparseMatrixCSC, Matrix}`: the F matrix for the training dataset
- `Chat_val::Matrix`: the Chat matrix for the validation dataset
- `A::SparseMatrixCSC`: the adjacency matrix
- `i2f::Dict`: the dictionary returning features given indices
- `C_train_ind::Array`: the gold paths' indices for the training dataset

# Optional Arguments
- `rC::Union{Nothing, Matrix}=nothing`: correlation Matrix of C and Chat, specify to save computing time
- `max_t::Int64=15`: maximum number of timesteps
- `max_can::Int64=10`: maximum number of candidates to consider
- `n_neighbors::Int64=10`: the top n form neighbors to be considered
- `grams::Int64=3`: the number n of grams that make up n-grams
- `tokenized::Bool=false`: if true, the dataset target is tokenized
- `sep_token::Union{Nothing, String, Char}=nothing`: separator
- `target_col::Union{String, :Symbol}=:Words`: the column name for target strings
- `if_pca::Bool=false`: turn on to enable pca mode
- `pca_eval_M::Matrix=nothing`: pass original F for pca mode
- `verbose::Bool=false`: if true, more information will be printed

# Examples
```julia
# training dataset
JudiLing.build_paths(
    latin_train,
    cue_obj_train.C,
    S_train,
    F_train,
    Chat_train,
    A,
    cue_obj_train.i2f,
    cue_obj_train.gold_ind,
    max_t=max_t,
    n_neighbors=10,
    verbose=false
    )

# validation dataset
JudiLing.build_paths(
    latin_val,
    cue_obj_train.C,
    S_val,
    F_train,
    Chat_val,
    A,
    cue_obj_train.i2f,
    cue_obj_train.gold_ind,
    max_t=max_t,
    n_neighbors=10,
    verbose=false
    )

# pca mode
res_build = JudiLing.build_paths(
    korean,
    Array(Cpcat),
    S,
    F,
    ChatPCA,
    A,
    cue_obj.i2f,
    cue_obj.gold_ind,
    max_t=max_t,
    if_pca=true,
    pca_eval_M=Fo,
    n_neighbors=3,
    verbose=true
    )
```
"""
function build_paths(
    data_val,
    C_train,
    S_val,
    F_train,
    Chat_val,
    A,
    i2f,
    C_train_ind;
    rC = nothing,
    max_t = 15,
    max_can = 10,
    n_neighbors = 10,
    grams = 3,
    tokenized = false,
    sep_token = nothing,
    target_col = :Words,
    start_end_token = "#",
    if_pca = false,
    pca_eval_M = nothing,
    ignore_nan = true,
    verbose = false,
)
    # initialize queues for storing paths
    n_val = size(data_val, 1)
    # working_q = Array{Queue{Array{Int64,1}},1}(undef, n_val)
    # res = Array{Array{Array{Int64}},1}(undef, n_val)
    res = Vector{Vector{Tuple{Vector{Int64},Int64}}}(undef, n_val)

    for j = 1:n_val
        res[j] = Tuple{Vector{Int64},Int64}[]
    end

    verbose && println("Finding all top features..")
    # findall features indices for all utterances
    isnothing(rC) && begin
        rC = cor(Chat_val, Matrix(C_train), dims = 2)
    end
    top_indices = find_top_feature_indices(
        # C_train,
        # Chat_val,
        rC,
        C_train_ind,
        n_neighbors = n_neighbors,
        verbose = verbose,
    )

    # verbose && println("="^10)
    # verbose && println("Timestep $i")

    verbose && println("Finding paths...")
    iter = 1:n_val
    if verbose
        pb = Progress(n_val)
    end
    for j in iter
        candidates_t = top_indices[j]

        # timestep 1
        working_q = Queue{Array{Int64,1}}()
        for c in candidates_t
            # check whether a n-gram is a start n-gram
            if isstart(c,
                i2f,
                tokenized = tokenized,
                sep_token = sep_token,
                start_end_token = start_end_token,
            )

                a = Int64[]
                push!(a, c)
                # check whether this n-gram is both start and complete
                if iscomplete(
                    a,
                    i2f,
                    tokenized = tokenized,
                    sep_token = sep_token,
                    start_end_token = start_end_token,
                )
                    push!(res[j], (a, 0))
                else
                    enqueue!(working_q, a)
                end
            end
        end

        for i = 2:max_t
            tmp_working_q = Queue{Array{Int64,1}}()
            while !isempty(working_q)
                a = dequeue!(working_q) ## a = [11] A[11,5] == 1 # candidates = [1, 5, 7]
                for c in candidates_t ## c = 5 # a = [11, 1, 5, 7] # a = [11, 1] [11, 5] [11, 7]
                    # if a n-grams is attachable then append it
                    if isattachable(a, c, A)
                        a_copy = deepcopy(a)
                        push!(a_copy, c)
                        # if the path is complete then move it to result list
                        if iscomplete(
                            a_copy,
                            i2f,
                            tokenized = tokenized,
                            sep_token = sep_token,
                            start_end_token = start_end_token,
                        )
                            push!(res[j], (a_copy, 0))
                        else
                            # otherwise enqueue it to the next timestep
                            enqueue!(tmp_working_q, a_copy)
                        end
                    end
                end
            end

            # refresh queue for the next timestep
            working_q = tmp_working_q
        end
        if verbose
            ProgressMeter.next!(pb)
        end
    end

    verbose && println("Evaluating paths...")
    eval_can(res, S_val, F_train, i2f, max_can, if_pca, pca_eval_M, ignore_nan, verbose)
end

"""
    learn_paths_rpi(
        data_train::DataFrame,
        data_val::DataFrame,
        C_train::Union{Matrix, SparseMatrixCSC},
        S_val::Union{Matrix, SparseMatrixCSC},
        F_train,
        Chat_val::Union{Matrix, SparseMatrixCSC},
        A::SparseMatrixCSC,
        i2f::Dict,
        f2i::Dict;
        gold_ind::Union{Nothing, Vector} = nothing,
        Shat_val::Union{Nothing, Matrix} = nothing,
        check_gold_path::Bool = false,
        max_t::Int = 15,
        max_can::Int = 10,
        threshold::Float64 = 0.1,
        is_tolerant::Bool = false,
        tolerance::Float64 = (-1000.0),
        max_tolerance::Int = 3,
        grams::Int = 3,
        tokenized::Bool = false,
        sep_token::Union{Nothing, String} = nothing,
        keep_sep::Bool = false,
        target_col::Union{Symbol, String} = "Words",
        start_end_token::String = "#",
        issparse::Union{Symbol, Bool} = :auto,
        sparse_ratio::Float64 = 0.05,
        if_pca::Bool = false,
        pca_eval_M::Union{Nothing, Matrix} = nothing,
        activation::Union{Nothing, Function} = nothing,
        ignore_nan::Bool = true,
        check_threshold_stat::Bool = false,
        verbose::Bool = false
    )

Calculate learn_paths with results indices supports as well.

# Obligatory Arguments
- `data::DataFrame`: the training dataset
- `data_val::DataFrame`: the validation dataset
- `C_train::Union{SparseMatrixCSC, Matrix}`: the C matrix for training dataset
- `S_val::Union{SparseMatrixCSC, Matrix}`: the S matrix for validation dataset
- `F_train::Union{SparseMatrixCSC, Matrix, Chain}`: the F matrix for training dataset, or a deep learning comprehension model trained on the training data
- `Chat_val::Union{SparseMatrixCSC, Matrix}`: the Chat matrix for validation dataset
- `A::SparseMatrixCSC`: the adjacency matrix
- `i2f::Dict`: the dictionary returning features given indices
- `f2i::Dict`: the dictionary returning indices given features

# Optional Arguments
- `gold_ind::Union{Nothing, Vector}=nothing`: gold paths' indices
- `Shat_val::Union{Nothing, Matrix}=nothing`: the Shat matrix for the validation dataset
- `check_gold_path::Bool=false`: if true, return a list of support values for the gold path; this information is returned as second output value
- `max_t::Int64=15`: maximum timestep
- `max_can::Int64=10`: maximum number of candidates to consider
- `threshold::Float64=0.1`:the value set for the support such that if the support of an n-gram is higher than this value, the n-gram will be taking into consideration
- `is_tolerant::Bool=false`: if true, select a specified number (given by `max_tolerance`) of n-grams whose supports are below threshold but above a second tolerance threshold to be added to the path
- `tolerance::Float64=(-1000.0)`: the value set for the second threshold (in tolerant mode) such that if the support for an n-gram is in between this value and the threshold and the max_tolerance number has not been reached, then allow this n-gram to be added to the path
- `max_tolerance::Int64=4`: maximum number of n-grams allowed in a path
- `grams::Int64=3`: the number n of grams that make up an n-gram
- `tokenized::Bool=false`: if true, the dataset target is tokenized
- `sep_token::Union{Nothing, String, Char}=nothing`: separator token
- `keep_sep::Bool=false`:if true, keep separators in cues
- `target_col::Union{String, :Symbol}=:Words`: the column name for target strings
- `start_end_token::Union{String, Char}="#"`: start and end token in boundary cues
- `issparse::Union{Symbol, Bool}=:auto`: control of whether output of Mt matrix is a dense matrix or a sparse matrix
- `sparse_ratio::Float64=0.05`: the ratio to decide whether a matrix is sparse
- `if_pca::Bool=false`: turn on to enable pca mode
- `pca_eval_M::Matrix=nothing`: pass original F for pca mode
- `activation::Function=nothing`: the activation function you want to pass
- `ignore_nan::Bool=true`: whether to ignore NaN when compare correlations, otherwise NaN will be selected as the max correlation value
- `check_threshold_stat::Bool=false`: if true, return a threshold and torlerance proportion for each timestep
- `verbose::Bool=false`: if true, more information is printed
"""
function learn_paths_rpi(
    data_train::DataFrame,
    data_val::DataFrame,
    C_train::Union{Matrix, SparseMatrixCSC},
    S_val::Union{Matrix, SparseMatrixCSC},
    F_train,
    Chat_val::Union{Matrix, SparseMatrixCSC},
    A::SparseMatrixCSC,
    i2f::Dict,
    f2i::Dict;
    gold_ind::Union{Nothing, Vector} = nothing,
    Shat_val::Union{Nothing, Matrix} = nothing,
    check_gold_path::Bool = false,
    max_t::Int = 15,
    max_can::Int = 10,
    threshold::Float64 = 0.1,
    is_tolerant::Bool = false,
    tolerance::Float64 = (-1000.0),
    max_tolerance::Int = 3,
    grams::Int = 3,
    tokenized::Bool = false,
    sep_token::Union{Nothing, String} = nothing,
    keep_sep::Bool = false,
    target_col::Union{Symbol, String} = "Words",
    start_end_token::String = "#",
    issparse::Union{Symbol, Bool} = :auto,
    sparse_ratio::Float64 = 0.05,
    if_pca::Bool = false,
    pca_eval_M::Union{Nothing, Matrix} = nothing,
    activation::Union{Nothing, Function} = nothing,
    ignore_nan::Bool = true,
    check_threshold_stat::Bool = false,
    verbose::Bool = false
)

    res = learn_paths(
        data_train,
        data_val,
        C_train,
        S_val,
        F_train,
        Chat_val,
        A,
        i2f,
        f2i,
        gold_ind = gold_ind,
        Shat_val = Shat_val,
        check_gold_path = check_gold_path,
        max_t = max_t,
        max_can = max_can,
        threshold = threshold,
        is_tolerant = is_tolerant,
        tolerance = tolerance,
        max_tolerance = max_tolerance,
        grams = grams,
        tokenized = tokenized,
        sep_token = sep_token,
        keep_sep = keep_sep,
        target_col = target_col,
        start_end_token = start_end_token,
        issparse = issparse,
        sparse_ratio = sparse_ratio,
        if_pca = if_pca,
        pca_eval_M = pca_eval_M,
        activation = activation,
        ignore_nan = ignore_nan,
        check_threshold_stat = check_threshold_stat,
        verbose = verbose
    )

    if check_gold_path
        gpi = res[2]
        res = res[1]
    end

    n = size(res)
    ngrams_ind = make_ngrams_ind(res, n)
    Shat = zeros(Float64, size(S_val))

    for i in 1:n[1]
        ci = ngrams_ind[i]
        # CHANGE needed
        Shat[i,:] = predict_shat(F_train, ci)
    end

    tmp, rpi = learn_paths(
        data_train,
        data_val,
        C_train,
        S_val,
        F_train,
        Chat_val,
        A,
        i2f,
        f2i,
        gold_ind = ngrams_ind,
        Shat_val = Shat,
        check_gold_path = true,
        max_t = max_t,
        max_can = 1,
        threshold = 1.,
        is_tolerant = false,
        tolerance = 1.,
        max_tolerance = 1,
        grams = grams,
        tokenized = tokenized,
        sep_token = sep_token,
        keep_sep = keep_sep,
        target_col = target_col,
        start_end_token = start_end_token,
        issparse = issparse,
        sparse_ratio = sparse_ratio,
        if_pca = if_pca,
        pca_eval_M = pca_eval_M,
        activation = activation,
        ignore_nan = ignore_nan,
        check_threshold_stat = check_threshold_stat,
        verbose = false
    )

    if check_gold_path
        return res, gpi, rpi
    else
        return res, rpi
    end

end



"""
    eval_can(candidates, S, F::Union{Matrix,SparseMatrixCSC, Chain}, i2f, max_can, if_pca, pca_eval_M)

Calculate for each candidate path the correlation between predicted semantic
vector and the gold standard semantic vector, and select as target for production
the path with the highest correlation.
"""
function eval_can(
    candidates,
    S,
    F,
    i2f,
    max_can,
    if_pca,
    pca_eval_M,
    ignore_nan = true,
    verbose = false,
)

    verbose &&
        println("average $(mean(length.(candidates))) of paths to evaluate")

    res_l = Array{Array{Result_Path_Info_Struct,1},1}(undef, size(S, 1))
    iter = 1:size(S, 1)
    if verbose
        pb = Progress(size(S, 1))
    end

    if if_pca
        F = pca_eval_M
    end

    Threads.@threads for i in iter
        tid = Threads.threadid()
        res = Result_Path_Info_Struct[]
        if size(candidates[i], 1) > 0
            for (ci, n) in candidates[i] # ci = [1,3,4]
                # CHANGE needed
                Shat = predict_shat(F, ci)
                Scor = cor(Shat[1, :], S[i, :])
                push!(res, Result_Path_Info_Struct(ci, n, Scor))
            end
        end

        if ignore_nan
            res = filter(x -> !isnan(x.support), res)
        end

        # we collect only top x candidates from the top
        res_l[i] = collect(Iterators.take(
            sort!(res, by = x -> x.support, rev = true),
            max_can,
        ))
        if verbose
            ProgressMeter.next!(pb)
        end
    end

    res_l
end

"""
    find_top_feature_indices(rC, C_train_ind)

Find all indices for the n-grams of the top n closest neighbors of
a given target.
"""
function find_top_feature_indices(
    rC,
    C_train_ind;
    n_neighbors = 10,
    verbose = false,
)

    # collect num of val data
    n_val = size(rC, 1)

    # calculate correlation matrix
    # rC = cor(Chat_val, Matrix(C_train), dims=2)
    # display(rC)

    # initialize features list for all candidates
    features_all = Vector{Vector{Int64}}(undef, n_val)

    # create iter for progress bar
    verbose && println("finding all n_neighbors features...")
    iter = 1:n_val
    if verbose
        pb = Progress(n_val)
    end

    # find all features of n_neighbors
    for i in iter
        features = [
            C_train_ind[j]
            for j in sortperm(rC[i, :], rev = true)[1:n_neighbors]
        ]
        features_all[i] = unique(collect(Iterators.flatten(features)))
        if verbose
            ProgressMeter.next!(pb)
        end
    end

    features_all
end

"""
    make_ngrams_ind(res, n)

Construct ngrams indices.
"""
function make_ngrams_ind(
    res,
    n
)
    ngrams_ind = Array{Array{Int, 1}, 1}(undef, n)
    for i = 1:n[1]
        if isempty(res[i])
            ngrams_ind[i] = []
        else
            ngrams_ind[i] = res[i][1].ngrams_ind
        end
    end
    ngrams_ind
end

"""
    predict_shat(F::Union{Matrix, SparseMatrixCSC},
                 ci::Vector{Int})
Predicts semantic vector shat given a comprehension matrix `F` and a
list of indices of ngrams `ci`.

# Obligatory arguments
- `F::Union{Matrix, SparseMatrixCSC}`: Comprehension matrix F.
- `ci::Vector{Int}`: Vector of indices of ngrams in c vector. Essentially, this is a vector indicating which ngrams in a c vector are absent and which are present.

"""
function predict_shat(F::Union{Matrix, SparseMatrixCSC},
                      ci::Vector{Int})
    return(sum(F[ci,:], dims=1))
end
