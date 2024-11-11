using Dates
using Distributions
using Flux

# --------------------------------------------------------------------------

function place_all_boxes_on_goals!(board::Board)
    for (□,v) in enumerate(board)
        if (v & GOAL) > 0
            v |= BOX
        else
            v &= ~BOX
        end
        board[□] = v
    end
    return board
end

function remove_player!(board::Board)
    for □ in tile_indices(board)
        board[□] &= ~PLAYER
    end
    return board
end

# --------------------------------------------------------------------------

function calc_gradient_l2_norm_helper(item)
    if item isa NamedTuple
        sum_sq = 0.0
        for val in values(item)
            sum_sq += calc_gradient_l2_norm_helper(val)
        end
        return sum_sq
    elseif item isa Tuple || item isa Vector
        sum_sq = 0.0
        for entry in item
            sum_sq += calc_gradient_l2_norm_helper(entry)
        end
        return sum_sq
    elseif item isa AbstractArray{Float32}
        return sum(v^2 for v in item)
    elseif item === nothing
        return 0.0
    elseif item isa Real
        return item^2
    else
        error("Unsupported type encountered: $(typeof(item))")
    end
end

function calc_gradient_l2_norm(v::Any)
    return sqrt(calc_gradient_l2_norm_helper(v))
end


# --------------------------------------------------------------------------

"""
We accomplished our goal if all goals in 's_goal' have boxes on them in 'board'.
"""
function completed_goal(board::Board, s_goal::Board)
    for (□,v) in enumerate(s_goal)
        if is_set(v, GOAL) && not_set(board[□], BOX)
            return false
        end
    end
    return true
end

# --------------------------------------------------------------------------

# A training entry is an initial board, a goal, and a sequence of moves.
# If the problem is unsolved, there will be no moves.

struct TrainingEntry0
    s_start::Board
    s_goal::Board
    moves::Vector{Direction}
end

function to_text(entry::TrainingEntry0)::String
    fout = IOBuffer()

    println(fout, "start:")
    println(fout, board_to_text(entry.s_start))

    println(fout, "goal:")
    println(fout, board_to_text(entry.s_goal))

    print(fout, "moves: ")
    for dir in entry.moves
        print(fout, dir_to_char(dir))
    end
    println(fout, "")

    return String(take!(fout))
end

function print_summary(entry::TrainingEntry0)
    println("moves:")
    print("\t")
    for dir in entry.moves
        print(dir_to_char(dir), " ")
    end
    println("")
end

# Create a new training entry that is based on the given training entry, rotated
# 90 degrees nsteps times.
function rotate_training_entry(entry::TrainingEntry0, nsteps::Int)
    return TrainingEntry0(
            rotr90(entry.s_start, nsteps),
            rotr90(entry.s_goal, nsteps),
            rotr90_dir.(entry.moves, nsteps))
end

# Create a new training entry that is based on the given training entry, transposed.
function transpose_training_entry(entry::TrainingEntry0)
    return TrainingEntry0(
            entry.s_start',
            entry.s_goal',
            transpose_dir.(entry.moves))
end

# Any given training entry can be rotated or transposed to produce 8 different training entries.
function diversify_training_entry(entry, rng = Random.default_rng())
    entry = rotate_training_entry(entry, rand(rng, 0:3))
    if rand() > 0.5
        return transpose_training_entry(entry)
    end
    return entry
end

# Produce all variations of every training entry by expanding the input list.
function diversify_dataset!(training_entries::Vector)
    append!(training_entries, [transpose_training_entry(entry) for entry in training_entries])
    append!(training_entries, [rotate_training_entry(entry, 1) for entry in training_entries])
    append!(training_entries, [rotate_training_entry(entry, 2) for entry in training_entries])
    return training_entries
end

# Load a training entry .txt file
function load_training_entry(filename::String)
    # Open the file and read the contents
    lines = readlines(filename)
    n_lines = length(lines)

    # Expect the first line to be "start"
    i_line = 1
    @assert lines[i_line] == "start:"
    i_line += 1

    # Load the start board
    board_lines = String[]

    line = strip(lines[i_line])
    i_line += 1

    while !isempty(line) && i_line < n_lines
        push!(board_lines, line)
        line = strip(lines[i_line])
        i_line += 1
    end

    s_start = Game(join(board_lines, "\n")).board_start

    # Expect the next line to be "goal:"
    @assert lines[i_line] == "goal:"
    i_line += 1

    # Load the goal board
    board_lines = String[]

    line = strip(lines[i_line])
    i_line += 1

    while !isempty(line) && i_line < n_lines
        push!(board_lines, line)
        line = strip(lines[i_line])
        i_line += 1
    end

    s_goal = Game(join(board_lines, "\n")).board_start


    line = strip(lines[i_line])
    i_line += 1
    @assert startswith(line, "moves:")
    if length(line) > 7
        moves = [char_to_dir(c) for c in line[8:end]]
    else
        moves = Direction[]
    end


    return TrainingEntry0(copy(s_start), copy(s_goal), moves)
end

function get_maximum_sequence_length(training_entries::Vector{TrainingEntry0})
    return maximum(length(entry.moves) for entry in training_entries)
end


function construct_solved_training_entry(game::Game, solve_data::AStarData)
    state = State(game, game.board_start)
    moves = convert_to_moves(game, state, solve_data.pushes[1:solve_data.sol_depth])
    n_moves = length(moves)

    # Walk state to the end to get the final (goal) state
    for dir in moves
        maybe_move!(state, game, dir)
    end

    goal_board = deepcopy(state.board)

    # Remove the player from the goal board
    goal_board[state.□_player] &= ~PLAYER

    return TrainingEntry0(
        deepcopy(game.board_start),
        goal_board,
        moves,
    )
end

function construct_failed_training_entry(game::Game)
    # Construct the goal, which has the boxes on the goals
    goal_board = deepcopy(game.board_start)
    for (□, v) in enumerate(goal_board)
        if (v & BOX) > 0
            # remove the box from this tile
            goal_board[□] &= ~BOX
        end
        if (v & GOAL) > 0
            # add a box on this tile
            goal_board[□] |= BOX
        end
    end

    # Remove the player from the goal board
    goal_board[game.□_player_start] &= ~PLAYER

    return TrainingEntry0(
        deepcopy(game.board_start), # s_start
        deepcopy(goal_board), # s_goal
        Direction[],
    )
end

# --------------------------------------------------------------------------

# A higher-level training entry for the level-1 model.
# Such a training entry operates on push moves rather than individual player moves.
# We have an initial board, a goal, a sequence of states, the associated goals,
# and whether the board ends up solved. If the problem is unsolved, there will be no moves.

# Note that we do not store an "action", but rather have a more general state + goal sequence.
# This lets us train on more complicated actions that involve moving more than one box.
# For initial supervised training, actions will only move one box, and the goal will set all other
# boxes to walls (as well as clearing all goals except the one we're pushing the box to).

# If the problem it solved, the moves may contain undos. When that happens, we can either generate
# training examples without the undos (where we back up and remove the action that led to them),
# or we can include the undo but mask the bad action out so we don't learn to execute it.

struct TrainingEntry01
    s_start::Board # The starting board
    states::Vector{Board} # In order, from first successor state post s-start to final state.
                          # The model's goal is the final state (sans player).
                          # Of length n if there are n actions.
    goals::Vector{Board}  # The goals for each state transition. Also of length n.
    bad_moves::BitVector  # If set, then a given move should not be encouraged when training
                          # Also of length n.
    solved::Bool          # Whether this training entry ends up solving the problem.
end

function to_text(entry::TrainingEntry01)::String
    fout = IOBuffer()

    println(fout, "start:")
    println(fout, board_to_text(entry.s_start))

    println(fout, entry.solved ? "solved" : "unsolved")

    println(fout, "n_actions: $(length(entry.states))")

    println(fout, "states:")
    for s in entry.states
        println(fout, board_to_text(s))
    end

    println(fout, "goals:")
    for s in entry.goals
        println(fout, board_to_text(s))
    end

    print(fout, "isbad: ")
    for is_bad in entry.bad_moves
        print(fout, is_bad ? 'x' : '-')
    end
    println(fout, "")

    return String(take!(fout))
end

# Create a new training entry that is based on the given training entry, rotated
# 90 degrees nsteps times.
function rotate_training_entry(entry::TrainingEntry01, nsteps::Int)
    return TrainingEntry01(
            rotr90(entry.s_start, nsteps),
            [rotr90(s, nsteps) for s in entry.states],
            [rotr90(s, nsteps) for s in entry.goals],
            copy(entry.bad_moves),
            entry.solved)
end

# Create a new training entry that is based on the given training entry, transposed.
function transpose_training_entry(entry::TrainingEntry01)
    return TrainingEntry01(
            entry.s_start',
            [s' for s in entry.states],
            [s' for s in entry.goals],
            copy(entry.bad_moves),
            entry.solved)
end

# Load a training entry .txt file for higher-order sequences,
# which will only contain one solution example.
function load_training_entries2(filename::String)
    entries = TrainingEntry01[]

    # Open the file and read the contents
    lines = readlines(filename)
    n_lines = length(lines)
    i_line = 1

    s_start = Board(undef, 1, 1)
    let # Load the start board

        # Expect the first line to be "start"
        if lines[i_line] != "start:"
            @show "exit at bad start"
            return entries
        end
        i_line += 1


        line = strip(lines[i_line])
        i_line += 1

        board_lines = String[]
        while !isempty(line) && i_line < n_lines
            push!(board_lines, line)
            line = strip(lines[i_line])
            i_line += 1
        end

        s_start = Game(join(board_lines, "\n")).board_start
    end

    # Load whether it was solved
    line = strip(lines[i_line])
    i_line += 1
    solved = line == "solved"

    # Load the number of actions
    n_actions = 0
    let
        m = match(r"n_actions: (\d+)", lines[i_line])
        if isnothing(m)
            @show "exit at bad n_actions"
            return entries
        end
        i_line += 1

        n_actions = parse(Int, m.captures[1])
    end

    # Load the states
    states = Array{Board}(undef, n_actions)
    i_line += 1
    for i_action in 1:n_actions

        board_lines = String[]

        line = strip(lines[i_line])
        i_line += 1

        while !isempty(line) && i_line < n_lines
            push!(board_lines, line)
            line = strip(lines[i_line])
            i_line += 1
        end

        states[i_action] = Game(join(board_lines, "\n")).board_start
    end

    # Load the goals
    goals = Array{Board}(undef, n_actions)
    i_line += 1
    for i_action in 1:n_actions

        board_lines = String[]

        line = strip(lines[i_line])
        i_line += 1

        while !isempty(line) && i_line < n_lines
            push!(board_lines, line)
            line = strip(lines[i_line])
            i_line += 1
        end

        goals[i_action] = Game(join(board_lines, "\n")).board_start
    end


    # Parse isbad
    bad_moves = falses(0)
    let
        line = strip(lines[i_line])
        i_line += 1
        if !startswith(line, "isbad:")
            @show "exit at bad isbad"
            return entries
        end
        if length(line) > 7
            bad_moves = [c != '-' for c in line[8:end]]
        end
    end

    push!(entries, TrainingEntry01(
        s_start, states, goals, bad_moves, solved
        ))

    return entries
end

function get_maximum_sequence_length(training_entries::Vector{TrainingEntry01})
    return maximum(length(entry.states) for entry in training_entries2)
end

# The goal should be equal to the given successor state,
# except all unpushed boxes should be walls and
# all goals are replaced with a single goal at the target location.
# We include the player's final position.
function build_goal_board_for_transition(game::Game, s::State, i_box::Integer)::Board
    goal = deepcopy(s.board)

    # replace all boxes except the target box with walls
    for (i,□) in enumerate(s.□_boxes)
        if i == i_box
            continue
        end
        goal[□] &= ~BOX
        goal[□] |= WALL
    end

    # remove all goals
    for □ in game.□_goals
        goal[□] &= ~GOAL
    end

    # place a goal at the final box position
    goal[s.□_boxes[i_box]] |= GOAL

    return goal
end

function construct_solved_training_entry2(game::Game, solve_data::AStarData)

    state = State(game, game.board_start)
    s_start = deepcopy(game.board_start)
    solved = solve_data.solved

    # Walk through the game, committing moves every time we change which box
    # we are pushing.
    states = Board[]
    goals = Board[]

    i_box_active = 1
    if solve_data.sol_depth > 0
        i_box_active = solve_data.pushes[1].i_box
    end
    for push in solve_data.pushes[1:solve_data.sol_depth]
        if push.i_box != i_box_active
            # We have a change in direction.
            # Commit the current transition.

            push!(states, deepcopy(state.board))
            push!(goals, build_goal_board_for_transition(game, state, i_box_active))
        end

        # Advance
        move!(state, game, push)
        i_box_active = push.i_box
    end

    # Append the last one
    push!(states, deepcopy(state.board))
    push!(goals, build_goal_board_for_transition(game, state, i_box_active))

    bad_moves = falses(length(states)) # no bad moves

    return TrainingEntry01(s_start, states, goals, bad_moves, solved)
end

function construct_failed_training_entry2(game::Game, rng = Random.default_rng())
    # Construct the goal, which has the goals on the boxes,
    # and store it.
    goal_board = deepcopy(game.board_start)
    for (□, v) in enumerate(goal_board)
        if (v & BOX) > 0
            # remove the box from this tile
            goal_board[□] &= ~BOX
        end
        if (v & GOAL) > 0
            # add a box on this tile
            goal_board[□] |= BOX
        end
    end

    # It isn't clear where to place the player, so
    # let's just place them at some random open location next to a goal.
    # For a valid board, there will always be at least one valid player location.
    # If there is no valid player location, then we just leave the player where they are.
    let
        □_player = game.□_player_start
        n_candidates = 0
        for (□, v) in enumerate(goal_board)
            col, row = tile_index_to_col_row(game, □)

            # check whether this tile is open and is next to a goal
            is_open = !is_set(v,  WALL + BOX)
            is_next_to_goal = (
                (col > 1 && is_set(goal_board[row, col-1], WALL+BOX)) ||
                (col < size(goal_board, 2) && is_set(goal_board[row, col+1], WALL+BOX)) ||
                (row > 1 && is_set(goal_board[row-1, col], WALL+BOX)) ||
                (row < size(goal_board, 1) && is_set(goal_board[row+1, col], WALL+BOX))
            )
            if is_open && is_next_to_goal
                n_candidates += 1
                if rand() < 1.0/n_candidates
                    goal_board[□_player] &= ~PLAYER
                    □_player = □
                    goal_board[□_player] |= PLAYER
                end
            end
        end
    end

    return TrainingEntry01(
        deepcopy(game.board_start), # s_start
        [deepcopy(goal_board)], # we need some sort of move (which we won't train on), so just use the goal
        [deepcopy(goal_board)], # goal
        falses(1), # one bad move
        false
    )
end



# --------------------------------------------------------------------------


function create_timestamped_directory(base_dir::String)
    # Get the current timestamp
    now = Dates.now()

    # Format the timestamp
    timestamp = Dates.format(now, "yyyymmdd_HHMMss")

    # Create the full directory path
    dir_path = joinpath(base_dir, timestamp)

    # Create the directory
    mkpath(dir_path)

    return dir_path
end


function print_directory_structure_with_counts_internal!(strs::Vector{String}, directory::AbstractString, indent_level::Integer, indent::String)

    n_entries = 0

    for content in readdir(directory)
        fullpath = joinpath(directory, content)
        if isdir(fullpath)
            n_entries += print_directory_structure_with_counts_internal!(strs, fullpath, indent_level + 1, indent)
        elseif isfile(fullpath)
            matches = endswith(content, ".txt")
            if matches
                n_entries += 1
            end
        end
    end

    push!(strs, "$(repeat(indent, indent_level))$(basename(directory)) - $(n_entries) entries")

    return n_entries
end

function print_directory_structure_with_counts(directory::AbstractString, indent="    ")

    strs = String[]
    n_entries = print_directory_structure_with_counts_internal!(strs, directory, 0, indent)

    for str in reverse(strs)
        println(str)
    end

    return n_entries
end



# Load all of the training entries in a directory and its subdirectories
function load_training_set(directory::String)
    entries = TrainingEntry0[]

    for content in readdir(directory)
        fullpath = joinpath(directory, content)
        if isdir(fullpath)
            append!(entries, load_training_set(fullpath))
        elseif isfile(fullpath)
            if endswith(content, ".txt")
                push!(entries, load_training_entry(fullpath))
            end
        end
    end

    return entries
end


# Load all of the training entries in a directory and its subdirectories
function load_training_set2(directory::String)
    entries = TrainingEntry01[]

    for content in readdir(directory)
        fullpath = joinpath(directory, content)
        if isdir(fullpath)
            append!(entries, load_training_set2(fullpath))
        elseif isfile(fullpath)
            if endswith(content, ".txt")
                push!(entries, load_training_entry2(fullpath))
            end
        end
    end

    return entries
end


# --------------------------------------------------------------------------

# Produce a training entry
function produce_unsolved_training_entry(s_start::Board, s_goal::Board)
    return TrainingEntry0(
        s_start,
        s_goal,
        Direction[]
        )
end

# Look at the given solution.
# If it does solve the problem, then we want to retain it as a solution.
# We retain a cleaned version where we remove invalid moves.
function produce_training_entry_from_rollout(
    s_start::Board,
    s_goal::Board,
    moves::Vector{Direction}
  )::Union{TrainingEntry0, Nothing}

    game = Game(s_start)
    state = State(game, deepcopy(game.board_start))

    # Run through the game, skipping bad moves
    good_moves = Direction[]

    for dir in moves
        succeeded = maybe_move!(state, game, dir)
        if succeeded
            push!(good_moves, dir)
        end
    end

    # Verify that the game is solved at the end
    solved = completed_goal(state.board, s_goal)

    if !solved
        # no training entry if not solved
        return nothing
    end

    return TrainingEntry0(s_start, s_goal, good_moves)
end


# --------------------------------------------------------------------------

function set_board_input!(
        inputs::Array{Float32}, # 8×8×5×s×b
        board::Board,
        i_seq::Int,
        i_batch::Int)
    ncols, nrows = size(board)
    for row in 1:nrows
        for col in 1:ncols
            v = board[col, row]
            inputs[col, row, 1, i_seq, i_batch] = Float32((v & BOX) > 0)
            inputs[col, row, 2, i_seq, i_batch] = Float32((v & FLOOR) > 0)
            inputs[col, row, 3, i_seq, i_batch] = Float32((v & GOAL) > 0)
            inputs[col, row, 4, i_seq, i_batch] = Float32((v & PLAYER) > 0)
            inputs[col, row, 5, i_seq, i_batch] = Float32((v & WALL) > 0)
        end
    end
    return inputs
end

function set_board_from_input!(
        board::Board,
        inputs::Array{Float32}, # 8×8×5×s×b
        i_seq::Int,
        i_batch::Int)
    ncols, nrows = size(board)
    for row in 1:nrows
        for col in 1:ncols
            board[col, row] =
                (inputs[col, row, 1, i_seq, i_batch] > 0) * BOX +
                (inputs[col, row, 2, i_seq, i_batch] > 0) * FLOOR +
                (inputs[col, row, 3, i_seq, i_batch] > 0) * GOAL +
                (inputs[col, row, 4, i_seq, i_batch] > 0) * PLAYER +
                (inputs[col, row, 5, i_seq, i_batch] > 0) * WALL
        end
    end
    return board
end

# --------------------------------------------------------------------------

function positional_encoding(max_seq_len, model_dim)
    pe = zeros(Float32, model_dim, max_seq_len)
    for pos in 1:max_seq_len
        for i in 0:(model_dim-1)
            if i % 2 == 0
                pe[i+1, pos] = sin(pos / 10000^(i / model_dim))
            else
                pe[i+1, pos] = cos(pos / 10000^((i-1) / model_dim))
            end
        end
    end
    return pe
end

# --------------------------------------------------------------------------

# A feed forward layer in a transformer has the same input and output dimension, but typially a 4x larger hidden dimension
struct FeedForward
    affine1::Dense
    affine2::Dense
    dropout::Dropout
end

Flux.@functor FeedForward

function FeedForward((dim_in, dim_hidden)::Pair; bias=true, init=Flux.glorot_uniform, dropout_prob = 0.0)
    affine1 = Dense(dim_in => dim_hidden; bias=bias, init=init)
    affine2 = Dense(dim_hidden => dim_in; bias=bias, init=init)
    dropout = Dropout(dropout_prob)
    return FeedForward(affine1, affine2, dropout)
end

function (m::FeedForward)(X::AbstractArray{Float32, 3}) # [dim × dim × batch_size]
    X = m.affine1(X)
    X = relu(X)
    X = m.affine2(X)
    return m.dropout(X)
end


# --------------------------------------------------------------------------

struct PolicyTransformerLayer
    Wq::Dense
    Wk::Dense
    Wv::Dense

    mha::MultiHeadAttention
    norm_1::LayerNorm

    ff::FeedForward
    norm_2::LayerNorm
end

Flux.@functor PolicyTransformerLayer

function PolicyTransformerLayer(
        dim::Int;
        nheads::Int = 8,
        hidden_dim_scale::Int = 4,
        init = Flux.glorot_uniform,
        dropout_prob = 0.0)
    Wq = Dense(dim => dim; bias=true, init=init)
    Wk = Dense(dim => dim; bias=true, init=init)
    Wv = Dense(dim => dim; bias=true, init=init)

    mha = Flux.MultiHeadAttention(dim => dim*nheads => dim, nheads=nheads, bias=true, init=init, dropout_prob=dropout_prob)
    norm_1 = Flux.LayerNorm(dim)

    dim_hidden = dim * hidden_dim_scale
    ff = FeedForward(dim=>dim_hidden, bias=true, init=init, dropout_prob=dropout_prob)
    norm_2 = Flux.LayerNorm(dim)
    return PolicyTransformerLayer(Wq, Wk, Wv, mha, norm_1, ff, norm_2)
end

function (m::PolicyTransformerLayer)(
        X::AbstractArray{Float32, 3}, # [dim × ntokens × batch_size]
        mask::AbstractMatrix{Bool},#  [ntokens × ntokens]
    )
    Q = m.Wq(X) # [dim × ntokens × batch_size]
    K = m.Wk(X) # [dim × ntokens × batch_size]
    V = m.Wv(X) # [dim × ntokens × batch_size]
    X′, activations = m.mha(Q, K, V, mask=mask) # [dim × ntokens × batch_size], [dim × ntokens × batch_size]
    X = m.norm_1(X + X′)      # [dim × ntokens × batch_size]
    X = m.norm_2(X + m.ff(X)) # [dim × ntokens × batch_size]
    return X
end

# --------------------------------------------------------------------------
# Create a mask that attends to the current and all previous inputs
#  mask[i,j] means input j attends to input i
function basic_causal_mask(dim::Int)
    mask = Matrix{Bool}(undef, dim, dim)
    fill!(mask, false)
    for i in 1:dim
        for j in i:dim
            mask[i,j] = true
        end
    end
    return mask
end

# Create a mask that only attends to the current input and the goal (input 1)
#  mask[i,j] means input j attends to input i
function input_only_mask(dim::Int)
    mask = Matrix{Bool}(undef, dim, dim)
    fill!(mask, false)
    for i in 1:dim
        mask[i,i] = true # attend to self
        mask[1,i] = true # attend to goal
    end
    return mask
end

# --------------------------------------------------------------------------

# The Level-0 Policy
struct SokobanPolicyLevel0
    batch_size::Int   # b, batch size
    max_seq_len::Int  # s, maximum input length, in number of tokens
    encoding_dim::Int # e, dimension of the embedding space

    encoder::Chain
    pos_enc::AbstractMatrix{Float32}
    dropout::Dropout
    mask::AbstractMatrix{Bool} # causal mask, [s × s], mask[i,j] means input j attends to input i
    trunk::Vector{PolicyTransformerLayer}
    action_head::Dense # logits for our 4 actions: {up,left,down,right}
    nsteps_head::Dense # logits for the number of remaining steps, plus inf: P(min(round(log(n+1)), 5)) plus inf
end

Flux.@functor SokobanPolicyLevel0

function SokobanPolicyLevel0(;
        batch_size::Int = 32,
        max_seq_len::Int = 32,
        encoding_dim::Int = 8,
        num_trunk_layers::Int = 3,
        n_mha_heads::Int = 8,
        trunk_hidden_dim_scale::Int = 4,
        encoder_conv_dim::Int = 8,
        init = Flux.glorot_uniform,
        dropout_prob::Float64 = 0.0,
        no_past_info::Bool = false, # if true, we create a causal mask that only attends to the current input and the goal
        )

    e = encoding_dim
    f = encoder_conv_dim
    b = batch_size
    s = max_seq_len

    norm_enc = Flux.LayerNorm(e)
    dropout = Dropout(dropout_prob)
    pos_enc = positional_encoding(s, e)
    mask = no_past_info ? input_only_mask(s) : basic_causal_mask(s)

    # The encoder encodes the states into token embeddings (8×8×5×s×b →  e×s×b)
    encoder = Chain(
        # 8×8×5×s×b
        x -> reshape(x, size(x)[1], size(x)[2], size(x)[3], size(x)[4]*size(x)[5]),
        # 8×8×5×sb
        Conv((5,5), 5=>f, relu, stride=1, pad=2),
        # 8×8×f×sb
        Conv((5,5), f=>f, relu, stride=1, pad=2),
        # 8×8×f×sb
        Conv((5,5), f=>2, relu, stride=1, pad=2),
        # 8×8×2×sb
        x -> reshape(Flux.flatten(x), 128, s, b),
        # 128×s×b (Dense layer expects inputs as (features, batch), thus another reshape is needed)
        x -> reshape(x, :, size(x, 2)*size(x, 3)),
        # 128×sb
        Dense(128 => e, relu),
        # e×sb
        x -> reshape(x, e, s, b),
        # e×s×b
        x -> norm_enc(x)
    )

    # The trunk is the workhorse of the transformer, and iteratively
    # applies self-attention and skip-connection nonlinearities
    trunk = Array{PolicyTransformerLayer}(undef, num_trunk_layers)
    for i_trunk_layer in 1:num_trunk_layers
        trunk[i_trunk_layer] = PolicyTransformerLayer(
            encoding_dim, init=init, dropout_prob=dropout_prob,
            nheads=n_mha_heads, hidden_dim_scale=trunk_hidden_dim_scale)
    end

    action_head = Dense(e => 4; bias=true, init=init)
    nsteps_head = Dense(e => 7; bias=true, init=init)

    return SokobanPolicyLevel0(
        batch_size, max_seq_len, encoding_dim,
        encoder, pos_enc, dropout, mask, trunk, action_head, nsteps_head)
end

function (m::SokobanPolicyLevel0)(input::AbstractArray{Float32, 5}) # [8×8×5×s×b]

    X = m.encoder(input) .+ m.pos_enc
    X = m.dropout(X)

    for layer in m.trunk
        X = layer(X, m.mask)
    end

    action_logits = m.action_head(X) # [4×s×b]
    nsteps_logits = m.nsteps_head(X) # [7×s×b]

    return (action_logits, nsteps_logits)
end

struct SokobanPolicyLevel0Data
    inputs::AbstractArray{Float32}     # 8×8×5×s×b
    policy_target::AbstractArray{Float32}  # 4×s×b
    nsteps_target::AbstractArray{Float32}  # 7×s×b
    policy_mask::AbstractArray{Int32}      # 4×s×b # TODO: Switch to UInt8
    nsteps_mask::AbstractArray{Int32}      # 7×s×b
end

function SokobanPolicyLevel0Data(max_seq_len::Integer, batch_size::Integer)
    inputs = zeros(Float32, (8, 8, 5, max_seq_len, batch_size))
    policy_target = zeros(Float32, (4, max_seq_len, batch_size))
    nsteps_target = zeros(Float32, (7, max_seq_len, batch_size))
    policy_mask = zeros(Int32, size(policy_target))
    nsteps_mask = zeros(Int32, size(nsteps_target))
    return SokobanPolicyLevel0Data(inputs, policy_target, nsteps_target, policy_mask, nsteps_mask)
end

function SokobanPolicyLevel0Data(m::SokobanPolicyLevel0)
    return SokobanPolicyLevel0Data(m.max_seq_len, m.batch_size)
end

function to_gpu(tpd::SokobanPolicyLevel0Data)
    return SokobanPolicyLevel0Data(
        gpu(tpd.inputs),
        gpu(tpd.policy_target),
        gpu(tpd.nsteps_target),
        gpu(tpd.policy_mask),
        gpu(tpd.nsteps_mask)
    )
end

function to_cpu(tpd::SokobanPolicyLevel0Data)
    return SokobanPolicyLevel0Data(
        cpu(tpd.inputs),
        cpu(tpd.policy_target),
        cpu(tpd.nsteps_target),
        cpu(tpd.policy_mask),
        cpu(tpd.nsteps_mask)
    )
end



# --------------------------------------------------------------------------

# Unpack a TrainingEntry0 into the tensor representation used by the model.
function convert_training_entry!(
        inputs::Array{Float32},    # 8×8×5×s×b
        policy_target::Array{Float32}, # 4×s×b
        nsteps_target::Array{Float32}, # 7×s×b
        policy_mask::Array{Int32},     # 4×s×b
        nsteps_mask::Array{Int32},     # 7×s×b
        training_entry::TrainingEntry0,
        batch_index::Int)

    # Ensure that our training entry is not longer than our model can handle
    n_moves = length(training_entry.moves)
    @assert n_moves < size(inputs, 4)

    b = batch_index

    # Only train on things we set to 1
    policy_mask[:, :, b] .= 0
    nsteps_mask[:, :, b] .= 0

    # The goal always goes into the first entry
    set_board_input!(inputs, training_entry.s_goal, 1, b)
    policy_target[:, 1, b] .= 1.0/size(policy_target, 1)
    nsteps_target[:, 1, b] .= 1.0/size(nsteps_target, 1)

    # Walk through the game and place the states
    game = Game(deepcopy(training_entry.s_start))
    state = State(game, game.board_start)

    s = 2 # seq index
    for dir in training_entry.moves

        set_board_input!(inputs, state.board, s, b)
        policy_target[:, s, b] .= Float32(0)
        policy_target[dir, s, b] = Float32(1)


        nsteps_target[:, s, b] .= Float32(0)
        nsteps = length(training_entry.moves)
        idx = Int(clamp(round(log(nsteps+1)), 0, 5)) + 1
        nsteps_target[idx, s, b] = Float32(1)

        policy_mask[:, s, b] .= 1 # train on the action
        nsteps_mask[:, s, b] .= 1 # train on the step count

        # Advance the state
        succeeded = maybe_move!(state, game, dir)
        @assert succeeded # we expect valid moves in all training examples
        s += 1
    end

    # Always set the final state (which is the first state in an unsolved entry),
    # but the policy target is not trained on.
    set_board_input!(inputs, state.board, s, b)
    policy_target[:, s, b] .= 1.0/size(policy_target, 1)
    nsteps_target[:, s, b] .= Float32(0)
    nsteps_mask[:, s, b] .= 1
    if n_moves > 0
        nsteps_target[1, s, b] = Float32(1) # solved.
    else
        nsteps_target[end, s, b] = Float32(1) # infinite steps to solve w/out undo
    end
end

function convert_training_entry!(
        data::SokobanPolicyLevel0Data,
        training_entry::TrainingEntry0,
        batch_index::Int)
    convert_training_entry!(
        data.inputs,
        data.policy_target,
        data.nsteps_target,
        data.policy_mask,
        data.nsteps_mask,
        training_entry,
        batch_index)
    return data
end

function clear!(data::SokobanPolicyLevel0Data, batch_index::Int)
    data.inputs[:, :, :, :, batch_index] .= 0
    data.policy_mask[:, :, batch_index] .= 0 # don't train on it
    data.nsteps_mask[:, :, batch_index] .= 0 # don't train on it
    data.policy_target[:, :, batch_index] .= 1.0/size(data.policy_target, 1)
    data.nsteps_target[:, :, batch_index] .= 1.0/size(data.nsteps_target, 1)
    return data
end

# --------------------------------------------------------------------------

function load_params(model_directory::AbstractString)
    params = JSON.parsefile(joinpath(model_directory, "params.json"))

    if (!haskey(params, "num_trunk_layers"))
        params["num_trunk_layers"] = 3
    end
    if (!haskey(params, "n_mha_heads"))
        params["n_mha_heads"] = 8
    end
    if (!haskey(params, "trunk_hidden_dim_scale"))
        params["trunk_hidden_dim_scale"] = 4
    end
    if (!haskey(params, "encoder_conv_dim"))
        params["encoder_conv_dim"] = 8
    end

    return params
end

function load_policy(::Type{SokobanPolicyLevel0}, model_directory::AbstractString, params::Dict{String,Any})

    policy = SokobanPolicyLevel0(
        batch_size = params["batch_size"],
        max_seq_len = params["max_seq_len"],
        encoding_dim = params["encoding_dim"],
        num_trunk_layers = params["num_trunk_layers"],
        n_mha_heads = parmas["n_mha_heads"],
        trunk_hidden_dim_scale = params["trunk_hidden_dim_scale"],
        encoder_conv_dim = params["encoder_conv_dim"],
        dropout_prob=params["dropout_prob"],
        no_past_info=params["no_past_info"])

    model_state = JLD2.load(joinpath(model_directory, "model0.jld2"), "model_state");
    Flux.loadmodel!(policy, model_state);

    return policy
end

# --------------------------------------------------------------------------

mutable struct BeamSearchData
    inputs::Array{Float32}  # [8×8×5×s×b]
    moves::Array{Direction} #       [s×b]  The move trajectories maintained by beam search
    scores::Vector{Float32} #         [b]  The running log likelihoods for the beam
    depth::Int              # How deep we got before returning. moves[depth] is the last move we used.
    solution::Int           # Batch index of the trajectory that solves the problem (or -1 otherwise)
end

function beam_search!(
        data::BeamSearchData,
        s_start::Board,
        s_goal::Board,
        policy::SokobanPolicyLevel0
    )

    inputs = data.inputs
    moves = data.moves
    scores = data.scores

    b = size(inputs, 5) # The beam width is equal to the batch size
    @assert b == size(moves, 2)

    max_seq_len = size(inputs, 4)
    @assert max_seq_len == size(moves, 1)

    # We're going to track a batch worth of states in our beam
    game = Game(deepcopy(s_start))
    states = [State(game, deepcopy(s_start)) for i in 1:b]
    fill!(scores, 0.0) # running log likelihood of the actions

    # The number of beams.
    # At the start, we only have a single beam (the root trajectory)
    n_beams = 1

    inputs_next = zeros(Float32, size(inputs))
    states_next = Array{State}(undef, b)
    moves_next = zeros(Direction, (policy.max_seq_len, policy.batch_size))

    # The array of beam scores, which has one value per action, per beam.
    scores_next = Array{Float32}(undef, 5*b)
    p = collect(1:length(scores_next)) # permutation vector

    # The goal goes into the first entry
    for beam_index in 1:b
        set_board_input!(inputs, s_goal, 1, beam_index)
    end

    # Advance the games in parallel
    for seq_index in 2:max_seq_len

        for beam_index in 1:n_beams
            set_board_input!(inputs, states[beam_index].board, seq_index, beam_index)
        end

        # Run the model
        # policy_logits are [5 × ntokens × batch_size]
        # nsteps_logits are [7 × ntokens × batch_size]
        policy_logits_gpu, nsteps_logits_gpu = policy(inputs |> gpu)

        # Compute the probabilities
        probabilities = softmax(policy_logits_gpu, dims=1) |> cpu # [5 × ntokens × batch_size]

        # Compute the score for each beam.
        # An invalid / unused beam gets a score of -Inf.
        fill!(scores_next, -Inf)
        for i in 1:5
            for j in 1:n_beams
                k = b*(i-1) + j
                scores_next[k] = scores[j] + probabilities[i,seq_index,j]
            end
        end

        # Run a partial sort to get the highest b entries.
        # Produes a permulation vector of length b that is sorted in descending order by score.
        partialsortperm!(p, scores_next, 1:b, rev=true)

        # Compute the successor states and update the scores
        n_beams = 0
        for beam_index_dst in 1:b
            permutation_index = p[beam_index_dst]

            if !isfinite(scores_next[permutation_index])
                break
            end

            n_beams += 1

            # The index of the beam this came from.
            beam_index_src = mod1(permutation_index, b)

            # The direction associated with that beam's step.
            dir = UInt8(div(permutation_index - beam_index_src, b) + 1)

            states_next[beam_index_dst] = deepcopy(states[beam_index_src])

            # Copy the moves in this sequence
            # Subtract 1 to avoid the space taken up by the goal
            moves_next[1:seq_index-2,beam_index_dst] = moves[1:seq_index-2,beam_index_src]
            moves_next[seq_index-1,beam_index_dst] = dir

            if dir != DIR_UNDO
                # Set the next state to the current state.
                maybe_move!(states_next[beam_index_dst], game, dir)

            elseif seq_index > 2 # can only undo if we've made a move
                # Revert to the previous state
                set_board_from_input!(
                    states_next[beam_index_dst].board,
                    inputs,
                    seq_index-1, # previous seq index
                    beam_index_src)
                states_next[beam_index_dst] = State(game, states_next[beam_index_dst].board)

            end

            # Copy the score
            scores[beam_index_dst] = scores_next[permutation_index]

            # Copy the inputs
            inputs_next[:,:,:,:,beam_index_dst] = inputs[:,:,:,:,beam_index_src]
        end

        states[:] = states_next
        moves[:] = moves_next
        inputs[:] = inputs_next

        # If any of these states reach the goal, we are done
        # We check this by looking at box positions (which isn't perfectly correct - we could wish to have the player on a certain side)
        for beam_index in 1:n_beams
            if completed_goal(states[beam_index].board, s_goal)
                # We found a solution!
                # And for us it is a shortest solution.
                data.solution = beam_index
                data.depth = seq_index-1
                return data
            end
        end
    end

    # Failed to find a solution.
    data.solution = -1
    data.depth = b
    return data
end

# --------------------------------------------------------------------------

mutable struct RolloutData
    inputs::Array{Float32}  # [8 × 8 × 5 × s × b]
    moves::Array{Direction} # [s × b] The moves for the rollouts
    rewards::Array{Float32} # [s × b] The per-step rewards
    depths::Vector{Int}     # [b] How deep each rollout is
end

# Run the model in parallel to produce rollouts
function rollout!(
    data::RolloutData,
    policy::SokobanPolicyLevel0,
    s_starts::Vector{Board}, # [b] the starting boards
    s_goals::Vector{Board};  # [b] the goal boards
    policy_dirichlet_prior::Float32 = 1.0f0,
    reward_bad_move::Float32 = -0.1f0, # reward obtained for a step that runs into a wall
    reward_solve::Float32    =  1.0f0, # reward obtained for solving a puzzle
    )

    inputs = data.inputs
    moves = data.moves
    rewards = data.rewards
    depths = data.depths

    s = size(inputs, 4)
    b = size(inputs, 5)

    @assert length(s_starts) == b
    @assert length(s_goals) == b
    @assert size(inputs) == (8, 8, 5, s, b)
    @assert size(moves) == (s,b)
    @assert size(rewards) == (s,b)
    @assert length(depths) == b

    fill!(moves, DIR_UP)
    fill!(rewards, zero(Float32))
    fill!(depths, 0)

    num_running = b
    is_done = falses(b) # keep track of whether each rollout is done
    states = [State(Game(board)) for board in s_starts]

    # Fill the goals into the first sequence channel
    for (bi, s_goal) in enumerate(s_goals)
        set_board_input!(inputs, s_goal, 1, bi)
    end

    # Fill the start states in the next sequence channel
    for (bi, s_start) in enumerate(s_starts)
        set_board_input!(inputs, s_start, 2, bi)
    end

    for si in 2:s
        if num_running ≤ 0
            break
        end

        # Run the model
        # policy_logits are [5 × s × b]
        # nsteps_logits are [7 × s × b]
        policy_logits, nsteps_logits = policy(inputs)

        # Pull out the logits for our current sequence index.
        policy_logits_seq = policy_logits[:,si,:] # [5 × b]

        # Optionally bias the policy logits.
        policy_logits_seq[:,:] .+= policy_dirichlet_prior # [5 × b]

        # Produce the action probabilities
        probabilities = softmax(policy_logits_seq, dims=1) # [5 × b]

        # Apply the actions
        for bi in 1:b
            if is_done[bi]
                continue # no need to simulate this one any further
            end

            # Create a Categorical distribution
            categorical = Categorical(probabilities[:,bi])

            # Sample from it to get the action
            ai = rand(categorical)
            dir = Direction(ai)
            moves[si,bi] = dir

            # Increase the depth by one
            depths[bi] += 1

            # Simulate the state forward
            if dir != DIR_UNDO
                # Set the next state to the current state.
                successful_move = maybe_move!(states[bi], dir)
                rewards[si, bi] += !successful_move * reward_bad_move

                if is_solved(states[bi])
                    is_done[bi] = true
                    num_running -= 1
                    rewards[si, bi] += reward_solve
                end

            elseif si > 2 # can only undo if we've made a move
                # Revert to the previous state
                set_board_from_input!(states[bi].board, inputs, si-1, bi)
                states[bi] = State(Game(states[bi].board)) # recompute from the board
            end

            # Fill in the next input
            if si < s
                set_board_input!(inputs, states[bi].board, si+1, bi)
            end
        end
    end

    return data
end

# --------------------------------------------------------------------------

function calc_metrics_gpu(
    policy::SokobanPolicyLevel0,
    data::SokobanPolicyLevel0Data,
    validation_set::Vector{TrainingEntry0},
    )

    policy = gpu(policy)

    m_validation_set = length(validation_set)

    policy_loss = 0.0
    nsteps_loss = 0.0

    n_policy_predictions = 0.0
    top_1_policy_accuracy = 0.0
    top_2_policy_accuracy = 0.0

    n_nsteps_predictions = 0.0
    top_1_nsteps_accuracy = 0.0
    top_2_nsteps_accuracy = 0.0
    solvability_accuracy = 0.0 # how good we are at predicting when a state is solvable or not

    i_batch = 0
    i_training_entry = 1
    while i_training_entry ≤ m_validation_set
        # Run another batch
        i_batch += 1
        n_nonzero_entries = 0
        for batch_index in 1:policy.batch_size
            if i_training_entry ≤ m_validation_set
                convert_training_entry!(data, validation_set[i_training_entry], batch_index)
                n_nonzero_entries += 1
            else
                clear!(data, batch_index)
            end
            i_training_entry += 1
        end

        inputs_gpu = data.inputs |> gpu
        policy_target_gpu = data.policy_target |> gpu
        nsteps_target_gpu = data.nsteps_target |> gpu
        policy_mask_gpu = data.policy_mask |> gpu
        nsteps_mask_gpu = data.nsteps_mask |> gpu

        # Run the model
        policy_logits_gpu, nsteps_logits_gpu = policy(inputs_gpu)

        # Compute the logit cross-entropy loss
        n_policy_targets = Flux.sum(policy_mask_gpu)
        if n_policy_targets > 0
            policy_loss += Flux.Losses.logitcrossentropy(policy_logits_gpu, policy_target_gpu,
                agg = losses -> sum(losses .* policy_mask_gpu) / n_policy_targets) * n_nonzero_entries

            # THIS PART RUNS ON THE CPU
            # Sort predictions. If policy_logits is [7 × s × b] then k_values is also [7 × s × b].
            # k_valies[1,i,j] will be the index of the top value for the predictions for (i,j)
            # k_valies[2,i,j] will be the index of the 2nd best value ... etc.
            policy_logits_cpu = cpu(policy_logits_gpu)
            k_values = mapslices(x -> sortperm(x, rev=true), policy_logits_cpu; dims=1)

            for seq_index in 2:policy.max_seq_len # skip the first one, since it is the goal
                for batch_index in 1:policy.batch_size
                    k1 = k_values[1,seq_index,batch_index]
                    k2 = k_values[2,seq_index,batch_index]
                    if data.policy_mask[k1,seq_index,batch_index] > 0.9
                        iscorrect1 = data.policy_target[k1,seq_index,batch_index] > 0.9
                        iscorrect2 = data.policy_target[k2,seq_index,batch_index] > 0.9
                        @assert iscorrect1 + iscorrect2 < 2
                        top_1_policy_accuracy += iscorrect1
                        top_2_policy_accuracy += iscorrect1 + iscorrect2
                        n_policy_predictions += 1
                    end
                end
            end
        end

        n_nsteps_targets = Flux.sum(nsteps_mask_gpu)
        if n_nsteps_targets > 0
            nsteps_loss += Flux.Losses.logitcrossentropy(nsteps_logits_gpu, nsteps_target_gpu,
                agg = losses -> sum(losses .* nsteps_mask_gpu) / n_nsteps_targets) * n_nonzero_entries

            # THIS PART RUNS ON THE CPU
            # Sort predictions. If nsteps_logits is [5 × s × b] then k_values is also [5 × s × b].
            # k_valies[1,i,j] will be the index of the top value for the predictions for (i,j)
            # k_valies[2,i,j] will be the index of the 2nd best value ... etc.
            nsteps_logits_cpu = cpu(nsteps_logits_gpu)
            k_values = mapslices(x -> sortperm(x, rev=true), nsteps_logits_cpu; dims=1)

            for seq_index in 2:policy.max_seq_len # skip the first one, since its the goal
                for batch_index in 1:policy.batch_size
                    k1 = k_values[1,seq_index,batch_index]
                    k2 = k_values[2,seq_index,batch_index]
                    if data.nsteps_mask[k1,seq_index,batch_index] > 0.9
                        iscorrect1 = data.nsteps_target[k1,seq_index,batch_index] > 0.9
                        iscorrect2 = data.nsteps_target[k2,seq_index,batch_index] > 0.9
                        @assert iscorrect1 + iscorrect2 < 2
                        top_1_nsteps_accuracy += iscorrect1
                        top_2_nsteps_accuracy += iscorrect1 + iscorrect2
                        n_nsteps_predictions += 1

                        not_solvable = data.nsteps_target[end,seq_index,batch_index] > 0.9
                        if not_solvable
                            solvability_accuracy += (k1 == size(data.nsteps_target,1))
                        else
                            solvability_accuracy += (k1 != size(data.nsteps_target,1))
                        end
                    end
                end
            end
        end
    end

    return Dict{String, Float64}(
        "m_validation_set" => m_validation_set,
        "n_batches"   => i_batch,
        "policy_loss" => policy_loss / m_validation_set,
        "nsteps_loss" => nsteps_loss / m_validation_set,
        "top_1_policy_accuracy" => top_1_policy_accuracy / n_policy_predictions,
        "top_2_policy_accuracy" => top_2_policy_accuracy / n_policy_predictions,
        "n_policy_predictions"  => n_policy_predictions,
        "top_1_nsteps_accuracy" => top_1_nsteps_accuracy / n_nsteps_predictions,
        "top_2_nsteps_accuracy" => top_2_nsteps_accuracy / n_nsteps_predictions,
        "solvability_accuracy"  => solvability_accuracy / n_nsteps_predictions,
        "n_nsteps_predictions"  => n_nsteps_predictions,
    )
end


function calc_metrics_gpu(
    policy::SokobanPolicyLevel0,
    data::BeamSearchData,
    validation_set::Vector{TrainingEntry0},
    )

    policy = gpu(policy)

    m_validation_set = length(validation_set)

    n_solved_tp = 0
    n_solved_fp = 0
    n_failed_tp = 0
    n_failed_fp = 0

    accumulated_solution_length = 0

    for training_entry in validation_set
        beam_search!(data, training_entry.s_start, training_entry.s_goal, policy)

        if data.solution == -1 # failed
            if training_entry.solved
                n_failed_fp += 1 # we should not have failed this one
            else
                n_failed_tp += 1 # we expected not to solve this one
            end
        else
            if training_entry.solved
                n_solved_tp += 1 # we had a stored solution for this one
            else
                n_solved_fp += 1 # this one should not be solvable
            end

            accumulated_solution_length += data.depth
        end
    end

    return Dict{String, Float64}(
        "m_validation_set" => m_validation_set,
        "n_solved_tp" => n_solved_tp,
        "n_solved_fp" => n_solved_fp,
        "n_failed_tp" => n_failed_tp,
        "n_failed_fp" => n_failed_fp,
        "solved_tp_rate" => n_solved_tp / m_validation_set,
        "solved_fp_rate" => n_solved_fp / m_validation_set,
        "failed_tp_rate" => n_failed_tp / m_validation_set,
        "failed_fp_rate" => n_failed_fp / m_validation_set,
        "average_solution_length" => accumulated_solution_length / (n_solved_tp + n_solved_fp),
    )
end

# --------------------------------------------------------------------------

# struct TransformerPolicy2
#     batch_size::Int   # batch size
#     max_seq_len::Int  # maximum input length, in number of tokens
#     encoding_dim::Int # dimension of the embedding space
#     n_actions::Int    # number of actions that this outputs

#     encoder::Chain
#     pos_enc::AbstractMatrix{Float32}
#     dropout::Dropout
#     mask::AbstractMatrix{Bool} # causal mask, [s × s], mask[i,j] means input j attends to input i
#     trunk::Vector{PolicyTransformerLayer}
#     goalem_seeds::AbstractMatrix{Float32} # learned goal embedding seeds
#     action_head::Dense # produces n_actions logits
#     goalem_head::Dense # produces n_actions goal embeddings
#     nsteps_head::Dense # P(min(round(log(n+1)), 5)) plus inf, produces logits
# end

# Flux.@functor TransformerPolicy2

# function TransformerPolicy2(;
#         batch_size::Int = 32,
#         max_seq_len::Int = 32,
#         encoding_dim::Int = 8,
#         num_trunk_layers::Int = 3,
#         init = Flux.glorot_uniform,
#         dropout_prob::Float64 = 0.0,
#         no_past_info::Bool = false, # if true, we create a causal mask that only attends to the current input and the goal
#         )

#     norm_enc = Flux.LayerNorm(encoding_dim)

#     encoder = Chain(
#         # 8×8×5×s×b
#         x -> reshape(x, size(x)[1], size(x)[2], size(x)[3], size(x)[4]*size(x)[5]),
#         # 8×8×5×sb
#         Conv((5,5), 5=>8, relu, stride=1, pad=2),
#         # 8×8×8×sb
#         Conv((5,5), 8=>8, relu, stride=1, pad=2),
#         # 8×8×8×sb
#         Conv((5,5), 8=>2, relu, stride=1, pad=2),
#         # 8×8×2×sb
#         x -> reshape(Flux.flatten(x), 128, max_seq_len, batch_size),
#         # 128×s×b (Dense layer expects inputs as (features, batch), thus another reshape is needed)
#         x -> reshape(x, :, size(x, 2)*size(x, 3)),
#         # 128×sb
#         Dense(128 => encoding_dim, relu),
#         # d×sb
#         x -> reshape(x, encoding_dim, max_seq_len, batch_size),
#         # d×s×b
#         x -> norm_enc(x)
#     )

#     dropout = Dropout(dropout_prob)

#     mask = basic_causal_mask(max_seq_len)
#     if no_past_info
#         mask = input_only_mask(max_seq_len)
#     end

#     pos_enc = positional_encoding(max_seq_len, encoding_dim)

#     trunk = Array{PolicyTransformerLayer}(undef, num_trunk_layers)
#     for i_trunk_layer in 1:num_trunk_layers
#         trunk[i_trunk_layer] = PolicyTransformerLayer(encoding_dim, init=init, dropout_prob=dropout_prob)
#     end

#     action_head = Dense(encoding_dim => 5; bias=true, init=init)
#     nsteps_head = Dense(encoding_dim => 7; bias=true, init=init)

#     return TransformerPolicy(
#         batch_size, max_seq_len, encoding_dim,
#         encoder, pos_enc, dropout, mask, trunk, action_head, nsteps_head)
# end

# function (m::TransformerPolicy)(input::AbstractArray{Float32, 5}) # [8 × 8 × 5 × ntokens × batch_size]

#     X = m.encoder(input) .+ m.pos_enc
#     X = m.dropout(X)

#     for layer in m.trunk
#         X = layer(X, m.mask)
#     end

#     action_logits = m.action_head(X) # [5 × ntokens × batch_size]
#     nsteps_logits = m.nsteps_head(X) # [7 × ntokens × batch_size]

#     return (action_logits, nsteps_logits)
# end

# struct TransformerPolicyData
#     inputs::AbstractArray{Float32} # 8 × 8 × 5 × max_seq_len × batch_size
#     policy_target::AbstractArray{Float32}  # 5 × max_seq_len × batch_size
#     nsteps_target::AbstractArray{Float32}  # 7 × max_seq_len × batch_size
#     policy_mask::AbstractArray{Int32}      # 5 × max_seq_len × batch_size
#     nsteps_mask::AbstractArray{Int32}      # 7 × max_seq_len × batch_size
# end

# function TransformerPolicyData(m::TransformerPolicy)
#     inputs = zeros(Float32, (8, 8, 5, m.max_seq_len, m.batch_size))
#     policy_target = zeros(Float32, (5, m.max_seq_len, m.batch_size))
#     nsteps_target = zeros(Float32, (7, m.max_seq_len, m.batch_size))
#     policy_mask = zeros(Int32, size(policy_target))
#     nsteps_mask = zeros(Int32, size(nsteps_target))
#     return TransformerPolicyData(inputs, policy_target, nsteps_target, policy_mask, nsteps_mask)
# end

# function to_gpu(tpd::TransformerPolicyData)
#     return TransformerPolicyData(
#         gpu(tpd.inputs),
#         gpu(tpd.policy_target),
#         gpu(tpd.nsteps_target),
#         gpu(tpd.policy_mask),
#         gpu(tpd.nsteps_mask)
#     )
# end

# function to_cpu(tpd::TransformerPolicyData)
#     return TransformerPolicyData(
#         cpu(tpd.inputs),
#         cpu(tpd.policy_target),
#         cpu(tpd.nsteps_target),
#         cpu(tpd.policy_mask),
#         cpu(tpd.nsteps_mask)
#     )
# end


# --------------------------------------------------------------------------