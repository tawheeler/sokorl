using Dates
using Distributions
using Flux
using NNlib
using OneHotArrays

# The dimension indices for the board sparse encoding
const DIM_WALL = 1
const DIM_FLOOR = 2
const DIM_GOAL = 3
const DIM_BOX = 4
const DIM_PLAYER = 5
const NUM_SPARSE_BOARD_FEATURES = 5

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

function calc_num_params(model)
    return sum(length, Flux.params(model))
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

function has_solution(entry::TrainingEntry0)::Bool
    return length(entry.moves) > 0
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
function load_training_entry(::Type{TrainingEntry0}, filename::String)
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

# Place all boxes on goals and remove the player.
function construct_naive_goal_board(board::Board)
    goal_board = deepcopy(board)
    for (□, v) in enumerate(goal_board)
        if (v & BOX) > 0
            # remove the box from this tile
            goal_board[□] &= ~BOX
        end
        if (v & GOAL) > 0
            # add a box on this tile
            goal_board[□] |= BOX
        end
        if (v & PLAYER) > 0
            # remove the player
            goal_board[□] &= ~PLAYER
        end
    end
    return goal_board
end

function construct_failed_training_entry(game::Game)
    # Construct the goal, which has the boxes on the goals and no player.
    goal_board = construct_naive_goal_board(game.board_start)

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

struct TrainingEntry1
    s_start::Board # The starting board
    states::Vector{Board} # In order, from first successor state post s-start to final state.
                          # The model's goal is the final state (sans player).
                          # Of length n if there are n actions.
    goals::Vector{Board}  # The goals for each state transition. Also of length n.
    solved::Bool          # Whether this training entry ends up solving the problem.
end

function to_text(entry::TrainingEntry1)::String
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

    return String(take!(fout))
end

# Create a new training entry that is based on the given training entry, rotated
# 90 degrees nsteps times.
function rotate_training_entry(entry::TrainingEntry1, nsteps::Int)
    return TrainingEntry1(
            rotr90(entry.s_start, nsteps),
            [rotr90(s, nsteps) for s in entry.states],
            [rotr90(s, nsteps) for s in entry.goals],
            entry.solved)
end

# Create a new training entry that is based on the given training entry, transposed.
function transpose_training_entry(entry::TrainingEntry1)
    return TrainingEntry1(
            entry.s_start',
            [s' for s in entry.states],
            [s' for s in entry.goals],
            entry.solved)
end

# Load a training entry .txt file for higher-order sequences,
# which will contain one solution example.
function load_training_entry(::Type{TrainingEntry1}, filename::String)
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


    return TrainingEntry1(
        s_start, states, goals, solved
        )
end

function get_maximum_sequence_length(training_entries::Vector{TrainingEntry1})
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

    return TrainingEntry1(s_start, states, goals, solved)
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

    return TrainingEntry1(
        deepcopy(game.board_start), # s_start
        Board[], # no moves
        [deepcopy(goal_board)], # goal
        false
    )
end


"""
Identify the box that moved between states s0 and s1.
Returns the tile index of the box in s0 and s1.
Returns `nothing` if no boxes or more than 1 box moved.
"""
function find_moved_box(A::Board, B::Board)::Union{Nothing, Pair{TileIndex,TileIndex}}
    moved_boxes_a = TileIndex[]
    for (□, v) in enumerate(A)
        if (v & BOX) > 0 && (B[□] & BOX) == 0
            # This is a box in A that is not in the same place in B
            push!(moved_boxes_a, □)
        end
    end
    if length(moved_boxes_a) != 1
        return nothing
    end

    moved_boxes_b = TileIndex[]
    for (□, v) in enumerate(B)
        if (v & BOX) > 0 && (A[□] & BOX) == 0
            # This is a box in B that is not in the same place in A
            push!(moved_boxes_b, □)
        end
    end
    if length(moved_boxes_b) != 1
        return nothing
    end

    return (moved_boxes_a[1] => moved_boxes_b[1])
end

"""
Construct a TrainingEntry0 based on a 1-box transition from A to B.
Training examples are the steps for which we have a moved box.
The goal is the transformed problem (replacing other boxes with walls, only goal is the box pos),
And the states are the real states.
"""
function build_level0_board_for_transition(
        board::Board,
        □_box_src::TileIndex,
        □_box_dst::TileIndex,
        remove_player::Bool)::Board
    retval = deepcopy(board)

    # replace all boxes except the target box with walls
    for (□, v) in enumerate(retval)
        if (v & BOX) > 0 && □ != □_box_src
            retval[□] &= ~BOX
            retval[□] |= WALL
        end
    end

    # remove all goals
    for □ in tile_indices(retval)
        retval[□] &= ~GOAL
    end

    # place a goal at the final box position
    retval[□_box_dst] |= GOAL

    if remove_player
        remove_player!(retval)
    end

    return retval
end

function construct_solved_training_entry_from_transition(A::Board, B::Board)::Union{Nothing, TrainingEntry0}

    # Ensure that we have one moved box
    box_pair = find_moved_box(A, B)
    if isnothing(box_pair)
        return nothing
    end

    # Build our simplified problem
    board_src = build_level0_board_for_transition(A, box_pair[1], box_pair[2], false)
    board_dst = build_level0_board_for_transition(B, box_pair[2], box_pair[2], true)

    # Solve the problem
    solver = AStar(
        100,
        1.0,
        5000000
    )
    subgame = Game(board_src)
    solve_data = solve(solver, subgame)
    if !solve_data.solved
        return nothing
    end

    state = State(subgame, subgame.board_start)
    moves = convert_to_moves(subgame, state, solve_data.pushes[1:solve_data.sol_depth])
    n_moves = length(moves)

    return TrainingEntry0(
        board_src,
        board_dst,
        moves,
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
function load_training_set(T, directory::String)
    entries = T[]

    for content in readdir(directory)
        fullpath = joinpath(directory, content)
        if isdir(fullpath)
            append!(entries, load_training_set(T, fullpath))
        elseif isfile(fullpath)
            if endswith(content, ".txt")
                push!(entries, load_training_entry(T, fullpath))
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
        inputs::Array{Bool}, # 8×8×5×s×b
        board::Board,
        i_seq::Int,
        i_batch::Int)
    ncols, nrows = size(board)
    for row in 1:nrows
        for col in 1:ncols
            v = board[col, row]
            inputs[col, row, DIM_WALL,   i_seq, i_batch] = is_set(v, WALL)
            inputs[col, row, DIM_FLOOR,  i_seq, i_batch] = !is_set(v, WALL)
            inputs[col, row, DIM_GOAL,   i_seq, i_batch] = is_set(v, GOAL)
            inputs[col, row, DIM_BOX,    i_seq, i_batch] = is_set(v, BOX)
            inputs[col, row, DIM_PLAYER, i_seq, i_batch] = is_set(v, PLAYER)
        end
    end
    return inputs
end

function set_board_from_input!(
        board::Board,
        inputs::Array{Bool}, # 8×8×5×s×b
        i_seq::Int,
        i_batch::Int)
    ncols, nrows = size(board)
    for row in 1:nrows
        for col in 1:ncols
            board[col, row] =
                inputs[col, row, DIM_WALL,   i_seq, i_batch] * WALL +
                inputs[col, row, DIM_GOAL,   i_seq, i_batch] * GOAL +
                inputs[col, row, DIM_BOX,    i_seq, i_batch] * BOX +
                inputs[col, row, DIM_PLAYER, i_seq, i_batch] * PLAYER
        end
    end
    return board
end

# --------------------------------------------------------------------------

"""
Shift the given tensor in its 1st and 2nd dimension with zero-padding.

e.g. shift_tensor(tensor, 1, 0, 0) will produce:

   [1 1 1 1 1]      [  1 1 1 1]
   [1 2     1]      [  1 2    ]
   [1   1   1]  ->  [  1   1  ]
   [1 1 1 1 1]      [  1 1 1 1]
"""
function shift_tensor(tensor::AbstractArray, d_row::Integer, d_col::Integer, pad_value)
    pad_up    = max( d_row, 0)
    pad_down  = max(-d_row, 0)
    pad_left  = max( d_col, 0)
    pad_right = max(-d_col, 0)

    tensor_padded = NNlib.pad_constant(
        tensor,
        (pad_up, pad_down, pad_left, pad_right, (0 for i in 1:2*(ndims(tensor)-2))...),
        pad_value)

    dims = size(tensor_padded)
    row_lo = 1 + pad_down
    row_hi = dims[1] - pad_up
    col_lo = 1 + pad_right
    col_hi = dims[2] - pad_left

    return tensor_padded[row_lo:row_hi, col_lo:col_hi, (Colon() for d in dims[3:end])...]
end

"""
Advance all sparse board states by moving one step in the row and column space.

  UP    = (d_row=-1, d_col= 0)
  LEFT  = (d_row= 0, d_col=-1)
  DOWN  = (d_row=+1, d_col= 0)
  RIGHT = (d_row= 0, d_col=+1)

Returns two new tensors:
    player_new: [h,w,s,b]
    boxes_new:  [h,w,s,b]
"""
function advance_boards(
    inputs::AbstractArray{Bool}, # [h,w,f,s,b]
    d_row::Integer,
    d_col::Integer)

    boxes  = inputs[:,:,DIM_BOX,   :,:] # [h,w,s,b]
    player = inputs[:,:,DIM_PLAYER,:,:] # [h,w,s,b]
    walls  = inputs[:,:,DIM_WALL,  :,:] # [h,w,s,b]

    player_shifted = shift_tensor(player, d_row, d_col, false)
    player_2_shift = shift_tensor(player_shifted, d_row, d_col, false)

    # A move is valid if the player destination is empty
    # or if its a box and the next space over is empty
    not_box_or_wall = .!(boxes .| walls)

    # 1 if it is a valid player destination tile for a basic player move (not a push)
    move_space_empty = player_shifted .& not_box_or_wall

    # 1 if the tile is a player destination tile containing a box
    move_space_isbox = player_shifted .& boxes

    # 1 if the tile is a player destination tile whose next one over is a valid box push receptor
    push_space_empty = player_shifted .& shift_tensor(not_box_or_wall, -d_row, -d_col, false)

    # 1 if it is a valid player move destination
    move_mask = move_space_empty

    # 1 if it is a valid player push destination
    # (which also means it currently has a box)
    push_mask = move_space_isbox .& push_space_empty

    # new player location
    mask = move_mask .| push_mask
    player_new = mask .| (player .* shift_tensor(.!mask, -d_row, -d_col, false))

    # new box location
    box_destinations = shift_tensor(boxes .* push_mask, d_row, d_col, false)
    boxes_new = (boxes .* (.!push_mask)) .| box_destinations

    return player_new, boxes_new
end

function are_solved(inputs::AbstractArray{Bool}) # [h,w,f,s,b]
    boxes = inputs[:,:,DIM_BOX,  :,:] # [h,w,s,b]
    goals = inputs[:,:,DIM_GOAL, :,:] # [h,w,s,b]
    box_not_on_goal = boxes .⊻ goals
    is_failed = reshape(any(box_not_on_goal, dims=(1,2)), (size(inputs, 4), size(inputs, 5))) # [s,b]
    return .!is_failed  # [s,b]
end

function advance_boards(
    inputs::AbstractArray{Bool}, # [h,w,f,s,b]
    actions::AbstractArray{Int}) #       [s,b]

    succ_u = advance_boards(inputs, -1,  0) # [h,w,s,d], [h,w,s,d]
    succ_l = advance_boards(inputs,  0, -1)
    succ_d = advance_boards(inputs,  1,  0)
    succ_r = advance_boards(inputs,  0,  1)

    size_u = size(succ_u[1])
    target_dims = (size_u[1], size_u[2], 1, size_u[3:end]...)
    player_news = cat(
        reshape(succ_u[1], target_dims),
        reshape(succ_l[1], target_dims),
        reshape(succ_d[1], target_dims),
        reshape(succ_r[1], target_dims), dims=3) # [h,w,a,s,d]
    box_news = cat(
        reshape(succ_u[2], target_dims),
        reshape(succ_l[2], target_dims),
        reshape(succ_d[2], target_dims),
        reshape(succ_r[2], target_dims), dims=3) # [h,w,a,s,d]

    actions_onehot = onehotbatch(actions, 1:4) # [a,s,d]
    actions_onehot = reshape(actions_onehot, (1,1,size(actions_onehot)...)) # [1,1,a,s,d]

    boxes_new = any(actions_onehot .& box_news, dims=3)
    player_new = any(actions_onehot .& player_news, dims=3)

    return cat(inputs[:,:,1:3,:,:], boxes_new, player_new, dims=3)
end

function advance_board_inputs(
    inputs::AbstractArray{Bool}, # [8,8,5,s,b]
    actions::AbstractArray{Int}) #       [s,b]

    inputs_new = advance_boards(inputs, actions)

    # Right shift and keep the goal and starting state
    return cat(inputs[:, :, :, 1:2, :], inputs_new[:, :, :, 2:end-1, :], dims=4) # [8,8,5,s,b]
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

struct BoardEncoder
    conv1::Conv
    conv2::Conv
    conv3::Conv
    dense::Dense
    layernorm::Flux.LayerNorm
end

Flux.@functor BoardEncoder

function BoardEncoder(encoding_dim::Int)
    e = encoding_dim

    conv1 = Conv((5,5), 5=>8, relu, stride=1, pad=2)
    conv2 = Conv((5,5), 8=>8, relu, stride=1, pad=2)
    conv3 = Conv((5,5), 8=>2, relu, stride=1, pad=2)
    dense = Dense(128 => e, relu)
    layernorm = Flux.LayerNorm(e)
    return BoardEncoder(conv1, conv2, conv3, dense, layernorm)
end

function (m::BoardEncoder)(X::AbstractArray{Bool}) # [8×8×5×...]
    orig_size = size(X)

    # 8×8×5×s×b
    # Combine the trailing dimensions into the batch dimension
    X = reshape(X, orig_size[1], orig_size[2], orig_size[3], :)
    # 8×8×5×sb
    X = m.conv1(X)
    # 8×8×8×sb
    X = m.conv2(X)
    # 8×8×8×sb
    X = m.conv3(X)
    # 8×8×2×sb
    X = reshape(Flux.flatten(X), 128, :)
    # 128×sb
    X = m.dense(X)
    # e×sb
    X = reshape(X, size(X)[1], orig_size[4:end]...)
    # e×s×b
    return m.layernorm(X)
end

# --------------------------------------------------------------------------

# The Level-0 Policy
struct SokobanPolicyLevel0
    batch_size::Int   # b, batch size
    max_seq_len::Int  # s, maximum input length, in number of tokens
    encoding_dim::Int # e, dimension of the embedding space

    encoder::BoardEncoder
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

    # The encoder encodes the states into token embeddings (8×8×5×s×b →  e×s×b)
    encoder = BoardEncoder(e)

    dropout = Dropout(dropout_prob)
    pos_enc = positional_encoding(s, e)
    mask = no_past_info ? input_only_mask(s) : basic_causal_mask(s)

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

# A call method for when we skip the board encoder
# TODO: Consolidate?
function (m::SokobanPolicyLevel0)(input::AbstractArray{Float32, 3}) # [e×s×b]

    X = input .+ m.pos_enc
    X = m.dropout(X)

    for layer in m.trunk
        X = layer(X, m.mask)
    end

    action_logits = m.action_head(X) # [4×s×b]
    nsteps_logits = m.nsteps_head(X) # [7×s×b]

    return (action_logits, nsteps_logits)
end

# Call method for when we run the board encoder
function (m::SokobanPolicyLevel0)(input::AbstractArray{Bool, 5}) # [8×8×5×s×b]

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
    inputs::AbstractArray{Bool}         # 8×8×5×s×b
    policy_target::AbstractArray{Bool}  # 4×s×b  onehot encoding of target action
    nsteps_target::AbstractArray{Bool}  # 7×s×b  onehot encoding of target discrete nsteps bin
    policy_mask::AbstractArray{Bool}    # 4×s×b  whether to train on a given sample
    nsteps_mask::AbstractArray{Bool}    # 7×s×b  whether to train on a given sample
end

function SokobanPolicyLevel0Data(s::Integer, b::Integer)
    inputs = zeros(Bool, (8, 8, 5, s, b))
    policy_target = zeros(Bool, (4, s, b))
    nsteps_target = zeros(Bool, (7, s, b))
    policy_mask = zeros(Bool, size(policy_target))
    nsteps_mask = zeros(Bool, size(nsteps_target))
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
        inputs::Array{Bool},    # 8×8×5×s×b
        policy_target::Array{Bool}, # 4×s×b
        nsteps_target::Array{Bool}, # 7×s×b
        policy_mask::Array{Bool},   # 4×s×b
        nsteps_mask::Array{Bool},   # 7×s×b
        training_entry::TrainingEntry0,
        batch_index::Int)

    # Ensure that our training entry is not longer than our model can handle
    n_moves = length(training_entry.moves)
    @assert n_moves < size(inputs, 4)

    bi = batch_index

    # Only train on things we set to 1
    policy_mask[:, :, bi] .= 0
    nsteps_mask[:, :, bi] .= 0

    # The goal always goes into the first entry
    set_board_input!(inputs, training_entry.s_goal, 1, bi)
    policy_target[:, 1, bi] .= true
    nsteps_target[:, 1, bi] .= true

    # Walk through the game and place the states
    game = Game(deepcopy(training_entry.s_start))
    state = State(game, game.board_start)

    si = 2 # seq index
    for dir in training_entry.moves

        set_board_input!(inputs, state.board, si, bi)
        policy_target[:, si, bi] .= false
        policy_target[dir, si, bi] = true


        nsteps_target[:, si, bi] .= false
        nsteps_remaining = length(training_entry.moves) + 1 - si
        idx = Int(clamp(round(log(nsteps_remaining+1)), 0, 5)) + 1
        nsteps_target[idx, si, bi] = true

        policy_mask[:, si, bi] .= true # train on the action
        nsteps_mask[:, si, bi] .= true # train on the step count

        # Advance the state
        succeeded = maybe_move!(state, game, dir)
        @assert succeeded # we expect valid moves in all training examples
        si += 1
    end

    # Always set the final state (which is the first state in an unsolved entry),
    # but the policy target is not trained on.
    set_board_input!(inputs, state.board, si, bi)
    policy_target[:, si, bi] .= true
    nsteps_target[:, si, bi] .= false
    nsteps_mask[:, si, bi] .= true
    if n_moves > 0
        nsteps_target[1, si, bi] = true # solved.
    else
        nsteps_target[end, si, bi] = true # cannot be solved
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
    data.inputs[:, :, :, :, batch_index] .= false
    data.policy_mask[:, :, batch_index] .= false
    data.nsteps_mask[:, :, batch_index] .= false
    data.policy_target[:, :, batch_index] .= true
    data.nsteps_target[:, :, batch_index] .= true
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
        n_mha_heads = params["n_mha_heads"],
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
    inputs::Array{Bool}     # [8×8×5×s×b]
    moves::Array{Direction} #       [s×b]  The move trajectories maintained by beam search
    scores::Vector{Float32} #         [b]  The running log likelihoods for the beam
    depth::Int              # How deep we got before returning. moves[depth] is the last move we used.
    solution::Int           # Batch index of the trajectory that solves the problem (or -1 otherwise)
end

function beam_search_old!(
        data::BeamSearchData,
        s_start::Board,
        s_goal::Board,
        policy::SokobanPolicyLevel0
    )

    policy = gpu(policy)
    inputs = cpu(data.inputs)
    moves = data.moves
    scores = data.scores

    b = size(inputs, 5) # The beam width is equal to the batch size
    @assert b == size(moves, 2)

    s = size(inputs, 4)
    @assert s == size(moves, 1)

    # We're going to track a batch worth of states in our beam
    game = Game(deepcopy(s_start))
    states = [State(game, deepcopy(s_start)) for i in 1:b]
    fill!(scores, 0.0) # running log likelihood of the actions

    # The number of beams.
    # At the start, we only have a single beam (the root trajectory)
    n_beams = 1

    inputs_next = zeros(Bool, size(inputs))
    states_next = Array{State}(undef, b)
    moves_next = zeros(Direction, (s, b))

    # The array of beam scores, which has one value per action, per beam.
    scores_next = Array{Float32}(undef, 5*b)
    p = collect(1:length(scores_next)) # permutation vector

    # The goal goes into the first entry
    for beam_index in 1:b
        set_board_input!(inputs, s_goal, 1, beam_index)
    end

    # Advance the games in parallel
    for seq_index in 2:s

        for beam_index in 1:n_beams
            set_board_input!(inputs, states[beam_index].board, seq_index, beam_index)
        end

        # Run the model
        # policy_logits are [4 × ntokens × batch_size]
        # nsteps_logits are [7 × ntokens × batch_size]
        policy_logits_gpu, nsteps_logits_gpu = policy(gpu(inputs))

        # Compute the probabilities
        probabilities = softmax(policy_logits_gpu, dims=1) |> cpu # [4 × ntokens × batch_size]

        # Compute the score for each beam.
        # An invalid / unused beam gets a score of -Inf.
        fill!(scores_next, -Inf)
        for i in 1:size(probabilities, 1)
            for j in 1:n_beams
                k = b*(i-1) + j
                scores_next[k] = scores[j] + log(probabilities[i,seq_index,j])
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

            # Set the next state to the current state.
            maybe_move!(states_next[beam_index_dst], game, dir)

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

function beam_search!(
    inputs::Array{Bool, 5},      # [h×w×f×s×b]
    policy0::SokobanPolicyLevel0,
    s_start::Board,
    s_goal::Board)

    policy0 = gpu(policy0)

    h, w, f, s, b = size(inputs)

    # Fill the goals and starting states into the first sequence channel
    for bi in 1:b
        set_board_input!(inputs, s_goal, 1, bi)
        set_board_input!(inputs, s_start, 2, bi)
    end

    # Keep track of the actual actions
    actions = ones(Int, s, b) |> gpu # [s, b]

    inputs_gpu = gpu(inputs)

    # Advance the games in parallel
    si = 2
    while si < s

        # Run the model
        # policy_logits are [4 × s × b]
        # nsteps_logits are [7 × s × b]
        policy_logits, nsteps_logits = policy0(inputs_gpu)

        # Compute the probabilities
        action_probs = softmax(policy_logits, dims=1) # [4 × s × b]
        action_logls = log.(action_probs) # [4 × s × b]

        # Base beam scores are how likely each beam thus far is to be solved
        nsteps_probs = softmax(nsteps_logits, dims=1) # [7 × s × b]
        nsteps_logls = log.(nsteps_probs) # [7 × s × b]
        beam_scores = Float32(1.0) .- nsteps_logls[7:7, si, :] # [1, b]

        # The beam scores are the base beam scores plus the action log likelihoods
        candidate_beam_scores = beam_scores .+ action_logls[:, si, :] # [4, b]

        # Get the top 'b' beams
        candidate_beam_scores_flat = vec(candidate_beam_scores) # [4b]
        topk_indices = partialsortperm(candidate_beam_scores_flat, 1:b; rev=true)

        # Convert flat indices back to action and beam indices
        selected_actions = (topk_indices .- 1) .÷ b .+ 1  # [b] action indices (1 to 4)
        selected_beams   = (topk_indices .- 1) .% b .+ 1  # [b] beam indices (1 to b)
        selected_scores  = candidate_beam_scores_flat[topk_indices]  # [b]
        inputs_gpu = inputs_gpu[:,:,:,:,selected_beams]

        actions[si,:] = selected_actions

        # Apply the actions to the selected beams
        inputs_gpu = advance_board_inputs(inputs_gpu, actions)

        if any(are_solved(inputs_gpu)[si,:])
            break
        end

        si += 1
    end

    return (cpu(inputs_gpu), cpu(actions), si)
end

# --------------------------------------------------------------------------

function sample_gumbel_noise(s::Integer, b::Integer)
    return rand(Gumbel{Float32}(0.0f0, 1.0f0), (4, s, b))
end

"""
Run parallel rollouts using the given model, updating the provided inputs tensor.
The rollout happens on the GPU.
"""
function rollouts!(
    inputs::Array{Bool, 5},      # [h×w×f×s×b]
    gumbel_noise::Array{Float32, 3}, # [4×s×b]
    policy0::SokobanPolicyLevel0,
    s_starts::Vector{Board}, # [b]
    s_goals::Vector{Board}) # [b]

    policy0 = gpu(policy0)

    h, w, f, s, b = size(inputs)

    @assert length(s_starts) == b
    @assert length(s_goals) == b

    # Fill the goals into the first sequence channel
    for (bi, s_goal) in enumerate(s_goals)
        set_board_input!(inputs, s_goal, 1, bi)
    end

    # Fill the start states in the second sequence channel
    for (bi, s_start) in enumerate(s_starts)
        set_board_input!(inputs, s_start, 2, bi)
    end

    inputs_gpu = gpu(inputs)
    gumbel_noise_gpu = gpu(gumbel_noise)

    for si in 2:s-1

        # Run the model
        # policy_logits are [4 × s × b]
        # nsteps_logits are [7 × s × b]
        policy_logits_gpu, nsteps_logits_gpu = policy0(inputs_gpu)

        #     # Optionally bias the policy logits.
        #     policy_logits_seq[:,:] .+= policy_dirichlet_prior # [4 × b]

        # Sample from the action logits using the Gumbel-max trick
        actions_gpu = argmax(policy_logits_gpu .+ gumbel_noise_gpu, dims=1) # CartesianIndex{3}[1 × s × ,b]
        actions_gpu = getindex.(actions_gpu, 1) # Int64[1 × s × b]
        actions_gpu = dropdims(actions_gpu, dims=1) # Int64[s × b]

        # Apply the actions
        inputs_gpu = advance_board_inputs(inputs_gpu, actions_gpu)
    end

    return cpu(inputs_gpu)
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
            # k_values[1,i,j] will be the index of the top value for the predictions for (i,j)
            # k_values[2,i,j] will be the index of the 2nd best value ... etc.
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
            # k_values[1,i,j] will be the index of the top value for the predictions for (i,j)
            # k_values[2,i,j] will be the index of the 2nd best value ... etc.
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
    inputs::Array{Bool, 5},      # [h×w×f×s×b]
    validation_set::Vector{TrainingEntry0},
    )

    s = size(inputs,4)

    policy = gpu(policy)

    m_validation_set = length(validation_set)

    n_solved_tp = 0
    n_solved_fp = 0
    n_failed_tp = 0
    n_failed_fp = 0

    accumulated_solution_length = 0

    for training_entry in validation_set

        gumbel_noise = sample_gumbel_noise(policy.max_seq_len, policy.batch_size)
        inputs, actions, depth = beam_search!(inputs, policy, training_entry.s_start, training_entry.s_goal)

        is_solved = depth < s

        if !is_solved
            if has_solution(training_entry)
                n_failed_fp += 1 # we should not have failed this one
            else
                n_failed_tp += 1 # we expected not to solve this one
            end
        else # solved
            if has_solution(training_entry)
                n_solved_tp += 1 # we had a stored solution for this one
            else
                n_solved_fp += 1 # this one should not be solvable
            end

            accumulated_solution_length += depth
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
        "solved_rate" => n_solved_tp / (n_solved_tp + n_failed_fp),
        "average_solution_length" => accumulated_solution_length / (n_solved_tp + n_solved_fp),
    )
end


# --------------------------------------------------------------------------
struct GoalEmbeddingHead
    seeds::AbstractArray{Float32} # learned goal embedding seeds [e×a×1×1]
    affine1::Dense
    affine2::Dense
end

Flux.@functor GoalEmbeddingHead

function GoalEmbeddingHead(encoding_dim::Int, n_actions::Int, hidden_dim_scale::Int; bias=true, init=Flux.glorot_uniform)
    hidden_dim = encoding_dim * hidden_dim_scale
    seeds = Float32.(Flux.glorot_uniform(encoding_dim, n_actions, 1, 1))
    affine1 = Dense(encoding_dim => hidden_dim; bias=bias, init=init)
    affine2 = Dense(hidden_dim => encoding_dim; bias=bias, init=init)
    return GoalEmbeddingHead(seeds, affine1, affine2)
end

function (m::GoalEmbeddingHead)(X::AbstractArray{Float32, 3}) # [e×s×b]
    e,s,b = size(X)

    X = reshape(X, (e, 1, s, b)) # [e×1×s×b]
    A = X .+ m.seeds # [e×a×s×b]

    A = m.affine1(A)
    A = relu(A)
    return m.affine2(A)
end

# --------------------------------------------------------------------------

struct SokobanPolicyLevel1
    batch_size::Int   # batch size
    max_seq_len::Int  # maximum input length, in number of tokens
    encoding_dim::Int # dimension of the embedding space
    n_actions::Int    # number of actions that this outputs

    encoder::BoardEncoder
    pos_enc::AbstractMatrix{Float32}
    dropout::Dropout
    mask::AbstractMatrix{Bool} # causal mask, [s × s], mask[i,j] means input j attends to input i
    trunk::Vector{PolicyTransformerLayer}

    action_head::Dense # produces n_actions logits
    nsteps_head::Dense # P(min(round(log(n+1)), 5)) plus inf, produces logits
    goalem_head_μ::GoalEmbeddingHead # produces n_actions goal embeddings
    goalem_head_logν::GoalEmbeddingHead # the log variance of the goal embedding means
end

Flux.@functor SokobanPolicyLevel1

function SokobanPolicyLevel1(;
        batch_size::Int = 32,
        max_seq_len::Int = 32,
        encoding_dim::Int = 8,
        n_actions::Int = 4,
        num_trunk_layers::Int = 3,
        n_mha_heads::Int = 8,
        trunk_hidden_dim_scale::Int = 4,
        action_hidden_dim_scale::Int = 2,
        encoder_conv_dim::Int = 8,
        init = Flux.glorot_uniform,
        dropout_prob::Float64 = 0.0,
        )

    e = encoding_dim
    f = encoder_conv_dim
    a = n_actions
    b = batch_size
    s = max_seq_len

    dropout = Dropout(dropout_prob)
    pos_enc = positional_encoding(s, e)
    mask = basic_causal_mask(s)

    encoder = BoardEncoder(e)

    # The trunk is the workhorse of the transformer, and iteratively
    # applies self-attention and skip-connection nonlinearities
    trunk = Array{PolicyTransformerLayer}(undef, num_trunk_layers)
    for i_trunk_layer in 1:num_trunk_layers
        trunk[i_trunk_layer] = PolicyTransformerLayer(
            e, init=init, dropout_prob=dropout_prob,
            nheads=n_mha_heads, hidden_dim_scale=trunk_hidden_dim_scale)
    end

    goalem_hidden_dim = e*action_hidden_dim_scale

    action_head = Dense(e => a; bias=true, init=init)
    nsteps_head = Dense(e => 7; bias=true, init=init)
    goalem_head_μ = GoalEmbeddingHead(e, a, action_hidden_dim_scale, bias=true, init=init)
    goalem_head_logν = GoalEmbeddingHead(e, a, action_hidden_dim_scale, bias=true, init=init)

    return SokobanPolicyLevel1(
        batch_size, max_seq_len, encoding_dim, n_actions,
        encoder, pos_enc, dropout, mask, trunk, action_head, nsteps_head, goalem_head_μ, goalem_head_logν)
end

function load_policy(::Type{SokobanPolicyLevel1}, model_directory::AbstractString, params::Dict{String,Any})

    policy = SokobanPolicyLevel1(
        batch_size = params["batch_size"],
        max_seq_len = params["max_seq_len"],
        encoding_dim = params["encoding_dim"],
        n_actions = params["n_actions"],
        num_trunk_layers = params["num_trunk_layers"],
        n_mha_heads = params["n_mha_heads"],
        trunk_hidden_dim_scale = params["trunk_hidden_dim_scale"],
        action_hidden_dim_scale = params["action_hidden_dim_scale"],
        encoder_conv_dim = params["encoder_conv_dim"],
        dropout_prob=params["dropout_prob"])

    model_state = JLD2.load(joinpath(model_directory, "model1.jld2"), "model_state");
    Flux.loadmodel!(policy, model_state);

    return policy
end

function (m::SokobanPolicyLevel1)(input::AbstractArray{Bool, 5}) # [8×8×5×s×b]

    X = m.encoder(input) .+ m.pos_enc
    X = m.dropout(X)

    for layer in m.trunk
        X = layer(X, m.mask)
    end

    action_logits        = m.action_head(X)      # [  a×s×b]
    nsteps_logits        = m.nsteps_head(X)      # [  7×s×b]
    goal_embeddings_μ    = m.goalem_head_μ(X)    # [e×a×s×b]
    goal_embeddings_logν = m.goalem_head_logν(X) # [e×a×s×b]

    return (action_logits, nsteps_logits, goal_embeddings_μ, goal_embeddings_logν)
end

# TODO: Could most of these be UInt8 since they are sparse?
struct SokobanPolicyLevel1Data
    inputs::AbstractArray{Bool}   # 8 × 8 × 5 × s × b
    policy_target::AbstractArray{Float32} # e × s × b (embeddings)
    nsteps_target::AbstractArray{Bool}    # 7 × s × b
    policy_mask::AbstractArray{Bool}      # 1 × s × b
    nsteps_mask::AbstractArray{Bool}      # 7 × s × b
end

function SokobanPolicyLevel1Data(max_seq_len::Int, batch_size::Int, encoding_dim::Int, n_actions::Int)
    s, b, e, a = max_seq_len, batch_size, encoding_dim, n_actions

    inputs     = zeros(Bool, (8, 8, 5, s, b))
    policy_target = zeros(Float32, (e, s, b))
    nsteps_target = zeros(Bool,    (7, s, b))
    policy_mask   = zeros(Bool,    (1, s, b))
    nsteps_mask   = zeros(Bool,    (7, s, b))
    return SokobanPolicyLevel1Data(inputs, policy_target, nsteps_target, policy_mask, nsteps_mask)
end

function SokobanPolicyLevel1Data(m::SokobanPolicyLevel1)
    return SokobanPolicyLevel1Data(m.max_seq_len, m.batch_size, m.encoding_dim, m.n_actions)
end

function to_gpu(tpd::SokobanPolicyLevel1Data)
    return SokobanPolicyLevel1Data(
        gpu(tpd.inputs),
        gpu(tpd.policy_target),
        gpu(tpd.nsteps_target),
        gpu(tpd.policy_mask),
        gpu(tpd.nsteps_mask)
    )
end

function to_cpu(tpd::SokobanPolicyLevel1Data)
    return SokobanPolicyLevel1Data(
        cpu(tpd.inputs),
        cpu(tpd.policy_target),
        cpu(tpd.nsteps_target),
        cpu(tpd.policy_mask),
        cpu(tpd.nsteps_mask)
    )
end


# --------------------------------------------------------------------------

# # Unpack a TrainingEntry1 into the tensor representation used by the model.
function convert_training_entry!(
        inputs::Array{Bool},       # 8×8×5×s×b
        policy_target::Array{Float32}, # e×s×b
        nsteps_target::Array{Bool},    # 7×s×b
        policy_mask::Array{Bool},      # a×s×b
        nsteps_mask::Array{Bool},      # 7×s×b
        training_entry::TrainingEntry1,
        encoder::BoardEncoder,
        batch_index::Int)

    h, w, f, s, b = size(inputs)

    # Ensure that our training entry is not longer than our model can handle
    n_states = length(training_entry.states)
    @assert n_states < s

    bi = batch_index

    # Only train on things we set to true
    policy_mask[:, :, bi] .= false
    nsteps_mask[:, :, bi] .= false

    # The overall goal is the final state (sans player),
    # Or the one goal if it exists.
    goal_board = deepcopy(training_entry.s_start)
    if length(training_entry.states) > 0
        goal_board[:] = training_entry.states[end]
    elseif length(training_entry.goals) > 0
        goal_board[:] = training_entry.goals[end]
    end
    remove_player!(goal_board)

    # The goal always goes into the first entry
    set_board_input!(inputs, goal_board, 1, bi)
    policy_target[:, 1, bi] .= true
    nsteps_target[:, 1, bi] .= true

    # Walk through the game and place the states
    all_states = Board[]
    push!(all_states, training_entry.s_start)
    append!(all_states, training_entry.states)

    goal_inputs = Array{Bool}(undef, h, w, f, 1, 1)

    si = 2 # seq index

    # Note that all_states is 1 longer than goals, so we end up with an extra state.
    for (board, goal) in zip(all_states, training_entry.goals)

        set_board_input!(inputs, board, si, bi)
        set_board_input!(goal_inputs, goal, 1, 1)
        policy_target[:, si, bi] = encoder(goal_inputs)[:,:,:,1,1]

        nsteps_target[:, si, bi] .= false
        nsteps_remaining = length(training_entry.states) + 1 - si
        idx = Int(clamp(round(log(nsteps_remaining+1)), 0, 5)) + 1
        nsteps_target[idx, si, bi] = true

        policy_mask[:, si, bi] .= true # train on the action
        nsteps_mask[:, si, bi] .= true # train on the step count

        si += 1
    end

    # Always set the final state (which is the first state in an unsolved entry),
    # but the policy target is not trained on.
    set_board_input!(inputs, all_states[end], si, bi)
    policy_target[:, si, bi] .= false
    nsteps_target[:, si, bi] .= false
    nsteps_mask[:, si, bi] .= 1
    if training_entry.solved
        nsteps_target[1, si, bi] = true # solved.
    else
        nsteps_target[end, si, bi] = true # infinite steps to solve w/out undo
    end
end

function convert_training_entry!(
        data::SokobanPolicyLevel1Data,
        training_entry::TrainingEntry1,
        encoder::BoardEncoder,
        batch_index::Int)
    convert_training_entry!(
        data.inputs,
        data.policy_target,
        data.nsteps_target,
        data.policy_mask,
        data.nsteps_mask,
        training_entry,
        encoder,
        batch_index)
    return data
end

function clear!(data::SokobanPolicyLevel1Data, batch_index::Int)
    data.inputs[:, :, :, :, batch_index] .= false
    data.policy_mask[:, :, batch_index] .= 0 # don't train on it
    data.nsteps_mask[:, :, batch_index] .= false # don't train on it
    data.policy_target[:, :, batch_index] .= false # empty embedding
    data.nsteps_target[:, :, batch_index] .= true
    return data
end

# --------------------------------------------------------------------------

"""
Gaussian mixture model loss

 ℓ are the mixture logits, of shape   [a×...]
 μ is the mean prediction, of shape [e×a×...]
 logν is the diagonal log variance  [e×a×...]
 y_true is the true data              [e×...]
 mask                                 [1×...]

This loss assumes diagonal Gaussians. We ignore constant terms and get:

  pdf = exp(-0.5*sum( νᵢ(xᵢ - μᵢ)² )

And then we apply a mixture weight over this and take the log
"""
function gmm_loss(
    ℓ::AbstractArray{Float32},
    μ::AbstractArray{Float32},
    logν::AbstractArray{Float32},
    y_true::AbstractArray{Float32},
    mask::AbstractArray{Bool},
    )

    w = softmax(ℓ; dims=1)  # mixing coeffs
    ν = exp.(logν)  # variances

    # Note that this pdf ignores constant terms
    y_true = reshape(y_true, size(y_true)[1], 1, size(y_true)[2:end]...) # [e×1×...]
    Δ = y_true .- μ                                                      # [e×a×...]
    pdfs = exp.(-0.5f0 .* sum(ν .* (Δ.^2); dims=1))                      # [1×a×...]
    pdfs = reshape(pdfs, size(ℓ))                                        # [  a×...]

    weigted_pdfs = sum(w .* pdfs; dims=1) # [1×...]

    # negative log likelihood
    loss = -sum(log.(weigted_pdfs) .* mask)

    return loss
end

function calc_metrics_gpu(
    policy::SokobanPolicyLevel1,
    data::SokobanPolicyLevel1Data,
    validation_set::Vector{TrainingEntry1},
    encoder::BoardEncoder,
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
                convert_training_entry!(data, validation_set[i_training_entry], encoder, batch_index)
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
        ℓ_gpu, nsteps_logits_gpu, μ_gpu, logν_gpu = policy(inputs_gpu)

        # Compute the GMM loss
        n_policy_targets = Flux.sum(policy_mask_gpu)
        if n_policy_targets > 0
            policy_loss += (gmm_loss(ℓ_gpu, μ_gpu, logν_gpu, policy_target_gpu, policy_mask_gpu) / n_policy_targets) * n_nonzero_entries

            # THIS PART RUNS ON THE CPU
            # Sort μ_gpu embeddings by L2 distance to the target goal embeddings.
            embedding_dists_gpu = let
                x = reshape(policy_target_gpu, size(policy_target_gpu)[1], 1, size(policy_target_gpu)[2:end]...) # [e×1×...]
                Δ = x .- μ_gpu # [e×a×...]
                dists = sqrt.(sum(x.^2; dims=2)) # [e×...]
                dropdims(dists; dims=2) # [e×...]
            end
            embedding_dists_cpu = cpu(embedding_dists_gpu)
            ek_values = mapslices(x -> sortperm(x, rev=false), embedding_dists_cpu; dims=1)

            # Then sort the predictions.
            # If policy_logits is [a×s×b] then k_values is also [a×s×b].
            # k_values[1,s,b] will be the index of the top value for the predictions for (s,b)
            # k_values[2,s,b] will be the index of the 2nd best value ... etc.
            policy_logits_cpu = cpu(ℓ_gpu)
            k_values = mapslices(x -> sortperm(x, rev=true), policy_logits_cpu; dims=1)

            for s in 2:policy.max_seq_len # skip the first one, since it is the goal
                for b in 1:policy.batch_size
                    if data.policy_mask[1,s,b] > 0.9
                        ek1 = ek_values[1,s,b] # the index of the action whose embedding is closest to the target embedding
                        k1 = k_values[1,s,b]
                        k2 = k_values[2,s,b]
                        iscorrect1 = k1 == ek1
                        iscorrect2 = k2 == ek1
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
            # k_values[1,i,j] will be the index of the top value for the predictions for (i,j)
            # k_values[2,i,j] will be the index of the 2nd best value ... etc.
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

# --------------------------------------------------------------------------

mutable struct RolloutResults1
    # Moves are stored as just a long list that we append to
    moves::Vector{Vector{Direction}} # [b]

    # Delineations between L0 and L1 are captured here
    # The length of moves[bi] is added here at the start of every L0 call
    level_delineations::Vector{Vector{Int}} # [b]

    # Whether each run was solved
    solved::BitVector
end

function rollout(
    policy0::SokobanPolicyLevel0,
    policy1::SokobanPolicyLevel1,
    inputs0_cpu::Array{Float32}, # [8×8×5×s×b]
    inputs1_cpu::Array{Float32}, # [    e×s×b]
    board::Board)

    policy1 = gpu(policy1)
    policy0 = gpu(policy0)

    # TODO: Add to inputs
    board_input = Array{Float32}(undef, 8, 8, 5, 1, 1)

    s0 = policy0.max_seq_len
    s1 = policy1.max_seq_len
    b = policy1.batch_size
    e = policy1.encoding_dim

    num_running = b
    solved1 = falses(b) # keep track of whether each rollout is done
    solved0 = falses(b)
    n_steps = zeros(Int, b)
    states = [State(Game(deepcopy(board))) for i in 1:b]

    moves = [Direction[] for i in 1:b]
    level_delineations = [Int[] for i in 1:b]

    gumbel = Gumbel{Float32}(0.0f0, 1.0f0)

    # Construct a goal board where all boxes are on goals.
    goal_board = construct_naive_goal_board(board)

    # Fill the goals into the first sequence channel
    for bi in 1:b
        set_board_input!(inputs1_cpu, goal_board, 1, bi)
    end

    # Fill the start states in the next sequence channel
    for (bi, s_start) in enumerate(states)
        set_board_input!(inputs1_cpu, board, 2, bi)
    end

    # Simulate forward
    for si1 in 2:s1
        if num_running ≤ 0
            break
        end

        inputs1_gpu = gpu(inputs1_cpu)
        ℓ_gpu, nsteps_logits_gpu, μ_gpu, logν_gpu = policy1(inputs1_gpu)

        ℓ_cpu = cpu(ℓ_gpu)[:,si1,:] #   [a×b]
        μ_cpu = cpu(μ_gpu)[:,:,si1,:] # [e×a×b]
        σ_cpu = cpu(sqrt.(exp.(logν_gpu)))[:,:,si1,:] # [e×a×b]

        # Sample from the actions using the Gumbel-max trick

        gumbel_noise = rand(gumbel, size(ℓ_cpu))
        sampled_actions = argmax(ℓ_cpu .+ gumbel_noise, dims=1) # [b]

        # Apply the selected goal embeddings to the level-0 problem.
        fill!(inputs0_cpu, zero(Float32))
        for (bi, is_finished) in enumerate(solved1)
            if is_finished
                continue
            end

            ai = sampled_actions[bi][1]
            inputs0_cpu[:,1,bi] = μ_cpu[:,ai,bi] + σ_cpu[:,ai,bi].*randn(e)

            # TODO: This is horribly inefficient
            inputs0_cpu[:,2,bi] = cpu(policy0.encoder(inputs1_gpu))[:,si1,bi]

            push!(level_delineations[bi], length(moves[bi]))
        end

        # Run the level-0 problem
        num_running2 = num_running
        solved0[:] = solved1
        for si0 in 2:s0
            if num_running2 ≤ 0
                break
            end

            # Run the model
            # policy_logits are [4 × s × b]
            # nsteps_logits are [7 × s × b]
            policy_logits_gpu, nsteps_logits_gpu = policy0(gpu(inputs0_cpu))

            nsteps_probs = softmax(cpu(nsteps_logits_gpu), dims=1) # [7 × s × b]

            # Pull out the logits for our current sequence index.
            policy_logits_seq = cpu(policy_logits_gpu)[:,si0,:] # [4 × b]

            # Sample from the action logits using the Gumbel-max trick
            gumbel_noise = rand(gumbel, size(policy_logits_seq))
            sampled_actions = argmax(policy_logits_seq .+ gumbel_noise, dims=1)

            # Apply the actions
            for bi in 1:b
                if solved0[bi]
                    continue # no need to simulate this one any further
                end

                # Grab the sampled action
                ai = sampled_actions[bi][1]
                dir = Direction(ai)
                push!(moves[bi], dir)

                # Simulate the state forward
                successful_move = maybe_move!(states[bi], dir)

                # Stop when the stopping likelihood is high
                # Note that this does not mean that the overall rollout is done, just this L0 rollout is.
                if nsteps_probs[si0] > 0.5
                    solved0[bi] = true
                    num_running2 -= 1
                end

                # Fill in the next input
                if si0 < s0
                    # TODO: This is all inefficient
                    set_board_input!(board_input, states[bi].board, 1, 1)
                    inputs0_cpu[:,si0+1,bi] = cpu(policy0.encoder(gpu(board_input)))[:,1,1]
                end
            end
        end

        for bi in 1:b
            if !solved1[bi] && is_solved(states[bi])
                solved1[bi] = true
                num_running -= 1
            end
        end
    end

    return RolloutResults1(moves, level_delineations, solved1)
end

# Run a batch worth of rollouts per training entry, and see how well we do in terms of solving it.
# The goal in each case is just to solve each board.
function calc_rollout_metrics(
    policy0::SokobanPolicyLevel0,
    policy1::SokobanPolicyLevel1,
    boards::Vector{Board})

    b = policy1.batch_size
    s0 = policy0.max_seq_len
    s1 = policy1.max_seq_len
    e0 = policy0.encoding_dim
    e1 = policy1.encoding_dim

    n_boards = length(boards)
    n_rollouts = b*n_boards

    # The number of total rollouts that resulted in a solved board
    n_solved_per_rollout = 0

    # The number of boards for which at least one rollout solved the board
    n_solved_per_board = 0

    # The mean step-solution length
    n_steps_per_rollout = 0 # across all rollouts
    n_steps_per_solution = 0 # across all solved rollouts
    n_steps_per_board = 0 # min per board

    # The mean L1-action length
    n_l1act_per_rollout = 0
    n_l1act_per_solution = 0
    n_l1act_per_board = 0

    @assert policy0.batch_size == b # assumed for now

    inputs0_cpu = Array{Float32}(undef, e0, s0, b)
    inputs1_cpu = Array{Float32}(undef, 8, 8, 5, s1, b)

    for board in boards
        res = rollout(policy0, policy1, inputs0_cpu, inputs1_cpu, board)
        n_solved_per_rollout += sum(res.solved)
        n_solved_per_board += any(res.solved)
        n_steps_per_rollout += sum(length.(res.moves))
        n_steps_per_solution += sum(res.solved[bi] * length(res.moves[bi]) for bi in 1:b)
        n_steps_per_board += any(res.solved) ? minimum(res.solved[bi] * length(res.moves[bi]) for bi in 1:b) : 0
        n_l1act_per_rollout += sum(length.(res.level_delineations))
        n_l1act_per_solution += sum(res.solved[bi] * length(res.level_delineations[bi]) for bi in 1:b)
        n_l1act_per_board += any(res.solved) ? minimum(res.solved[bi] * length(res.level_delineations[bi]) for bi in 1:b) : 0
    end

    return Dict{String, Float64}(
        "n_boards"                => n_boards,
        "n_rollouts_per_board"    => b,
        "solve_rate_all_rollouts" => n_solved_per_rollout / n_rollouts,
        "solve_rate_per_board"    => n_solved_per_board / n_boards,
        "ave_nsteps_per_rollout"  => n_steps_per_rollout / n_rollouts,
        "ave_nsteps_per_solution" => n_steps_per_solution / n_solved_per_rollout,
        "ave_nsteps_per_board"    => n_steps_per_board / n_boards,
        "ave_l1act_per_rollout"   => n_l1act_per_rollout / n_rollouts,
        "ave_l1act_per_solution"  => n_l1act_per_solution / n_solved_per_rollout,
        "ave_l1act_per_board"     => n_l1act_per_board / n_boards,
    )
end

# GOAL: evaluate rollouts for validation set
#  1. run it as-is and see how bad performance is
#  2. validate rollouts by
#            1. running with real board encoding
#            2. printouts
#  3. on GPU


# Dict{String, Float64} with 10 entries:
#   "ave_nsteps_per_rollout"  => 375.083
#   "n_rollouts_per_board"    => 32.0
#   "ave_nsteps_per_board"    => 0.0
#   "ave_l1act_per_board"     => 0.0
#   "solve_rate_per_board"    => 0.08
#   "ave_nsteps_per_solution" => 127.167
#   "ave_l1act_per_solution"  => 21.1429
#   "ave_l1act_per_rollout"   => 62.4506
#   "n_boards"                => 100.0
#   "solve_rate_all_rollouts" => 0.013125