# Problem generation

abstract type BoardGenAction end

# Generate a room that takes up a subset of the space, but is otherwise completely empty.
# There are no special checks to ensure that we don't overlap with existing rooms.
#
                ########
                ########
                #   ####
                #   ####
                #   ####
                #   ####
                ########
                ########
#
struct GenerateSubRoom <: BoardGenAction
    minimum_num_floor_times::Int
end
function apply_board_gen_action!(M::GenerateSubRoom, board::Board, rng)

    board_height, board_width = size(board)

    row_lo = row_hi = 1
    col_lo = col_hi = 1
    while (row_hi - row_lo + 1) * (col_hi - col_lo + 1) < M.minimum_num_floor_times
        row_span = rand(rng, 2:board_height-1, 2)
        row_lo = min(row_span[1], row_span[2])
        row_hi = max(row_span[1], row_span[2])
        col_span = rand(rng, 2:board_width-1, 2)
        col_lo = min(col_span[1], col_span[2])
        col_hi = max(col_span[1], col_span[2])
    end

    for row in row_lo:row_hi
        for col in col_lo:col_hi
            board[row,col] = FLOOR
        end
    end

    return board
end

struct SpawnGoal <: BoardGenAction end
function apply_board_gen_action!(M::SpawnGoal, board::Board, rng)

    # Select a random non-wall (non-goal) tile to be a goal
    n_tries = 100
    □_goal = 1
    while is_set(board[□_goal], WALL | GOAL) && n_tries > 0
        □_goal = rand(rng, 1:length(board))
        n_tries -= 1
    end
    board[□_goal] |= GOAL

    return board
end

struct SpawnBox <: BoardGenAction end
function apply_board_gen_action!(M::SpawnBox, board::Board, rng)

    # Select a random non-wall, non-goal (and non-box) tile to be the box starting position
    n_tries = 100
    □_box_start = 1
    while is_set(board[□_box_start], WALL | GOAL | BOX) && n_tries > 0
        □_box_start = rand(rng, 1:length(board))
        n_tries -= 1
    end
    board[□_box_start] |= BOX

    return board
end

struct SpawnPlayer <: BoardGenAction end
function apply_board_gen_action!(M::SpawnPlayer, board::Board, rng)

    # Select a random non-wall, non-box tile to be the player starting position
    n_tries = 100
    □_player_start = 1
    while is_set(board[□_player_start], WALL | BOX) && n_tries > 0
        □_player_start = rand(rng, 1:length(board))
        n_tries -= 1
    end
    board[□_player_start] |= PLAYER

    return board
end

struct SpeckleWalls <: BoardGenAction
    n_walls_lo::Int
    n_walls_hi::Int
end
function apply_board_gen_action!(M::SpeckleWalls, board::Board, rng)

    # Select random non-wall, non-player, non-box, non-goal tile to be walls.
    candidates = TileIndex[]
    for (□, v) in enumerate(board)
        if !is_set(v, WALL | BOX | GOAL | PLAYER)
            push!(candidates, □)
        end
    end

    # Determine how many walls to generate
    n_walls = rand(M.n_walls_lo:M.n_walls_hi)
    n_walls = min(n_walls, length(candidates))

    for □ in sample(rng, candidates, n_walls, replace=false)
        board[□] = WALL
    end

    return board
end


function generate_board(rng, board_height::Int, board_width::Int, steps::Vector{BoardGenAction})
    board = Matrix{TileValue}(undef, (board_height, board_width))
    fill!(board, WALL)

    for action in steps
        apply_board_gen_action!(action, board, rng)
    end

    return board
end