"""
A game defines the static information for a Sokoban puzzle.
It also contains some additional precomputed information.
"""
struct Game
    # The start board state, which contains a board
    # that includes some culling / precomputation
    board_start::Board

    # The player starting tile. Used to determine when a pull is legel.
    □_player_start::TileIndex

    # The goal locations. These are static and unchanging.
    # We always have the same number of boxes as we do goals.
    □_goals::Vector{TileIndex}

    step_fore::Vector{TileIndex} # □ + step_fore[dir] yields the square one step in the given direction from □
    step_left::Vector{TileIndex} # □ + step_left[dir] is the same as □ + step_fore[rot_left(dir)]
    step_right::Vector{TileIndex} # □ + step_right[dir] is the same as □ + step_fore[rot_right(dir)]

    # Zobrist hash components
    hash_components::Vector{UInt64}
end

tile_indices(board::Board) = TileIndex(1):TileIndex(length(board))
tile_indices(game::Game) = TileIndex(1):TileIndex(length(game.board))
box_indices(game::Game) = BoxIndex(1):BoxIndex(length(game.□_goals))
goal_indices(game::Game) = box_indices(game)

num_rows(board::Board) = size(board, 1)
num_rows(game::Game) = size(game.board_start, 1)
num_cols(board::Board) = size(board, 2)
num_cols(game::Game) = size(game.board_start, 2)


function Game(board::Board)

    # Compute our starting values
    □_player_start = 0
    □_goals = TileIndex[]
    for (□, v) in enumerate(board)
        if (v & GOAL) > 0
            push!(□_goals, □)
        end
        if (v & PLAYER) > 0
            □_player_start = □
        end
    end

    # Construct our steps
    board_height, board_width = size(board)
    step_fore = TileIndex[-1, -board_height, 1, board_height]
    step_left = TileIndex[step_fore[2], step_fore[3], step_fore[4], step_fore[1]]
    step_right = TileIndex[step_fore[4], step_fore[1], step_fore[2], step_fore[3]]

    # Construct our hash components
    hash_components = construct_zobrist_hash_components(length(board))

    return Game(
        copy(board),
        □_player_start,
        □_goals,
        step_fore,
        step_left,
        step_right,
        hash_components
        )
end

function Game(str::String)

    # Figure out the board height and width,
    # and verify that the board width does not change
    running_board_width = 0
    board_width = 0
    board_height = 1
    for c in str
        if c == '\n'
            if board_width == 0
                board_width = running_board_width
            else
                @assert board_width == running_board_width
            end
            running_board_width = 0
            board_height += 1
        else
            running_board_width += 1
        end
    end

    # Allocate the board
    board = zeros(TileValue, (board_height, board_width))

    place_border!(board)

    # Populate the board
    let
        row = 1
        col = 1
        for c in str

            if c == '\n'
                row += 1
                col = 1
                continue
            end

            # Only set the value if it is within the border
            if 1 < row < board_height && 1 < col < board_width
                board[row, col] = char_to_square(c)
            end

            col += 1
        end
    end

    return Game(board)
end

function Game(strings::Vector{String})
    max_len = maximum(length(str) for str in strings)

    big_string = ""
    for (str_index,str) in enumerate(strings)
        big_string *= str
        n_chars_to_add = max_len-length(str)
        for i in 1:n_chars_to_add
            big_string *= " "
        end
        if str_index != length(strings)
            big_string *= "\n"
        end
    end

    return Game(big_string)
end

"""
The dynamic state for a Sokoban game, which includes both the lean state information
and the board information for multiple forms of fast analysis.
IE `boxes` lets one quickly identify box locations,
whereas `board` lets one quickly verify if a given location has a box (or player), etc.
This struct is mutable.
"""
mutable struct State
    □_player::TileIndex
    □_boxes::Vector{TileIndex}
    board::Board
    zhash::UInt64
end

function State(n_boxes::Int, board_height::Int, board_width::Int)
    return State(
        zero(TileIndex),
        Array{TileIndex}(undef, length(n_boxes)),
        Array{TileValue}(undef, board_height, board_width),
        zero(UInt64)
    )
end

function State(game::Game, board::Board = game.board_start)
    □_player = zero(TileIndex)
    □_boxes = TileIndex[]
    for (□, v) in enumerate(board)
        if (v & PLAYER) > 0
            □_player = □
        end
        if (v & BOX) > 0
            push!(□_boxes, □)
        end
    end
    zhash = calculate_zobrist_hash(□_boxes, game.hash_components)
    return State(□_player, □_boxes, copy(board), zhash)
end

"""
Remove the player from a board state
"""
function remove_player!(s::State)
    s.board[s.□_player] &= ~PLAYER
    s.□_player = 0
end

"""
Move the player from one location to another.
"""
function move_player!(s::State, □_player::TileIndex)
    # remove the player from the board
    s.board[s.□_player] &= ~PLAYER

    # update the player position
    s.□_player = □_player

    # add the player to the board
    s.board[s.□_player] |= PLAYER
end

"""
Unset the box flag in the board entry of the given state.
This is useful sometimes when calculating reachability for ideal cases.
"""
function remove_all_boxes_from_board!(s::State)
    for □ in s.□_boxes
        s.board[□] &= ~BOX
    end
    return s
end

"""
Set the box flag in the board entry of the given state for each box.
"""
function place_all_boxes_on_board!(s::State)
    for □ in s.□_boxes
        s.board[□] |= BOX
    end
    return s
end

"""
Returns true if our state has been solved.
IE every box is on a goal.
"""
function is_solved(s::State)
    for □ in s.□_boxes
        if (s.board[□] & GOAL) == 0
            return false
        end
    end
    return true
end

"""
Attempt to move the player one tile in the given direction.
If there is a box, and the next square over is vacant, push the box.
If the move is invalid (when walking into a wall, or into an unpushable box),
then do not do anything.
We return whether the move was successful.
NOTE: We do not normalize the resulting player position
"""
function maybe_move!(s::State, game::Game, dir::Direction)::Bool
    □ = s.□_player # where the player starts
    ▩ = □ + game.step_fore[dir] # where the player potentially ends up

    if (s.board[▩] & WALL) > 0
        return false # We would be walking into a wall
    end

    if (s.board[▩] & BOX) > 0
        # We would be walking into a box.
        # This is only a legal move if we can push the box.
        ◰ = ▩ + game.step_fore[dir] # where box ends up
        if is_set(s.board[◰],  WALL + BOX)
            return false # We would be pushing the box into a box or wall
        end

        # Move the box
        # This should always succeed
        i_box = findfirst(isequal(▩), s.□_boxes)

        s.board[▩] &= ~BOX # Clear the box
        s.zhash ⊻= game.hash_components[▩] # Undo the box's hash component

        s.□_boxes[i_box] = ◰
        s.board[◰] |= BOX # Add the box
        s.zhash ⊻= game.hash_components[◰] # Add the box's hash component
    end

    # At this point we have established this as a legal move.
    # Finish by moving the player
    move_player!(s, ▩)

    return true
end

function maybe_move!(s::State, dir::Direction)::Bool
    step_fore = get_step_fore(s.board, dir)

    □ = s.□_player # where the player starts
    ▩ = □ + step_fore # where the player potentially ends up

    if (s.board[▩] & WALL) > 0
        return false # We would be walking into a wall
    end

    if (s.board[▩] & BOX) > 0
        # We would be walking into a box.
        # This is only a legal move if we can push the box.
        ◰ = ▩ + step_fore # where box ends up
        if is_set(s.board[◰],  WALL + BOX)
            return false # We would be pushing the box into a box or wall
        end

        # Move the box
        # This should always succeed
        i_box = findfirst(isequal(▩), s.□_boxes)

        s.board[▩] &= ~BOX # Clear the box
        s.zhash ⊻= game.hash_components[▩] # Undo the box's hash component

        s.□_boxes[i_box] = ◰
        s.board[◰] |= BOX # Add the box
        s.zhash ⊻= game.hash_components[◰] # Add the box's hash component
    end

    # At this point we have established this as a legal move.
    # Finish by moving the player
    move_player!(s, ▩)

    return true
end


"""
A push is defined as moving one box one step in one direction.
"""
struct Push
    i_box::BoxIndex
    dir::Direction
end

"""
Apply a push to a state, producing a successor state.
We assume the push is valid, meaning that the box is reachable by the player.
The player is placed in the box's location.
NOTE: We do not normalize the resulting player position
"""
function move!(s::State, game::Game, a::Push)
    □ = s.□_boxes[a.i_box] # where box starts
    ▩ = □ + game.step_fore[a.dir] # where box ends up

    s.board[s.□_player] &= ~PLAYER # Clear the player
    s.board[□] &= ~BOX # Clear the box
    s.zhash ⊻= game.hash_components[□] # Undo the box's hash component

    s.□_player = □ # Player ends up where the box is.
    s.□_boxes[a.i_box] = ▩

    s.board[s.□_player] |= PLAYER # Add the player
    s.board[▩] |= BOX # Add the box
    s.zhash ⊻= game.hash_components[▩] # Add the box's hash component

    return s
end

"""
Reverse a push on a state, producing the predecessor state.
We assume that the push is valid.
The player is placed in the location prior to push.
NOTE: We do not normalize the resulting player position.
"""
function unmove!(s::State, game::Game, a::Push)
    □ = s.□_boxes[a.i_box] # where box starts
    ▩ = □ - game.step_fore[a.dir] # where box ends up

    s.board[s.□_player] &= ~PLAYER # Clear the player
    s.board[□] &= ~BOX # Clear the box
    s.zhash ⊻= game.hash_components[□] # Undo the box's hash component

    s.□_player = ▩ - game.step_fore[a.dir] # Player ends up behind the box.
    s.□_boxes[a.i_box] = ▩

    s.board[s.□_player] |= PLAYER # Add the player
    s.board[▩] |= BOX # Add the box
    s.zhash ⊻= game.hash_components[▩] # Add the box's hash component

    return s
end

function tile_index_to_col_row(game::Game, □::Integer)::Tuple{Int,Int}
    colrow = CartesianIndices(size(game.board_start))[□]
    return (colrow[1], colrow[2])
end

"""
Get the Manhattan distance between two tile indices
"""
function get_manhattan_dist(game::Game, □::TileIndex, ▩::TileIndex)
    □_col, □_row = tile_index_to_col_row(game, □)
    ▩_col, ▩_row = tile_index_to_col_row(game, ▩)
    return abs(□_col - ▩_col) + abs(□_row - ▩_row)
end