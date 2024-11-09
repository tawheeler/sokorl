"""
Check to see if the given value is either its max or min extrema
"""
is_type_extrema(v::T) where T <: Real = v == typemax(T) || v == typemin(T)

function is_set(v::TileValue, flags::TileValue)
    return (v & flags) > 0
end

function not_set(v::TileValue, flags::TileValue)
    return (v & flags) == 0
end

"""
Get a tile's char representation.
"""
function square_to_char(v::TileValue)::Char
    retval = CH_FLOOR
    pbgw = v & (PLAYER+BOX+GOAL+WALL)
    if pbgw == PLAYER
        retval = CH_PLAYER
    elseif pbgw == PLAYER+GOAL
        retval = CH_PLAYER_ON_GOAL
    elseif pbgw == BOX || pbgw == PLAYER+BOX # can happen for deadlock-sets
        retval = CH_BOX
    elseif pbgw == BOX+GOAL || pbgw == PLAYER+BOX+GOAL # can happen for deadlock-sets
        retval = CH_BOX_ON_GOAL
    elseif pbgw == GOAL
        retval = CH_GOAL
    elseif pbgw == WALL
        retval = CH_WALL
    end
    return retval
end

function char_to_square(c::Char)::TileValue
    if c == CH_BOX || c == '$'
        return BOX
    elseif c == CH_BOX_ON_GOAL || c == '*'
        return BOX+GOAL
    elseif c == CH_GOAL
        return GOAL
    elseif c == CH_FLOOR || c == CH_NON_BLANK_FLOOR
        return FLOOR
    elseif c == CH_PLAYER || c == '@'
        return PLAYER
    elseif c == CH_PLAYER_ON_GOAL || c == '+'
        return PLAYER+GOAL
    elseif c == CH_SQUARE_SET
        return FLAG_PLAYER_REACHABLE_SQUARE
    elseif c == CH_WALL
        return WALL
    else
        return FLOOR
    end
end

function dir_to_char(dir::Direction)::Char
    retval = '?'
    if dir == DIR_UP
        retval = '↑'
    elseif dir == DIR_LEFT
        retval = '←'
    elseif dir == DIR_DOWN
        retval = '↓'
    elseif dir == DIR_RIGHT
        retval = '→'
    end
    return retval
end

"""
Convert a dx and dy between two tiles to a direction
"""
function dxdy_to_direction(dx::Integer, dy::Integer)::Direction
    @assert(dx == 0 || dy == 0)
    if dx == 0
        return dy ≤ 0 ? DIR_UP : DIR_DOWN
    else
        return dx ≤ 0 ? DIR_LEFT : DIR_RIGHT
    end
end

function print_char_matrix(chars)
    for i in 1:length(chars)
        for c in chars[i]
            print(c)
        end
        println("")
    end
end

function rotr90_dir(dir::Direction, k::Integer)
    return mod1(dir - k, N_DIRS)
end

function transpose_dir(dir::Direction)
    if dir == DIR_UP
        return DIR_LEFT
    elseif dir == DIR_LEFT
        return DIR_UP
    elseif dir == DIR_RIGHT
        return DIR_DOWN
    elseif dir == DIR_DOWN
        return DIR_RIGHT
    else
        return dir
    end
end

function board_to_text(board::Board)::String
    result = ""
    for row in 1:size(board,2)
        for col in 1:size(board,1)
            result *= square_to_char(board[row, col])
        end
        result *= "\n"
    end
    return result
end

function place_border!(board::Board)
    board_height, board_width = size(board)

    board[:,1] .= WALL
    board[:,board_width] .= WALL
    board[1,:] .= WALL
    board[board_height,:] .= WALL

    return board
end

function get_step_fore(board::Board, dir::Direction)::TileIndex
    if dir == DIR_UP
        return -1
    elseif dir == DIR_LEFT
        return -size(board, 1)
    elseif dir == DIR_DOWN
        return 1
    elseif dir == DIR_RIGHT
        return size(board, 1)
    else
        return 0 # should never happen
    end
end

# Find the first tile containing the player, or typemax if not found.
function find_player_tile(board::Board)::TileIndex
    for (□, v) in enumerate(board)
        if is_set(v, PLAYER)
            return □
        end
    end
    return typemax(TileIndex)
end


"""
Attempt to move the player one tile in the given direction.
If there is a box, and the next square over is vacant, push the box.
If the move is invalid (when walking into a wall, or into an unpushable box),
then do not do anything.
We return whether the move was successful.
"""
function maybe_move!(board::Board, dir::Direction, □_player::TileValue=find_player_tile(board))::Bool

    step_fore = get_step_fore(board, dir)

    □ = □_player # where the player starts
    ▩ = □ + step_fore # where the player potentially ends up

    if is_set(board[▩], WALL)
        return false # We would be walking into a wall
    end

    if is_set(board[▩], BOX)
        # We would be walking into a box.
        # This is only a legal move if we can push the box.
        ◰ = ▩ + step_fore # where box ends up
        if is_set(board[◰],  WALL + BOX)
            return false # We would be pushing the box into a box or wall
        end

        # Move the box
        board[▩] &= ~BOX # Clear the box
        board[◰] |= BOX # Add the box
    end

    # At this point we have established this as a legal move.
    # Finish by moving the player
    board[□] &= ~PLAYER # Clear the player
    board[▩] |= PLAYER # Add the player

    return true
end