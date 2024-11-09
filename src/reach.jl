"""
A struct for efficiently caculating the reachable tiles from a given source tile.
The structure stores a vector of integers. For any given run, we increase our calc_counter by 2
and then compute the reachable tiles.
A reachable tile has tiles[□] == calc_counter.
A tile with a box on it that neighbors a reachable square has tiles[□] == calc_counter+1.
Non-floor tiles will have value typemax(ReachabilityCount).
"""
mutable struct ReachableTiles
    n_reachable_tiles::Int
    □_min::TileIndex # The minimum (top-left) reachable tile. This is often used as a normalized player position.
    calc_counter::ReachabilityCount # incremented by 2 on each call to calculate_player_reachable_squares
    tile_counters::Vector{ReachabilityCount} # the player's reachable tiles have the value 'calc_counter'; tiles with neighboring boxes have the value 'calc_counter' + 1}

    function ReachableTiles(n_tiles::Int)
        return new(
            0,
            zero(TileIndex),
            zero(ReachabilityCount),
            zeros(ReachabilityCount, n_tiles)
            )
    end
end

is_reachable(□::Integer, reach::ReachableTiles) = reach.tile_counters[□] == reach.calc_counter
is_reachable_box(□::Integer, reach::ReachableTiles) = reach.tile_counters[□] == reach.calc_counter + 1
have_same_reachability(□::Integer, ▩::Integer, reach::ReachableTiles) = reach.tile_counters[□] == reach.tile_counters[▩]

"""
This method resets the reachable tiles struct, setting the counter for all floor values to 0 and
the counter for all non-floor objects to typemax.
Note: A complete reset of the square timestamps is only necessary on initialization and when the timestamp overflows
"""
function clear!(reach::ReachableTiles, board::Board)
    reach.□_min = 0
    reach.calc_counter = 0

    for (□, v) in enumerate(board)
        if (v & WALL) == 0
            # A potentially-reachable floor tile
            reach.tile_counters[□] = 0
        else
            # Marking non-floors with high-value makes calculating reachable tiles more efficient
            reach.tile_counters[□] = typemax(ReachabilityCount)
        end
    end
    return reach
end

function construct_and_init_reach(board::Board)
    n_tiles = length(board)
    reach = ReachableTiles(n_tiles)
    return clear!(reach, board)
end

"""
Compute the tiles reachable from the given source tile.
Returns the number of reachable tiles.
Here 'stack' must have at least the same length as 'board'.
NOTE: The reach must have been previous cleared at some point.
"""
function calc_reachable_tiles!(
    reach::ReachableTiles,
    game::Game,
    board::Board,
    □_start::TileIndex,
    stack::Vector{TileIndex}
    )

    # The calc counter wraps around when the upper bound is reached.
    if reach.calc_counter ≥ typemax(ReachabilityCount) - 2
        clear!(reach, board)
    end

    reach.calc_counter += 2
    reach.□_min = □_start
    reach.tile_counters[□_start] = reach.calc_counter
    reach.n_reachable_tiles = 1

    stack[1] = □_start
    stack_hi = 1 # index of most recently pushed item

    # stack contains items as long as stack_hi > 0
    while stack_hi > 0
        # pop stack
        □ = stack[stack_hi]
        stack_hi -= 1

        # Examine tile neighbors
        dir = DIR_UP
        while dir <= DIR_RIGHT
            ▩ = □ + game.step_fore[dir]
            if reach.tile_counters[▩] < reach.calc_counter
                # An unvisited floor tile, possibly with a box (non-floors are typemax ReachabilityCount)
                if not_set(board[▩], BOX) # not a box
                    reach.n_reachable_tiles += 1
                    # push the square to the stack
                    stack_hi += 1
                    stack[stack_hi] = ▩
                    reach.tile_counters[▩] = reach.calc_counter
                    reach.□_min = min(▩, reach.□_min)
                else
                    # a box
                    reach.tile_counters[▩] = reach.calc_counter + 1
                end
            end
            dir += one(Direction)
        end
    end

    return reach
end

"""
Check whether the given tile is a culdesac - ie has walls in three directions.
Here, `dir` must face into the culdesac.
Ex:
    ###
  →  □#
    ###
"""
function is_culdesac(game::Game, s::State, □::TileIndex, dir::Direction)
    return is_set(s.board[□+game.step_fore[dir]], WALL) &&
           is_set(s.board[□+game.step_left[dir]], WALL) &&
           is_set(s.board[□+game.step_right[dir]], WALL)
end


"""
Calculate the distance to each reachable square from the given player position.
We assume that the reachable tiles were pre-calculated.
"""
function calc_distances!(
    distances::Vector{Int32}, # same length as board
    game::Game,
    □_player::TileIndex,
    reach::ReachableTiles,
    queue::Vector{Tuple{TileIndex, Int32}}
    )

    queue_lo = 0 # one below next valid index
    queue_hi = 0 # index of most recently added item

    # Set all distances to max
    fill!(distances, typemax(Int32))

    # Distance to the current state is zero
    queue_hi += 1
    queue[queue_hi] = (□_player, zero(Int32))
    distances[□_player] = zero(Int32)

    # Run breadth-first-search
    while queue_lo != queue_hi
        queue_lo += 1
        □, d = queue[queue_lo]
        d′ = d + one(Int32)

        for dir in DIRECTIONS
            ▩ = □ + game.step_fore[dir]
            if is_reachable(▩, reach) && (d′ < distances[▩])
                # This is an improvement
                distances[▩] = d′
                queue_hi += 1
                queue[queue_hi] = (▩, d′)
            end
        end
    end

    return distances
end

"""
Calculates the distance to all tiles that can be reached
by pulling or pushing a box around the board, starting from □_box.
Once done, distances_by_direction[□, dir] will contain the distance to get to □, with the player in the dir-side direction.
"""
function calc_box_pull_or_push_distances!(
    distances_by_direction::Matrix{Int32}, # Of size board_size × N_DIRS
    game::Game,
    s::State,
    □_box::TileIndex,
    reach::ReachableTiles,
    stack::Vector{TileIndex},
    queue::Vector{Tuple{TileIndex, TileIndex, Int32}}, # Of length board_size * N_DIRS, (box, player, distance)
    push_box::Bool, # If false, we pull the box
    use_player_position::Bool,
    continue_calculation::Bool,
    )

    # Store original state
    □_player_orig = s.□_player
    □_box_orig = □_box
    v_box_orig = s.board[□_box]

    # Remove the box, if any, from the board
    s.board[□_box] &= ~BOX

    dist_init_value = zero(Int32)
    if !continue_calculation
        # We are starting from scratch. Initialize `distances_by_direction`.
        for □ in tile_indices(game)
            if (s.board[□] & WALL) > 0
                # Set wall squares to typemin(Int32) - the search cannot find a better path to the tile
                for dir in DIRECTIONS
                    distances_by_direction[□, dir] = typemin(Int32)
                end
            else
                # Set floor squares to typemax(Int32) - the search may find a better path to the tile
                for dir in DIRECTIONS
                    distances_by_direction[□, dir] = typemax(Int32)
                end
            end
        end
    else
        # We chose to continue the calculation, which means we do not overwrite `distances_by_direction`.
        # This will taint reachable tiles.
        # It does not treat each continuation tile as a new 0-distance starting point.
        for dir in DIRECTIONS
            if !is_type_extrema(distances_by_direction[□_box,dir])
                dist_init_value = max(dist_init_value, distances_by_direction[□_box,dir])
            end
        end
    end

    □_box_prev = zero(TileIndex)

    queue_lo = 1 # Index of the next entry to grab. If == queue_hi, then no entries to grab.
    queue_hi = 1 # Index of the next empty entry

    if use_player_position
        s.board[□_box] |= BOX # add the box
        calc_reachable_tiles!(reach, game, s.board, s.□_player, stack)
        s.board[□_box] &= ~BOX # remove the box
    end

    # Enqueue all valid first pushes or pulls around the box.
    # If we push, then the player starts in the tile opposite dir.
    # If we pull, then the player starts in the tile in that dir.
    for dir_box_move in DIRECTIONS
        dir_player = push_box ? OPPOSITE_DIRECTION[dir_box_move] : dir_box_move
        □_player = □_box + game.step_fore[dir_player]
        if ((s.board[□_player] & WALL) == 0) &&
           (!use_player_position || is_reachable(□_player, reach))
            # The player tile is not a wall and, if we care, it is player-reachable.
            distances_by_direction[□_box, dir_player] = dist_init_value
            queue[queue_hi] = (□_box, □_player, zero(Int32))
            queue_hi += 1
        elseif !use_player_position
            # POSSIBLE BUG: This seems to allow pushing/pulling from walls, which is odd
            #               The only difference here is that we do not add this to the queue
            distances_by_direction[□_box, dir_player] = dist_init_value = dist_init_value
        end
    end

    # Run breadth-first search
    while queue_lo != queue_hi
        □_box, □_player, d = queue[queue_lo]
        queue_lo += 1

        # Set the player location and put the box on the board
        s.□_player = □_player
        s.board[□_box] |= BOX

        @assert (s.board[□_box] & WALL) == 0
        @assert (s.board[□_player] & WALL) == 0

        if (□_box != □_box_prev) || !is_reachable(s.□_player, reach)
            # The current set of player-reachable tiles is not valid anymore; recalculate the set
            □_box_prev = □_box
            calc_reachable_tiles!(reach, game, s.board, s.□_player, stack)
        end

        d′ = d+1 # The distance after pushing a box from this position
        for dir_box_move in DIRECTIONS
            dir_player = push_box ? OPPOSITE_DIRECTION[dir_box_move] : dir_box_move
            ▩_box = □_box + game.step_fore[dir_box_move] # tile the box ends up in
            □_player = □_box + game.step_fore[dir_player] # tile the player pushes or pulls from
            ▩_player = ▩_box + game.step_fore[dir_player] # tile the player ends up in

            accept_move = false
            if push_box
                if (distances_by_direction[▩_box, dir_player] > d′) && # we found a quicker path
                   is_reachable(□_player, reach) &&
                   ((s.board[▩_box] & (WALL+BOX)) == 0)
                    accept_move = true
                end
            else
                # We are pulling the box
                if (distances_by_direction[▩_box, dir_player] > d′) &&
                   is_reachable(□_player, reach) &&
                   is_reachable(▩_player, reach) &&
                   ((s.board[▩_box] & (WALL+BOX)) == 0) &&
                   (
                    (!is_culdesac(game, s, ▩_player, dir_player)) ||
                    (▩_player == □_player_orig) # if the player after the pull ends at its starting position in the game, then the square is OK
                   )
                    accept_move = true
                end
            end

            if accept_move
                distances_by_direction[▩_box, dir_player] = d′
                queue[queue_hi] = (▩_box, ▩_player, d′)
                queue_hi += 1
            end
        end

        # Remove the box again
        s.board[□_box] &= ~BOX
    end

    # Restore the game state
    s.□_player = □_player_orig
    s.board[□_box_orig] = v_box_orig

    return distances_by_direction
end

"""
Calculates the distance to all tiles that can be reached
by pushing a box around the board, starting from □_box.
Once done, distances_by_direction[□, dir] will contain the distance to get to □,
with the player in the dir-side direction.
"""
function calc_box_push_distances!(
    distances_by_direction::Matrix{Int32}, # Of size board_size × N_DIRS
    game::Game,
    s::State,
    □_box::TileIndex,
    reach::ReachableTiles,
    stack::Vector{TileIndex},
    queue::Vector{Tuple{TileIndex, TileIndex, Int32}}, # Of length board_size * N_DIRS, (box, player, distance)
    )

    # Store original state
    □_player_orig = s.□_player
    □_box_orig = □_box
    v_box_orig = s.board[□_box]

    s.board[□_box] |= BOX # add the box (it may not exist)
    calc_reachable_tiles!(reach, game, s.board, s.□_player, stack)
    s.board[□_box] &= ~BOX # remove the box

    # We are starting from scratch. Initialize `distances_by_direction`
    for □ in tile_indices(game)
        # Set wall squares to typemin(Int32) - the search cannot find a better path to the tile
        # Set floor squares to typemax(Int32) - the search may find a better path to the tile
        value = (s.board[□] & WALL) > 0 ? typemin(Int32) : typemax(Int32)
        for dir in DIRECTIONS
            distances_by_direction[□, dir] = value
        end
    end


    queue_lo = 1 # Index of the next entry to grab. If == queue_hi, then no entries to grab.
    queue_hi = 1 # Index of the next empty entry

    # Enqueue all valid first pushes around the box
    for dir_box_move in DIRECTIONS
        dir_player = OPPOSITE_DIRECTION[dir_box_move]
        □_player = □_box + game.step_fore[dir_player]
        if ((s.board[□_player] & WALL) == 0) && is_reachable(□_player, reach)
            # The player tile is not a wall and is player-reachable.
            distances_by_direction[□_box, dir_player] = zero(Int32)
            queue[queue_hi] = (□_box, □_player, zero(Int32))
            queue_hi += 1
        end
    end

    # Run breadth-first search
    □_box_prev = zero(TileIndex)
    while queue_lo != queue_hi
        □_box, □_player, d = queue[queue_lo]
        queue_lo += 1

        # Set the player location and put the box on the board
        s.□_player = □_player
        s.board[□_box] |= BOX

        @assert (s.board[□_box] & WALL) == 0
        @assert (s.board[□_player] & WALL) == 0

        if (□_box != □_box_prev) || !is_reachable(s.□_player, reach)
            # The current set of player-reachable tiles is not valid anymore; recalculate the set
            □_box_prev = □_box
            calc_reachable_tiles!(reach, game, s.board, s.□_player, stack)
        end

        d′ = d+1 # The distance after pushing a box from this position
        for dir_box_move in DIRECTIONS
            dir_player = OPPOSITE_DIRECTION[dir_box_move]
            ▩_box = □_box + game.step_fore[dir_box_move] # tile the box ends up in
            □_player = □_box + game.step_fore[dir_player] # tile the player pushes or pulls from
            ▩_player = □_box # tile the player ends up in

            # Accept the move if we found a quicker path, can reach the push source,
            # and the target tile is legal
            if (distances_by_direction[▩_box, dir_player] > d′) && # we found a quicker path
               (is_reachable(□_player, reach)) &&
               ((s.board[▩_box] & (WALL+BOX)) == 0)
                distances_by_direction[▩_box, dir_player] = d′
                queue[queue_hi] = (▩_box, ▩_player, d′)
                queue_hi += 1
            end
        end

        # Remove the box again
        s.board[□_box] &= ~BOX
    end

    # Restore the game state
    s.□_player = □_player_orig
    s.board[□_box_orig] = v_box_orig

    return distances_by_direction
end

"""
The same calculation as above, but we additionally populate, for each state,
the state that leads to it
"""
function calc_box_push_distances!(
    distances_by_direction::Matrix{Int32}, # Of size board_size × N_DIRS
    previous_states::Matrix{Tuple{TileIndex, TileIndex}}, # Of size board_size × N_DIRS, (box index, player index)
    game::Game,
    s::State,
    □_box::TileIndex,
    reach::ReachableTiles,
    stack::Vector{TileIndex},
    queue::Vector{Tuple{TileIndex, TileIndex, Int32}}, # Of length board_size * N_DIRS, (box, player, distance)
    )

    # Store original state
    □_player_orig = s.□_player
    □_box_orig = □_box
    v_box_orig = s.board[□_box]

    s.board[□_box] |= BOX # add the box (it may not exist)
    calc_reachable_tiles!(reach, game, s.board, s.□_player, stack)
    s.board[□_box] &= ~BOX # remove the box

    # We are starting from scratch. Initialize `distances_by_direction`
    for □ in tile_indices(game)
        # Set wall squares to typemin(Int32) - the search cannot find a better path to the tile
        # Set floor squares to typemax(Int32) - the search may find a better path to the tile
        value = (s.board[□] & WALL) > 0 ? typemin(Int32) : typemax(Int32)
        for dir in DIRECTIONS
            distances_by_direction[□, dir] = value
        end
    end

    queue_lo = 1 # Index of the next entry to grab. If == queue_hi, then no entries to grab.
    queue_hi = 1 # Index of the next empty entry

    # Enqueue all valid first pushes or pulls around the box
    for dir_box_move in DIRECTIONS
        dir_player = OPPOSITE_DIRECTION[dir_box_move]
        □_player = □_box + game.step_fore[dir_player]
        if ((s.board[□_player] & WALL) == 0) && is_reachable(□_player, reach)
            # The player tile is not a wall and is player-reachable.
            distances_by_direction[□_box, dir_player] = zero(Int32)
            previous_states[□_box, dir_player] = (zero(TileIndex), zero(TileIndex)) # no previous state
            queue[queue_hi] = (□_box, □_player, zero(Int32))
            queue_hi += 1
        end
    end

    # Run breadth-first search
    □_box_prev = zero(TileIndex)
    while queue_lo != queue_hi
        □_box, □_player, d = queue[queue_lo]
        queue_lo += 1

        # Set the player location and put the box on the board
        s.□_player = □_player
        s.board[□_box] |= BOX

        @assert (s.board[□_box] & WALL) == 0
        @assert (s.board[□_player] & WALL) == 0

        if (□_box != □_box_prev) || !is_reachable(s.□_player, reach)
            # The current set of player-reachable tiles is not valid anymore; recalculate the set
            □_box_prev = □_box
            calc_reachable_tiles!(reach, game, s.board, s.□_player, stack)
        end

        d′ = d+1 # The distance after pushing a box from this position
        for dir_box_move in DIRECTIONS
            dir_player = OPPOSITE_DIRECTION[dir_box_move]
            ▩_box = □_box + game.step_fore[dir_box_move] # tile the box ends up in
            □_player_push = □_box + game.step_fore[dir_player] # tile the player pushes or pulls from
            ▩_player = □_box # tile the player ends up in

            # Accept the move if we found a quicker path, can reach the push source,
            # and the target tile is legal
            if (distances_by_direction[▩_box, dir_player] > d′) && # we found a quicker path
               (is_reachable(□_player_push, reach)) &&
               ((s.board[▩_box] & (WALL+BOX)) == 0)
                distances_by_direction[▩_box, dir_player] = d′
                previous_states[▩_box, dir_player] = (□_box, □_player)
                queue[queue_hi] = (▩_box, ▩_player, d′)
                queue_hi += 1
            end
        end

        # Remove the box again
        s.board[□_box] &= ~BOX
    end

    # Restore the game state
    s.□_player = □_player_orig
    s.board[□_box_orig] = v_box_orig

    return distances_by_direction
end

"""
Calculates the distance to all tiles that can be reached
by pulling a box around the board, starting from □_box.
Once done, distances_by_direction[□, dir] will contain the distance to get to □, with the player in the dir-side direction.
"""
function calc_box_pull_distances!(
    distances_by_direction::Matrix{Int32}, # Of size board_size × N_DIRS
    game::Game,
    s::State,
    □_box::TileIndex,
    reach::ReachableTiles,
    stack::Vector{TileIndex},
    queue::Vector{Tuple{TileIndex, TileIndex, Int32}}, # Of length board_size * N_DIRS, (box, player, distance)
    )

    # Store original state
    □_player_orig = s.□_player
    □_box_orig = □_box
    v_box_orig = s.board[□_box]

    # Remove the box, if any, from the board
    s.board[□_box] &= ~BOX

    # We are starting from scratch. Initialize `distances_by_direction`.
    for □ in tile_indices(game)
        if (s.board[□] & WALL) > 0
            # Set wall squares to typemin(Int32) - the search cannot find a better path to the tile
            for dir in DIRECTIONS
                distances_by_direction[□, dir] = typemin(Int32)
            end
        else
            # Set floor squares to typemax(Int32) - the search may find a better path to the tile
            for dir in DIRECTIONS
                distances_by_direction[□, dir] = typemax(Int32)
            end
        end
    end

    queue_lo = 1 # Index of the next entry to grab. If == queue_hi, then no entries to grab.
    queue_hi = 1 # Index of the next empty entry

    # Enqueue all valid first pulls around the box.
    for dir_box_move in DIRECTIONS
        dir_player = dir_box_move
        □_player = □_box + game.step_fore[dir_player]
        if ((s.board[□_player] & WALL) == 0) && is_reachable(□_player, reach)
            # The player tile is not a wall and, if we care, it is player-reachable.
            distances_by_direction[□_box, dir_player] = zero(Int32)
            queue[queue_hi] = (□_box, □_player, zero(Int32))
            queue_hi += 1
        end
    end

    # Run breadth-first search
    □_box_prev = zero(TileIndex)
    while queue_lo != queue_hi
        □_box, □_player, d = queue[queue_lo]
        queue_lo += 1

        # Set the player location and put the box on the board
        s.□_player = □_player
        s.board[□_box] |= BOX

        @assert (s.board[□_box] & WALL) == 0
        @assert (s.board[□_player] & WALL) == 0

        if (□_box != □_box_prev) || !is_reachable(s.□_player, reach)
            # The current set of player-reachable tiles is not valid anymore; recalculate the set
            □_box_prev = □_box
            calc_reachable_tiles!(reach, game, s.board, s.□_player, stack)
        end

        d′ = d+1 # The distance after pulling a box from this position
        for dir_box_move in DIRECTIONS
            dir_player = dir_box_move
            ▩_box = □_box + game.step_fore[dir_box_move] # tile the box ends up in
            □_player = □_box + game.step_fore[dir_player] # tile the player pulls from
            ▩_player = ▩_box + game.step_fore[dir_player] # tile the player ends up in

            # We are pulling the box
            if (distances_by_direction[▩_box, dir_player] > d′) &&
               is_reachable(□_player, reach) &&
               is_reachable(▩_player, reach) &&
               ((s.board[▩_box] & (WALL+BOX)) == 0) &&
               (
                (!is_culdesac(game, s, ▩_player, dir_player)) ||
                (▩_player == □_player_orig) # if the player after the pull ends at its starting position in the game, then the square is OK
               )
                # Accept the move
                distances_by_direction[▩_box, dir_player] = d′
                queue[queue_hi] = (▩_box, ▩_player, d′)
                queue_hi += 1
            end
        end

        # Remove the box again
        s.board[□_box] &= ~BOX
    end

    # Restore the game state
    s.□_player = □_player_orig
    s.board[□_box_orig] = v_box_orig

    return distances_by_direction
end

"""
The same calculation as above, but we additionally populate, for each state,
the state that leads to it.
After calling this, previous_states[▩_box, dir_player] = (□_box, □_player)
where ▩_box is where the box is pulled to
      dir_player is the direction of the player from the ▩_box after the pull
      □_box is where the box was before the pull
      □_player is the direction of the player from the box after its previous pull
"""
function calc_box_pull_distances!(
    distances_by_direction::Matrix{Int32}, # Of size board_size × N_DIRS
    previous_states::Matrix{Tuple{TileIndex, TileIndex}}, # Of size board_size × N_DIRS, (box index, player index)
    game::Game,
    s::State,
    □_box::TileIndex,
    reach::ReachableTiles,
    stack::Vector{TileIndex},
    queue::Vector{Tuple{TileIndex, TileIndex, Int32}}, # Of length board_size * N_DIRS, (box, player, distance)
    )

    # Store original state
    □_player_orig = s.□_player
    □_box_orig = □_box
    v_box_orig = s.board[□_box]

    # Remove the box, if any, from the board
    s.board[□_box] &= ~BOX

    # We are starting from scratch. Initialize `distances_by_direction`.
    for □ in tile_indices(game)
        if (s.board[□] & WALL) > 0
            # Set wall squares to typemin(Int32) - the search cannot find a better path to the tile
            for dir in DIRECTIONS
                distances_by_direction[□, dir] = typemin(Int32)
            end
        else
            # Set floor squares to typemax(Int32) - the search may find a better path to the tile
            for dir in DIRECTIONS
                distances_by_direction[□, dir] = typemax(Int32)
            end
        end
    end

    queue_lo = 1 # Index of the next entry to grab. If == queue_hi, then no entries to grab.
    queue_hi = 1 # Index of the next empty entry

    # Enqueue all valid first pulls around the box.
    for dir_box_move in DIRECTIONS
        dir_player = dir_box_move
        □_player = □_box + game.step_fore[dir_player]
        if ((s.board[□_player] & WALL) == 0) && is_reachable(□_player, reach)
            # The player tile is not a wall and, if we care, it is player-reachable.
            distances_by_direction[□_box, dir_player] = zero(Int32)
            previous_states[□_box, dir_player] = (zero(TileIndex), zero(TileIndex)) # no previous state
            queue[queue_hi] = (□_box, □_player, zero(Int32))
            queue_hi += 1
        end
    end

    # Run breadth-first search
    □_box_prev = zero(TileIndex)
    while queue_lo != queue_hi
        □_box, □_player, d = queue[queue_lo]
        queue_lo += 1

        # Set the player location and put the box on the board
        s.□_player = □_player
        s.board[□_box] |= BOX

        @assert (s.board[□_box] & WALL) == 0
        @assert (s.board[□_player] & WALL) == 0

        if (□_box != □_box_prev) || !is_reachable(s.□_player, reach)
            # The current set of player-reachable tiles is not valid anymore; recalculate
            □_box_prev = □_box
            calc_reachable_tiles!(reach, game, s.board, s.□_player, stack)
        end

        d′ = d+1 # The distance after pulling a box from this position
        for dir_box_move in DIRECTIONS
            dir_player = dir_box_move
            ▩_box = □_box + game.step_fore[dir_box_move] # tile the box ends up in
            □_player_pull = □_box + game.step_fore[dir_player] # tile the player pulls from
            ▩_player = ▩_box + game.step_fore[dir_player] # tile the player ends up in

            # We are pulling the box
            if (distances_by_direction[▩_box, dir_player] > d′) &&
               is_reachable(□_player_pull, reach) &&
               is_reachable(▩_player, reach) &&
               ((s.board[▩_box] & (WALL+BOX)) == 0) &&
               (
                (!is_culdesac(game, s, ▩_player, dir_player)) ||
                (▩_player == □_player_orig) # if the player after the pull ends at its starting position in the game, then the square is OK
               )
                # Accept the move
                distances_by_direction[▩_box, dir_player] = d′
                previous_states[▩_box, dir_player] = (□_box, □_player)
                queue[queue_hi] = (▩_box, ▩_player, d′)
                queue_hi += 1
            end
        end

        # Remove the box again
        s.board[□_box] &= ~BOX
    end

    # Restore the game state
    s.□_player = □_player_orig
    s.board[□_box_orig] = v_box_orig

    return distances_by_direction
end

function init_distances_by_direction!(
    distances_by_direction::Matrix{Int32}, # Of size board_size × N_DIRS
    board::Board
    )
    for □ in 1:length(board)
        val = (board[□] & WALL) > 0 ? typemin(Int32) : typemax(Int32)
        for dir in DIRECTIONS
            distances_by_direction[□, dir] = val
        end
    end
    return distances_by_direction
end


"""
Calculate the minimum number of pushes from each tile to the nearest target tile,
in the ideal case where there are no (other) boxes.
The target tiles are given by the first `n_targets` elements in `stack`.
Walls have distance typemin(Int32).
Tiles that cannot reach any target have distance typemax(Int32).
"""
function calc_push_dist_to_targets_for_all_squares!(
    distances::Vector{Int32}, # Of length board_size
    game::Game,
    s::State,
    reach::ReachableTiles,
    stack::Vector{TileIndex}, # Of length board_size, pre-filled with target tiles
    queue::Vector{Tuple{TileIndex, TileIndex, Int32}}, # Of length board_size * N_DIRS
    n_targets::Int,
    distances_by_direction::Matrix{Int32} # Of size board_size × N_DIRS
    )

    # Get the state ready
    □_player_orig = s.□_player
    remove_all_boxes_from_board!(s)

    init_distances_by_direction!(distances_by_direction, s.board)

    # Enqueue items
    # Queue of box tile index, player tile index, distance ::Int32
    # Keep track of our queue bounds
    queue_lo = 1 # Index of the next entry to grab. If == queue_hi, then no entries to grab.
    queue_hi = 1 # Index of the next empty entry
    for i in 1:n_targets
        □_box = stack[i] # The target tile that a box has been pushed onto
        for dir in DIRECTIONS
            distances_by_direction[□_box,dir] = zero(Int32) # Distance is zero as box is on the tile
            □_player = □_box + game.step_fore[dir] # Potential player tile
            if (s.board[□_player] & WALL) == 0
                # not a wall - player can be there
                queue[queue_hi] = (□_box, □_player, zero(Int32))
                queue_hi += 1
            end
        end
    end

    # Run breadth-first search
    □_box_prev = zero(Int32)
    while queue_lo != queue_hi
        □_box, □_player, d = queue[queue_lo]
        s.□_player = □_player
        d′ = d+1 # The distance after pushing a box to the current position
        queue_lo += 1

        # Place the box on the board
        s.board[□_box] |= BOX

        # Recalculate the reach if it is no longer valid
        if (□_box != □_box_prev) || !is_reachable(□_player, reach)
            □_box_prev = □_box
            calc_reachable_tiles!(reach, game, s.board, s.□_player, stack)
        end

        # Try all pushes
        for dir in DIRECTIONS
            □_box_from = □_box + game.step_fore[dir] # Where the box was pushed from
            □_player_from = TileIndex(□_box + 2*game.step_fore[dir]) # Where the player was pushing from
            if (distances_by_direction[□_box_from, dir] > d′) &&
               (is_reachable(□_box_from, reach)) &&
               ((s.board[□_box_from] & (WALL+BOX)) == 0) &&
               ((s.board[□_player_from] & (WALL+BOX)) == 0) &&
               (
                (!is_culdesac(game, s, □_player_from, dir)) ||
                (□_player_from == game.□_player_start) # if the player after the pull ends at their starting position, then the square is ok
               )
                distances_by_direction[□_box_from, dir] = d′
                queue[queue_hi] = (□_box_from, □_player_from, d′)
                queue_hi += 1
            end
        end

        # Remove the box from the board again
        s.board[□_box] &= ~BOX
    end

    # Now back out the min push distances per tile.
    for □ in 1:length(distances)
        distances[□] = typemax(Int32)
        if (s.board[□] & WALL) == 0
            for dir in DIRECTIONS
                distances[□] = min(distances[□], distances_by_direction[□,dir])
            end
        end
    end

    # Return to the original state
    place_all_boxes_on_board!(s)
    s.□_player = □_player_orig

    return distances
end

"""
Calculate the minimum number of pushes from each target tile to all other tiles on the board,
in the ideal case where there are no (other) boxes.
The target tiles are given by the first `n_targets` elements in `stack`.
Walls have distance typemin(Int32).
Tiles that cannot reach any target have distance typemax(Int32).
"""
function calc_push_dist_from_targets_for_all_squares!(
    distances::Vector{Int32}, # Of length board_size
    game::Game,
    s::State,
    reach::ReachableTiles,
    stack::Vector{TileIndex}, # Of length board_size, pre-filled with target tiles
    queue::Vector{Tuple{TileIndex, TileIndex, Int32}}, # Of length board_size * N_DIRS
    n_targets::Int,
    distances_by_direction::Matrix{Int32} # Of size board_size × N_DIRS
    )

    # Get the state ready
    □_player_orig = s.□_player
    remove_all_boxes_from_board!(s)

    init_distances_by_direction!(distances_by_direction, s.board)

    # Enqueue items
    # Queue of box tile index, player tile index, distance ::Int32
    # Keep track of our queue bounds
    queue_lo = 1 # Index of the next entry to grab. If == queue_hi, then no entries to grab.
    queue_hi = 1 # Index of the next empty entry
    for i in 1:n_targets
        □_box = stack[i] # The target tile that a box can be pushed from
        for dir in DIRECTIONS
            distances_by_direction[□_box,dir] = zero(Int32) # Distance is zero as box is on the tile
            □_player = □_box + game.step_fore[dir] # Potential player tile
            if (s.board[□_player] & WALL) == 0
                # not a wall - player can be there
                queue[queue_hi] = (□_box, □_player, zero(Int32))
                queue_hi += 1
            end
        end
    end

    # Run breadth-first search
    □_box_prev = zero(Int32)
    while queue_lo != queue_hi
        □_box, □_player, d = queue[queue_lo]
        s.□_player = □_player
        d′ = d+1 # The distance after pushing a box to the current position
        queue_lo += 1

        # Place the box on the board
        s.board[□_box] |= BOX

        # Recalculate the reach if it is no longer valid
        if (□_box != □_box_prev) || !is_reachable(□_player, reach)
            □_box_prev = □_box
            calc_reachable_tiles!(reach, game, s.board, s.□_player, stack)
        end

        # Try all pushes
        for dir in DIRECTIONS
            □_box_to = □_box + game.step_fore[dir] # Where the box was pushed from
            □_player_from = □_box - game.step_fore[dir] # Where the player was pushing from
            if (distances_by_direction[□_box_to, dir] > d′) &&
               (is_reachable(□_player_from, reach)) &&
               ((s.board[□_box_to] & (WALL+BOX)) == 0)
                distances_by_direction[□_box_to, dir] = d′
                queue[queue_hi] = (□_box_to, □_box, d′)
                queue_hi += 1
            end
        end

        # Remove the box from the board again
        s.board[□_box] &= ~BOX
    end

    # Now back out the min push distances per tile.
    for □ in 1:length(distances)
        distances[□] = typemax(Int32)
        if (s.board[□] & WALL) == 0
            for dir in DIRECTIONS
                distances[□] = min(distances[□], distances_by_direction[□,dir])
            end
        end
    end

    # Return to the original state
    place_all_boxes_on_board!(s)
    s.□_player = □_player_orig

    return distances
end

function count_reachable_tiles(distances_by_direction::Matrix{Int32})
    n_reachable_tiles = 0
    for □ in 1:size(distances_by_direction,1)
        if distances_by_direction[□, DIR_UP] != typemin(Int32)
            # Not a wall
            if distances_by_direction[□, DIR_UP] != typemax(Int32) ||
               distances_by_direction[□, DIR_LEFT] != typemax(Int32) ||
               distances_by_direction[□, DIR_DOWN] != typemax(Int32) ||
               distances_by_direction[□, DIR_RIGHT] != typemax(Int32)
                # Reachable
                n_reachable_tiles += 1
            end
        end
    end
    return n_reachable_tiles
end

"""
Calculate the minimum number of pushes from each tile to the nearest goal,
in the ideal case where there are no boxes.
This distance is often used to compute search lower bounds.
Walls have distance typemin(Int32).
Squares that cannot reach any goal have distance typemax(Int32).
"""
function calc_push_dist_to_nearest_goal_for_all_squares!(
    distances::Vector{Int32}, # Of length board_size
    game::Game,
    s::State,
    reach::ReachableTiles,
    stack::Vector{TileIndex}, # Of length board_size
    queue::Vector{Tuple{TileIndex, TileIndex, Int32}}, # Of length board_size * N_DIRS
    distances_by_direction::Matrix{Int32} # Of size board_size × N_DIRS
    )

    # Add all goal squares
    n_targets = 0
    for (i_goal, □_goal) in enumerate(game.□_goals)
        # Only proceed if the goal square has not been temporarily disabled
        if (s.board[□_goal] & GOAL) > 0
            n_targets += 1
            stack[n_targets] = □_goal
        end
    end

    return calc_push_dist_to_targets_for_all_squares!(
        distances, game, s, reach, stack, queue, n_targets, distances_by_direction)
end

"""
Calculate the minimum number of pushes from each tile to the nearest box starting position,
in the ideal case where there are no boxes.
Walls have distance typemin(Int32).
Squares that cannot reach any box position have distance typemax(Int32).
"""
function calc_push_dist_to_nearest_box_position_for_all_tiles!(
    distances::Vector{Int32},
    game::Game,
    s::State,
    reach::ReachableTiles,
    stack::Vector{TileIndex}, # Of length board size
    queue::Vector{Tuple{TileIndex, TileIndex, Int32}}, # Of length board_size * N_DIRS
    distances_by_direction::Matrix{Int32} # Of size board_size × N_DIRS
    )

    # Add all box start squares
    n_targets = 0
    for (i_box, □_box) in enumerate(s.□_boxes)
        # Only proceed if the box square has not been temporarily disabled
        if (s.board[□_box] & WALL) == 0
            n_targets += 1
            stack[n_targets] = □_box
        end
    end

    return calc_push_dist_from_targets_for_all_squares!(
        distances, game, s, reach, stack, queue, n_targets, distances_by_direction)
end


"""
Convert a trajectory in pushes from the given start state to a trajectory in moves.
The given state should have the player in their correct starting position.
NOTE: We allocate memory.
"""
function convert_to_moves(game::Game, s0::State, pushes::Vector{Push})
    # Allocate memory
    s = deepcopy(s0)
    reach = ReachableTiles(length(s.board))
    clear!(reach, s.board)
    stack = Array{TileIndex}(undef, length(game.board_start))
    distances = zeros(Int32, length(s.board))
    queue = Array{Tuple{TileIndex, Int32}}(undef, length(s.board))

    moves = Direction[]
    for a in pushes
        ▩ = s.□_boxes[a.i_box] # box location
        □ = ▩ - game.step_fore[a.dir] # player push start location

        calc_reachable_tiles!(reach, game, s.board, s.□_player, stack)

        # Get distance to target
        □_player_orig = s.□_player
        s.□_player = □
        calc_distances!(distances, game, s.□_player, reach, queue)
        s.□_player = □_player_orig

        while s.□_player != □
            # Find a neighbor square that is closer.
            # We assume that it exists
            step_dir = typemax(Direction)
            ⬓ = zero(TileIndex)
            for dir in DIRECTIONS
                ◳ = s.□_player + game.step_fore[dir]
                if ⬓ == 0 || distances[◳] < distances[⬓]
                    ⬓ = ◳
                    step_dir = dir
                end
            end
            @assert step_dir != typemax(Direction)

            push!(moves, step_dir)
            maybe_move!(s, game, step_dir)
        end

        # Now apply the push
        push!(moves, a.dir)
        move!(s, game, a)
    end

    return moves
end