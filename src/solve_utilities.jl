"""
The sum of the distance to the nearest goal for all boxes,
which is a lower bound on the number of pushes (and thus also moves) to complete the board.
"""
function calculate_simple_lower_bound(s::State, dist_to_nearest_goal::Vector{Int32})::Int32
    value = zero(Int32)
    for □ in s.□_boxes
        if □ != 0 # box is on the board / hasn't been temporarily removed
            distance = dist_to_nearest_goal[□]
            if distance == typemax(Int32)
                # This box cannot reachy any goal
                return distance
            end
            value += distance
        end
    end
    return value
end


"""
Get the list of valid pushes for the given state.
NOTE: We assume reachability has already been computed.
NOTE: This method is not all that efficient - it allocates a moves vector.
"""
function get_pushes(game::Game, s::State, reach::ReachableTiles)
    pushes = Push[]
    for (i_box, □) in enumerate(s.□_boxes)
        if is_reachable_box(□, reach)
            # Get a move for each direction we can get at it from,
            # provided that the opposing side is clear.
            for dir in DIRECTIONS
                □_dest = □ + game.step_fore[dir] # Destination tile
                □_player = □ - game.step_fore[dir] # Side of the player
                # If we can reach the player's pos and the destination is clear.
                if is_reachable(□_player, reach) && ((s.board[□_dest] & (FLOOR+BOX)) == FLOOR)
                    push!(pushes, Push(i_box, dir))
                end
            end
        end
    end
    return pushes
end

"""
Fill pushes with the list of valid pushes for the given state.
We return the number of valid pushes.
NOTE: We assume reachability has already been computed.
"""
function get_pushes!(pushes::Vector{Push}, game::Game, s::State, reach::ReachableTiles)::Int
    n_pushes = 0
    for (i_box, □) in enumerate(s.□_boxes)
        if is_reachable_box(□, reach)
            # Get a move for each direction we can get at it from,
            # provided that the opposing side is clear.
            for dir in DIRECTIONS
                □_dest = □ + game.step_fore[dir] # Destination tile
                □_player = □ - game.step_fore[dir] # Side of the player
                # If we can reach the player's pos and the destination is clear.
                if is_reachable(□_player, reach) && ((s.board[□_dest] & (FLOOR+BOX)) == FLOOR)
                    n_pushes += 1
                    pushes[n_pushes] = Push(i_box, dir)
                end
            end
        end
    end
    return n_pushes
end

"""
Given a push, produces the next valid push, with pushes ordered
by i_box and then by direction.
The first valid push can be obtained by passing in a box index of 1 and direction 0x00.
If there no valid next push, a push with box index 0 is returned.
"""
function get_next_push!(game::Game, s::State, reach::ReachableTiles, push::Push = Push(1, 0))
    i_box = push.i_box
    dir = push.dir
    while i_box ≤ length(s.□_boxes)
        □ = s.□_boxes[i_box]
        if is_reachable_box(□, reach)
            # Get a move for each direction we can get at it from,
            # provided that the opposing side is clear.
            while dir < N_DIRS
                dir += one(Direction)
                □_dest = □ + game.step_fore[dir] # Destination tile
                □_player = □ - game.step_fore[dir] # Side of the player
                # If we can reach the player's pos and the destination is clear.
                if is_reachable(□_player, reach) && ((s.□board[□_dest] & (FLOOR+BOX)) == FLOOR)
                    return Push(i_box, dir)
                end
            end
        end

        # Reset
        dir = zero(Direction)
        i_box += one(BoxIndex)
    end

    # No next push
    return Push(0, 0)
end