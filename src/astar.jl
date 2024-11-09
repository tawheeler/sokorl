"""
This solver uses A* via the simple lower bound.
It also has a transposition table.
"""
struct AStar
    d_max::Int       # maximum depth to evaluate to
    timeout::Float64 # maximum execution time in seconds
    n_positions::Int # number of positions to store in the transposition table
end
AStar(; d_max::Int=1, timeout::Float64=1.0, n_positions::Int=10) = AStar(d_max, timeout, n_positions)

get_params(solver::AStar) = Dict{String,Any}(
    "d_max"=>solver.d_max,
    "timeout"=>solver.timeout,
    "n_positions"=>solver.n_positions,
    )

mutable struct AStarData
    solved::Bool
    max_depth::Int # max depth reached during search
    n_pushes_evaled::Int # number of pushes evaluated during search
    n_pushes_skipped::Int # number of pushes skipped due to the transposition table
    n_infeasible_states::Int
    set_position_count_1::Int
    set_position_count_2::Int
    n_positions_used::Int
    setup_time::Float64 # time it takes to set up the problem
    total_time::Float64
    pushes::Vector{Push} # of length d_max
    sol_depth::Int # number of pushes in the solution
end

function print_results(data::AStarData, verbosity::Int=1)
    println("\t\tsolved: ", data.solved)
    println("\t\ttotal time: ", data.total_time)
    if verbosity > 0
        println("\t\tsetup_time: ", data.setup_time, " ($(round(data.setup_time/data.total_time*100,digits=2))%)")
        println("\t\tsolving time: ", data.total_time - data.setup_time)
        println("\t\tmax_depth: ", data.max_depth)
        println("\t\tsol_depth: ", data.sol_depth)
        println("\t\tn_pushes_evaled: ", data.n_pushes_evaled)
        println("\t\tn_pushes_skipped: ", data.n_pushes_skipped)
        println("\t\tn_infeasible_states: ", data.n_infeasible_states)
        println("\t\tset_position_count_1: ", data.set_position_count_1)
        println("\t\tset_position_count_2: ", data.set_position_count_2)
        println("\t\tn_positions_used: ", data.n_positions_used)
    end
end

# An index in the transposition table
const PositionIndex = UInt32

mutable struct Position
    pred::PositionIndex # predecessor index in positions array
    push::Push          # the push leading to this position
    push_depth::Int32   # search depth
    lowerbound::Int32   # heuristic cost-to-go
    on_path::Bool       # whether this move is on the active search path
end

"""
Set the state to the state represented by the position at the given index.
This is done by backtracking until we reach a shared predecessor state, and then applying
moves to end up in the same position.
"""
function set_position!(
    s::State,
    game::Game,
    positions::Vector{Position},
    dest_pos_index::PositionIndex,
    curr_pos_index::PositionIndex,
    data,
    )

    data.set_position_count_1 += 1

    # Backtrack from the destination state until we find a common ancestor,
    # as indicated by `on_path`. We reverse the `pred` pointers as we go
    # so we can find our way back.
    pos_index = dest_pos_index
    next_pos_index = zero(PositionIndex)
    while (pos_index != 0) && (!positions[pos_index].on_path)
        temp = positions[pos_index].pred
        positions[pos_index].pred = next_pos_index
        next_pos_index = pos_index
        pos_index = temp
    end

    # Backtrack from the source state until we reach a common ancestor.
    while (curr_pos_index != pos_index) && (curr_pos_index != 0)
        positions[curr_pos_index].on_path = false
        unmove!(s, game, positions[curr_pos_index].push)
        curr_pos_index = positions[curr_pos_index].pred
        data.set_position_count_2 += 1
    end

    # Traverse up the tree to the destination state, reversing the pointers again.
    while next_pos_index != 0
        # TODO: Have to check that the push is legal if we ever re-stitch together positions
        move!(s, game, positions[next_pos_index].push)
        positions[next_pos_index].on_path = true

        # Reverse the pointers again
        temp = positions[next_pos_index].pred
        positions[next_pos_index].pred = curr_pos_index
        curr_pos_index = next_pos_index
        next_pos_index = temp
    end

    return s
end

function solve(solver::AStar, game::Game)
    t_start = time()

    s = State(game, deepcopy(game.board_start))
    d_max, timeout, n_positions = solver.d_max, solver.timeout, solver.n_positions

    # Allocate memory and recalculate some things
    data = AStarData(false, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, Array{Push}(undef, d_max), 0)

    n_tiles = length(s.board)
    reach =construct_and_init_reach(s.board)
    stack = Array{TileIndex}(undef, n_tiles)

    queue_big = Array{Tuple{TileIndex, TileIndex, Int32}}(undef, length(s.board) * N_DIRS)
    distances_by_direction = Array{Int32}(undef, n_tiles, N_DIRS)
    dist_to_nearest_goal = zeros(Int32, n_tiles)
    calc_push_dist_to_nearest_goal_for_all_squares!(dist_to_nearest_goal, game, s, reach, stack, queue_big, distances_by_direction)

    # Calculate reachability and set player min pos
    calc_reachable_tiles!(reach, game, s.board, s.□_player, stack)
    move_player!(s, reach.□_min)

    # Compute the lower bound
    simple_lower_bound = calculate_simple_lower_bound(s, dist_to_nearest_goal)

    pushes = Array{Push}(undef, 4*length(s.□_boxes))

    # Allocate a bunch of positions
    positions = Array{Position}(undef, n_positions)
    for i in 1:length(positions)
        positions[i] = Position(
            zero(PositionIndex),
            Push(0,0),
            zero(Int32),
            zero(Int32),
            false
        )
    end

    # Index of the first free position
    # NOTE: For now we don't re-use positions
    free_index = one(PositionIndex)

    # A priority queue used to order the states that we want to process,
    # in order by total cost (push_depth + lowerbound cost-to-go)
    to_process = PriorityQueue{PositionIndex, Int32}()

    # A dictionary that maps (player pos, zhash) to position index
    # TODO: This is a terrible way of doing this. Use the linked-list approach.
    closed_set = Dict{Tuple{TileIndex, UInt64}, PositionIndex}()

    # Store an infeasible state in the first index
    # We point to this position any time we get an infeasible state
    infeasible_state_index = free_index
    positions[infeasible_state_index].lowerbound = typemax(Int32)
    positions[infeasible_state_index].push_depth = zero(Int32)
    free_index += one(PositionIndex)

    # Enqueue the root state
    current_pos_index = zero(PositionIndex)
    enqueue!(to_process, current_pos_index, simple_lower_bound)
    closed_set[(s.□_player, s.zhash)] = current_pos_index

    data.setup_time = time() - t_start

    done = false
    while !done && !isempty(to_process) && (time() - t_start < timeout)
        pos_index = dequeue!(to_process)
        set_position!(s, game, positions, pos_index, current_pos_index, data)
        current_pos_index = pos_index
        push_depth = pos_index > 0 ? positions[pos_index].push_depth : zero(Int32)
        data.max_depth = max(data.max_depth, push_depth+1)
        calc_reachable_tiles!(reach, game, s.board, s.□_player, stack)


        # Iterate over pushes
        n_pushes = get_pushes!(pushes, game, s, reach)
        for push_index in 1:n_pushes
            push = pushes[push_index]

            data.n_pushes_evaled += 1
            move!(s, game, push)
            calc_reachable_tiles!(reach, game, s.board, s.□_player, stack)
            move_player!(s, reach.□_min)

            # If we have not seen this state before so early, and it isn't an infeasible state.
            new_pos_index = get(closed_set, (s.□_player, s.zhash), zero(PositionIndex))
            if new_pos_index == 0 || push_depth+1 < positions[new_pos_index].push_depth

                lowerbound = calculate_simple_lower_bound(s, dist_to_nearest_goal)

                # Check to see if we are done
                if lowerbound == 0
                    # We are done!
                    done = true

                    # Back out the solution
                    data.solved = true
                    data.sol_depth = push_depth + 1
                    data.pushes[data.sol_depth] = push

                    pred = pos_index
                    while pred > 0
                        pos = positions[pred]
                        data.pushes[pos.push_depth] = pos.push
                        pred = pos.pred
                    end

                    break
                end

                # Enqueue and add to closed set if the lower bound indicates feasibility
                if lowerbound != typemax(Int32)

                    # Add the position
                    pos′ = positions[free_index]
                    pos′.pred = pos_index
                    pos′.push = push
                    pos′.push_depth = push_depth + 1
                    pos′.lowerbound = lowerbound

                    to_process[free_index] = pos′.push_depth + pos′.lowerbound
                    closed_set[(s.□_player, s.zhash)] = free_index
                    free_index += one(PositionIndex)
                else
                    # Store infeasible state
                    data.n_infeasible_states += 1
                    closed_set[(s.□_player, s.zhash)] = infeasible_state_index
                end
            else
                data.n_pushes_skipped += 1
            end

            unmove!(s, game, push)
        end
    end

    data.n_positions_used = free_index - 1
    data.total_time = time() - t_start
    return data
end