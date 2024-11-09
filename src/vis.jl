using Luxor
using Colors

const COLOR_WALL = (0.2, 0.2, 0.2)
const COLOR_FLOOR = (0.4, 0.4, 0.4)
const COLOR_TILE = (0.5, 0.5, 0.5)
const COLOR_GOAL = (0x1B/0xFF, 0xA1/0xFF, 0xEA/0xFF)
const COLOR_PLAYER = (0xF5/0xFF, 0x61/0xFF, 0x5C/0xFF)
const COLOR_BOX = (0x70/0xFF, 0xA1/0xFF, 0x7C/0xFF)
const COLOR_MOVE = (0xF5/0xFF, 0xCE/0xFF, 0x42/0xFF)

const PIX_PER_SIDE = 25

get_drawing_pix_width(game::Game) = num_cols(game)*PIX_PER_SIDE
get_drawing_pix_height(game::Game) = num_rows(game)*PIX_PER_SIDE

function create_drawing(game::Game)
    return Drawing(
        get_drawing_pix_width(game),
        get_drawing_pix_height(game),
        :png)
end

function draw_walls_and_floors!(board::Board)
    # Render all walls and floors
    for row in 1:size(board, 2)
        y = (row-1)*PIX_PER_SIDE
        for col in 1:size(board,1)
            x = (col-1)*PIX_PER_SIDE
            v = board[row,col]
            if (v & WALL) > 0
                sethue(COLOR_WALL...)
                rect(x, y, PIX_PER_SIDE, PIX_PER_SIDE, :fill)
            else
                sethue(COLOR_FLOOR...)
                rect(x, y, PIX_PER_SIDE, PIX_PER_SIDE, :fill)
                sethue(COLOR_TILE...)
                rect(x + 2, y + 2, PIX_PER_SIDE - 4, PIX_PER_SIDE - 4, :fill)
            end
        end
    end
end

function draw_goals!(game::Game)
    # Render all goals
    sethue(COLOR_GOAL...)
    for □ in game.□_goals
        col, row = tile_index_to_col_row(game, □)
        x = (row-0.5)*PIX_PER_SIDE
        y = (col-0.5)*PIX_PER_SIDE
        circle(x, y, PIX_PER_SIDE/4, :fill)
    end
end

function draw_boxes_and_player!(board::Board)
    # Render boxes and player
    for col in 1:size(board, 2)
        x = (col-0.5)*PIX_PER_SIDE
        for row in 1:size(board,1)
            y = (row-0.5)*PIX_PER_SIDE
            v = board[row,col]
            if (v & BOX) > 0
                # color according to whether it is on goal or not
                if (v & GOAL) > 0
                    sethue(COLOR_GOAL...)
                else
                    sethue(COLOR_BOX...)
                end
                box(Point(x, y), PIX_PER_SIDE-2, PIX_PER_SIDE-2, 5, :fill)
            end
            if (v & PLAYER) > 0
                sethue(COLOR_PLAYER...)
                star(Point(x,y), PIX_PER_SIDE/3, 5, 0.5, 0.0, :fill)
            end
        end
    end
end

function draw_board(game::Game, board::Board)
    d = create_drawing(game)
    draw_walls_and_floors!(board)
    draw_goals!(game)
    draw_boxes_and_player!(board)
    finish()
    d
end

function get_trajectory_states(game::Game, s0::State, pushes::Vector{Push})
    s = deepcopy(s0)
    sol_depth = length(pushes)
    states = Array{State}(undef, 1+sol_depth)
    states[1] = deepcopy(s)
    for (d, push) in enumerate(pushes)
        move!(s, game, push)
        states[d+1] = deepcopy(s)
    end
    return states
end

function get_trajectory_states(game::Game, s0::State, moves::Vector{Direction})
    s = deepcopy(s0)
    sol_depth = length(moves)
    states = Array{State}(undef, 1+sol_depth)
    states[1] = deepcopy(s)
    for (d, move) in enumerate(moves)
        maybe_move!(s, game, move)
        states[d+1] = deepcopy(s)
    end
    return states
end

function render_states_to_gif(game::Game, states::Vector{State}, filename_no_ext::String, framerate_hz)
    width = get_drawing_pix_width(game)
    height = get_drawing_pix_height(game)
    movie = Movie(width, height, filename_no_ext)

    draw_frame = (scene, framenumber) -> begin
        Luxor.translate(-get_drawing_pix_width(game)/2, -get_drawing_pix_height(game)/2)

        s = states[framenumber]
        draw_walls_and_floors!(s.board)
        draw_goals!(game)
        draw_boxes_and_player!(s.board)

        # fontsize(20)
        # fontface("JuliaMono-Regular")
        # sethue("white")
        # text(string("state $(framenumber-1)"), Point(10.0, 18.0), halign=:left)
    end

    return Luxor.animate(
        movie,
        [Scene(movie, draw_frame, 1:length(states))],
        framerate = framerate_hz,
        creategif = true)
end