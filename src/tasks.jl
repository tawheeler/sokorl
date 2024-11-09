using Printf
using JSON

include("sokoban.jl")
include("sokorl.jl")

###################################

function generate_supervised_dataset(
	board_height::Int,
	n_training_entries_solved::Int,  # The number of solved entries to generate
	n_training_entries_unsolved::Int, # The number of unsolveable entries to generate
	board_gen_actions::Vector{BoardGenAction}, # The board generation actions to use to generate each board
	generated_training_entries_directory::String;
	subdirectory::String = "", # The directory to save them to. If empty, a name will be generated
	seed::UInt64 = zero(UInt64),
	)

	# Create an output directory
	if subdirectory == ""
		# look at the existing directories and just pick the next digit
		idx = 1
		subdirectory = "1_box_randroom$(idx)"
		while isdir(joinpath(generated_training_entries_directory, subdirectory))
			idx += 1
			subdirectory = "1_box_randroom$(idx)"
		end
	end

	directory = joinpath(generated_training_entries_directory, subdirectory)
	if isdir(directory)
		@error "Directory $directory already exists!"
		return
	end

	mkpath(directory)
	println("Generating dataset to ", directory)

	# Save the params
	open(joinpath(directory, "params.json"), "w") do file
	    write(file,
	          JSON.json(Dict{String,Any}(
	                "board_height" => BOARD_HEIGHT,
	                "n_training_entries_solved" => n_training_entries_solved,
	                "n_training_entries_unsolved" => n_training_entries_unsolved,
	                "seed" => seed,
	                "board_gen_actions" => string(board_gen_actions),
	    )))
	end


	solver = AStar(
	    100,
	    10.0,
	    5000000
	)

	rng = MersenneTwister(seed)

	max_n_tries = 10*(n_training_entries_solved + n_training_entries_unsolved)

	n_tries = 0
	n_solved = 0
	n_unsolved = 0

	t_start = time()
	while (n_solved < n_training_entries_solved || n_unsolved < n_training_entries_unsolved) && n_tries < max_n_tries
		n_tries += 1

	    added = false
        board = generate_board(rng, board_height, board_height, board_gen_actions)
        game = Game(board)
        data = solve(solver, game)
        training_entry = nothing

        if data.solved
            if n_solved < n_training_entries_solved
                training_entry = construct_solved_training_entry(game, data)
                n_solved += 1
                added = true
            end
        else
            if n_unsolved < n_training_entries_unsolved
                training_entry = construct_failed_training_entry(game)
                n_unsolved += 1
                added = true
            end
        end

        if added
		    filename = @sprintf "%04d.txt" (n_solved + n_unsolved)
		    open(joinpath(directory, filename), "w") do io
		        str = to_text(training_entry)
		        write(io, str)
		    end
		end

		println("$n_solved ($n_training_entries_solved), $n_unsolved ($n_training_entries_unsolved), $n_tries, $(round(time() - t_start, digits=1))s elapsed")
	end

	println("DONE: $(round(time() - t_start, digits=1))s elapsed")
end

###################################

# Define the parameters to use in EC2.
# These are overwritten if we are running locally.

BOARD_HEIGHT = 8

SOKORL_DIRECTORY = "/home/ec2-user/sokorl"

N_TRAINING_ENTRIES_SOLVED = 5 # 2000
N_TRAINING_ENTRIES_UNSOLVED = 5 # 2000

BOARD_GEN_ACTIONS = [
           GenerateSubRoom(10),
           GenerateSubRoom(2),
           SpawnGoal(),
           SpawnBox(),
           SpawnPlayer(),
           SpeckleWalls(0,5)
       ]

SEED = rand(UInt64)

username = get(ENV, "USER", nothing)
if username == "twheeler"
	println("Running locally")
	SOKORL_DIRECTORY = "/home/twheeler/Documents/projects/sokoban/sokorl"
	N_TRAINING_ENTRIES_SOLVED = 10
	N_TRAINING_ENTRIES_UNSOLVED = 10
	SEED = UInt64(1)
end

###

# Derived Params

TRAINING_ENTRIES_DIRECTORY = joinpath(SOKORL_DIRECTORY, "training_entries")
GENERATED_TRAINING_ENTRIES_DIRECTORY = joinpath(TRAINING_ENTRIES_DIRECTORY, "generated")


# Run

println("running tasks!")

generate_supervised_dataset(BOARD_HEIGHT, N_TRAINING_ENTRIES_SOLVED, N_TRAINING_ENTRIES_UNSOLVED, BOARD_GEN_ACTIONS, GENERATED_TRAINING_ENTRIES_DIRECTORY, subdirectory="", seed=SEED)