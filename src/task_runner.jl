using JSON
using Dates
using Printf

# Run tasks in the task folder until there are none.
# Whenever we take on a task, we either move it to the completed folder or the attempted folder

const TASKS_FILEPATH = "/home/tim/Documents/personal/sokorl/tasks"
const FOLDER_TODO = joinpath(TASKS_FILEPATH, "todo")
const FOLDER_DONE = joinpath(TASKS_FILEPATH, "done")
const FOLDER_TRIED = joinpath(TASKS_FILEPATH, "tried")

# Prepend this header to every task
const TASK_HEADER = """
using CUDA
using MLUtils
using Printf
using Plots
using JLD2
using JSON

const PROJECT_DIR = "/home/tim/Documents/personal/sokorl"
const SRC_DIRECTORY = joinpath(PROJECT_DIR, "src")
const TRAINING_DIRECTORY = joinpath(PROJECT_DIR, "training_entries")
const VALIDATION_DIRECTORY = joinpath(PROJECT_DIR, "validation")
const MODEL_DIRECTORY = joinpath(PROJECT_DIR, "models")

include(joinpath(SRC_DIRECTORY, "sokoban.jl"))
include(joinpath(SRC_DIRECTORY, "vis.jl"))
include(joinpath(SRC_DIRECTORY, "sokorl.jl"))

if !CUDA.functional()
    println(stdout, "CUDA is not functional")
    println(stderr, "CUDA is not functional")
    exit(1)
end
"""



# Get the oldest task (by modification)
function get_next_task()

    oldest_task_filename::Union{Nothing, String} = nothing
    oldest_task_modification_time = 0.0

    for content in readdir(FOLDER_TODO)
        fullpath = joinpath(FOLDER_TODO, content)

        if !isfile(fullpath)
            continue # must be a file
        end
        if !endswith(content, ".task")
            continue # must be a task file
        end

        modification_time = mtime(fullpath)
        if isnothing(oldest_task_filename) || modification_time < oldest_task_modification_time
            oldest_task_filename = content
            oldest_task_modification_time = modification_time
        end
    end

    return oldest_task_filename
end


mutable struct TaskResult
    succeeded::Bool
    message::String
    t_start::Float64
    t_elapsed::Float64
end

# Run the task in the given file.
function run_task(task_filepath::AbstractString)
    res = TaskResult(false, "", time(), NaN)

    try
        content = read(task_filepath, String)

        temp_file, temp_io = mktemp()
        write(temp_io, TASK_HEADER)
        write(temp_io, "\n")
        write(temp_io, content)
        close(temp_io)

        output = read(`julia -q $(temp_file)`, String)

        res.succeeded = true
        res.message = output
    catch e
        # We failed to
        res.succeeded = false
        res.message = string(e)
    end

    res.t_elapsed = time() - res.t_start

    return res
end

function write_out_result(res::TaskResult, task_file::String, dest_folder::String)
    task_name_without_ext, _ = splitext(task_file)
    open(joinpath(dest_folder, "$(task_name_without_ext).res"), "w") do file
        write(file,
            JSON.json(Dict{String,Any}(
                "succeeded" => res.succeeded,
                "message" => res.message,
                "t_start" => res.t_start,
                "t_elapsed" => res.t_elapsed,
        )))
    end
end


function run_tasks()
    t_start = time()
    n_tasks = 0

    done = false
    while !done
        println("Running tasks!")
        println("It is currently ", Dates.format(now(), "HH:MM:SS"))

        task_file = get_next_task()
        if !isa(task_file, String)
            println("No tasks left!")
            done = true
            break
        end

        n_tasks += 1

        println("Running $(n_tasks)th task: $(task_file)")
        println("="^20)

        task_filepath = joinpath(FOLDER_TODO, task_file::String)
        res = run_task(task_filepath)
        dest_folder = res.succeeded ? FOLDER_DONE : FOLDER_TRIED

        # name the task according to the time
        task_dst_name = Dates.format(Dates.now(), "yyyymmdd_HHMMss") * ".task"
        mv(task_filepath, joinpath(dest_folder, task_dst_name))
        write_out_result(res, task_dst_name, dest_folder)
        println(res.succeeded ? "SUCCEEDED" : "FAILED")
        println("="^20)
    end

    @printf("Ran for a total of %.2f seconds and completed %d tasks\n", time() - t_start, n_tasks)
end

run_tasks()




