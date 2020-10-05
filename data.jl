using FileIO
using LightGraphs

include("graph_generator.jl")

# DATA GENERTOR FUNCTIONS ******************************************************
#=
Create a dictionary where key is the index and value is a graph. Write graphs
to a .lgz file. We also write meta data such as max_num_node and max_prev_node
to .jld file.

in:     num_graphs: int, number of ER graphs in this dataset
        file_name: file name WITHOUT extension
        max_num_node: int, maximum number of nodes in any graphs
        min_num_node: int, minimum number of nodes in any graphs
        max_probability: float, max probability of connection
        min_probability:float, min probability of connection

out:    nothing, write 2 files:
            file_name_meta.jld holds meta data such as max_num_node, max_prev_node,
                and num_graphs
            file_name_data.jld holds training data: x, y, and len
=#
function create_er_dataset(num_graphs=1000, file_name="train_ER", max_node=100, min_node=50, max_probability=0.8, min_probability=0.04)
    graph_dict = Dict()

    max_num_node = 0
    max_prev_node = 0
    all_matrix = []
    #  for each graph
    for i = 1:num_graphs
        # randomize parameters for ER graphs
        num_node = rand(min_node: max_node)
        probability = rand(min_probability:0.01: max_probability)

        # generate the graph
        g = er_graph(num_node, probability)

        # add g's adjacency matrix to all_matrix array
        push!(all_matrix, Matrix(adjacency_matrix(g)))

        # update max_num_node
        max_num_node = max(max_num_node, size(g,1))

        # add g to our graph dict
        graph_dict[i] = g
    end

    # find the max_prev_node
    max_prev_node = find_max_prev_node(all_matrix, 100, 1) # 1 is root for bfs

    # transform our matrix to training data.
    (all_x, all_y, all_len) = transform(all_matrix, max_num_node, max_prev_node)

    # save graph_dict to file
    # savegraph(file_name, graph_dict)

    # save meta data
    meta_file_name = string(file_name, "_meta.jld")
    save(meta_file_name, "max_prev_node", max_prev_node, "max_num_node", max_num_node, "num_graphs", num_graphs)

    # save training data
    training_file_name = string(file_name, "_data.jld")
    save(training_file_name, "all_x", all_x, "all_y", all_y, "all_len", all_len)
end


# LOAD DATA FUNCTIONS ******************************************************
#=
Load dataset from a file

in:     file_name: string, full file name
out:    a tuple: (all_x, all_y, all_len)
=#
function load_dataset(file_name::String)
    training_file_name = string(file_name)
    (all_x, all_y, all_len) = load(training_file_name, "all_x", "all_y", "all_len")
    return (all_x, all_y, all_len)
end

#=
Load meta data from a file

in:     file_name: string, full file name
out:    a tuple: (max_num_node, max_prev_node, num_graphs)
=#
function load_meta_data(file_name::String)
    # load meta data
    meta_file_name = string(file_name)
    (max_num_node, max_prev_node, num_graphs) = load(meta_file_name, "max_num_node", "max_prev_node", "num_graphs")
    return (max_num_node, max_prev_node, num_graphs)
end


# load_meta_data("train_ER_meta.jld")
# (all_x, all_y, all_len) = load_dataset("train_ER_data.jld")
# println(all_x[1])
# println(all_y[1])
# println(all_len[1])

create_er_dataset(1000, "Train_ER", 30,10,0.8,0.04)
