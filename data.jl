using FileIO
using LightGraphs

include("graph_generator.jl")

# DATA GENERTOR FUNCTIONS ******************************************************
#=
Create dataset for grid graphs along with a meta data file to store the information
about this dataset.

in:     num_graphs: int, number of gird graphs in this dataset
        file_name: file name WITHOUT extension
        max_row: int, maximum number of nodes in a row
        min_row: int, minimum number of nodes in a row
        max_col: int, maximum number of nodes in a column
        min_col:innt, maximum number of nodes in a column

out:    nothing, write 2 files:
            file_name_meta.jld holds meta data such as max_num_node, max_prev_node,
                and num_graphs
            file_name_data.jld holds training data: x, y, and len
=#
function create_grid_dataset(num_graphs=1000;
    file_name="grid",
    for_training=true,
    max_row=10,
    min_row=5,
    max_col=10,
    min_col=5)

    graph_dict = Dict()

    max_num_node = 0
    max_prev_node = 0
    all_matrix = []
    #  for each graph
    for i = 1:num_graphs
        # randomize parameters for the Grid graphs
        num_row = rand(min_row:max_row)
        num_col = rand(min_col:max_col)

        # generate the graph
        g = grid_graph(num_row, num_col)

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
    n_train = Int(round(num_graphs * 0.8))

    # set file name based on purpose train or not
    if for_training
        file_name = string("train_", file_name)
    else
        file_name = string("test_", file_name)
    end

    # save meta data
    meta_file_name = string(file_name, "_meta.jld")
    save(meta_file_name, "max_prev_node", max_prev_node, "max_num_node", max_num_node, "num_graphs", num_graphs)

    @info "Saving training and testing data... "
    # save training data
    training_file_name = string("train_", file_name, "_data.jld")
    save(training_file_name, "all_x", all_x[:, :, 1:n_train], "all_y", all_y[:, :, 1:n_train], "all_len", all_len[1:n_train])
    testing_file_name = string("test_", file_name, "_data.jld")
    save(testing_file_name, "all_x", all_x[:, :, (n_train+1):end], "all_y", all_y[:, :, (n_train+1):end], "all_len", all_len[(n_train+1):end])
    @info "Done! max_prev_node: $(max_prev_node) \t max_num_node $(max_num_node) \t num_graphs $(num_graphs)"
end




#=
Create dataset for ladder graphs along with a meta data file to store the information
about this dataset.

in:     num_graphs: int, number of ladder graphs in this dataset
        file_name: file name WITHOUT extension
        max_n: int, maximum number of variable n
        min_n: int, minimum number of variable n

        the number of nodes = 2n
        the number of edges = 3n - 2

out:    nothing, write 2 files:
            file_name_meta.jld holds meta data such as max_num_node, max_prev_node,
                and num_graphs
            file_name_data.jld holds training data: x, y, and len
=#
function create_ladder_dataset(num_graphs=1000;
    file_name="ladder",
    for_training=true,
    max_n = 30,
    min_n = 10)


    graph_dict = Dict()

    max_num_node = 0
    max_prev_node = 0
    all_matrix = []
    #  for each graph
    for i = 1:num_graphs
        # randomize parameters for the ladder graphs
        n = rand(min_n:max_n) # number of node = 2n, number of edges = 3n-2

        # generate the graph
        g = ladder_graph(n)

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
    n_train = Int(round(num_graphs * 0.8))

    # set file name based on purpose train or not
    if for_training
        file_name = string("train_", file_name)
    else
        file_name = string("test_", file_name)
    end

    # save meta data
    meta_file_name = string(file_name, "_meta.jld")
    save(meta_file_name, "max_prev_node", max_prev_node, "max_num_node", max_num_node, "num_graphs", num_graphs)
    @info "Saving training and testing data... "
    # save training data
    training_file_name = string("train_", file_name, "_data.jld")
    save(training_file_name, "all_x", all_x[:, :, 1:n_train], "all_y", all_y[:, :, 1:n_train], "all_len", all_len[1:n_train])
    testing_file_name = string("test_", file_name, "_data.jld")
    save(testing_file_name, "all_x", all_x[:, :, (n_train+1):end], "all_y", all_y[:, :, (n_train+1):end], "all_len", all_len[(n_train+1):end])
    @info "Done! max_prev_node: $(max_prev_node) \t max_num_node $(max_num_node) \t num_graphs $(num_graphs)"
end




#=
Create dataset for stochastic block model graphs along with a meta data file to store the information
about this dataset. Limited to 2 communities per graph.

in:     num_graphs: int, number of sbm graphs in this dataset.
        file_name: file name WITHOUT extension
        max_num_vertices_per_community: int, maximum number of nodes in a community
        min_num_vertices_per_community: int, minimum number of nodes in a community

out:    nothing, write 2 files:
            file_name_meta.jld holds meta data such as max_num_node, max_prev_node,
                and num_graph
            file_name_data.jld holds training data: x, y, and len
=#
function create_sbm_dataset(num_graphs;
    file_name="sbm",
    for_training=true,
    max_num_vertices_per_community=10,
    min_num_vertices_per_community=7)


    graph_dict = Dict()

    max_num_node = 0
    max_prev_node = 0
    all_matrix = []
    #  for each graph
    for i = 1:num_graphs
        # create parameter for the sbm graph
        n_per_community = [0, 0]
        n_per_community[1] = rand(min_num_vertices_per_community:max_num_vertices_per_community)
        n_per_community[2] = rand(min_num_vertices_per_community:max_num_vertices_per_community)

        ave_degrees = [n_per_community[1]*0.3 n_per_community[1]*0.05;
                        n_per_community[2]*0.05 n_per_community[2]*0.3]

        # generate the graph
        g = sbm_graph(ave_degrees, n_per_community)

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
    n_train = Int(round(0.8 * num_graphs))

    # set file name based on purpose train or not
    if for_training
        file_name = string("train_", file_name)
    else
        file_name = string("test_", file_name)
    end

    # save meta data
    meta_file_name = string(file_name, "_meta.jld")
    save(meta_file_name, "max_prev_node", max_prev_node, "max_num_node", max_num_node, "num_graphs", num_graphs)
    @info "Saving training and testing data... "

    # save training data
    training_file_name = string("train_", file_name, "_data.jld")
    save(training_file_name, "all_x", all_x[:, :, 1:n_train], "all_y", all_y[:, :, 1:n_train], "all_len", all_len[1:n_train])
    testing_file_name = string("test_", file_name, "_data.jld")
    save(testing_file_name, "all_x", all_x[:, :, (n_train+1):end], "all_y", all_y[:, :, (n_train+1):end], "all_len", all_len[(n_train+1):end])
    @info "Done! max_prev_node: $(max_prev_node) \t max_num_node $(max_num_node) \t num_graphs $(num_graphs)"
end




#=
Create dataset for stochastic complete bipartite along with a meta data file to store the information
about this dataset.

in:     num_graphs: int, number of complete bipartite graphs in this dataset
        file_name: file name WITHOUT extension
        for_training: boolean, create for training or not
        max_n1: int, maximum number of nodes in a partition 1
        min_n1: int, minimum number of nodes in a partition 1
        max_n2: int, maximum number of nodes in a partition 2
        min_n2: int, minimum number of nodes in a partition 2

out:    nothing, write 2 files:
            file_name_meta.jld holds meta data such as max_num_node, max_prev_node,
                and num_graphs
            file_name_data.jld holds training data: x, y, and len
=#
function create_complete_bipartite_dataset(num_graphs;
    file_name="complete_bipartite",
    for_training=true,
    max_n1=20,
    min_n1=15,
    max_n2=20,
    min_n2=15)


    graph_dict = Dict()

    max_num_node = 0
    max_prev_node = 0
    all_matrix = []
    #  for each graph
    for i = 1:num_graphs
        # create parameter for the complete bipartite graph
        n1 = rand(min_n1: max_n1)
        n2 = rand(min_n2: max_n2)

        # generate the graph
        g = complete_bipartite_graph(n1, n2)

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

    # set file name based on purpose train or not
    if for_training
        file_name = string("train_", file_name)
    else
        file_name = string("test_", file_name)
    end

    # save meta data
    meta_file_name = string(file_name, "_meta.jld")
    save(meta_file_name, "max_prev_node", max_prev_node, "max_num_node", max_num_node, "num_graphs", num_graphs)

    # save training data
    training_file_name = string(file_name, "_data.jld")
    save(training_file_name, "all_x", all_x, "all_y", all_y, "all_len", all_len)
end




# LOAD DATA FUNCTIONS **********************************************************
#=
Load dataset from a file and partition

in:     file_name: string, full file name
        train_ratio: float, the ratio of train_size / dataset_size
out:    1 tuple contains 2 smaller triples:
            ((train_x, train_y, train_len), (test_x, test_y, test_len))
=#
function load_dataset_with_partition(file_name::String, train_ratio::Float64)
    training_file_name = string(file_name)
    (all_x, all_y, all_len) = load(training_file_name, "all_x", "all_y", "all_len")

    dataset_size = size(all_x, 3)
    train_size = round(Int, size(all_x, 3) * train_ratio)

    train_x = all_x[:,:,1:train_size]
    train_y = all_y[:,:,1:train_size]
    train_len = all_len[1:train_size]

    test_x = all_x[:,:,train_size + 1:dataset_size]
    test_y = all_y[:,:,train_size + 1:dataset_size]
    test_len = all_len[train_size + 1:dataset_size]

    return ((train_x, train_y, train_len), (test_x, test_y, test_len))
end




#=
Load dataset from a file
in:     file_name: string, full file name
out:    1 triple: (x, y, len)
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
