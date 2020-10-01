using LightGraphs
using JLD2

include("data_helpers.jl")


# HELPER FUNCTIONS ***************************************************************************
#=
in: graph: a graph
out: an adjacency matrix of biggest component.
=#
function get_biggest_component_adj_matrix(graph)
    # array of arrays of nodes in same component
    components = connected_components(graph)

    # get the max size
    @show size_arr = map(length, components)
    max_size = reduce(max,size_arr)

    # arrays of nodes in biggest component
    biggest_component_idx = components[findall(x->x==max_size, size_arr)[1]]

    # create adjacency matrix for biggest component
    return adjacency_matrix(graph)[biggest_component_idx, biggest_component_idx]
end

#=
in: graph: a graph
out: biggest component: a graph.
=#
function get_biggest_component(graph)
    return Graph(get_biggest_component_adj_matrix(graph))
end


# GRAPH GENERTOR FUNCTIONS ********************************************************************
function ba_adj_matrix(total_node, init_node, new_edges)
    temp = barabasi_albert(total_node, init_node, new_edges)
    return get_biggest_component_adj_matrix(temp)
end

function ba_graph(total_node, init_node, new_edges)
    temp = barabasi_albert(total_node, init_node, new_edges)
    return Graph(get_biggest_component_adj_matrix(temp))
end

function ne_adj_matrix(total_node, probability)
    temp = erdos_renyi(total_node, probability * 1.0)
    return get_biggest_component_adj_matrix(temp)
end

function er_graph(total_node, probability)
    temp = barabasi_albert(total_node, init_node, new_edges)
    return Graph(get_biggest_component_adj_matrix(temp))
end

# Create data set

#=
Create a dictionary where key is the index and value is a graph. Write graphs to a .lgz file.
We also write meta data such as max_num_node and max_prev_node to .jld file.

in:     num_graphs: int, number of ER graphs in this dataset
        file_name: file name, end with .lgz extension
        max_num_node: int, maximum number of nodes in any graphs
        min_num_node: int, minimum number of nodes in any graphs
        max_probability: float, max probability of connection
        min_probability:float, min probability of connection

out:    nothing, write 2 files:
            file_name.lgz holds the graphs dataset
            file_name.jld holds meta data such as max_num_node, max_prev_node, and num_graphs
=#
function create_er_dataset(num_graphs, file_name, max_num_node=nothing, min_num_node=nothing, max_probability=nothing, min_probability=nothing)
    graph_dict = Dict()

    if max_num_node == nothing
        max_num_node = 200
    end

    if min_num_node == nothing
        min_num_node = 50
    end

    if max_probability == nothing
        max_probability = 0.8
    end

    if min_probability == nothing
        min_probability = 0.04
    end

    max_num_node = 0
    max_prev_node = 0
    for i = 1:num_graphs
        num_node = rand(min_num_node, max_num_node)
        probability = rand(min_probability, 0.01, max_probability)
        g = er_graph(num_node, probability)

        # update max_num_node
        max_num_node = max(max_num_node, size(g,1))
        # update max_prev_node
        encoded = encode(adjacency_matrix(g))
        max_prev_node = max(max_prev_node, size(encoded,1))

        graph_dict[i] = g
    end

    savegraph(file_name, graph_dict, num_graphs)

    meta_file_name = string(file_name[1:findlast('.',file_name) - 1], ".jld")
    save(meta_file_name, "max_prev_node", max_prev_node, "max_num_node", max_num_node, "num_graphs", num_graphs)
end


#=
Load dataset from a file

in:     file_name: string, file name with .lgz extension
out:    a tuple: (all_matrix, max_num_node, max_prev_node, num_graphs) where
            all_matrix: array of adjacency matrices of every graph in the dataset
            max_num_node: int, maximum number of nodes of a graph in the dataset
            max_prev_node: int, maximum "width" of encoded adjacency matrix from the dataset.
            num_graphs: int, the number of graphs in the dataset
=#
function load_dataset(file_name):
    # load meta data
    meta_file_name = string(file_name[1:findlast('.',file_name) - 1], ".jld")

    max_prev_node = load(meta_file_name)["max_prev_node"]
    max_num_node = load(max_num_node)["max_num_node"]
    num_graphs = load(num_graphs)["num_graphs"]

    # load dict
    all_matrix = map(adjacency_matrix(values(loadgraphs(file_name)))

    return (all_matrix, max_num_node, max_prev_node, num_graphs)
end
