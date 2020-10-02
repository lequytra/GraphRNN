using LightGraphs
include("data_helpers.jl")


# HELPER FUNCTIONS ***************************************************************************
#=
in:     graph: a graph
out:    an adjacency matrix of the biggest component.
=#
function get_biggest_component_adj_matrix(graph)
    # array of arrays of nodes in same component
    components = connected_components(graph)

    # get the max size
    size_arr = map(length, components)
    max_size = reduce(max,size_arr)

    # arrays of nodes in biggest component
    biggest_component_idx = components[findall(x->x==max_size, size_arr)[1]]

    # create adjacency matrix for biggest component
    return Matrix((adjacency_matrix(graph))[biggest_component_idx, biggest_component_idx])
end

#=
in:     graph: a graph
out:    biggest component: a graph.
=#
function get_biggest_component(graph)
    return Graph(get_biggest_component_adj_matrix(graph))
end



# GRAPH GENERTOR FUNCTIONS *****************************************************
function ba_adj_matrix(total_node, init_node, new_edges)
    temp = barabasi_albert(total_node, init_node, new_edges)
    return get_biggest_component_adj_matrix(temp)
end

function ba_graph(total_node, init_node, new_edges)
    temp = barabasi_albert(total_node, init_node, new_edges)
    return Graph(get_biggest_component_adj_matrix(temp))
end

function er_adj_matrix(total_node=50, probability=0.01)
    temp = erdos_renyi(total_node, probability)
    return get_biggest_component_adj_matrix(temp)
end

function er_graph(total_node=50, probability=0.01)
    temp = erdos_renyi(total_node, probability)
    return Graph(get_biggest_component_adj_matrix(temp))
end
