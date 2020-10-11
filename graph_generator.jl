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
# BA network
function ba_adj_matrix(total_node, init_node, new_edges)
    temp = barabasi_albert(total_node, init_node, new_edges)
    return get_biggest_component_adj_matrix(temp)
end

function ba_graph(total_node, init_node, new_edges)
    temp = barabasi_albert(total_node, init_node, new_edges)
    return Graph(get_biggest_component_adj_matrix(temp))
end



# ER network
function er_adj_matrix(total_node=50, probability=0.01)
    temp = erdos_renyi(total_node, probability)
    return get_biggest_component_adj_matrix(temp)
end

function er_graph(total_node=50, probability=0.01)
    temp = erdos_renyi(total_node, probability)
    return Graph(get_biggest_component_adj_matrix(temp))
end



# Grid network
function grid_2D_adj_matrix(num_row=15, num_col=5)
    temp = Grid([num_row, num_col])
    return get_biggest_component_adj_matrix(temp)
end

function grid_2D_graph(num_row=15, num_col=5)
    temp = Grid([num_row, num_col])
    return Graph(get_biggest_component_adj_matrix(temp))
end



# Ladder network
function ladder_adj_matrix(n=20)
    # number of node = 2n, number of edges = 3n-2
    temp = ladder_graph(n)
    return get_biggest_component_adj_matrix(temp)
end



# SBM netowrk
function sbm_graph(ave_degrees , n_per_community)
    # make sure the size of community is appropriate, not too big nor too small
    @assert size(ave_degrees, 1) == length(n_per_community) "size of ave_degrees must agree with n_per_community"

    # create sbm_graph, notice that the rule of ave_degrees and n_per_community
    # must follow the rule of sbm_graph.
    temp = stochastic_block_model(ave_degrees, n_per_community)
    return Graph(get_biggest_component_adj_matrix(temp))
end

function sbm_adj_matrix(ave_degrees , n_per_community)
    # make sure the size of community is appropriate, not too big nor too small
    @assert size(ave_degrees, 1) == length(n_per_community) "size of ave_degrees must agree with n_per_community"

    # create sbm_graph, notice that the rule of ave_degrees and n_per_community
    # must follow the rule of sbm_graph.
    temp = stochastic_block_model(ave_degrees, n_per_community)
    return get_biggest_component_adj_matrix(temp)
end



# Complete Bipartite Graph
function complete_bipartite_adj_matrix(n1=10, n2=5)
    temp = complete_bipartite_graph(n1, n2)
    return adjacency_matrix(temp)
end
