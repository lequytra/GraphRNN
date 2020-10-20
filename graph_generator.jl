using LightGraphs
include("data_helpers.jl")


# HELPER FUNCTIONS ***************************************************************************
#=
Params:
    graph: a graph
Return:
    an adjacency matrix of the biggest component.
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
Return the biggest component of a graph

Params:
    graph: a graph

Return:
    biggest component: a graph.
=#
function get_biggest_component(graph)
    return Graph(get_biggest_component_adj_matrix(graph))
end



# GRAPH GENERTOR FUNCTIONS *****************************************************
# BA network
#=
return adjacency_matrix of the biggest component in a BA GRAPH

Params:
    total_node: Int: the total number of nodes
    init_node: Int: the begin node
    new_edges: Int: number of new edges that connect new node to constructing
        graphs based on preferential attachment.

returns:
    adjacency_matrix: Array{64,2}.
=#
function ba_adj_matrix(total_node, init_node, new_edges)
    temp = barabasi_albert(total_node, init_node, new_edges)
    return get_biggest_component_adj_matrix(temp)
end

#=
return the biggest component in a BA GRAPH

Params:
    total_node: Int: the total number of nodes
    init_node: Int: the begin node
    new_edges: Int: number of new edges that connect new node to constructing
        graphs based on preferential attachment.

returns:
    a BA graph.
=#
function ba_graph(total_node, init_node, new_edges)
    temp = barabasi_albert(total_node, init_node, new_edges)
    return Graph(get_biggest_component_adj_matrix(temp))
end



# ER network
#=
return the adjacency matrix of the biggest component in an ER graph

Params:
    total_node: Int: the total number of nodes in the graph.
    probability: Float: the probability that two nodes are connected.

Return:
    adjacency_matrix of an ER graph: Array{64, 2}
=#
function er_adj_matrix(total_node=50, probability=0.01)
    temp = erdos_renyi(total_node, probability)
    return get_biggest_component_adj_matrix(temp)
end

#=
return the biggest component in ER graph

Params:
    total_node: Int: the total number of nodes in the graph.
    probability: Float: the probability that two nodes are connected.

Return:
    an ER graph.
=#
function er_graph(total_node=50, probability=0.01)
    temp = erdos_renyi(total_node, probability)
    return Graph(get_biggest_component_adj_matrix(temp))
end



# Grid network
#=
return the adjacency matrix of a grid graph.

Params:
    num_row: Int: number of row in the grid
    num_col: Int: number of col in the grid

Return:
    adjacency matrix of the grid graph: Array{Int, 2}
=#
function grid_adj_matrix(num_row=15, num_col=5)
    temp = Grid([num_row, num_col])
    return get_biggest_component_adj_matrix(temp)
end

#=
return a grid graph.

Params:
    num_row: Int: number of row in the grid
    num_col: Int: number of col in the grid

Return:
    a grid graph.
=#
function grid_graph(num_row=15, num_col=5)
    temp = Grid([num_row, num_col])
    return Graph(get_biggest_component_adj_matrix(temp))
end



# Ladder network
#=
Return an adjacency_matrix of a ladder_graph

Params:
    n: Int: the number of steps in the ladder

Return:
    the adjacency matrix of a ladder graph
=#
function ladder_adj_matrix(n=20)
    # number of node = 2n, number of edges = 3n-2
    temp = ladder_graph(n)
    return get_biggest_component_adj_matrix(temp)
end



# SBM netowrk
#=
Return the an sbm graph.

Params:
    ave_degrees: Array{64, 2}: dimension = number of community x number of community.
        ave_degrees[x,y] is the average degrees that a node in community x has with
            other nodes in community y.
        ave_degrees[x,x] therefore is the average degrees of community x.
    n_per_community: Array{64, 1}: length = number of community.
        n_per_community[i]: is the number of nodes in community i.
Return:
    a sbm graph
=#
function sbm_graph(ave_degrees , n_per_community)
    # make sure the size of community is appropriate, not too big nor too small
    @assert size(ave_degrees, 1) == length(n_per_community) "size of ave_degrees must agree with n_per_community"

    # create sbm_graph, notice that the rule of ave_degrees and n_per_community
    # must follow the rule of sbm_graph.
    temp = stochastic_block_model(ave_degrees, n_per_community)
    return Graph(get_biggest_component_adj_matrix(temp))
end

#=
Return the adjacency matrix of an sbm graph.

Params:
    ave_degrees: Array{64, 2}: dimension = number of community x number of community.
        ave_degrees[x,y] is the average degrees that a node in community x has with
            other nodes in community y.
        ave_degrees[x,x] therefore is the average degrees of community x.
    n_per_community: Array{64, 1}: length = number of community.
        n_per_community[i]: is the number of nodes in community i.
Return:
    the adjacency matrix of a sbm graph
=#
function sbm_adj_matrix(ave_degrees , n_per_community)
    # make sure the size of community is appropriate, not too big nor too small
    @assert size(ave_degrees, 1) == length(n_per_community) "size of ave_degrees must agree with n_per_community"

    # create sbm_graph, notice that the rule of ave_degrees and n_per_community
    # must follow the rule of sbm_graph.
    temp = stochastic_block_model(ave_degrees, n_per_community)
    return get_biggest_component_adj_matrix(temp)
end



# Complete Bipartite Graph
#=
Return the adjacency matrix of a complete bipartite graph.

Params:
    n1: Int: number of nodes in partition 1
    n2: Int: number of nodes in partition 2
Return:
    The adjacency matrix of a complete bipartite graph.
=#
function complete_bipartite_adj_matrix(n1=10, n2=5)
    temp = complete_bipartite_graph(n1, n2)
    return adjacency_matrix(temp)
end
