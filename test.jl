using LightGraphs
include("data_helpers.jl")
include("graph_generator.jl")
include("data.jl")
include("graph_visualization.jl")

# Not important. Just to test implementation of different files.

# test encode / decode: check
function test_encode_decode_er()
    for i in 1:100
        g = er_graph(100, 0.05)
        adj_matrix = adjacency_matrix(g)
        adj_matrix = bfs_adj_matrix(adj_matrix)
        encoded_adj = encode(adj_matrix)
        decoded_adj = decode(encoded_adj)

        # test encode
        if adj_matrix != decoded_adj
            print("false")
        end
    end
end

function test_encode_decode_er_w_input(adj_matrix)
    @show adj_matrix
    @show encoded_adj = encode(adj_matrix)
    @show decoded_adj = decode(encoded_adj)

    if adj_matrix != decoded_adj
       print("False")
    end
end

test_encode_decode_er()

# test_encode_decode_full
function test_encode_decode_full_er()
    for i in 1:100
        g = er_graph(100, 0.05)
        adj_matrix = adjacency_matrix(g)
        adj_matrix = bfs_adj_matrix(adj_matrix)
        encoded_adj = encode_full(adj_matrix)
        decoded_adj = decode_full(encoded_adj)

        # test encode
        if adj_matrix != decoded_adj
            print("false")
        end
    end
end

function test_encode_decode_full_er_w_input(adj_matrix)
    @show adj_matrix
    @show encoded_adj = encode_full(adj_matrix)
    @show decoded_adj = decode_full(encoded_adj)

    if adj_matrix != decoded_adj
       print("False")
    end
end

# adj = er_adj_matrix(10, 0.1)
# adj = bfs_adj_matrix(adj)
# adj = test_encode_decode_full_er_w_input(adj)

# for i = 1:1
#     # w = [6.0 0.1;0.1 8.0]
#     # n_per_community = [40; 40]
#     # g = stochastic_block_model(w, n_per_community)
#
#     # sbm_viz(g, file_name=string("sbm",i,".png"))
#
#     # g = Grid([10,10])
#     # grid_viz(g, file_name="Grid.png")
#
#     g = ladder_graph(10)
#     ladder_viz(g, file_name="ladder.png")
#
#     g = complete_bipartite_graph(4,6)
#     complete_bipartite_viz(g, file_name="cb.png")
# end
