using LightGraphs
include("data_helpers.jl")
include("graph_generator.jl")


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

adj = er_adj_matrix(10, 0.1)
bfs_adj_matrix(adj)

test_encode_decode_full_er()
