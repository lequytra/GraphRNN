using LinearAlgebra
using Random
using LightGraphs

function encode_full(adj_matrix, max_node=nothing)
    if max_node == nothing
        max_node = size(adj_matrix)[1] - 1
    end

    adj_matrix = tril(adj_matrix, -1)
    n = size(adj_matrix)[1]
    adj_matrix = adj_matrix[2:n, 1:n-1]

    output = fill(0, (n, max_node))

    for i in 1:n
        start = max(0, i - max_node + 1)
        output_start = max_node + start - i
        output[i, output_start:max_node - 1] = adj_matrix[i, start:i]
    end
    output = reverse(output, dims=2)
    return output
end

function decode_full(encoded_matrix)
    n, max_node = size(encoded_matrix)
    decoded = fill(0, (n, n))
    encoded_matrix = reverse(encoded_matrix, dims=2)

    for i in range(encoded_matrix)
        start = max(0, i - max_node + 1)
        output_start = max_node + start - i
        decoded[i, start:i] = encoded_matrix[i, output_start:max_node - 1]
    end
    output = fill(0, (n+1, n+1))
    output[2:n, 1:n-1] = tril(decoded, 0)
    output = output + transpose(output)
    return output
end

function encode(adj_matrix)
    #=
    Return sequential form of the graph represented
    by adjacency matrix.
    =#
    adj_matrix = tril(adj_matrix, -1)
    n = size(adj_matrix)[1]
    adj_matrix = adj_matrix[2:n, 1:n - 1]
    output = []
    start = 0
    for i in 1:n
        arr = adj_matrix[i, start:i]
        push!(output, arr)
        first_zero = findall(x -> x == 0, arr)
        start = end - length(arr) + first_zero
    end
    return output
end


function decode(encoded_adj)
    n_nodes = length(encoded_adj)
    adj = fill(0, (n_nodes, n_nodes))
    for i in 1:n_nodes
        start = i - length(encoded_adj[i])
        adj[i, start:i] = encoded_adj[i]
    end
    adj_matrix = fill(0, (n_nodes + 1, n_nodes + 1))
    adj_matrix[2:n_nodes, 1:n_nodes - 1] = tril(adj, 0)
    adj_matrix = adj_matrix + transpose(adj_matrix)
    return adj_matrix
end

function bfs_ordering(graph, root=1)
    #=
    Return a list of vertices in BFS ordering
    Params:
        graph (SimpleGraph): graph.
    Returns:
        Array{Int32}: a list of vertex indices.
    =#

    #Todo: Implement BFS ordering here.
    return [i for i in 1:nv(graph)]
end

function find_max_node(all_matrix, n_sample=nothing)
    #=
    Estimate the maximum number of prev nodes to keep.
    =#
    if n_sample == nothing
        random_indices = 1:size(all_matrix)[1]
    end
    # randomly pick n_sample
    random_indices = rand(1:size(all_matrix)[1], n_sample)
    all = all_matrix[random_indices]
    max_so_far = 0
    for matrix in all
        encoded = encode(matrix)
        max_so_far = max(max_so_far, max(map(x -> length(x), encoded)))
    end

    return max_so_far
end

function transform(all_matrix, max_node=nothing, n_sample=nothing)
    all_size = [size(matrix)[1] for matrix in all_matrix]
    if max_node == nothing
        n = max(all_size)
    else
        n = max_node
    end

    if max_node == nothing
        max_node = find_max_node(all_matrix, n_sample=n_sample)
    end
    all_data = []
    for matrix in all_matrix
        x = fill(0, (n + 1, max_node))
        x[1, :] .= 1
        y = fill(0, (n + 1, max_node))
        random_indices = shuffle(1:size(matrix)[1])
        matrix_shuffle = matrix[random_indices, random_indices]
        graph = SimpleGraph(matrix_shuffle)
        indices = bfs_ordering(graph)
        matrix_shuffle = matrix_shuffle[indices, indices]
        encoded = encode(matrix_shuffle, max_node=max_node)
        y[1:size(encoded)[1], :] = encoded
        x[2:size(encoded)[1] + 1, :] = encoded
        push!(all_data, Dict([("x": x), ("y": y), ("len": size(matrix)[1])))
    end
    return all_data
end