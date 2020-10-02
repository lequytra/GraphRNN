using LinearAlgebra
using Random
using LightGraphs
using Flux

# ENCODE or DECODE *************************************************************
#=
Generate a sequences of step to construct the given adj_matrix.
A step in this case is an array with fixed size.

This can be decoded by decode_full function
=#
function encode_full(adj_matrix, max_prev_node=nothing, is_full = false)
    if max_prev_node == nothing || is_full
        max_prev_node = size(adj_matrix)[1] - 1
    end

    # pick up the lower triangular matrix
    adj_matrix = tril(adj_matrix, -1)

    # n = number of nodes in the network
    n = size(adj_matrix)[1]

    # remove the first row and last column of adj_matrix (right most)
    adj_matrix = adj_matrix[2:n, 1:n-1]

    # init the output matrix
    output = fill(0, (size(adj_matrix)[1], max_prev_node))

    # for each node, create a construction step.
    for i in 1:size(adj_matrix, 1)
        # start and end indices of row i in adj_matrix
        input_start = max(1, i - max_prev_node + 1)
        input_end = i

        # start and end indices of row i in output
        output_start = max_prev_node - (input_end - input_start)
        output_end = max_prev_node

        # write to output
        output[i, output_start:output_end] = adj_matrix[i, input_start:input_end]
    end
    # reverse order. Can we test without reverse?
    output = reverse(output, dims=2)
    return output
end


#=
Create adjacency matrix from the given encoded_matrix. The encoded_matrix is
created from encode_full function.
=#
function decode_full(encoded_matrix)
    # n = number of nodes - 1, max_prev_node = size of a step.
    (n, max_prev_node) = size(encoded_matrix)

    # decoded is the lower triangular matrix of the original adjacency matrix
    decoded_tril = fill(0, (n, n))
    # reverse order. Can we test without reverse in both encode and decode?
    encoded_matrix = reverse(encoded_matrix, dims=2)

    # for each encoded step
    for i in 1:size(encoded_matrix, 1)
        # start and end of row i in decoded
        decoded_start = max(1, i - max_prev_node + 1)
        decoded_end = i

        # start and end of row i in encoded_matrix
        encoded_start = max_prev_node - (decoded_end - decoded_start)
        encoded_end = max_prev_node

        # copy to decoded_matrix
        decoded_tril[i, decoded_start:decoded_end] = encoded_matrix[i, encoded_start:encoded_end]
    end
    # final output, reconstruct the full matrix
    decoded_matrix = fill(0, (n+1, n+1))
    decoded_matrix[2:n+1, 1:n] = tril(decoded_tril, 0)
    decoded_matrix = decoded_matrix + transpose(decoded_matrix)
    return decoded_matrix
end


#=
Return sequential form of the graph represented
by adjacency matrix.

This can be decoded by decode function

# IMPORTANT: A bfs_ordering rearange must be called before calling this function
=#
function encode(adj_matrix)
    adj_matrix = tril(adj_matrix, -1)

    # n = number of nodes
    n = size(adj_matrix, 1)

    # remove the top row and right most column of the adj_matrix
    # the size is now (n-1)*(n-1)
    adj_matrix = adj_matrix[2:n, 1:n - 1]

    # encoded output matrix
    output = []

    # start index of adj_matrix
    input_start = 1
    for i in 1:size(adj_matrix, 1)
        # end index of adj_matrix
        input_end = i

        # get the encoded step
        step = adj_matrix[i, input_start:input_end]

        push!(output, step)
        non_zero = findall(x -> x != 0, step)[1]
        # update input_start for next encoded step
        input_start = input_end - length(step) + non_zero
    end
    # return encoded output
    return output
end

#=
This function call bfs_ordering before actually encode. This will avoid some
potential crash.
=#
function safe_encode(adj_matrix, root=nothing)
    g = Graph(adj_matrix)

    # bfs ordering
    indices = bfs_ordering(g, root)
    adj_matrix = adj_matrix[indices, indices]

    return encode(adj_matrix)

end

#=
This decode the encoded_matrix. This matrix is encoded by encode function
=#
function decode(encoded_matrix)
    # n = number of node - 1
    n = size(encoded_matrix, 1)

    # init the decoded matrix
    decoded_tril = fill(0, (n, n))

    # for each step
    for i in 1:size(decoded_tril, 1)
        decoded_start = i - length(encoded_matrix[i]) + 1
        decoded_end = i
        decoded_tril[i, decoded_start:decoded_end] = encoded_matrix[i]
    end
    decoded_matrix = fill(0, (n + 1, n + 1))
    decoded_matrix[2:n+1, 1:n] = tril(decoded_tril, 0)
    decoded_matrix = decoded_matrix + transpose(decoded_matrix)
    return decoded_matrix
end



# PERMUTE AND RE-ORDERING ******************************************************
#=
This function permute row and col of a matrix.
=#
function permute_matrix(matrix)
    random_indices = shuffle(1:size(matrix)[1])
    permuted_matrix = matrix[random_indices, random_indices]

    return permuted_matrix
end

#=
Return a list of vertices in BFS ordering
Params:
    graph (SimpleGraph): graph.
Returns:
    Array{Int32}: a list of vertex indices.
=#
function bfs_ordering(graph, root=nothing)
    if root == nothing
        # choose a random node to start
        rand_start_vertex = rand(1:nv(graph))
    else
        rand_start_vertex = root
    end

    # set up visit visit_queue
    visit_queue = [rand_start_vertex]       # node in visit_queue must also be
                                            # in bfs_seq
    bfs_seq = [rand_start_vertex]           # return sequence of node
    while size(visit_queue,1) > 0
        # dequeue visit queue (aka remove element 1)
        current_node = visit_queue[1]
        deleteat!(visit_queue,1)

        # get the neighbor of current node
        neighbors = outneighbors(graph, current_node)
        for neighbor in neighbors
            if !(neighbor in bfs_seq)
                push!(bfs_seq, neighbor)
                push!(visit_queue, neighbor)
            end
        end
    end
    return bfs_seq
end

#=
Reorder the nodes in a graph in bfs ordering
in:     adj_matrix: adjacency matrix of that graph
        root: the starting node for bfs. If == nothing, the start node would
            begin randomized and therefore, the outputs are different between
            different run.
out:    adjacency matrix of the graph after got bfs_ordering
=#
function bfs_adj_matrix(adj_matrix, root=nothing)
    g = Graph(adj_matrix)

    indices = bfs_ordering(g, root)
    return adj_matrix[indices, indices]
end



# MAIN HELPER ******************************************************************
#=
Estimate the maximum number of prev nodes to keep.
in:     all_matrix: array of adjacency matrix
        n_sample: number of sampled matrices being used to find max prev_node.

out:    the max prev_node among sampled matrix

note prev_node = number of cols of (encoded matrix)
=#
function find_max_prev_node(all_matrix, n_sample=nothing, root=nothing)
    # set number of sample if it is nothing.
    if n_sample == nothing
        n_sample = max(20000, size(all_matrix)[1])
    end
    println(n_sample)

    # create a random indices
    random_indices = rand(1:size(all_matrix)[1], n_sample)

    # create sampled matrices array
    sampled_matrix = all_matrix[random_indices]
    max_node_so_far = 0

    # find max_node_so_far
    for matrix in sampled_matrix
        encoded = safe_encode(matrix, root)
        max_node_so_far = max(max_node_so_far, reduce(max,map(length, encoded)))
    end

    return max_node_so_far
end

#=
Generate training data from adjacency matrices.

in:     all_matrix: array of adjacency matrices. This is our dataset
        max_num_node: maximum number of nodes in matrices in all_matrix
        max_prev_node: int, the maximum length of the cols in adjacency matrix
        n_sample = int, number of sample

out:    all_x = array of encoded matrices. size = batch_size
        all_y = our array of targets correspond to . size = batch_size
        all_len = int, number of nodes in each matrix

IMPORTANT for consistency, we
=#
function transform(all_matrix, max_num_node=nothing, max_prev_node=nothing, n_sample=nothing, root=1)
    # get an array of number of nodes of each matrix in all_matrix
    ones = fill(1, length(all_matrix))
    all_size = map(size, all_matrix, ones)

    # get the max prev_node (max width of matrices in all_matrix)
    if max_prev_node == nothing
        max_prev_node = find_max_prev_node(all_matrix, n_sample=n_sample)
    end

    # set up n (maximum number of nodes)
    if max_num_node == nothing
        n = reduce(max, all_size)
    else
        n = max_num_node
    end

    # output
    all_x = []
    all_y = []
    all_len = []

    for matrix in all_matrix
        # init the training sequence
        x = fill(0, (n, max_prev_node))
        x[1, :] .= 1

        # init the training label (target)
        y = fill(0, (n, max_prev_node))

        # permute order of nodes
        shuffled_matrix = permute_matrix(matrix)

        # generate new graph from permuted adj_matrix
        shuffled_matrix = bfs_adj_matrix(shuffled_matrix, root=root)

        # encode the matrix
        encoded = encode_full(shuffled_matrix, max_prev_node=max_prev_node)

        y[1:size(encoded)[1], :] = encoded
        x[2:size(encoded)[1] + 1, :] = encoded

        push!(all_x, Flux.batchseq([x[:, i] for i in 1:size(x)[2]]))
        push!(all_y, Flux.batchseq(y[:, i] for i in 1:size(y)[2]))
        push!(all_len, size(matrix)[1])
    end
    return (all_x, all_y, all_len)
end
