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

# adj = er_adj_matrix(10, 0.1)
# adj = bfs_adj_matrix(adj)
# adj = test_encode_decode_full_er_w_input(adj)

using TensorBoardLogger #import the TensorBoardLogger package
using Logging #import Logging package
using TestImages
using Flux: Data
using Random
logger = TBLogger("imagelogs", tb_append) #create tensorboard logger

################log images example: mri################
mri = testimage("mri")

#using logger interface
with_logger(logger) do
    @info "image/mri/loggerinterface" mri
end
#using explicit function interface
log_image(logger, "image/mri/explicitinterface", mri, step = 0)


################log images example: MNIST data################
images = shuffle(Data.MNIST.images())[1:5]

#using logger interface
with_logger(logger) do
    @info "image/mnist/loggerinterface" images = TBImages(images, HW)
end
#using explicit function interface
log_images(logger, "image/mnist/explicitinterface", images, step = 0)


################log images example: random arrays################
noise = rand(16, 16, 3, 4) #Format is HWCN
