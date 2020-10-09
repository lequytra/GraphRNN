using Flux, CUDA
import YAML
using Dates
using Statistics: mean
using BSON: @save, @load

include("model.jl")
include("data_helpers.jl")
include("data.jl")


function train(model, lr, dataloader, epochs, resume_from=1, checkpoints_folder="", model_path=nothing)
	if model_path != nothing && ispath(model_path)
		@load model_path model opt
	else
		opt = Flux.Optimise.ADAM(lr)
	end
	pms = Flux.params(model)
	loss(x, y) = mean(Flux.Losses.logitcrossentropy.(model.(x), y))
	
	for i in resume_from:epochs
		Flux.train!(loss, pms, dataloader, opt)
		println("Finish 1 epoch.")

		@save "$(checkpoints_folder)/model-$(Dates.now()).bson" model opt
		println("Model saved.")
		Flux.reset!(model)
		println("Reset.")
	end
end


function main(config_path)
	#TODO fill in main here including loading data, train, test etc.
	args = YAML.load_file(config_path)

	x, y, len = load_dataset(args["data_path"]) |> gpu
	x, y = [x[:, :, i] for i in 1:size(x, 3)], [y[:, :, i] for i in 1:size(y, 3)]
	# x, y = [rand(28, 30) for i in 1:2], [rand(28, 30) for i in 1:2]

	@show size(x), size(y)

	dataloader = Flux.Data.DataLoader((x, y), batchsize=args["batch_size"], shuffle=true)
	# TODO: default to 100 max_prev_node for now since data is encoded.
	max_prev_node = args["max_prev_node"] != -1 ? args["max_prev_node"] : 100

	model = GraphRNN(max_prev_node,
		args["node_embedding_size"],
		args["edge_embedding_size"],
		args["node_hidden_size"],
		args["node_output_size"])  

	if !isdir(args["checkpoints"])
		mkdir(args["checkpoints"])
	end

	train(model, args["lr"], dataloader, args["epochs"], args["resume_from"], args["checkpoints"])

end

# main("configs/test.yaml")

# function test_rnn_epoch (epoch, rnn, output, max_num_node, max_prev_node, test_batch_size=16)
# 	# TODO: init hidden layers and eval output

# 	# generate graphs
# 	y_prediction_long = fill(0.0, test_batch_size, max_num_node, max_prev_node) #|> gpu
# 	x_steps = fill(1.0, test_batch_size, 1, max_prev_node)

# 	for i in 1:max_num_node
# 		h = rnn(x_steps)

# 	end
# end
