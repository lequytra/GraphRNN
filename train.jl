using Flux, CUDA
import YAML
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
	loss(x, y) = Flux.Losses.logitcrossentropy(model(x), y)

	for i in resume_from:epochs
		println("start $i")
		Flux.train!(loss, pms, dataloader, opt)
		println("Finish 1 epoch.")

		@save "$(checkpoints_folder)/model-$(now()).bson" model opt
		println("Model saved.")
		Flux.reset!(model)
		println("Reset.")
	end
end


function main(config_path)
	#TODO fill in main here including loading data, train, test etc. 
	args = YAML.load_file(config_path)

	# x, y, len = load_dataset(args["data_path"]) |> gpu
	x, y = [rand(28, 30) for i in 1:1000], rand(28, 30, 1000)

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
