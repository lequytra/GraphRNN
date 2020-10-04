using Flux
using Filesystem
import YAML
using BSON: @save, @load

include("model.jl")
include("datahelper.jl")


function train(model, lr, dataloader, epochs, resume_from=1, checkpoints_folder="", model_path=nothing)
	if model_path != nothing and ispath(model_path)
		@load model_path model opt 
	else 
		opt = Flux.Optimise.ADAM(lr)
	end

	pms = Flux.params(model)
	loss(x, y) = Flux.Losses.logitcrossentropy(model(x), y)

	for i in resume_from:epochs
		Flux.train!(loss, pms, dataloader, opt)

		@save "$(checkpoints_folder)/model-$(now()).bson" model opt

		Flux.reset!(model)
	end
end


function main(config_path)
	#TODO fill in main here including loading data, train, test etc. 
	args = YAML.load_file(config_path)

	x, y, len = load_data(args["data_path"])

	dataloader = Flux.Data.DataLoader((x, y, len), batchsize=args["batch_size"], shuffle=true)
	max_prev_node = args["max_prev_node"] != -1 ? args["max_prev_node"] : # TODO: find max_prev_node
	model = Model(max_prev_node, 
		args["node_embedding_size"], 
		args["edge_embedding_size"], 
		args["node_hidden_size"], 
		args["node_output_size"]) 

	if !isdir(args["checkpoints"])
		mkdir(args["checkpoints"])
	end

	train(model, lr, dataloader, args["epochs"], args["resume_from"], args["checkpoints"])

end
