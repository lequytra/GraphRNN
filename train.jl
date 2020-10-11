using Flux
using Torch: torch
using TensorBoardLogger
import YAML
using Dates
using ProgressMeter
using Statistics: mean
using BSON: @save, @load

include("model.jl")
include("data_helpers.jl")
include("data.jl")

if has_cuda()
	@info "CUDA is on."
	using CUDA
	device = gpu
else 
	device = cpu
end

function TBLogger(logger; kws...)
	with_logger(logger) do 
		for (k, v) in kws
			#TODO: log information here
		end
	end
end


function test(testloader, loss_func)
	total_loss = 0
	iter = 0
	for (x, y) in testloader
		total_loss += loss_func(x, y)
		iter += 1
	end
	@show "Test loss: $(total_loss/iter)"
end

function train(model, lr, trainloader, testloader, epochs, resume_from=1, checkpoints_folder="", model_path=nothing)
	if model_path != nothing && ispath(model_path)
		@load model_path model opt
	else
		opt = Flux.Optimise.ADAM(lr)
	end

	pms = Flux.params(model)
	loss(x, y) = mean(Flux.Losses.logitcrossentropy.(model.(x), y))
	
	for i in resume_from:epochs
		# Train on entire dataset
		p_train = Progress(length(trainloader),
							dt=0.5, 
							barglyphs=BarGlyphs("[=> ]"), 
							barlen=50, 
							color=:yellow)
		total_loss = 0
		for (x, y) in trainloader
			loss_val = nothing
			grads = gradient(pms) do 
				loss_val = loss(x, y)
			end
			total_loss += loss_val
			Flux.update!(opt, pms, grads)
			ProgressMeter.next!(p_train; showvalues = [(:loss, loss_val)])
		end
		println("Training loss at iteration $(i + 1): $(total_loss/length(trainloader))")
		p_test = Progress(length(testloader),
							dt=0.5, 
							barglyphs=BarGlyphs("[=> ]"), 
							barlen=50, 
							color=:green)	
		total_loss = 0
		for (x, y) in testloader 
			loss_val = loss(x, y)
			total_loss += loss_val
			ProgressMeter.next!(p_test; showvalues = [(:loss, loss_val)])
		end
		println("Testing loss at iteration $(i + 1): $(total_loss/length(testloader))")

		println("Saving model ...")
		@save "$(checkpoints_folder)/model-$(Dates.now()).bson" model opt
		
		Flux.reset!(model)
	end
end


function main(config_path)
	#TODO fill in main here including loading data, train, test etc.
	args = YAML.load_file(config_path)

	X_train, y_train, len_train = load_dataset(args["data"]["train"]) 
	X_train, y_train = [X_train[:, :, i] for i in 1:size(X_train, 3)] |> device, 
	[y_train[:, :, i] for i in 1:size(y_train, 3)] |> device

	X_test, y_test, len_test = load_dataset(args["data"]["test"]) 
	X_test, y_test = [X_test[:, :, i] for i in 1:size(X_test, 3)] |> device, 
	[y_test[:, :, i] for i in 1:size(y_test, 3)] |> device


	train_loader = Flux.Data.DataLoader((X_train, y_train), batchsize=args["batch_size"], shuffle=true)
	test_loader = Flux.Data.DataLoader((X_test, y_test), batchsize=args["batch_size"], shuffle=true)
	# TODO: default to 100 max_prev_node for now since data is encoded.
	max_prev_node = args["max_prev_node"] != -1 ? args["max_prev_node"] : 100

	println("Training on $(length(X_train)) examples.")
	println("Testing on $(length(X_test)) examples")

	model = GraphRNN(max_prev_node,
		args["node_embedding_size"],
		args["edge_embedding_size"],
		args["node_hidden_size"],
		args["node_output_size"];
		device=device) |> torch

	if !isdir(args["checkpoints"])
		mkdir(args["checkpoints"])
	end

	train(model, args["lr"], 
		train_loader, 
		test_loader,
		args["epochs"], 
		args["resume_from"], 
		args["checkpoints"])

end;

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
