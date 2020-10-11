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
	CUDA.allowscalar(false)
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

function test_rnn_epoch (epoch, graph_level, edge_level, max_num_node, max_prev_node, test_batch_size=16)
	# we sample one graph at a time
	test_sample_graphs = []

	for batch_id = 1:test_batch_size
		# init hidden of graph level to 0 and set eval
		set_hidden!(graph_level, fill(0.0, size(hidden(graph_level))))

		# generate graphs
		# shape of y_pred = matrix (max_prev_node X max_num_node)
		# shape of x_pred = matrix (max_prev_node X 1)
		y_pred = fill(0, max_prev_node, max_num_node)
		x_step = fill(1, max_prev_node, 1)

		for i in 1:max_num_node
			h = edge_level(x_step)

			# these lines of code need to be check, we need to permute h and truncate
			# hidden null to match the shape of hidden layers
			hidden_null = fill(0, max_prev_node, max_num_node)
			edge_level.hidden = cat(permute(h), hidden_null, dims=1)

			x_step = fill(0, max_prev_node, max_num_node)
			edge_level_x_step = fill(1, 1)

			for j in 1:min(max_prev_node, i)
				# this could be just a number
				edge_level_y_pred = edge_level(edge_level_x_step)
				edge_level_x_step = sample_sigmoid(edge_level_y_pred, True, 0.5, 1)
				x_step[j] = edge_level_x_step
				set_hidden!(edge_level, hidden(edge_level) # is this correct?
			end

			y_pred[:, i] = x_step
			set_hidden!(graph_level, hidden(graph_level))
		end
		encoded_seq = convert(Array{Int64, 2}, y_pred)

		test_sample_graphs.append(Graph(decode_full(encoded_seq)))
	end
	return test_sample_graphs
end


# since we only work with batch size of 1, we can simplify this function a bit
# compares to original version.
# edge_level_y is expected to be a 2D matrix only
function sample_sigmoid(edge_level_y, sample=True, thresh=0.5, sample_time=2)
	# calculate sigmoid
	edge_level_y = sigmoid(edge_level_y)

	# do sampling
	if sample
		if sample_time > 1
			y_result = rand(size(edge_level_y)) # verify the shape of edge_level output

			# since we only do one batch
			for i in 1:1
				# sample multiple time
				for j in 1:sample_time
					y_thresh = rand(size(edge_level_y))
					y_result = map(>, y, y_thresh)
					if sum(y_result) > 0
						break
					end
				end
			end
		else
			y_thresh = rand(size(edge_level_y))
			y_result = map(>, y, y_thresh)
		end
	else
		y_thresh = fill(1, size(edge_level_y)) .* thresh
		y_result = map(>, y, y_thresh)
	end
	# notice that in our case y_result is only a 2D matrix
	return y_result
end
