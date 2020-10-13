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
include("graph_visualization.jl")

if has_cuda()
	@info "CUDA is on."
	using CUDA
	# Currently doesn't work on gpu so use cpu for now. 
	device = cpu
else
	device = cpu
end

function load_checkpoint(;path=nothing, checkpointdir="checkpoints")
	model, opt, i = nothing, nothing, 1
	if path != nothing
		@info "Loading pre-trained model from $path"
		@load path model opt epoch 
	elseif isdir(checkpointdir) 
		try
			path = maximum(readdir(checkpointdir))
			@load string(checkpointdir, "/", path) model opt i
			@info "Loading pre-trained model from $path"

		catch e 
			@warn "No pre-trained weights found. Train/eval on random weights."
		end
	end
	return model, opt, i
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

function train(model, lr, trainloader, testloader, epochs; resume_from=1, opt=nothing, checkpointsdir=nothing, eval_period=1)

	if opt == nothing
		opt = Flux.Optimise.ADAM(lr)
	end

	pms = Flux.params(model)
	loss(x, y) = mean(Flux.Losses.logitbinarycrossentropy.(model.(x, reset=true), y))

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
			ProgressMeter.next!(p_train; showvalues = [(:iter, i), (:loss, loss_val)])
		end
		println("Training loss at iteration $(i): $(total_loss/length(trainloader))")
		if i % eval_period == 0
			p_test = Progress(length(testloader),
								dt=0.5,
								barglyphs=BarGlyphs("[=> ]"),
								barlen=50,
								color=:green)
			total_loss = 0
			for (x, y) in testloader
				loss_val = loss(x, y)
				total_loss += loss_val
				ProgressMeter.next!(p_test; showvalues = [(:iter, i), (:loss, loss_val)])
			end
			println("Testing loss at iteration $(i): $(total_loss/length(testloader))")
		end
			if checkpointsdir != nothing && i % 100 == 0
				println("Saving model ...")
				@save "$(checkpointsdir)/model-$(Dates.now()).bson" model opt i
			end
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
	max_prev_node = args["max_prev_node"] != -1 ? args["max_prev_node"] : 30

	println("Training on $(length(X_train)) examples.")
	println("Testing on $(length(X_test)) examples")

	model, opt, resume_epoch = nothing, nothing, 1

	if args["auto_resume"]["enable"]
		model_path = args["auto_resume"]["model_path"] != "" ? args["auto_resume"]["model_path"] : nothing
		model, opt, resume_epoch = load_checkpoint(path=model_path, 
			checkpointdir=args["auto_resume"]["checkpointsdir"])
	end

	if model == nothing
		model = GraphRNN(args["max_prev_node"],
			args["node_embedding_size"],
			args["edge_embedding_size"],
			args["node_hidden_size"],
			args["node_output_size"];
			device=device) |> torch
	end

	if !isdir(args["auto_resume"]["checkpointsdir"])
		@info "Model weights, optimizer and epoch index will be automatically saved to $(args["auto_resume"]["checkpointsdir"])"
		mkdir(args["auto_resume"]["checkpointsdir"])
	end

	@info "Train for $(args["epochs"]) epochs."

	train(model, args["lr"],
		train_loader,
		test_loader,
		args["epochs"],
		resume_from=resume_epoch, 
		opt=opt,
		checkpointsdir=args["auto_resume"]["checkpointsdir"],
		eval_period=args["eval_period"])
	
	# Clear model's hidden state
	Flux.reset!(model)

	G_pred = test_rnn_epoch(model, args["max_num_node"], args["max_prev_node"]; test_batch_size=20)
	
	if !isdir("predictions")
		mkdir("predictions")
	end
	for (i, g) in enumerate(G_pred)
		grid_viz(g, file_name="predictions/$i.png")
	end
	@save "prediction_result.bson" G_pred
end;


function test_rnn_epoch(model, max_num_node, max_prev_node; test_batch_size=16)
	# we sample one graph at a time
	test_sample_graphs = []

	@showprogress for batch_id = 1:test_batch_size
		# init hidden of graph level to 0 and set eval
		set_hidden!(model.graph_level, fill(0.0, size(hidden(model.graph_level))))

		# generate graphs
		# shape of y_pred = matrix (max_prev_node X max_num_node)
		# shape of x_pred = matrix (max_prev_node X 1)
		y_pred = fill(0, max_prev_node, max_num_node)
		x_step = fill(1, max_prev_node) # Edge step

		for i in 1:max_num_node
			# Predict next node
			h = model.graph_level(x_step, reset=false) # max_prev_node x 1

			set_hidden!(model.edge_level, h)

			x_step = fill(0, max_prev_node) # (max_prev_node,)
			edge_level_x_step = fill(1, (1, 1))

			for j in 1:min(max_prev_node, i)
				# this could be just a number
				edge_level_y_pred = model.edge_level(edge_level_x_step, reset=false)
				edge_level_x_step = sample_sigmoid(edge_level_y_pred; sample=false, thresh=0.555, sample_time=1)
				x_step[j:j] = edge_level_x_step
			end

			y_pred[:, i] = x_step
			Flux.reset!(model)
		end
		encoded_seq = convert(Array{Int64, 2}, y_pred)
		g = Graph(decode_full(transpose(encoded_seq)))
		@show ne(g)/(nv(g)^2)
		push!(test_sample_graphs, g)

	end
	return test_sample_graphs
end;


# since we only work with batch size of 1, we can simplify this function a bit
# compares to original version.
# edge_level_y is expected to be a 2D matrix only
function sample_sigmoid(edge_level_y; sample=true, thresh=0.5, sample_time=2)
	# calculate sigmoid
	edge_level_y = sigmoid.(edge_level_y)

	# do sampling
	if sample
		if sample_time > 1
			y_result = rand(size(edge_level_y)) # verify the shape of edge_level output

			# sample multiple time
			for j in 1:sample_time
				y_thresh = rand(size(edge_level_y))
				y_result = map(>, edge_level_y, y_thresh)
				if sum(y_result) > 0
					break
				end
			end

		else
			y_thresh = rand(size(edge_level_y))
			y_result = map(>, edge_level_y, y_thresh)
		end
	else
		y_thresh = fill(1, size(edge_level_y)) .* thresh
		y_result = map(>, edge_level_y, y_thresh)
	end
	# notice that in our case y_result is only a 2D matrix
	return map(Int, y_result)
end
