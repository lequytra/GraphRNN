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
include("tsb_logger.jl")

const TEMP_IMG_FILE_PATH = "temp.png"


#=
Load saved model, optimizer and epoch index from a given path or
checkpoint folder. If no saved model is found, return nothing. 
Args:
	path(Optional[str]): path to saved model. 
	checkpointdir (Optional[str]): path to checkpoints folder. 

Returns
	model (GraphRNN): the saved model from previous training session. 
	opt (Optimizer): optimizer state for the saved model. 
	epoch (int): epoch number when the model was saved. 
=#
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

#=
Train and evaluate the GraphRNN model. 
Args:
	args (dict): configs. 
	model (GraphRNN): an initialized GraphRNN model. 
	trainloader (Flux.Data.Dataloader): training dataset. 
	testloader (Flux.Data.Dataloader): testing dataset. 
	resume_from (Optional[int]): resume training from certain epoch. 
	opt (Optional[Optimizer]): optimizer to use. 
	tblogger (TBLogger): Tensorboard logger object. 
=#
function train(args, model, trainloader, testloader; resume_from=1, opt=nothing, tblogger=nothing)
	# set up optimizer if we don't have any
	if opt == nothing
		opt = Flux.Optimise.ADAM(args["lr"])
	end

	pms = Flux.params(model)
	loss(x, y) = mean(Flux.Losses.logitbinarycrossentropy.(model.(x, reset=true), y))

	# for each epoch
	for i in resume_from:args["epochs"]
		# Train on entire dataset
		p_train = Progress(length(trainloader),
							dt=0.5,
							barglyphs=BarGlyphs("[=> ]"),
							barlen=50,
							color=:yellow)
		total_loss = 0
		for (idx, (x, y)) in enumerate(trainloader)
			loss_val = nothing
			grads = gradient(pms) do
				loss_val = loss(x, y)
			end
			total_loss += loss_val
			Flux.update!(opt, pms, grads)
			ProgressMeter.next!(p_train; showvalues = [(:iter, i), (:loss, loss_val)])
			# log test loss
			if args["TBlog"]["enable"]
				iter_ = length(trainloader) * (i - 1) + idx 
				TB_log_scalar(tblogger, args["TBlog"]["train_loss_tag"], loss_val, iter_)
			end
		end

		# printing training loss
		println("Training loss at iteration $(i): $(total_loss/length(trainloader))")


		if i % args["eval_period"] == 0

			p_test = Progress(length(testloader),
								dt=0.5,
								barglyphs=BarGlyphs("[=> ]"),
								barlen=50,
								color=:green)
			total_loss = 0
			for (idx, (x, y)) in enumerate(testloader)
				loss_val = loss(x, y)
				total_loss += loss_val
				ProgressMeter.next!(p_test; showvalues = [(:iter, i), (:loss, loss_val)])
				# log test loss
				if args["TBlog"]["enable"] && i % args["TBlog"]["test_loss_log_period"] == 0
					iter_ = length(testloader) * (i - 1) + idx 
					TB_log_scalar(tblogger, args["TBlog"]["test_loss_tag"], loss_val, iter_)
				end
			end

			# printing and logging testing loss
			println("Testing loss at iteration $(i): $(total_loss/length(testloader))")

		end

		# Log test image
		if args["TBlog"]["enable"] && i % args["TBlog"]["img_log_period"] == 0
			G_pred = test_rnn_epoch(model, args["max_num_node"], args["max_prev_node"]; test_batch_size=1)

			# Write image to TEMP_IMG_FILE_PATH
			# "sbm" can be replaced by "ladder", "grid", "complete_bipartite"
			graph_viz(G_pred[1], "sbm", TEMP_IMG_FILE_PATH)

			# Log image from TEMP_IMG_FILE_PATH
			TB_log_img(tblogger, string(args["TBlog"]["img_tag"], " ", i), TEMP_IMG_FILE_PATH, 1)
		end


		if args["auto_resume"]["save_model"] != 0 && i % args["auto_resume"]["save_model"] == 0
			println("Saving model ...")
			@save "$(args["auto_resume"]["checkpointsdir"])/model-$(Dates.now()).bson" model opt i
		end
	end
end


#=
Prepare dataset, train and test GraphRNN
Args:
	config_path (str): path to the YAML config file. 
					See `configs/test.yaml` for reference. 
Returns:
	model (GraphRNN): the trained GraphRNN model. 
=#
function main(config_path)
	# Load the config file. 
	args = YAML.load_file(config_path)

	# Load training dataset. 
	X_train, y_train, len_train = load_dataset(args["data"]["train"])
	X_train, y_train = [X_train[:, :, i] for i in 1:size(X_train, 3)],
	[y_train[:, :, i] for i in 1:size(y_train, 3)]

	# Load eval dataset
	X_test, y_test, len_test = load_dataset(args["data"]["test"])
	X_test, y_test = [X_test[:, :, i] for i in 1:size(X_test, 3)] ,
	[y_test[:, :, i] for i in 1:size(y_test, 3)] 


	train_loader = Flux.Data.DataLoader((X_train, y_train), batchsize=args["batch_size"], shuffle=true)
	test_loader = Flux.Data.DataLoader((X_test, y_test), batchsize=args["batch_size"], shuffle=true)
	# TODO: default to 100 max_prev_node for now since data is encoded.
	max_prev_node = args["max_prev_node"] != -1 ? args["max_prev_node"] : 30

	println("Training on $(length(X_train)) examples.")
	println("Testing on $(length(X_test)) examples")

	model, opt, resume_epoch = nothing, nothing, 1
	# If auto_resume is enabled, try to find saved model paths. 
	if args["auto_resume"]["enable"]
		model_path = args["auto_resume"]["model_path"] != "" ? args["auto_resume"]["model_path"] : nothing
		model, opt, resume_epoch = load_checkpoint(path=model_path,
			checkpointdir=args["auto_resume"]["checkpointsdir"])
	end

	# If a pre-trained model is not found, initialize GraphRNN with random weights. 
	if model == nothing
		model = GraphRNN(args["max_prev_node"],
			args["node_embedding_size"],
			args["edge_embedding_size"],
			args["node_hidden_size"],
			args["node_output_size"];
			device=cpu) |> torch
	end

	# Create a folder for saving checkpoints. 
	if !isdir(args["auto_resume"]["checkpointsdir"])
		@info "Model weights, optimizer and epoch index will be automatically saved to $(args["auto_resume"]["checkpointsdir"])"
		mkdir(args["auto_resume"]["checkpointsdir"])
	end

	@info "Train for $(args["epochs"]) epochs."

	logger = nothing
	# create TB logger object. 
	if args["TBlog"]["enable"]
		logger = TB_set_up(args["TBlog"]["log_dir"])
	end
	# Train and eval. 
	train(args, model,
		train_loader,
		test_loader;
		resume_from=resume_epoch,
		opt=opt,
		tblogger=logger
		)

	# Clear model's hidden state
	Flux.reset!(model)

	if args["inference"]["enable"]
		# Run inference using the trained model. 
		G_pred = test_rnn_epoch(model, args["max_num_node"], args["max_prev_node"]; test_batch_size=args["inference"]["num_preds"])

		if !isdir("predictions")
			mkdir("predictions")
		end

		# Writing the visualized predictions to predictions folder. 
		for (i, g) in enumerate(G_pred)
			grid_viz(g, file_name="predictions/$i.png")
		end
		@save "prediction_result.bson" G_pred
	end

	return model
end;


#=
Inference logic for GraphRNN.
Args:
	model (GraphRNN): GraphRNN model. 
	max_num_node (int): max number of nodes in the graphs. 
	max_prev_node (int): longest possible path. 
					See https://arxiv.org/pdf/1802.08773.pdf
	test_batch_size (Optional[int]): number of graphs to generate. 
Returns:
	test_sample_graphs (list[Graph]): a list of generated graphs using GraphRNN. 
=#

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
				edge_level_x_step = sample_sigmoid(edge_level_y_pred; sample=false, thresh=0.5, sample_time=1)
				x_step[j:j] = edge_level_x_step
			end

			y_pred[:, i] = x_step
		end

		Flux.reset!(model)

		encoded_seq = convert(Array{Int64, 2}, y_pred)
		g = Graph(decode_full(transpose(encoded_seq)))
		push!(test_sample_graphs, g)

	end
	return test_sample_graphs
end;


#=
Function to sample edges from an array of edge prediction scores. 
Args:
	edge_level_y (AbstractArray): an array of score for possible edges. 
	sample (Optional[true]): If true, sample edges independently using 
							nultivariate Bernoulli distribution. 
							If false, pick all edges with scores higher 
							than a threshold. 
	thresh (Optional[float]): threshold for picking edges. 
	sample_time (Optional[int]): number of times to run sampling method. 
Returns:
	y_result (AbstractArray): a binary array denoting predicted edges for a node. 
=#
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
