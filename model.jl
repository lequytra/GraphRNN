using Flux, CUDA
using Zygote: @adjoint, @showgrad, @nograd
using LinearAlgebra

#=
OutputModule struct defines the output layer of GraphRNN. 
It allow initialization of 2 Dense layers that are connected
to each other. 
Args:
	hidden_size (int): hidden size of preceding rnn in GraphRNN. 
	embedding_size (int): node/edge embedding size. 
	output_size (int): desired output size. 
=#
mutable struct OutputModule
	output_layer

	OutputModule(hidden_size, embedding_size, output_size; device=cpu) = new(Chain(Flux.Dense(hidden_size, embedding_size, relu), 
		Flux.Dense(embedding_size, output_size, sigmoid)) |> device) 
end

Flux.@functor OutputModule

#=
This function makes OutputModule struct callable. 
It defines the behavior of calling OutputModule.
Args:
	x (AbstractArray): input for the OutputModule. 
=#
function (m::OutputModule)(x)
	return m.output_layer(x)
end

#=
Defines a GRUBlock in GraphRNN. It consists of a Dense layer, 
an RNN layer and an OutputModule. 
Args:
	input_size (int):
	embedding_size (int): node/edge embedding size. 
	hidden_size (int): hidden size for the RNN layer. 
	has_input (Optional[bool]): whether the GRUBlock takes input. 
								Default to true. 
	has_output (Optional[bool]): whether the GRUBlock returns an output. 
								Default to true. 
	output_size (Optional[int]): If has_output is false, then this argument is 
								not needed. Otherwise, define the desired output size.
	device (`cpu` or `gpu`): device to use. 
=#
mutable struct GRUBlock
	has_input::Bool
	has_output::Bool
	linear::Dense
	rnn::Flux.Recur
	output_module::OutputModule
end

GRUBlock(input_size, embedding_size, hidden_size; has_input=true, has_output=true, output_size=nothing, device=cpu) =
	GRUBlock(has_input,
		has_output,
		Flux.Dense(input_size, embedding_size) |> device,
		GRU(embedding_size, hidden_size) |> device,
		OutputModule(hidden_size, embedding_size, output_size; device=device))


hidden(m::GRUBlock) = m.rnn.state # Return the current hidden state in the GRUBlock. 

#=
Set the hidden state of the GRUBlock with the given hidden state. 
Args:
	m (GRUBlock): the GRUBlock to set hidden state for. 
	h (AbstractArray): hidden state. 
=#
function set_hidden!(m::GRUBlock, h)
	m.rnn.state = h
end


# Define trainable weights for the GRUBlock
Flux.trainable(gru::GRUBlock) = (gru.linear, gru.rnn, gru.output_module)
# Function to reset hidden state for the GRUBlock
Flux.reset!(gru::GRUBlock) = Flux.reset!(gru.rnn.state)

#=
Function to make GRUBlock struct callable. This defines the behavior
when calling GRUBlock. 
Args:
	inp (AbstractArray): input. 
	hidden (Optional[AbstractArray]): hidden state of the GRUBlock. 
	reset (Optional[bool]): whether to reset the hidden state of the GRUBlock
							after running. Default to true. 
Returns:
	the computational result of running input through GRUBlock. 
=#
function (m::GRUBlock)(inp; hidden=nothing, reset=true)
	if m.has_input
		inp = m.linear(inp)
	end
	if hidden != nothing
		set_hidden!(m, hidden)
	end
	inp = m.rnn(inp)
	Flux.reset!(m.rnn)
	if m.has_output
		inp = m.output_module(inp)
	end
	return inp
end


#=
Defines the architecture of GraphRNN, consisting of 2 GRUBlock: graph-level, edge-level. 
Args:
	input_size (int): input size. 
	node_embedding_size (int): size of the vector representing a node.
	edge_embedding_size (int): size of the vector representing an edge. 
	node_hidden_size (int): hidden size for the graph-level GRUBlock. 
	node_output_size (int): desired output for graph-level GRUBlock.
	device (Optional): device to use. 
=#
mutable struct GraphRNN
	device
	graph_level::GRUBlock
	edge_level::GRUBlock
	GraphRNN(input_size, node_embedding_size, edge_embedding_size, node_hidden_size, node_output_size; device=cpu) =
		new(device,
			GRUBlock(input_size, node_embedding_size, node_hidden_size; output_size=node_output_size, device=device),
			GRUBlock(1, edge_embedding_size, node_output_size; output_size=1, device=device))

end

# define trainable weights for GraphRNN
Flux.trainable(m::GraphRNN) = (m.graph_level, m.edge_level)
# Function to reset hidden states in GraphRNN model. 
function Flux.reset!(m::GraphRNN)
	Flux.reset!(m.graph_level)
	Flux.reset!(m.edge_level)
end

#=
This function makes GraphRNN struct callable. It defines the forward pass for
GraphRNN. 
Args:
	inp (AbstractArray): input
	reset (Optional[bool]): whether to reset hidden states after the forward pass. 
							Default to true.
=#
function (m::GraphRNN)(inp; reset=true)
	n_nodes = size(inp, 2)

	inp2 = m.graph_level(inp, reset=true) |> cpu
	inp = inp |> cpu

	partial = inp[:, 1:end- 1]
	edge_inp = cat([fill(1.0, size(inp, 1)), partial]..., dims=2)
	edge_inp = reshape(edge_inp, (1, size(edge_inp)...))
	edge_inp = [edge_inp[:, :, i] for i in 1:size(edge_inp, 3)]

	# Turn hidden state matrix into list of vectors
	inp2 = [inp2[:, i] for i in 1:size(inp2, 2)]
	hidden_in = zip(inp2 |> m.device, edge_inp |> m.device)

	all_output = [m.edge_level(in_, hidden=hidden, reset=true) for (hidden, in_) in hidden_in]

	all_output = cat(all_output..., dims=1)
	all_output = transpose(all_output)

	return all_output
end

# define autograd behavior for zip function. 
@adjoint function Base.Iterators.Zip(is)
  Base.Iterators.Zip(is), Δ -> (collect(zip(Δ...))...,)
end

