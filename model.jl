using Flux, CUDA
using Zygote: @adjoint, @showgrad, @nograd
using LinearAlgebra

mutable struct OutputModule
	output_layer

	OutputModule(hidden_size, embedding_size, output_size; device=cpu) = new(Chain(Flux.Dense(hidden_size, embedding_size, relu), 
		Flux.Dense(embedding_size, output_size, sigmoid)) |> device) 
end

Flux.@functor OutputModule

function (m::OutputModule)(x)
	return m.output_layer(x)
end

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

hidden(m::GRUBlock) = m.rnn.state

function set_hidden!(m::GRUBlock, h)
	m.rnn.state = h
end

Flux.trainable(gru::GRUBlock) = (gru.linear, gru.rnn, gru.output_module)
Flux.reset!(gru::GRUBlock) = Flux.reset!(gru.rnn.state)

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

mutable struct GraphRNN
	device
	graph_level::GRUBlock
	edge_level::GRUBlock
	GraphRNN(input_size, node_embedding_size, edge_embedding_size, node_hidden_size, node_output_size; device=cpu) =
		new(device,
			GRUBlock(input_size, node_embedding_size, node_hidden_size; output_size=node_output_size, device=device),
			GRUBlock(1, edge_embedding_size, node_output_size; output_size=1, device=device))

end

Flux.trainable(m::GraphRNN) = (m.graph_level, m.edge_level)
function Flux.reset!(m::GraphRNN)
	Flux.reset!(m.graph_level)
	Flux.reset!(m.edge_level)
end

# re
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


@adjoint function Base.Iterators.Zip(is)
  Base.Iterators.Zip(is), Δ -> (collect(zip(Δ...))...,)
end

# Hidden: hidden_size x timesteps
