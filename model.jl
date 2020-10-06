using Flux
using Zygote: @adjoint
using LinearAlgebra

mutable struct OutputModule 
	output_layer::Chain
	OutputModule(hidden_size, embedding_size, output_size) = new(Chain(Flux.Dense(hidden_size, embedding_size, relu), 
		Flux.Dense(embedding_size, output_size, sigmoid)))
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

GRUBlock(input_size, embedding_size, hidden_size; has_input=true, has_output=true, output_size=nothing) =
	GRUBlock(has_input, 
		has_output, 
		Flux.Dense(input_size, embedding_size), 
		GRU(embedding_size, hidden_size), 
		OutputModule(hidden_size, embedding_size, output_size))

hidden(m::GRUBlock) = m.rnn.state

function set_hidden!(m::GRUBlock, h)
	m.rnn.state = h 
end

Flux.trainable(gru::GRUBlock) = (gru.linear, gru.rnn, gru.output_module)
Flux.reset!(gru::GRUBlock) = Flux.reset!(gru.rnn.chain)

function (m::GRUBlock)(inp; hidden=nothing)
	if m.has_input
		inp = m.linear(inp)
	end
	if hidden != nothing
		set_hidden!(m, hidden)
	end
	inp = m.rnn(inp)
	if m.has_output
		inp = m.output_module(inp)
	end
	return inp 
end

mutable struct GraphRNN 
	graph_level::GRUBlock
	edge_level::GRUBlock
	GraphRNN(input_size, node_embedding_size, edge_embedding_size, node_hidden_size, node_output_size) = 
		new(GRUBlock(input_size, node_embedding_size, node_hidden_size; output_size=node_output_size), 
			GRUBlock(1, edge_embedding_size, node_output_size; output_size=1))

end

Flux.trainable(m::GraphRNN) = (m.graph_level, m.edge_level)
Flux.reset!(m::GraphRNN) = Flux.reset!(m.graph_level) && Flux.reset!(m.edge_level)

function (m::GraphRNN)(inp) 
	n_nodes = size(inp[1], 2)
	inp2 = m.graph_level.(inp)
	inp2 = cat(inp2..., dims=2)

	partial = [item[:, 1:end-1] for item in inp]
	partial = cat([reshape(item, (size(item)..., 1)) for item in partial]..., dims=3)
	edge_inp = cat([fill(1.0, (size(inp[1], 1), 1, size(inp)[1])), partial]..., dims=2)

	edge_inp = cat([edge_inp[:, :, i] for i in 1:size(inp, 1)]..., dims=2)

	edge_inp = [reshape(edge_inp[:, i], (1, size(edge_inp, 1))) for i in 1:size(edge_inp, 2)]

	all_output = []

	inp2 = [inp2[:, i] for i in 1:size(inp2, 2)]
	hidden_in = zip(inp2, edge_inp)
	all_output = [m.edge_level(in_, hidden=hidden) for (hidden, in_) in hidden_in]
	
	all_output = cat(all_output..., dims=1)
	all_output = reshape(all_output, size(inp, 1), n_nodes, :)
	all_output = [transpose(all_output[i, :, :]) for i in 1:size(inp, 1)]
	all_output = [reshape(item, size(item)..., 1) for item in all_output]
	all_output = cat(all_output..., dims=3)
	println("done")
	return all_output
end

@adjoint function Base.Iterators.Zip(is)
  Base.Iterators.Zip(is), Δ -> (collect(zip(Δ...))...,)
end

# Hidden: hidden_size x timesteps