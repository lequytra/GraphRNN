using Flux

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

function (m::GRUBlock)(inp)
	if m.has_input
		inp = m.linear(inp)
	end
	inp = m.rnn(inp)
	if m.has_output
		inp = m.output_module(inp)
	end
	return inp 
end

struct Model 
	graph_level
	edge_level
end

Flux.trainable(m::Model) = (m.graph_level, m.edge_level)
Flux.reset!(m::Model) = Flux.reset!(m.graph_level) && Flux.reset!(m.edge_level)

function (m::Model)(inp) 
	println("Input size: $(size(inp))")
	inp2 = m.graph_level(inp)
	set_hidden!(edge_level, inp2)
	edge_inp = fill(0.0, size(inp))
	edge_inp[:, :, 1] .= 1.0
	edge_inp[:, :, 2:end] = inp[:, :, 1:end - 1]
	output = m.edge_level(edge_inp)
	return output
end


function Model(input_size, node_embedding_size, edge_embedding_size, node_hidden_size, node_output_size) 
	Model(GRUBlock(input_size, node_embedding_size, node_hidden_size; output_size=node_output_size),
		GRUBlock(1, edge_embedding_size, node_output_size; output_size=1))
end;
