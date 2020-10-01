using Flux

@net type GRUBlock
	has_input::Bool 
	has_output::Bool 
	linear::Dense
	rnn::Flux.Recur
	output_module
	function (inp)
		if has_input
			inp = linear(inp)
		end
		inp = rnn(inp)
		if has_output
			inp = output_module(inp)
		end
		return inp 
	end
end

@net type OutputModule 
	linear
	linear
end

@net type Model 
	graph_level
	edge_level

	function (inp)
		inp2 = graph_level(inp)
		hidden(edge_level) = inp2
		edge_inp = fill(0.0, size(inp))
		edge_inp[:, :, 1] .= 1.0
		edge_inp [:, :, 2:end] = inp[:, :, 1:end - 1]
		output = edge_level(edge_inp)
		return output
	end
end

OutputModule(hidden_size, embedding_size, output_size) = 
	OutputModule(Flux.Dense(hidden_size, embedding_size, relu), Flux.Dense(embedding_size, output_size, sigmoid))

GRUBlock(input_size, embedding_size, hidden_size, has_input=true, has_output=true, output_size=nothing) =
	GRUBlock(has_input=has_input, 
			has_output=has_output, 
			linear=Flux.Dense(input_size, embedding_size), 
			rnn=GRU(embedding_size, hidden_size), 
			output_module=OutputModule(hidden_size, embedding_size, output_size))

hidden(m::GRUBlock) = hidden(m.rnn.state)


function Model(input_size, node_embedding_size, edge_embedding_size, node_hidden_size, edge_hidden_size, node_output_size) 
	Model(GRUBlock(input_size, node_embedding_size, node_hidden_size, has_input=true, has_output=true, output_size=hidden_output_size),
		GRUBlock(1, edge_embedding_size, edge_hidden_size, has_input=true, has_output=true, output_size=1))
end
