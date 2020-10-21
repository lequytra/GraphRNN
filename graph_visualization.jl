using LightGraphs

using GraphRecipes, Plots

import Cairo
import Fontconfig
using GraphPlot, Compose

# HELPER FUNCTIONS *************************************************************
#=
Return a layout for bipartite graph. This is used by the drawing function.

Params:
    g: graph: must be a bipartite graph.

Returns:
    x, y: 2 Arrays{Int64, 1}. Position x and y of each nodes.
=#
function bipartite_layout(g)
    bipart_map = bipartite_map(g)

    v_gap = 0.5
    h_gap = 0.5

    locs_x = []
    locs_y = []

    first_partition_counter = 0
    second_partition_counter = 0

    for i in 1:nv(g)
        if bipart_map[i] == 1
            append!(locs_x, 0)
            append!(locs_y, first_partition_counter * h_gap)
            first_partition_counter += 1
        else
            append!(locs_x, v_gap)
            append!(locs_y, second_partition_counter * h_gap)
            second_partition_counter += 1
        end
    end

    return convert(Array{Float64,1}, locs_x), convert(Array{Float64,1}, locs_y)
end




# VIZUALIZATION FUNCTIONS ******************************************************
#=
visualize a ladder graph

Params:
    g: graph.
    file_name: String: name of the plotting image file. If == nothing then no file
        is created.
=#
function ladder_viz(g; file_name=nothing)
    temp = graphplot(g, curves=false)

    if file_name!= nothing
        png(file_name)
    end
    return temp
end



#=
visualize a grid graph

Params:
    g: graph.
    file_name: String: name of the plotting image file. If == nothing then no file
        is created.
=#
function grid_viz(g; file_name=nothing)
    temp = graphplot(g, curves=false)

    if file_name != nothing
        png(file_name)
    end
    return temp
end



#=
visualize a complete bipartite graph

Params:
    g: graph.
    file_name: String: name of the plotting image file. If == nothing then no file
        is created.

    other optional Params:
        node_fill_color: color of node.
        node_stroke_color: color of node's stroke.
        edge_stroke_color: color of edge.
=#
function complete_bipartite_viz(g, node_fill_color=colorant"blue", node_stroke_color=colorant"black", edge_stroke_color=colorant"black"; file_name=nothing)
    locs_x, locs_y = bipartite_layout(g);
    # plot
    temp = gplot(g,
        layout=bipartite_layout,
        nodefillc=node_fill_color,
        nodestrokec=node_stroke_color,
        nodestrokelw=0.5,
        edgestrokec=edge_stroke_color)

    if file_name != nothing
        draw(PNG(file_name, 16cm, 16cm), temp)
    end
    return temp
end


#=
visualize a sbm graph

Params:
    g: graph.
    file_name: String: name of the plotting image file. If == nothing then no file
        is created.

    other optional Params:
        node_fill_color: color of node.
        node_stroke_color: color of node's stroke.
        edge_stroke_color: color of edge.
=#
function sbm_viz(g, node_fill_color=colorant"blue", node_stroke_color=colorant"black", edge_stroke_color=colorant"black"; file_name=nothing)
  # plot
    temp = gplot(g,
        nodefillc=node_fill_color,
        nodestrokec=node_stroke_color,
        nodestrokelw=0.5,
        edgestrokec=edge_stroke_color)

    if file_name != nothing
        draw(PNG(file_name, 16cm, 16cm), temp)
    end
    return temp
end



#=
Draw given graph g of given type (provided via type_name). If file name is not
nothing, the output will be write to file with file_name.

Params:
    g: graph.
    type_name: String: specifies the type of graph. It can be "sbm",
        "complete_bipartite", "ladder", or "grid".
    file_name: String: name of the plotting image file. If == nothing then no file
        is created.
=#
function graph_viz(g, type_name, file_name=nothing)
    if type_name == "sbm"
        g = sbm_viz(g, file_name=file_name)

    elseif type_name == "complete_bipartite"
        g = complete_bipartite_viz(g, file_name=file_name)

    elseif type_name == "ladder"
        g = ladder_viz(g, file_name=file_name)

    elseif type_name == "grid"
        g = grid_viz(g,file_name=file_name)

    else
        throw(ArgumentError("type_name must be: \"sbm\", \"complete_bipartite\", \"ladder\", or \"grid\""))
    end
    return g
end

