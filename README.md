# GraphRNN

GraphRNN is a Julia re-implementation of the [GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models](https://arxiv.org/pdf/1802.08773.pdf) paper. 

## Introduction

## License
GraphRNN is released under the [GNU General Public License](LICENSE).

## Overview
The project has the following files:
* data_helper.jl (support file): handling encode/decode adjacency matrix, BFS ordering, transforming list of adjacency matrix to training format.
* graph_generator.jl (support file): generating different types of graph and other utilities such as getting the biggest component.
* graph_visualization.jl (support file): handling graph visualization.
* tsb_logger.jl (support file): handling creating TensorBoard logger, used in train.jl.
* configs/test.yaml: containing configuration parameters, can be directly modified based on the need of user.

* data.jl (main file): creating/loading data sets for different type of graphs.
* model.jl (main file): containing GraphRNN model definition.
* train.jl (main file): containing the main function which handles loading the data set, training, and inferencing. 

## Installation

Our project use the following packages: 

For data and graph generating files: FileIO, Flux, LightGraphs, LinearAlgebra, Random. 
```
# install packages
using Pkg
Pkg.add("FileIO")
Pkg.add("Flux")
Pkg.add("LightGraphs")
Pkg.add("LinearAlgebra")
Pkg.add("Random")

# include packages
using FileIO, Flux, LightGraphs, LinearAlgebra, Random
```
* For model and training: BSON, CUDA, Dates, Flux, LinearAlgebra, ProgressMeter, Statistics, TensorBoardLogger, Torch, YAML, Zygote. 
```
# installing packages
using Pkg
Pkg.add("BSON")
Pkg.add("CUDA")
Pkg.add("Dates")
Pkg.add("Flux")
Pkg.add("LinearAlgebra")
Pkg.add("ProgressMeter")
Pkg.add("Statistics")
Pkg.add("TensorBoardLogger")
Pkg.add("Torch")
Pkg.add("YAML")
Pkg.add("Zygote")

# include packages
using BSON: @save, @load
using Statistics: mean
using Zygote: @adjoint, @showgrad, @nograd
using Torch:torch
using CUDA, Dates, Flux, LinearAlgebra, ProgressMeter, TensorBoardLogger, YAML
```
* For graph visualization: Compose, Cairo, Fontconfig, GraphPlot, GraphRecipes, Lightgraphs, Plots. 
```
# installing packages
using Pkg
Pkg.add("Compose")
Pkg.add("Cairo")
Pkg.add("Fontconfig")
Pkg.add("GraphPlot")
Pkg.add("GraphRecipes")
Pkg.add("LightGraphs")
Pkg.add("Plots")

import Cairo
import Fontconfig
using LightGraphs, GraphRecipes, Plots, GraphPlot, Compose
```
Since the model and training also use files that involves data processing, it is recommended that you include packages for data before using model and training files.

## Generate Dataset

Our project supports creating data set for 4 types of graphs: SBM model, Complete Bipartite, Ladder, and Grid. To create a single graph, you can use function in [graph_generator.jl](https://github.com/lequytra/GraphRNN/blob/master/graph_generator.jl). To create a new data set for a specific type of graphs, you can use function in [data.jl](https://github.com/lequytra/GraphRNN/blob/master/data.jl). There will be 2 files being generated, the training data and the metadata of those training data. User can load these files using two load functions in [data.jl](https://github.com/lequytra/GraphRNN/blob/master/data.jl). To visualize a graph, you can use function in [graph_visualization.jl](https://github.com/lequytra/GraphRNN/blob/master/graph_visualization.jl). 

## Train

## Inference

