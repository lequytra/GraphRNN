# GraphRNN

GraphRNN is a Julia re-implementation of the [GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models](https://arxiv.org/pdf/1802.08773.pdf) paper. 

## Introduction

## License
GraphRNN is released under the [GNU General Public License](LICENSE).

## Overview
The project has the following files:
* [data_helper.jl](https://github.com/lequytra/GraphRNN/blob/master/data_helpers.jl) (support file): handling encode/decode adjacency matrix, BFS ordering, transforming list of adjacency matrix to training format.
* [graph_generator.jl](https://github.com/lequytra/GraphRNN/blob/master/graph_generator.jl) (support file): generating different types of graph and other utilities such as getting the biggest component.
  * include [data_helper.jl](https://github.com/lequytra/GraphRNN/blob/master/data_helpers.jl)
* [graph_visualization.jl](https://github.com/lequytra/GraphRNN/blob/master/graph_visualization.jl) (support file): handling graph visualization.
* [tsb_logger.jl](https://github.com/lequytra/GraphRNN/blob/master/tsb_logger.jl) (support file): handling creating TensorBoard logger, used in train.jl.
* [configs/test.yaml](https://github.com/lequytra/GraphRNN/blob/master/configs/test.yaml): containing configuration parameters, can be directly modified based on the need of user.  
  
* [data.jl](https://github.com/lequytra/GraphRNN/blob/master/data.jl) (main file): creating/loading data sets for different type of graphs.
  * include [graph_generator.jl](https://github.com/lequytra/GraphRNN/blob/master/graph_generator.jl)
* [model.jl](https://github.com/lequytra/GraphRNN/blob/master/model.jl) (main file): containing GraphRNN model definition.
* [train.jl](https://github.com/lequytra/GraphRNN/blob/master/train.jl) (main file): containing the main function which handles loading the data set, training, and inferencing. 
  * include [data_helper.jl](https://github.com/lequytra/GraphRNN/blob/master/data_helpers.jl), [graph_visualization.jl](https://github.com/lequytra/GraphRNN/blob/master/graph_visualization.jl), [tsb_logger.jl](https://github.com/lequytra/GraphRNN/blob/master/tsb_logger.jl), [data.jl](https://github.com/lequytra/GraphRNN/blob/master/data.jl), [model.jl](https://github.com/lequytra/GraphRNN/blob/master/model.jl).

## Installation

To install dependencies, please run `julia packages.jl`.

## Generate Dataset

Our project supports creating data set for 4 types of graphs: SBM model, Complete Bipartite, Ladder, and Grid. To create a single graph, you can use function in [graph_generator.jl](https://github.com/lequytra/GraphRNN/blob/master/graph_generator.jl). To create a new data set for a specific type of graphs, you can use function in [data.jl](https://github.com/lequytra/GraphRNN/blob/master/data.jl). There will be 2 files being generated, the training data and the metadata of those training data. User can load these files using two load functions in [data.jl](https://github.com/lequytra/GraphRNN/blob/master/data.jl). To visualize a graph, you can use function in [graph_visualization.jl](https://github.com/lequytra/GraphRNN/blob/master/graph_visualization.jl).  


## Train & Inference

For more detail, please refer to our [interactive tutorial](tutorial.ipynb)

## References
Jiaxuan  You  et  al.  “GraphRNN:  A  Deep  Generative  Model  for  Graphs”.  In:  (Feb.2018).url:https://arxiv.org/pdf/1802.08773.pdf.




