using Pkg

dependencies = [
    "FileIO",
    "Random",
    "BSON",
    "Dates",
    "Flux",
    "CUDA", 
    "LinearAlgebra",
    "ProgressMeter",
    "Statistics",
    "TensorBoardLogger",
    "Torch",
    "YAML",
    "Zygote",
    "Compose",
    "Cairo",
    "Fontconfig",
    "GraphPlot",
    "GraphRecipes",
    "LightGraphs",
    "Plots"
]

Pkg.add(dependencies)
