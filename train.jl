using Flux
using BSON: @save


function train(model, epochs, dataloader, resume_from=1, checkpoints_folder="")
	for i in resume_from:epochs
		loss(x, y) = Flux.Losses.logitcrossentropy(model(x), y)
		opt = Flux.Optimise.RMSProp()
		pms = flux.params(model)
		Flux.train!(loss, pms, dataloader, opt)
		@save "$(checkpoints_folder)/model-$(now()).bson" model opt
	end
end


function main()
	#TODO fill in main here including loading data, train, test etc. 
end
