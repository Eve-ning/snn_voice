 # trainer=[fast_dev, dev, prod]
trainer=fast_dev
# device=[cpu, gpu]
device=gpu
# test_config=[False, True]
# If True, doesn't run the fit, only prints the configs
# Useful to check if the multirun works as expected.
test_config=False

# Runs for one experiment
run-one:
	python -m\
	snn_voice.scripts.experiment\
	 model=m5_snn\
	 trainer.accelerator=${device}\
	 +test_config=${test_config}\
	 trainer=${trainer}

run-all:
	python -m\
	snn_voice.scripts.experiment\
	 -m model=glob(*,exclude=[model,model_snn,hjh_scnn])\
	 trainer=${trainer}\
	 trainer.accelerator=${device}\
	 ++model.snn.n_steps=1,2,5,15,40\
	 ++model.snn.learn_beta=False,True\
	 +test_config=${test_config}
