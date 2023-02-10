
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
	python3 -m\
	snn_voice.scripts.experiment\
	 model=m5_snn\
	 trainer.accelerator=${device}\
	 trainer=${trainer}\
	 +test_config=${test_config}


# Runs for two experiments
run-two:
	#!/bin/bash
	python3 -m\
	snn_voice.scripts.experiment\
     -m\
	 model=m5_snn,m5_cnn\
	 trainer.accelerator=${device}\
	 trainer=${trainer}\
	 +test_config=${test_config}

run-all:
	#!/bin/bash
	python3 -m\
	snn_voice.scripts.experiment\
	 -m "model=glob(*,exclude=[model,model_snn,hjh_scnn])"\
	 trainer.accelerator=${device}\
	 trainer=${trainer}\
	 +test_config=${test_config}\
	 ++model.snn.n_steps=1,2,5,15,40\
	 ++model.snn.learn_beta=False,True
