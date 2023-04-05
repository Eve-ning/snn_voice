
# trainer=[fast_dev, dev, prod]
trainer=fast_dev
# device=[cpu, gpu]
device=gpu
# test_config=[False, True]
# If True, doesn't run the fit, only prints the configs
# Useful to check if the multirun works as expected.
test_config=False
batch_size=128

# Runs for one experiment
run-one:
	python3 -m\
	snn_voice.scripts.experiment\
	 model=piczak_cnn\
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

run-all-cnn:
	#!/bin/bash
	python3 -m\
	snn_voice.scripts.experiment\
	 -m "model=glob(*_nn*,exclude=[model,model_snn,hjh_scnn])"\
	 trainer.accelerator=${device}\
	 trainer=${trainer}\
	 +test_config=${test_config}\

run-all-snn:
	#!/bin/bash
	python3 -m\
	snn_voice.scripts.experiment\
	 -m "model=glob(*_snn*,exclude=[model,model_snn,hjh_scnn])"\
	 trainer.accelerator=${device}\
	 trainer=${trainer}\
	 +test_config=${test_config}\
	 ++model.snn.n_steps=1,5,15,30\
	 ++model.snn.learn_beta=False,True
