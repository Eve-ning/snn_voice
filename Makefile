trainer=fast_dev
device=gpu

run-one:
	python -m\
	snn_voice.scripts.experiment\
	 model=m5_snn\
	 trainer.accelerator=${device}\
	 +sample=True\
	 trainer=${trainer}

run-all:
	python -m\
	snn_voice.scripts.experiment\
	 -m model=glob(*,exclude=[model,model_snn,hjh_scnn])\
	 trainer=${trainer}\
	 trainer.accelerator=${device}\
	 model.snn.n_steps=1,2,5,15,40\
	 model.snn.learn_beta=False,True
