run-fast-dev-1:
	python -m\
	snn_voice.scripts.experiment\
	 model=m5_snn\
	 trainer.accelerator=cpu\
	 +sample=True

run-fast-dev:
	python -m\
	snn_voice.scripts.experiment\
	 -m model=glob(*,exclude=[model,model_snn])\
	 trainer.accelerator=cpu\
	 +sample=True

run-dev:
	python -m\
	snn_voice.scripts.experiment\
	 -m model=glob(*,exclude=[model,model_snn])\
	 trainer=dev\
	 trainer.accelerator=gpu

run-prod:
	python -m\
	snn_voice.scripts.experiment\
	 -m model=glob(*,exclude=[model,model_snn])\
	 trainer=prod\
	 trainer.accelerator=gpu
