run-experiment:
	python -m snn_voice.scripts.experiment -m model=glob(*,exclude=[model,model_snn]) trainer.accelerator=cpu