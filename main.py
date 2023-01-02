from matplotlib import pyplot as plt

from snn_voice.datamodule.speech_command_datamodule import SpeechCommandsDataModule
from snn_voice.model.mx.m5_cnn import M5CNN
from snn_voice.model.mx.m5_trsafs import M5TRSAFS
from snn_voice.utils.plot_cnn import PlotCNN
from snn_voice.utils.plot_snn import PlotSNN

# %%

dm = SpeechCommandsDataModule(batch_size=16)
dm.prepare_data()
dm.setup()
val_samples = dm.load_samples('validation')
sample = val_samples['backward']
# %%
model_cnn = M5CNN()
h_cnn = PlotCNN(model_cnn, )
h_cnn.plot(sample)
plt.show()
# %%
model_snn = M5TRSAFS(n_steps=15)
# %%
h_snn = PlotSNN(model_snn, padding_f=0.5)
h_snn.plot(sample.repeat(15, 1, 1, 1), contrast=2,
           resize=(50 * 15, 500))
plt.gcf().set_figheight(13 * 4)
plt.gcf().set_figwidth(5)
plt.tight_layout()
plt.show()
# %%
