import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.io import wavfile

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]
        return result


def _conv_stack(dilations, in_channels, out_channels, kernel_size):
    """
    Create stack of dilated convolutional layers, outlined in WaveNet paper:
    https://arxiv.org/pdf/1609.03499.pdf
    """
    return nn.ModuleList(
        [
            CausalConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                dilation=d,
                kernel_size=kernel_size,
            )
            for i, d in enumerate(dilations)
        ]
    )


class WaveNet(nn.Module):
    def __init__(self, num_channels, dilation_depth, num_repeat, kernel_size=2):
        super(WaveNet, self).__init__()
        dilations = [2 ** d for d in range(dilation_depth)] * num_repeat
        internal_channels = int(num_channels * 2)
        self.hidden = _conv_stack(dilations, num_channels, internal_channels, kernel_size)
        self.residuals = _conv_stack(dilations, num_channels, num_channels, 1)
        self.input_layer = CausalConv1d(
            in_channels=1,
            out_channels=num_channels,
            kernel_size=1,
        )

        self.linear_mix = nn.Conv1d(
            in_channels=num_channels * dilation_depth * num_repeat,
            out_channels=1,
            kernel_size=1,
        )
        self.num_channels = num_channels

    def forward(self, x):
        out = x
        skips = []
        out = self.input_layer(out)

        for hidden, residual in zip(self.hidden, self.residuals):
            x = out
            out_hidden = hidden(x)

            # gated activation
            #   split (32,16,3) into two (16,16,3) for tanh and sigm calculations
            out_hidden_split = torch.split(out_hidden, self.num_channels, dim=1)
            out = torch.tanh(out_hidden_split[0]) * torch.sigmoid(out_hidden_split[1])

            skips.append(out)

            out = residual(out)
            out = out + x[:, :, -out.size(2) :]

        # modified "postprocess" step:
        out = torch.cat([s[:, :, -out.size(2) :] for s in skips], dim=1)
        out = self.linear_mix(out)
        return out


def error_to_signal(y, y_pred):
    """
    Error to signal ratio with pre-emphasis filter:
    https://www.mdpi.com/2076-3417/10/3/766/htm
    """
    y, y_pred = pre_emphasis_filter(y), pre_emphasis_filter(y_pred)
    return (y - y_pred).pow(2).sum(dim=2) / (y.pow(2).sum(dim=2) + 1e-10)


def pre_emphasis_filter(x, coeff=0.95):
    return torch.cat((x[:, :, 0:1], x[:, :, 1:] - coeff * x[:, :, :-1]), dim=2)


class SatNet(pl.LightningModule):
    def __init__(self, hparams):
        super(SatNet, self).__init__()
        self.wavenet = WaveNet(
            num_channels=hparams["num_channels"],
            dilation_depth=hparams["dilation_depth"],
            num_repeat=hparams["num_repeat"],
            kernel_size=hparams["kernel_size"],
        )
        self.hparams = hparams
        self.test_ds = TensorDataset()

    def prepare_data(self):

        createTensorDataset = lambda x, y: TensorDataset(torch.from_numpy(x).unsqueeze(1), torch.from_numpy(y).unsqueeze(1))
        
        inRate, inData = wavfile.read(hparams["in_file"])
        outRate, outData = wavfile.read(hparams["out_file"])
        sampleTime = 0.1
        sampleSize = int(inRate * sampleTime)
        length = len(inData) - len(inData) % sampleSize

        #Each row in this table represents the waveform samples seen in 0.1 seconds (4410 samples)
        x = inData[:length].reshape((-1, sampleSize)).astype(np.float32)
        y = outData[:length].reshape((-1, sampleSize)).astype(np.float32)

        splitLocA = int(len(x) * 0.6)
        splitLocB = int(len(x) * 0.8)

        X_train, X_valid, X_test = np.split(x, [splitLocA, splitLocB])
        y_train, y_valid, y_test = np.split(y, [splitLocA, splitLocB])

        self.train_ds = createTensorDataset(X_train, y_train)
        self.valid_ds = createTensorDataset(X_valid, y_valid)
        self.test_ds = createTensorDataset(X_test, y_test)

    def configure_optimizers(self):
        return torch.optim.Adam(self.wavenet.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.hparams.batch_size, num_workers=4)

    def forward(self, x):
        return self.wavenet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = error_to_signal(y[:, :, -y_pred.size(2) :], y_pred).mean()
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = error_to_signal(y[:, :, -y_pred.size(2) :], y_pred).mean()
        return loss

    def validation_epoch_end(self, outs):
        lossArray = []
        for tensor in outs:
            lossArray.append(tensor.item())
        avg_loss = (np.asarray(lossArray)).mean()
        self.log("avg_val_loss", avg_loss)

hparams = {
    "in_file": "data\\y_input_data.wav",
    "out_file": "data\\x_input_data.wav",
    "num_channels": 12,
    "dilation_depth": 10,
    "num_repeat": 1,
    "kernel_size": 3,
    "learning_rate": 3e-3,
    "batch_size": 64
}

model = SatNet.load_from_checkpoint("models\Tube Amp\sample-mnist-epoch=1495-avg_val_loss=0.0330.ckpt")
print (model)