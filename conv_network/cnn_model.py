# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
# Defining CNN models

# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Parvaneh Janbakhshi <parvaneh.janbakhshi@idiap.ch>

# This file is part of distance-based-cnn
#
# distance-based-cnn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# distance-based-cnn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with distance-based-cnn. If not, see <http://www.gnu.org/licenses/>.
"""*********************************************************************************************"""

import torch
import torch.nn as nn
import numpy as np


class NonlinRem(nn.Module):
    """Removing nonlinearity module
    Inheritance:
        nn.Module:
    """

    def __init__(self):
        super(NonlinRem, self).__init__()

    def forward(self, data):
        """passing input as output
        Args:
            data (tensor): input
        Returns:
            (tensor): passing input data as output
        """
        return data


ACT2FN = {"relu": nn.ReLU(), "leaky-relu": nn.LeakyReLU(), " ": NonlinRem()}


class CNN1d(nn.Module):
    def __init__(
        self,
        freqlen,
        seqlen,
        output_size,
        hidden_units,
        kernelsize,
        poolingsize,
        convchannels,
        nonlinearity,
        dropout_prob,
        batchnorm,
    ):
        """initializing the parameters of the CNN1d model (Baseline 1)
        Args:
            freqlen (int): freq dim (number of features)
            seqlen (int): temporal dim (number of frames)
            output_size (int): output size (number of classes)
            hidden_units (list): a list indicating the number of hidden units of
            linear layers between flattened CNN outputs and the final output. if list
            is empty means we have
            input > output (1 layer only)
            kernelsize((int or tuple)): kernel size in conv layers.
            poolingsize (int or tuple): pooling size in conv layers.
            convchannels (list): channels of convs, e.g.,  [1, outchannel1, outchannel2, ...]
            also indicates the number of convs in CNN part.
            nonlinearity (str): str indicating nonlinearity in between layers
            dropout_prob (float): the dropout rate.
            batchnorm (bool): If True batch norm is applied for conv layers
        """
        super(CNN1d, self).__init__()
        self.output_dim = output_size
        self.hidden_units = hidden_units
        self.nonlinearity = ACT2FN[nonlinearity]
        self.dropout = nn.Dropout(dropout_prob)
        self.maxpool = nn.MaxPool2d(1, poolingsize)  # 1d pooling across time
        self.seqlen = seqlen
        self.freqlen = freqlen
        self.all_layers_uints = self.hidden_units + [self.output_dim]
        # for one-channel spectrograms
        self.convchannels = [1] + convchannels
        self.psize = poolingsize
        self.ksize = kernelsize
        conv_layers = []
        frontend_layers = []
        tsize = seqlen
        # firs convolution layer applies filter-bank across frequency (F X 1),
        # and after that all are across time (1 X K)
        # first front-end
        for layers_num in range(len(self.convchannels) - 1):
            if layers_num == 0:
                conv = nn.Conv1d(
                    self.convchannels[layers_num],
                    self.convchannels[layers_num + 1],
                    kernel_size=(freqlen, 1),
                )
            else:
                conv = nn.Conv1d(
                    self.convchannels[layers_num],
                    self.convchannels[layers_num + 1],
                    kernel_size=(1, kernelsize),
                )
                tsize = int(np.floor(tsize - kernelsize + 1))
            tsize = int(
                np.floor(((tsize - (poolingsize - 1) - 1) / (poolingsize)) + 1)
            )  # after pooling
            conv_bn = nn.BatchNorm2d(self.convchannels[layers_num + 1])

            if layers_num == 0:
                frontend_layers = [conv, self.maxpool, self.nonlinearity]
            else:
                conv_layers += (
                    [conv, self.maxpool, conv_bn, self.nonlinearity]
                    if batchnorm
                    else [conv, self.maxpool, self.nonlinearity]
                )

        conv_layers = conv_layers + [self.dropout]  # add dropout before MLP

        self.FrontEnd = nn.Sequential(*frontend_layers)
        self.Conv = nn.Sequential(*conv_layers)

        self.lsize = tsize * self.convchannels[layers_num + 1]

        linear_layers = []
        for layers_num in range(len(self.all_layers_uints)):
            in_dim = (
                self.lsize if layers_num == 0 else self.all_layers_uints[layers_num - 1]
            )
            out_dim = self.all_layers_uints[layers_num]
            if layers_num != len(self.all_layers_uints) - 1:
                linear_layers += [nn.Linear(in_dim, out_dim), self.nonlinearity]
            else:
                linear_layers += [nn.Linear(in_dim, out_dim)]

        self.MLP = nn.Sequential(*linear_layers)

    def forward(self, input):
        """
        Input:
            input (tensor): [B X D X T] a 3d-tensor representing the input features.
        Return:
            (tensor): predicted output [B X H]
        """
        input = torch.unsqueeze(input, dim=1)  # Add 1-channel > B X 1 X D X T
        fe_out = self.FrontEnd(input)
        cnn_encout = self.Conv(fe_out)
        output = self.MLP(cnn_encout.view(-1, self.lsize))
        return output


class CNN2d(nn.Module):
    def __init__(
        self,
        freqlen,
        seqlen,
        output_size,
        hidden_units,
        kernelsize,
        poolingsize,
        convchannels,
        nonlinearity,
        dropout_prob,
        batchnorm,
    ):
        """initializing the parameters of the CNN2d model (Baseline 2)
        Args:
            freqlen (int): freq dim (number of features)
            seqlen (int): temporal dim (number of frames)
            output_size (int): output size (number of classes)
            hidden_units (list): a list indicating the number of hidden units of
            linear layers between flattened CNN outputs and  the final output. if list
            is empty means we have
            input > output (1 layer only)
            kernelsize((int or tuple)): kernel size in conv layers.
            poolingsize (int or tuple): pooling size in conv layers.
            convchannels (list): channels of convs, e.g., [1, outchannel1, outchannel2, ...]
            also indicates the number of convs in CNN part.
            nonlinearity (str): str indicating nonlinearity in between layers
            dropout_prob (float): the dropout rate.
            batchnorm (bool): If True batch norm is applied for conv layers
        """
        super(CNN2d, self).__init__()
        self.output_dim = output_size
        self.hidden_units = hidden_units
        self.nonlinearity = ACT2FN[nonlinearity]
        self.dropout = nn.Dropout(dropout_prob)
        self.maxpool = nn.MaxPool2d(poolingsize)
        self.seqlen = seqlen
        self.freqlen = freqlen
        self.all_layers_uints = self.hidden_units + [self.output_dim]
        # for one-channel spectrograms
        self.convchannels = [1] + convchannels
        self.psize = poolingsize
        self.ksize = kernelsize
        conv_layers = []
        tsize = seqlen
        fsize = freqlen
        for layers_num in range(len(self.convchannels) - 1):
            conv = nn.Conv2d(
                self.convchannels[layers_num],
                self.convchannels[layers_num + 1],
                kernel_size=kernelsize,
            )

            tsize = int(np.floor(tsize - kernelsize + 1))
            tsize = int(
                np.floor(((tsize - (poolingsize - 1) - 1) / (poolingsize)) + 1)
            )  # after pooling

            fsize = int(np.floor(fsize - kernelsize + 1))
            fsize = int(np.floor(((fsize - (poolingsize - 1) - 1) / (poolingsize)) + 1))

            conv_bn = nn.BatchNorm2d(self.convchannels[layers_num + 1])

            conv_layers += (
                [conv, self.maxpool, conv_bn, self.nonlinearity]
                if batchnorm
                else [conv, self.maxpool, self.nonlinearity]
            )

        conv_layers = conv_layers + [self.dropout]  # add dropout before MLP

        self.FrontEnd = nn.Linear(1, 1)  # just a dummy module
        self.Conv = nn.Sequential(*conv_layers)

        self.lsize = tsize * fsize * self.convchannels[layers_num + 1]

        linear_layers = []
        for layers_num in range(len(self.all_layers_uints)):
            in_dim = (
                self.lsize if layers_num == 0 else self.all_layers_uints[layers_num - 1]
            )
            out_dim = self.all_layers_uints[layers_num]
            if layers_num != len(self.all_layers_uints) - 1:
                linear_layers += [nn.Linear(in_dim, out_dim), self.nonlinearity]
            else:
                linear_layers += [nn.Linear(in_dim, out_dim)]

        self.MLP = nn.Sequential(*linear_layers)

    def forward(self, input):
        """
        Input:
            input (tensor): [B X D X T] a 3d-tensor representing the input features.
        Return:
            (tensor): predicted output [B X H]
        """
        input = torch.unsqueeze(input, dim=1)  # Add 1-channel > B X 1
        cnn_encout = self.Conv(input)
        output = self.MLP(cnn_encout.view(-1, self.lsize))
        return output


#%%
class CNNDist(nn.Module):
    def __init__(
        self,
        freqlen,
        seqlen,
        output_size,
        hidden_units,
        kernelsize,
        poolingsize,
        convchannels,
        nonlinearity,
        dropout_prob,
        batchnorm,
    ):
        """initializing the parameters of the CNNDist model (e2e distance-based CNN)
        Args:
            freqlen (int): freq dim (number of features)
            seqlen (int): temporal dim (number of frames)
            output_size (int): output size (number of classes)
            hidden_units (list): a list indicating the number of hidden units of
            linear layers between flattened CNN outputs and output. if list
            is empty means we have
            input > output (1 layer only)
            kernelsize((int or tuple)): kernel size in conv layers.
            poolingsize (int or tuple): pooling size in conv layers.
            convchannels (list): channels of convs, first channel is a 1d convolution
            , therefore is used to control the number of extracted feature
            in front-end layer to extract features, After that we have e.g.,
            [1, outchannel2, outchannel3, ...] for 2d CNN on distance matrices which
            also indicates the number of convs in the CNN part.
            nonlinearity (str): str indicating nonlinearity in between layers
            dropout_prob (float): the dropout rate.
            batchnorm (bool): If True batch norm is applied for conv layers
        """
        super(CNNDist, self).__init__()
        self.output_dim = output_size
        self.hidden_units = hidden_units
        self.nonlinearity = ACT2FN[nonlinearity]
        self.dropout = nn.Dropout(dropout_prob)
        self.maxpool = nn.MaxPool2d(poolingsize)
        self.seqlen = seqlen
        self.freqlen = freqlen
        self.all_layers_uints = self.hidden_units + [self.output_dim]
        # for one-channel spectrograms
        self.convchannels = [1] + convchannels
        self.psize = poolingsize
        self.ksize = kernelsize
        conv_layers = []
        frontend_layers = []
        tsize = seqlen
        # firs convolution (front-end) layer applies filter-bank across frequency (F X 1)
        # and after computing distance matrices all convolutions are 2d

        conv = nn.Conv1d(
            self.convchannels[0], self.convchannels[1], kernel_size=(freqlen, 1)
        )
        fsize = tsize  # square dist matrices are created after this layer
        frontend_layers = [conv, self.nonlinearity]
        self.convchannels = [1] + convchannels[1:]

        for layers_num in range(len(self.convchannels) - 1):
            conv = nn.Conv2d(
                self.convchannels[layers_num],
                self.convchannels[layers_num + 1],
                kernel_size=kernelsize,
            )
            tsize = int(np.floor(tsize - kernelsize + 1))
            tsize = int(
                np.floor(((tsize - (poolingsize - 1) - 1) / (poolingsize)) + 1)
            )  # after pooling
            fsize = int(np.floor(fsize - kernelsize + 1))
            fsize = int(
                np.floor(((fsize - (poolingsize - 1) - 1) / (poolingsize)) + 1)
            )  # after pooling

            conv_bn = nn.BatchNorm2d(self.convchannels[layers_num + 1])

            conv_layers += (
                [conv, self.maxpool, conv_bn, self.nonlinearity]
                if batchnorm
                else [conv, self.maxpool, self.nonlinearity]
            )

        conv_layers = conv_layers + [self.dropout]  # add dropout before MLP

        self.FrontEnd = nn.Sequential(*frontend_layers)
        self.Conv = nn.Sequential(*conv_layers)

        self.lsize = tsize * fsize * self.convchannels[layers_num + 1]

        linear_layers = []
        for layers_num in range(len(self.all_layers_uints)):
            in_dim = (
                self.lsize if layers_num == 0 else self.all_layers_uints[layers_num - 1]
            )
            out_dim = self.all_layers_uints[layers_num]
            if layers_num != len(self.all_layers_uints) - 1:
                linear_layers += [nn.Linear(in_dim, out_dim), self.nonlinearity]
            else:
                linear_layers += [nn.Linear(in_dim, out_dim)]

        self.MLP = nn.Sequential(*linear_layers)

    @staticmethod
    def FroNorm(x, y):
        """Computing frame-level Euclidean distance matrix from two input tensors
        d_ij = |xi|^2− 2x_i^T x_j + |xj|^2
        Input:
            x (tensor): [B X C X 1 X T] a 4d-tensor representing the input features.
            y (tensor): [B X C X 1 X T] a 4d-tensor representing the input features.
        Return:
            (tensor): Distance matrix [B X 1 X T X T]
        """

        x = x.squeeze(2)  # [B X C X T]
        y = y.squeeze(2)  # [B X C X T]
        yNorm = (torch.norm(y, dim=1) ** 2).unsqueeze(1).repeat(1, y.shape[-1], 1)
        xNorm = (torch.norm(x, dim=1) ** 2).unsqueeze(2).repeat(1, 1, x.shape[-1])
        return (
            torch.clamp(
                (yNorm + xNorm - 2 * torch.matmul(x.transpose(-1, -2), y)).unsqueeze(1),
                0.0,
                np.inf,
            )
            + np.finfo(np.float).eps
        ) ** (1 / 2)

    def forward(self, input):
        """
        Input:
            input (list) containing:
                - [B X D X T] a 3d-tensor representing the test input features.
                - [B X D X T] a 3d-tensor representing the reference input features.
        Return:
            (tensor): predicted output [B X H]
        """
        test = torch.unsqueeze(input[0], dim=1)  # Add 1-channel > B X 1
        ref = torch.unsqueeze(input[1], dim=1)  # Add 1-channel > B X 1
        test_feat = self.FrontEnd(test)
        ref_feat = self.FrontEnd(ref)
        cnn_encout = self.Conv(self.FroNorm(test_feat, ref_feat))
        output = self.MLP(cnn_encout.view(-1, self.lsize))
        return output


if __name__ == "__main__":
    # Cheking...
    # ---------------------- changing path for local run------------------ #
    from pathlib import Path 
    import sys, os
    file = Path(__file__).resolve()
    parent, root, subroot = file.parent, file.parents[1], file.parents[2]
    sys.path.append(str(subroot))
    sys.path.append(str(root))
    os.chdir(root)
    # -------------------------------------------------------------------- #
    from torch.utils.data import DataLoader
    from audio.audio_utils import get_config_args, create_transform
    from audio.audio_dataset import get_dataset
    import matplotlib.pyplot as plt
    import numpy as np
    import random

    cpath = "config/NNtrain_config_online.yaml"
    dataloading_config = get_config_args(cpath)
    cpath_off = "config/NNtrain_config.yaml"
    dataloading_config_off = get_config_args(cpath_off)
    feat_path = "config/audio_config.yaml"
    feat_config = get_config_args(feat_path)
    feat_type = feat_config.get("feat_type")
    if (feat_config["Pairwise-Distance"]) | (feat_config["Pairwise-Reps"]):
        file_path_on = "preprocess/dummy_database/folds/test_fold1_isowords_online.csv"
    else:
        file_path_on = "preprocess/dummy_database/folds/test_fold1_online.csv"

    if feat_config["Pairwise-Distance"]:
        file_path_off = (
            f"preprocess/dummy_database/folds/test_fold1_{feat_type}_Dist_offline.csv"
        )
    elif feat_config["Pairwise-Reps"]:
        file_path_off = (
            f"preprocess/dummy_database/folds/test_fold1_resized_words_{feat_type}_offline.csv"
        )
    else:
        file_path_off = f"preprocess/dummy_database/folds/test_fold1_{feat_type}_offline.csv"

    dataset_on = get_dataset(dataloading_config, file_path_on, feat_config)
    dataset_off = get_dataset(dataloading_config_off, file_path_off, feat_config)

    freqdim, seqlen = dataset_on.getDimension()
    freqdim, seqlen = dataset_off.getDimension()

    transforms = create_transform(feat_config, 16000)

    def seed_torch(seed=0):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    # offline testing
    Network_config = dataloading_config
    selected_model = Network_config["SelectedNetwork"]
    model_config = Network_config[selected_model]

    seed = 0
    seed_torch(seed)
    model = eval(f"{selected_model}")(freqdim, seqlen, **model_config)
    data_loader_on = torch.utils.data.DataLoader(
        dataset_on, batch_size=10, shuffle=True, num_workers=0
    )
    data_loader_off = torch.utils.data.DataLoader(
        dataset_off, batch_size=10, shuffle=True, num_workers=0
    )

    for batch_idx, data_batch in enumerate(data_loader_off):
        print("batch index: ", batch_idx)
        data, ID, targets = data_batch
        try:
            print("input size: ", data.shape)
        except:
            print("input size: ", data[0].shape)
        output = model(data)
        print("output/latent size: ", output[0].shape, "\n")
        break
    for batch_idx, data_batch in enumerate(data_loader_on):
        print("batch index: ", batch_idx)
        data, ID, targets = data_batch
        try:
            print("input size: ", data.shape)
        except:
            print("input size: ", data[0].shape)
        output = model(data)
        print("output/latent size: ", output[0].shape, "\n")
        break
# %%
