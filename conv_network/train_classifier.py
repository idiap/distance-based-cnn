# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
# Defining classifier model wrapper for training

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
import importlib


class ClassifierTrain(nn.Module):
    """
    Defining the classification module for training (has saving and loading methods (for the checkpoint))
    Based on the model configuration, initializes the model (selecting the model and its architecture)
    """

    def __init__(self, freqlen, seqlen, config_file, **kwargs):
        """initializing the RepresentationPretrain module
        Args:
            freqlen (int): freq dim (number of features)
            seqlen (int): temporal dim (number of frames)
            config_file (dict): configuration for network and training params
        """
        super(ClassifierTrain, self).__init__()
        self.Network_config = config_file
        print("\n[NN Classifier] - Initializing model...")
        self.selected_model = self.Network_config["SelectedNetwork"]
        model_config = self.Network_config[self.selected_model]
        NNmodel = getattr(
            importlib.import_module("conv_network.cnn_model"), self.selected_model
        )
        # print(Dstream, model_config)
        self.model = NNmodel(freqlen, seqlen, **model_config)
        print(
            "ClassifierTrain: "
            + self.selected_model
            + " - Number of parameters: "
            + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
            + "\n\n"
        )

    # Interface
    def load_model(self, all_states):
        """loading model from saved states
        Args:
            all_states (dict): dictionary of the states
        """
        self.model.FrontEnd.load_state_dict(all_states["FrontEnd"])
        self.model.Conv.load_state_dict(all_states["Conv"])
        self.model.MLP.load_state_dict(all_states["MLP"])

    # Interface
    def add_state_to_save(self, all_states):
        """Saving the states to the "all_states"
        Args:
            all_states (dict): dictionary of the states
        """
        all_states["FrontEnd"] = self.model.FrontEnd.state_dict()
        all_states["Conv"] = self.model.Conv.state_dict()
        all_states["MLP"] = self.model.MLP.state_dict()
        all_states["NetworkType"] = self.selected_model
        all_states["NetConfig"] = self.Network_config
        return all_states

    def forward(self, data, **kwargs):
        """
        Args:
            data:
                (tensor): [B X D X T] input features
        Return:
            (tensor): [B X num of classes] predicted output classes
        """
        return self.model(data)


if __name__ == "__main__":
    # ------------------------------ Sanity check ----------------------------- #
    from pathlib import Path 
    import sys, os
    file = Path(__file__).resolve()
    parent, root, subroot = file.parent, file.parents[1], file.parents[2]
    sys.path.append(str(subroot))
    sys.path.append(str(root))
    os.chdir(root)
    # ---------------------------------------------------------------------------- #
    import matplotlib.pyplot as plt
    from audio.audio_utils import get_config_args, create_transform
    from audio.audio_dataset import get_dataset
    from torch.utils.data import DataLoader
    import numpy as np
    import random

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
            f"preprocess/dummy_database/folds/test_fold1_{feat_type}_dist_offline.csv"
        )
    elif feat_config["Pairwise-Reps"]:
        file_path_off = (
            f"preprocess/dummy_database/folds/test_fold1_resized_words_{feat_type}_offline.csv"
        )
    else:
        file_path_off = f"preprocess/dummy_database/folds/test_fold1_{feat_type}_offline.csv"

    dataset_on = get_dataset(dataloading_config, file_path_on, feat_config)
    dataset_off = get_dataset(dataloading_config_off, file_path_off, feat_config)

    freqdim, seqlen = dataset_off.getDimension()

    model = ClassifierTrain(freqdim, seqlen, dataloading_config)
    seed_torch()
    data_loader_off = torch.utils.data.DataLoader(
        dataset_off, batch_size=10, shuffle=True, num_workers=2
    )
    for batch_idx, data_batch in enumerate(data_loader_off):
        print("batch index: ", batch_idx)
        data, ID, targets = data_batch
        out1 = model(data)
        print("output size: ", out1.shape, ID, targets)
        break
    seed_torch()
    data_loader_on = torch.utils.data.DataLoader(
        dataset_on, batch_size=10, shuffle=True, num_workers=2
    )
    for batch_idx, data_batch in enumerate(data_loader_on):
        print("batch index: ", batch_idx)
        data, ID, targets = data_batch
        out2 = model(data)
        print("output size: ", out2.shape, ID, targets)
        break
    print(torch.norm(out1 - out2))
