# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
# defining runner module for training/evaluating the classifier model

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

import os
import random
import torch
import torch.nn as nn
import numpy as np
import glob
import sys
from torch.utils.data import DataLoader
# sys.path.append('../')        # in case of local run
from audio.audio_dataset import get_dataset
from audio.audio_utils import get_config_args
from torch.autograd import Variable
from collections import defaultdict
import warnings

# from train_classifier import ClassifierTrain # in case of local run
from .train_classifier import ClassifierTrain


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


class NNRunner:
    """
    runner for NN training
    creating train/val/test data loaders, training/evaluation
    loops, checkpoint saving and loading (resume training)
    """

    def __init__(self, args):
        """Initializing the runner for training
        Args:
            args (argparse.Namespace): arguments for training
            including paths of config files, training params, fold, seed, ...
        """
        self.args = args
        self.config = self.args.config
        self.config_file = get_config_args(self.config)

        self.modelname = self.config_file["SelectedNetwork"]

        self.init_ckpt_path = (
            f"results/{self.args.expname}/{self.modelname.lower()}-{self.args.expname}"
            "/outputs/saved_model/"
        )

        self.b1_init_ckpt_path = (
            f"results/{self.args.expname}/cnn1d-{self.args.expname}"
            "/outputs/saved_model/"
        )
        self.b2_init_ckpt_path = (
            f"results/{self.args.expname}/cnn2d-{self.args.expname}"
            "/outputs/saved_model/"
        )

        os.makedirs(self.init_ckpt_path, exist_ok=True)
        # getting number of folds from pre-processed data and other
        # please check lack of overlap between test and train data for different
        # model training

        data_fold = len(
            glob.glob1(
                self.config_file["dataloader"]["data_path"],
                "test_fold*_isowords_online.csv",
            )
        )
        print(f"\n{data_fold} data folds are found for training.\n")
        if (self.modelname == "CNNDist") and self.args.preinit:
            b1model_num = len(glob.glob1(self.b1_init_ckpt_path, "NN_fold*.ckpt"))
            b2model_num = len(glob.glob1(self.b2_init_ckpt_path, "NN_fold*.ckpt"))

            print(
                f"\n{b1model_num} pre-trained B1 models and {b2model_num}"
                " pre-trained B2 models are found.\n"
            )

            if not all(elem == data_fold for elem in [b1model_num, b2model_num]):
                warnings.warn(
                    "mismatch between number of data folds for NN training "
                    "and number of pre-trained baseline models for initialization\n"
                    " If baseline models are not found for some folds, random"
                    " initialization is used"
                )

        self.init_ckpt_path += f"NN_fold{self.args.fold}.ckpt"
        self.b1_init_ckpt_path += f"NN_fold{self.args.fold}.ckpt"
        self.b2_init_ckpt_path += f"NN_fold{self.args.fold}.ckpt"
        self.feat_config = self.args.audio_config

        self.train_loader = self._get_dataloader(set="train", shuffle=True)

        if self.args.valmonitor:
            self.val_loader = self._get_dataloader(set="val", shuffle=False)
            print("\nMonitoring training based on validation set.")
        else:
            self.val_loader = self._get_dataloader(set="train", shuffle=False)
            print(
                "\nTraining is not monitored; validation set = non-shuffled train set."
            )

        self.test_loader = self._get_dataloader(set="test", shuffle=False)

        self.freqlen, self.seqlen = self.train_loader.dataset.getDimension()

        self.model = None
        self.optimizer = None
        self._get_model()  # initialize the NN model
        self._get_optimizer()  # initialize optimizer
        # initialization for Earlystoping method if used
        self.Estop_counter = 0
        self.Estop_best_score = None
        self.Estop = False
        self.Estop_min_loss = np.Inf
        self.Saved_counter = 0

        loss = {"CE": nn.CrossEntropyLoss(), "MSE": nn.MSELoss()}
        self.loss = loss[self.config_file["runner"]["optimizer"]["loss"]]

    def _get_model(self):
        """get NN model based on config(initialized either randomly or from
        previously saved baseline models if they are applicable
        """
        self.model = ClassifierTrain(self.freqlen, self.seqlen, self.config_file).to(
            self.args.device
        )
        if (self.modelname == "CNNDist") and self.args.preinit:
            b1_init_ckpt = (
                torch.load(self.b1_init_ckpt_path, map_location="cpu")
                if os.path.exists(self.b1_init_ckpt_path)
                else {}
            )
            b2_init_ckpt = (
                torch.load(self.b2_init_ckpt_path, map_location="cpu")
                if os.path.exists(self.b2_init_ckpt_path)
                else {}
            )
            if b1_init_ckpt:
                print(
                    "[ClassifierTrain] - Loading NN model front-end layer weights"
                    " from the 1st baseline init ckpt: ",
                    f"{self.b1_init_ckpt_path}",
                )
                with torch.no_grad():  # copy the parameters of the first layer
                    self.model.model.FrontEnd[0].weight.copy_(
                        b1_init_ckpt["FrontEnd"]["0.weight"]
                    )
                    self.model.model.FrontEnd[0].bias.copy_(
                        b1_init_ckpt["FrontEnd"]["0.bias"]
                    )
            if b2_init_ckpt:
                print(
                    "[ClassifierTrain] - Loading NN model Convolutional layer weights"
                    " from the 2nd baseline init ckpt: ",
                    f"{self.b2_init_ckpt_path}",
                )
                self.model.model.Conv.load_state_dict(b2_init_ckpt["Conv"])

    def _init_fn(self, worker_id):
        """Setting randomness seed for multi-process data loading
        Args:
            worker_id
        """
        seed_torch(self.args.seed)

    def _get_dataloader(self, set="train", shuffle=True):
        """Get dataloader (depending on config we get online or offline dataloader)
        Args:
            set (str, optional): data splits (train/test/validation). Defaults to 'train'.
            shuffle (bool, optional): shuffling data segments. Defaults to True.
        Returns:
            (Dataloader)
        """
        feat_config = get_config_args(self.feat_config)
        feat_type = feat_config["feat_type"]
        config = get_config_args(self.config)
        dataloader_config = config.get("dataloader")

        if dataloader_config.get("online"):  #  online

            if (feat_config["Pairwise-Distance"]) | (feat_config["Pairwise-Reps"]):
                suffix = "_isowords"
            else:
                suffix = ""
            file_path = os.path.join(
                dataloader_config.get("data_path"),
                f"{set}_fold{self.args.fold}{suffix}_online.csv",
            )
        else:  # offline

            if feat_config["Pairwise-Distance"]:
                file_path = os.path.join(
                    dataloader_config.get("data_path"),
                    f"{set}_fold{self.args.fold}_{feat_type}_Dist_offline.csv",
                )

            elif feat_config["Pairwise-Reps"]:
                file_path = os.path.join(
                    dataloader_config.get("data_path"),
                    f"{set}_fold{self.args.fold}_resized_words_{feat_type}_offline.csv",
                )
            else:
                file_path = os.path.join(
                    dataloader_config.get("data_path"),
                    f"{set}_fold{self.args.fold}_{feat_type}_offline.csv",
                )
        print(f"\n\n{set} Data...")
        dataset = get_dataset(config, file_path, feat_config)

        net = self.config_file["SelectedNetwork"]
        arch = self.config_file[net]
        drop_last = arch.get("batchnorm", False)
        return DataLoader(
            dataset,
            batch_size=dataloader_config.get("batch_size"),
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=dataloader_config.get("num_workers"),
            worker_init_fn=self._init_fn,
        )

    def _get_optimizer(self):
        """Set optimizer models for NN model
        (initialized)
        """
        if self.optimizer is None:  # initialize for the first time
            optimizer_config = self.config_file.get("runner").get("optimizer")
            optimizer_name = optimizer_config.get("type")
            assert self.model is not None, (
                "define NN model" " first before defining the optimizer."
            )
            self.lr = float(optimizer_config["lr"])
            model_params = list(self.model.parameters())
            param_list = [{"params": model_params, "lr": self.lr, "name": "NN-model"}]
            if optimizer_name == "SGD":
                self.optimizer = eval(f"torch.optim.{optimizer_name}")(
                    param_list, momentum=optimizer_config.get("momentum")
                )
            else:
                self.optimizer = eval(f"torch.optim.{optimizer_name}")(param_list)

            self.max_epoch = self.config_file["runner"]["Max_epoch"]
            self.minlr = float(optimizer_config["minlr"])

    def _save_ckpt(self):
        """save models and optimizers params into the checkpoint"""
        if os.path.exists(self.init_ckpt_path):
            init_ckpt = torch.load(self.init_ckpt_path, map_location="cpu")
        else:
            init_ckpt = {
                "Optimizer": "",
                "NetConfig": self.config_file,
            }
        init_ckpt = self.model.add_state_to_save(init_ckpt)
        init_ckpt["Optimizer"] = self.optimizer.state_dict()
        torch.save(init_ckpt, self.init_ckpt_path)

    def _load_ckpt(self):
        """load models and optimizers params from the checkpoint"""
        init_ckpt = torch.load(self.init_ckpt_path, map_location="cpu")
        self.model.load_model(init_ckpt)
        init_optimizer = init_ckpt.get("Optimizer")
        self.optimizer.load_state_dict(init_optimizer)

    def _earlystopping(
        self, val_loss, patience=5, verbose=True, delta=0, lr_factor=0.5
    ):
        """Early stops the training if validation loss doesn't improve after a given patience.
        it saves best model based on validation on checkpoint path
        Adapted from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
        Args:
            val_loss (float): validation loss
            patience (int, optional): How long to wait after last time validation loss
            improved. Defaults to 5.
            verbose (bool, optional): If True, prints a message for each validation loss
            improvement. Defaults to True.
            delta (int, optional): Minimum change in the monitored quantity to qualify
            as an improvement. Defaults to 0.
            lr_factor (float): after the patience epochs, multiple learning by lr_factor.
            Defaults to 0.5.
        """
        self.Estop_delta = delta
        score = -val_loss
        if self.Estop_best_score is None:
            self.Estop_best_score = score
            if verbose:
                print(
                    f"Validation loss decreased ({self.Estop_min_loss:.6f} -->"
                    f" {val_loss:.6f}).  Saving model ...",
                    flush=True,
                )
            # save model and optimizer
            self.Estop_min_loss = val_loss
            self._save_ckpt()

        elif score < self.Estop_best_score + self.Estop_delta:
            self.Estop_counter += 1
            if verbose:
                print(f"EarlyStopping counter: {self.Estop_counter} out of {patience}")
            if self.Estop_counter >= patience:
                # load previously saved models and optimizer while multiplying the leanring rate by lr_factor
                print("No improvement since last best model", flush=True)
                self._load_ckpt()  # start from the last best model
                self.lr = self.lr * lr_factor
                for g in self.optimizer.param_groups:  # decreasing the learning rate
                    g["lr"] = self.lr
                print(f"Learning rate decreased, new lr: ({self.lr:.2e})", flush=True)
                if self.lr < self.minlr:
                    print(
                        "Early STOP, after saving model for ",
                        self.Saved_counter,
                        " times",
                        flush=True,
                    )
                    self.Estop = True
                self.Estop_counter = 0
        else:
            if verbose:
                print(
                    f"Validation loss decreased ({self.Estop_min_loss:.6f} -->"
                    f" {val_loss:.6f}).  Saving model ...",
                    flush=True,
                )
            self.Estop_best_score = score
            self.Estop_min_loss = val_loss
            self._save_ckpt()
            self.Saved_counter += 1
            self.Estop_counter = 0

    def _train_epoch(self, train_loader):
        """training epoch
        Args:
            train_loader (DataLoader): downstream train DataLoader
        Returns:
            (float): loss value
        """
        self.model.train()
        batch_loss = []
        for batch_idx, data_batch in enumerate(train_loader):
            data, spk_ID, target = data_batch
            if isinstance(data, list):
                data = [Variable(item.to(self.args.device)) for item in data]
                target = Variable(target.to(self.args.device))
            else:
                data, target = Variable(data.to(self.args.device)), Variable(
                    target.to(self.args.device)
                )

            self.optimizer.zero_grad()
            predicted_output = self.model(data)
            loss = self.loss(predicted_output, target.long())
            loss.backward()
            self.optimizer.step()
            batch_loss.append(loss.item())
        return np.mean(batch_loss)

    @torch.no_grad()
    def _test_epoch(self, test_loader, get_spk_target=False):
        """testing epoch
        Args:
            test_loader (DataLoader): upstream test DataLoader
            get_spk_target (bool, optional): If True returns speaker-level targets.
            Defaults to False.

        Returns:
            (tuple): a tuple containing:
                - (numpy.ndarray): loss
                - (numpy.ndarray): chunk-level accuracy
                - (numpy.ndarray): predicted speaker-level scores [N, num of classes]
        """
        self.model.eval()
        batch_loss = []
        sum_acc = 0
        Spk_level_scores = defaultdict(list)
        Spk_target = defaultdict(list)
        for batch_idx, data_batch in enumerate(test_loader):
            data, spk_ID, target = data_batch

            if isinstance(data, list):
                data = [Variable(item.to(self.args.device)) for item in data]
                target = Variable(target.to(self.args.device))
            else:
                data, target = Variable(data.to(self.args.device)), Variable(
                    target.to(self.args.device)
                )
            predicted_output = self.model(data)
            loss = self.loss(predicted_output, target.long())
            batch_loss.append(loss.item())
            predicted_scores = nn.functional.softmax(
                predicted_output, dim=1
            )  # B X (number of classes)
            _, predicted_labels = torch.max(predicted_output, 1)
            sum_acc = sum_acc + torch.sum(predicted_labels.data == target.long())
            for b_ind in range(len(target)):
                Spk_level_scores[f"SPK_{int(spk_ID[b_ind])}"].append(
                    predicted_scores[b_ind, :].cpu().detach().numpy()
                )
                if get_spk_target:
                    Spk_target[f"SPK_{int(spk_ID[b_ind])}"].append(
                        target[b_ind].cpu().detach().numpy().item()
                    )
        test_chunk_acc = (
            (sum_acc.cpu().numpy().item()) / (test_loader.dataset.__len__())
        ) * 100
        test_loss = np.mean(batch_loss)
        spk_index_sorted = [
            int(i.split("SPK_")[1]) for i in list(Spk_level_scores.keys())
        ]
        predicted_score_spk_level = np.zeros(
            (len(spk_index_sorted), predicted_output.shape[1])
        )
        spk_target = np.zeros(len(spk_index_sorted))
        for idx, spk_indx in enumerate(spk_index_sorted):
            predicted_score_spk_level[idx, :] = np.mean(
                Spk_level_scores["SPK_{:d}".format(spk_indx)], axis=0
            )
            if get_spk_target:
                assert (
                    len(set(Spk_target["SPK_{:d}".format(spk_indx)])) == 1
                ), f" targets of spk index {spk_indx} are not in-agreement with utterance targets"
                spk_target[idx] = Spk_target["SPK_{:d}".format(spk_indx)][0]

        if get_spk_target:
            return (
                np.mean(batch_loss),
                test_chunk_acc,
                predicted_score_spk_level,
                spk_target,
            )
        else:
            return np.mean(batch_loss), test_chunk_acc, predicted_score_spk_level

    def train(self):
        """Main training for all epochs, and save the final model"""
        print(" - Loss Function: ", self.loss, "\n")
        epoch_len = len(str(self.max_epoch))
        for epoch in range(self.max_epoch):
            train_loss = self._train_epoch(self.train_loader)
            val_loss, val_chunk_acc, val_spk_scores = self._test_epoch(self.val_loader)
            print(
                f"\nEpoch [{epoch:>{epoch_len}}/{self.max_epoch:>{epoch_len}}]  train_loss: {train_loss:.5f} "
                + f"  val_loss: {val_loss:.5f}, val chunk acc: {val_chunk_acc:.3f}",
                flush=True,
            )
            if self.args.valmonitor:
                self._earlystopping(
                    val_loss,
                    patience=5,
                    verbose=self.args.verbose,
                    delta=0,
                    lr_factor=0.5,
                )
                if self.Estop:
                    print(
                        f"---Early STOP, after saving model for {self.Saved_counter} times",
                        flush=True,
                    )
                    break
        if not self.args.valmonitor:
            self._save_ckpt()
        self._load_ckpt()

        (
            last_val_loss,
            last_val_chunk_acc,
            last_val_spk_scores,
            val_spk_target,
        ) = self._test_epoch(self.val_loader, get_spk_target=True)
        min_val_loss_estop = self.Estop_min_loss
        print(
            f"\nFinal Model -->  val loss: {last_val_loss:.5f} val chunk acc: {last_val_chunk_acc:.3f} "
            + f"val loss Estop: {min_val_loss_estop:.5f}\n",
            flush=True,
        )
        self.val_chunk_acc = last_val_chunk_acc
        self.val_spk_scores = last_val_spk_scores
        self.val_spk_target = val_spk_target
        self._save_ckpt()

    def evaluation(self, set="test"):
        self._load_ckpt()
        test_loss, test_chunk_acc, test_spk_scores, test_spk_target = self._test_epoch(
            eval(f"self.{set}_loader"), get_spk_target=True
        )
        print(
            f"\nEvaluation Final Model --> {set} loss: {test_loss:.5f} {set} chunk acc: {test_chunk_acc:.3f}\n",
            flush=True,
        )

        self.test_chunk_acc = test_chunk_acc
        self.test_spk_scores = test_spk_scores
        self.test_spk_target = test_spk_target
        return test_loss, test_chunk_acc, test_spk_scores, test_spk_target
