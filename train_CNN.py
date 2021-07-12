# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
# Scripts for training the CNN models

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
import argparse
import shutil
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from conv_network.runner import NNRunner
from conv_network.runner import seed_torch
from audio.audio_utils import get_config_args


def remove_files(dir_path):
    """removing files or dirs in the path"""
    if not os.path.isdir(dir_path):
        os.remove(dir_path)
    else:
        for filename in os.listdir(dir_path):
            filepath = os.path.join(dir_path, filename)
            try:
                shutil.rmtree(filepath)
            except (OSError):
                try:
                    os.remove(filepath)
                except:
                    pass


def copy_file(scr, dest):
    """copy file"""
    try:
        shutil.copy(scr, dest)
    except shutil.SameFileError:
        pass


def get_fold_args():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0), flush=True)
    parser = argparse.ArgumentParser(description="arguments for upstream experiments")
    parser.add_argument(
        "-c",
        "--config",
        default="config/NNtrain_config.yaml",
        help="The path of config yaml file for configuring the NN training",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluation"],
        default="train",
        help="train or evaluation mode",
    )
    parser.add_argument(
        "--preinit",
        action="store_true",
        default=False,
        help="initialize the model using parts of trained baseline models if they exist"
        " and applicable, otherwise use random initialization",
    )
    parser.add_argument("-s", "--seed", type=int, default=0, help="seed number")
    parser.add_argument("--fold", type=int, default=1, help="fold number")
    parser.add_argument(
        "-d", "--device", type=str, default=device.type, help="cpu or cuda:0"
    )
    parser.add_argument(
        "-a",
        "--audio_config",
        default="config/audio_config.yaml",
        help="The path of config yaml file for features extraction",
    )
    parser.add_argument(
        "-n",
        "--expname",
        default="test",
        help="Save experiment at /results/{cnnmodel}-{expname}/outputs/saved_model/",
    )
    parser.add_argument(
        "-vm",
        "--valmonitor",
        action="store_true",
        default=False,
        help="if True the validation set is used to monitor/early-stoping the training",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="verbose for early stoping (If True prints a message for each validation"
        " loss improvement)",
    )

    if any("SPYDER" in name for name in os.environ) | any(
        "VSCODE_AGENT_FOLDER" in name for name in os.environ
    ):
        args = parser.parse_args(args=[])
    else:
        args, unknown = parser.parse_known_args()

    NN_modelname = get_config_args(args.config)["SelectedNetwork"]

    main_dir = f"results/{args.expname}/{NN_modelname.lower()}-{args.expname}"
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    result_path = (
        f"results/{args.expname}/{NN_modelname.lower()}-{args.expname}"
        f"/outputs/saved_model/NN_fold{args.fold}.ckpt"
    )

    # remove files from previous saved experiments
    if args.mode == "train":
        if os.path.exists(result_path):
            print("...remove previous downstream experiment", flush=True)
            remove_files(result_path)
        copy_file(args.config, main_dir + "/config.yaml")
        copy_file(args.audio_config, main_dir + "/feat_config.yaml")

    if args.mode == "evaluation":
        assert os.path.exists(
            result_path
        ), "No pre-trained models are found for evaluation mode."

    args.config = main_dir + "/config.yaml"
    args.audio_config = main_dir + "/feat_config.yaml"

    return args


def main_fold():
    args = get_fold_args()
    spk_acc_val = 0
    spk_acc_train = 0
    seed_factor = 0
    iter_num = 0
    # If the model are not trained (train accuracy < 55), we try with different seed
    # (set the upperlimit for iter_num times)
    while (spk_acc_val < 55) and (spk_acc_train < 55) and iter_num < 1:
        args.seed = args.seed + seed_factor
        seed_torch(args.seed)
        runner = NNRunner(args)

        if args.mode == "train":
            runner.train()

        save_path = f"results/{args.expname}/{runner.modelname.lower()}-{args.expname}/outputs/output_{args.fold}"
        saved_results = {}
        Data_sets = ["test", "val", "train"]
        for sets in Data_sets:
            loss, acc, spk_scores, target = runner.evaluation(set=sets)
            predicted_label = np.argmax(spk_scores, axis=1)
            message = ""
            message += f"\n---------{sets} data---------- \n\n"
            message += f"\tloss: {loss}\n"
            message += f"\tchunk-level accuracy: {acc} \n"
            message += f"\t----- speaker level evaluation \n"
            spk_acc = 100 * (predicted_label == target).sum() / len(target)
            spk_sen = (
                100
                * ((predicted_label == 1) * (target == 1)).sum()
                / (target == 1).sum()
            )
            spk_spe = (
                100
                * ((predicted_label == 0) * (target == 0)).sum()
                / (target == 0).sum()
            )
            AUC = np.round(roc_auc_score(target, 1 - spk_scores[:, 0]), decimals=3)
            message += f"\tAcc: {spk_acc}\n"
            message += f"\tSen: {spk_sen}\n"
            message += f"\tSpe: {spk_spe}\n\t"
            message += (
                r"[Spe & Sen & Spe: {:.3f} & {:.3f} & {:.3f}]".format(
                    spk_acc, spk_sen, spk_spe
                )
                + " \n"
            )
            message += f"\tAUC: {AUC}\n"
            message += "\tAvg scores of patients (▼): {:f} +- {:f}\n".format(
                np.mean(spk_scores[(target == 1), 0]),
                np.std(spk_scores[(target == 1), 0]),
            )
            message += "\tAvg scores of controls (▲): {:f} +- {:f}\n".format(
                np.mean(spk_scores[(target == 0), 0]),
                np.std(spk_scores[(target == 0), 0]),
            )
            print(message, flush=True)
            saved_results[sets] = [
                acc,
                spk_scores,
                target,
                spk_acc,
                spk_sen,
                spk_spe,
                AUC,
            ]

        np.save(save_path, saved_results)

        loss, acc, spk_scores, target = runner.evaluation(set="val")
        predicted_label = np.argmax(spk_scores, axis=1)
        spk_acc_val = 100 * (predicted_label == target).sum() / len(target)
        loss, acc, spk_scores, target = runner.evaluation(set="train")
        predicted_label = np.argmax(spk_scores, axis=1)
        spk_acc_train = 100 * (predicted_label == target).sum() / len(target)
        seed_factor += 10
        iter_num += 1


if __name__ == "__main__":
    main_fold()
