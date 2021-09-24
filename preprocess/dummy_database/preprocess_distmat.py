# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   preprocess a sample database
#   1) computing/saving features from words utterances (parallel processing)
#   2) Computing distance matrices from pairs of word feature representations
#   3) making tables (train/test/validation) for wav data (for online feature
#   extraction) and offline feature data (for offline dataloading)
#   Note: we consider cross fold validation here, therefore we create different
#   set of tables for each fold

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
import sys
from pathlib import Path

# ------------------------------ Path change ----------------------------- #
file = Path(__file__).resolve()
parent, root, subroot = file.parent, file.parents[1], file.parents[2]
sys.path.append(str(subroot))
sys.path.append(str(root))
os.chdir(root)
# ------------------------------------------------------------------------- #

import shutil
from joblib import Parallel, delayed
from os.path import basename
from os.path import dirname
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from audio.audio_utils import get_feat, get_config_args, get_distfeat, get_resizedfeat
from sklearn.model_selection import StratifiedKFold


def feature_extraction(wav_path, saved_path_dir, feat_config_path):
    """extracting features from audio file
    Args:
        wav_path (str): path of wav file
        saved_path_dir (str): directory path for saving features
        feat_config_path (str): path of feature extraction config file
    Returns:
        (tuple): a tuple containing:
            - (str): path of saved feature file (.npy)
            - (int): length of feature data (e.g., number of time frames)
    """
    feat_config = get_config_args(feat_config_path)
    features = get_feat(wav_path, feat_config_path).T
    if not os.path.exists(saved_path_dir):
        os.makedirs(saved_path_dir, exist_ok=True)
    new_filename = basename(wav_path).split(Path(wav_path).suffix)[0]
    new_path = os.path.join(saved_path_dir, new_filename)
    np.save(new_path, features)  # D X T
    return new_path + ".npy", features.shape[0]


def feature_extraction_resize(feat_path, saved_path_dir, feat_config_path):
    """extracting resized features from original feature files
    i.e., downsampling/upsampling is applied to the features to make them the same
    (temporal) size specified by "DistMatResize" in audio config file
    Args:
        feat_path (str): path of saved feature file
        saved_path_dir (str): directory path for saving resized features
        feat_config_path (str): path of feature extraction config file
    Returns:
        (tuple): a tuple containing:
            - (str): path of saved feature file (.npy)
            - (int): length of feature data (e.g., number of time frames)
    """
    feat_config = get_config_args(feat_config_path)
    features = get_resizedfeat(feat_path, feat_config_path).T
    if not os.path.exists(saved_path_dir):
        os.makedirs(saved_path_dir, exist_ok=True)
    new_filename = basename(feat_path).split(Path(feat_path).suffix)[0]
    new_path = os.path.join(saved_path_dir, new_filename)
    np.save(new_path, features)  # D X T
    return new_path + ".npy", features.shape[0]


def feature_dist_extraction(
    test_feat_path_list,
    ref_feat_path_list,
    test_spk_ID,
    ref_spk_ID,
    saved_path_dir,
    feat_config_path,
):
    """copmutes distance matrices between many utterance of two speakers, 
    so each each input test/ref list is from one speaker but for diff utterances
    Args:
        test_feat_path_list (list): list of utterance paths for the test speaker
        ref_feat_path_list (list): list of utterance paths for the reference speaker
        test_spk_ID (str): test speaker ID
        ref_spk_ID (str): reference speaker ID
        saved_path_dir (str): path for saving the distance features
        (pairwise distance between test and reference utterances)
        feat_config_path (str): path of feature extraction config file
    Returns:
        (tuple): a tuple containing:
            - (str): path of saved distance feature file (.npy)
            - (int): number of distance matrices (e.g., number of utterances)
    """
    feat_config = get_config_args(feat_config_path)
    feat_size = feat_config["DistMatResize"]
    features = np.zeros(
        (feat_size, feat_size, len(test_feat_path_list)), dtype="float32"
    )
    for num, (test_feat_path, ref_feat_path) in enumerate(
        zip(test_feat_path_list, ref_feat_path_list)
    ):
        features[:, :, num] = get_distfeat(
            test_feat_path, ref_feat_path, feat_config_path
        ).T
        uttr_t = basename(dirname(test_feat_path))
        uttr_f = basename(dirname(ref_feat_path))
        assert uttr_t == uttr_f, (
            "Disparity in test and reference utterances,"
            " Using different utterances for computing"
            " distance matrices is not possible!"
        )
        assert (
            test_spk_ID
            in basename(test_feat_path).split(Path(test_feat_path).suffix)[0]
        ) & (
            ref_spk_ID in basename(ref_feat_path).split(Path(ref_feat_path).suffix)[0]
        ), "spk IDs are not paired with paths"
    if not os.path.exists(saved_path_dir):
        os.makedirs(saved_path_dir, exist_ok=True)
    new_filename = test_spk_ID + "_" + ref_spk_ID
    new_path = os.path.join(saved_path_dir, new_filename)
    np.save(new_path, features)
    return new_path + ".npy", features.shape[2]


def preprocess_feat_saving(
    wav_path, ID, SPK_ID, label, save_path_feat, feat_config_path
):
    """saving and extracting features from audio files
    Args:
        wav_path (str): path of wav file
        ID (int): speaker index
        SPK_ID (str): speaker ID
        label (int): speaker label (healthy or pathological)
        save_path_feat (str): path for saving features
        feat_config_path (str): path of feature extraction config file

    Returns:
        (DataFrame): dataframe including information of features data
    """
    feat_path, feat_length = feature_extraction(
        wav_path, save_path_feat, feat_config_path
    )
    df_feat = pd.DataFrame(
        data={
            "ID": ["{:d}".format(ID)],
            "file_path": [feat_path],
            "length": ["{:d}".format(1)],
            "label": ["{:d}".format(label)],
            "spk_ID": [SPK_ID],
        }
    ).rename_axis("uttr_num")
    return df_feat.rename_axis("uttr_num")


def preprocess_feat_resizing(
    feat_path, ID, SPK_ID, label, save_path_feat, feat_config_path
):
    """resizing saved features
    Args:
        feat_path (str): path of saved feature data
        ID (int): speaker index
        SPK_ID (str): speaker ID
        label (int): speaker label (healthy or pathological)
        save_path_feat (str): path for saving the resized features
        feat_config_path (str): path of feature extraction config file
    Returns:
        (DataFrame): dataframe including information of resized features data
    """
    feat_path, feat_length = feature_extraction_resize(
        feat_path, save_path_feat, feat_config_path
    )
    df_feat = pd.DataFrame(
        data={
            "ID": ["{:d}".format(ID)],
            "file_path": [feat_path],
            "length": ["{:d}".format(1)],
            "label": ["{:d}".format(label)],
            "spk_ID": [SPK_ID],
        }
    ).rename_axis("uttr_num")
    return df_feat.rename_axis("uttr_num")


def preprocess_dist(
    test_feat_list,
    ref_feat_list,
    test_ID,
    ref_ID,
    test_SPK_ID,
    ref_SPK_ID,
    test_label,
    save_path_feat,
    feat_config_path,
):
    """saving the distance matrices between many utterance of two speakers,
    so each each input test/ref list is from one speaker but for diff utterances
    Args:
        test_feat_list (list): list of utterance paths for the test speaker
        ref_feat_list (list): list of utterance paths for the reference speaker
        test_ID (int): test speaker index
        ref_ID (int): reference speaker index
        test_SPK_ID (str): test speaker ID
        ref_SPK_ID (str): reference speaker ID
        test_label (int): test speaker label (healthy or pathological)
        save_path_feat (str): path for saving the distance features
        (pairwise distance between test and reference utterances)
        feat_config_path (str): path of feature extraction config file
    Returns:
        (DataFrame): dataframe including information of distance features data
    """

    feat_path, feat_num = feature_dist_extraction(
        test_feat_list,
        ref_feat_list,
        test_SPK_ID,
        ref_SPK_ID,
        save_path_feat,
        feat_config_path,
    )
    df_feat = pd.DataFrame(
        data={
            "ID": ["{:d}".format(test_ID)],
            "ref_ID": ["{:d}".format(ref_ID)],
            "file_path": [feat_path],
            "length": [feat_num],
            "label": ["{:d}".format(test_label)],
            "test_spk_ID": [test_SPK_ID],
            "ref_spk_ID": [ref_SPK_ID],
        }
    ).rename_axis("uttr_num")
    return df_feat.rename_axis("uttr_num")


def remove_dirs(path):
    """removing files in the path"""
    if os.path.exists(path):
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            try:
                shutil.rmtree(filepath)
            except (OSError):
                try:
                    os.remove(filepath)
                except:
                    pass


def folds_making(
    dist_path_csv, wav_path_csv, feat_path_csv, folds_path, feat_type, spk_index, labels
):
    """making fold-wise tables from tables of distance features, audio and feature data
    Args:
        feat_path_csv (str): path of feature data table
        wav_path_csv (str): path of audio data table
        folds_path (str): path f directory folds tables to be saved in
        feat_type (str): feature type
        spk_index (numpy.ndarray): speaker indices
        labels (numpy.ndarray): speaker labels

    Args:
        dist_path_csv (str): path of distance feature data table
        wav_path_csv (str): path of audio data table
        feat_path_csv (str): path of feature data table
        feat_type (str): feature type
        spk_index (numpy.ndarray): speaker indices
        labels (numpy.ndarray): (test) speaker labels
    """
    table_feat_dist = pd.read_csv(os.path.join(dist_path_csv))
    table_wav = pd.read_csv(os.path.join(wav_path_csv))
    table_feat = pd.read_csv(os.path.join(feat_path_csv))

    folds_num = 3
    main_Kfold_obj = StratifiedKFold(n_splits=folds_num, shuffle=True)
    val_folds_num = folds_num - 1
    val_Kfold_obj = StratifiedKFold(n_splits=val_folds_num, shuffle=True)

    for test_fold in range(1, folds_num + 1):
        # outer CV (main CV loop)
        D = main_Kfold_obj.split(spk_index, labels)
        for i in range(test_fold):
            train_index, test_index = next(D)
        train_tot_index = spk_index[train_index]
        train_tot_label = labels[train_index]
        test_fold_label = labels[test_index]
        test_fold_index = spk_index[test_index]

        gen = val_Kfold_obj.split(train_tot_index, train_tot_label)
        for train_nested_index, val_nested_index in gen:
            train_fold_index = train_tot_index[train_nested_index]
            val_fold_index = train_tot_index[val_nested_index]
            train_fold_label = train_tot_label[train_nested_index]
            val_fold_label = train_tot_label[val_nested_index]
            gen.close()

        print("Saving data table for fold: ", test_fold)

        table_wav[table_wav.ID.isin(list(map(str, val_fold_index)))].to_csv(
            os.path.join(folds_path, f"val_fold{test_fold}_isowords_online.csv"),
            index=False,
        )
        table_feat[table_feat.ID.isin(list(map(str, val_fold_index)))].to_csv(
            os.path.join(
                folds_path, f"val_fold{test_fold}_resized_words_{feat_type}_offline.csv"
            ),
            index=False,
        )
        table_feat_dist[table_feat_dist.ID.isin(list(map(str, val_fold_index)))][
            table_feat_dist.ref_ID.isin(
                list(map(str,train_fold_index[train_fold_label == 0]))
            )
        ].to_csv(
            os.path.join(
                folds_path, f"val_fold{test_fold}_{feat_type}_dist_offline.csv"
            ),
            index=False,
        )

        table_wav[table_wav.ID.isin(list(map(str, test_fold_index)))].to_csv(
            os.path.join(folds_path, f"test_fold{test_fold}_isowords_online.csv"),
            index=False,
        )
        table_feat[table_feat.ID.isin(list(map(str, test_fold_index)))].to_csv(
            os.path.join(
                folds_path,
                f"test_fold{test_fold}_resized_words_{feat_type}_offline.csv",
            ),
            index=False,
        )
        table_feat_dist[table_feat_dist.ID.isin(list(map(str, test_fold_index)))][
            table_feat_dist.ref_ID.isin(
                list(map(str, train_fold_index[train_fold_label == 0]))
            )
        ].to_csv(
            os.path.join(
                folds_path, f"test_fold{test_fold}_{feat_type}_dist_offline.csv"
            ),
            index=False,
        )

        table_wav[table_wav.ID.isin(list(map(str, train_fold_index)))].to_csv(
            os.path.join(folds_path, f"train_fold{test_fold}_isowords_online.csv"),
            index=False,
        )
        table_feat[table_feat.ID.isin(list(map(str, train_fold_index)))].to_csv(
            os.path.join(
                folds_path,
                f"train_fold{test_fold}_resized_words_{feat_type}_offline.csv",
            ),
            index=False,
        )
        table_feat_dist[table_feat_dist.ID.isin(list(map(str, train_fold_index)))][
            table_feat_dist.ref_ID.isin(
                list(map(str, train_fold_index[train_fold_label == 0]))
            )
        ].to_csv(
            os.path.join(
                folds_path, f"train_fold{test_fold}_{feat_type}_dist_offline.csv"
            ),
            index=False,
        )


def test(uttr_num, word_num, feat_config):
    import matplotlib.pyplot as plt

    feat_type = feat_config.get("feat_type")
    final_dist_df = pd.read_csv(os.path.join(main_dir, f"{feat_type}_distance_data.csv"))
    rel_path = final_dist_df["file_path"].tolist()[
        uttr_num
    ]  # os.path.relpath(final_dist_df['file_path'].tolist()[uttr_num], os.path.basename(root))
    spectr = np.load(rel_path, allow_pickle=True)
    final_feat_df = pd.read_csv(
        os.path.join(main_dir, f"{feat_type}_features_words_data.csv")
    )
    print(final_dist_df["file_path"].tolist()[uttr_num])
    print(final_feat_df["file_path"].tolist()[24 * (uttr_num // 50) + (word_num)])
    print(final_feat_df["file_path"].tolist()[24 * (uttr_num % 50) + (word_num)])
    rel_path = final_feat_df["file_path"].tolist()[24 * (uttr_num // 50) + (word_num)]
    test = np.load(rel_path, allow_pickle=True)
    rel_path = final_feat_df["file_path"].tolist()[24 * (uttr_num % 50) + (word_num)]
    ref = np.load(rel_path, allow_pickle=True)
    plt.subplot(311)
    plt.imshow(spectr[:, :, word_num], cmap="jet", aspect="auto")
    plt.subplot(312)
    plt.imshow(test, cmap="jet", aspect="auto")
    plt.subplot(313)
    plt.imshow(ref, cmap="jet", aspect="auto")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="preprocess arguments for the dataset."
    )
    parser.add_argument(
        "--Database",
        type=str,
        default="dummy_database/audio_data_words",
        help="database audio directory",
    )
    parser.add_argument(
        "--init",
        action="store_false",
        default=True,
        help="If True, removes previous saved preprocessed data",
    )
    parser.add_argument(
        "--njobs",
        default=4,
        type=int,
        help="number of parallel jobs for offline feature extraction",
    )
    parser.add_argument(
        "--features_data_path",
        default="dummy_database/features_data_words",
        type=str,
        help="Path of extracted features to be saved",
    )
    parser.add_argument(
        "--resized_features_data_path",
        default="dummy_database/resized_features_data_words",
        type=str,
        help="Path of resized features to be saved",
    )
    parser.add_argument(
        "--config_path",
        default="../config/audio_config.yaml",
        type=str,
        help="Path of feature extraction config file",
    )

    args = parser.parse_args(args=[])
    # args = parser.parse_args()

    #%%

    spk_index = []
    spk_ID = []
    all_path = []
    word_list = ["w1", "w2"]
    for dirpath, dirnames, filenames in os.walk(args.Database):
        for f_ind, filename in enumerate(
            [f for f in sorted(filenames) if f.endswith(".wav")]
        ):
            complete_path = os.path.join(dirpath, filename)
            all_path.append(complete_path)
            spk_ID.append(os.path.splitext(filename)[0].split("-")[0])

    spk_ID = np.unique(spk_ID)
    test_path = [None] * len(spk_ID)

    for spk_idx, spk in enumerate(spk_ID):
        spk_index.append(spk_idx)
        test_path[spk_idx] = []
        for uttr in word_list:
            for path in all_path:
                if (spk in path) and (uttr in path):
                    test_path[spk_idx].append(path)

    total_num = len(spk_index)  # creating dummy labels for the dummy database
    spk_index = np.array(spk_index)
    labels = np.zeros_like(spk_index)
    labels[len(spk_index) // 2 :] = 1  # healthy: 0, pathological: 1
    # test_path: list of all speakers, each list element is list of all words
    # uttered by the speaker
    control_num = (labels == 0).sum()
    ref_path = list(np.array(test_path)[np.array(spk_index)[labels == 0]])
    # ref_path: list of all control (reference) speakers, each list element is list
    # of all words uttered by the control speaker

    #%%

    feat_config = get_config_args(args.config_path)

    save_feat_path = os.path.join(
        args.features_data_path, feat_config.get("feat_type")
    )  # word-level features
    save_feat_path_resized = os.path.join(
        args.resized_features_data_path, feat_config.get("feat_type")
    )  # resized word-level features
    save_distfeat_path = os.path.join(
        args.features_data_path + "_Dist", feat_config.get("feat_type")
    )  # distance features

    main_dir = os.path.dirname(args.Database)
    if args.init:
        remove_dirs(save_feat_path_resized)  # remove dirs and files inside
        remove_dirs(save_distfeat_path)
        remove_dirs(save_feat_path)

    if not os.path.exists(save_feat_path_resized):
        os.makedirs(save_feat_path_resized)
    if not os.path.exists(save_distfeat_path):
        os.makedirs(save_distfeat_path)
    if not os.path.exists(save_feat_path):
        os.makedirs(save_feat_path)

    # here we can only handle the situation where each speaker utter
    # the exact same set of words, number of words have to be the same
    # for each speakers (otherwise in distance computation dataloader
    # we will have problem!)

    df_wav_pairedwords = pd.DataFrame(
        data={"ID": [], "file_path": [], "length": [], "label": [], "spk_ID": []}
    ).rename_axis("uttr_num")
    for spk_test_id in spk_index:
        for utt in range(len(word_list)):
            df_wav_subj = pd.DataFrame(
                data={
                    "ID": ["{:d}".format(spk_test_id)],
                    "file_path": [test_path[spk_test_id][utt]],
                    "length": ["{:d}".format(1)],
                    "label": ["{:d}".format(labels[spk_test_id])],
                    "spk_ID": [spk_ID[spk_test_id]],
                }
            ).rename_axis("uttr_num")
            df_wav_pairedwords = df_wav_pairedwords.append(
                df_wav_subj, ignore_index=True
            )
    df_wav_pairedwords = df_wav_pairedwords.rename_axis("uttr_num")
    df_wav_pairedwords.to_csv(os.path.join(main_dir, "audio_words_data.csv"))

    print(len(test_path) * len(test_path[0]), f"audio files found in {args.Database}")

    assert os.path.exists(
        os.path.join(main_dir, "audio_words_data.csv")
    ), "audio path file (audio_words_data.csv) is needed"

    table = pd.read_csv(os.path.join(main_dir, "audio_words_data.csv"))
    path_all_utr = table["file_path"].tolist()
    labels_all_utr = table["label"].tolist()
    spk_IDs_all_utr = table["spk_ID"].tolist()
    ID_all_utr = table["ID"].tolist()

    # computing main representations at word-level, for each word, we get feature reps
    df_feat_list = Parallel(n_jobs=args.njobs)(
        delayed(preprocess_feat_saving)(
            path_all_utr[i],
            ID_all_utr[i],
            spk_IDs_all_utr[i],
            labels_all_utr[i],
            save_feat_path,
            args.config_path,
        )
        for i in tqdm(range(len(path_all_utr)))
    )
    final_feat_df = pd.concat(df_feat_list)
    feat_type = feat_config.get("feat_type")
    final_feat_df.to_csv(os.path.join(main_dir, f"{feat_type}_features_words_data.csv"))

    table = pd.read_csv(os.path.join(main_dir, f"{feat_type}_features_words_data.csv"))
    path_all_utr = table["file_path"].tolist()
    labels_all_utr = table["label"].tolist()
    spk_IDs_all_utr = table["spk_ID"].tolist()
    ID_all_utr = table["ID"].tolist()

    # computing resized representations at word-level, for each word, we get feature reps
    df_feat_list = Parallel(n_jobs=args.njobs)(
        delayed(preprocess_feat_resizing)(
            path_all_utr[i],
            ID_all_utr[i],
            spk_IDs_all_utr[i],
            labels_all_utr[i],
            save_feat_path_resized,
            args.config_path,
        )
        for i in tqdm(range(len(path_all_utr)))
    )
    final_feat_df = pd.concat(df_feat_list)
    feat_type = feat_config.get("feat_type")
    final_feat_df.to_csv(
        os.path.join(main_dir, f"{feat_type}_resized_features_words_data.csv")
    )

    table = pd.read_csv(os.path.join(main_dir, f"{feat_type}_features_words_data.csv"))
    # creating list of speakers where each element is list of utterances itself
    test_path_read = []
    spk_IDs_read = []
    labels_read = []
    for idx in range(total_num):
        test_path_read.append(table["file_path"][table["ID"] == idx].tolist())
        spk_IDs_read.append(table["spk_ID"][table["ID"] == idx].tolist()[0])
        labels_read.append(table["label"][table["ID"] == idx].tolist()[0])
    ref_path_read = test_path_read[:control_num]

    # computing distance matrix from computed word representations
    total_num = len(test_path)
    df_feat_list = Parallel(n_jobs=args.njobs)(
        delayed(preprocess_dist)(
            test_path_read[i],
            ref_path_read[j],
            i,
            j,
            spk_IDs_read[i],
            spk_IDs_read[j],
            labels_read[i],
            save_distfeat_path,
            args.config_path,
        )
        for i in tqdm(range(total_num))
        for j in range(control_num)
    )
    final_feat_df = pd.concat(df_feat_list)
    feat_type = feat_config.get("feat_type")
    final_feat_df.to_csv(os.path.join(main_dir, f"{feat_type}_distance_data.csv"))
    # The name of folds are the same for different modules

    folds_path = os.path.join(main_dir, "folds")
    if not os.path.exists(folds_path):
        os.mkdir(folds_path)

    print("..testing...")
    test(0, 1, feat_config)

    # ugly solution for relative paths
    root_dir = os.path.basename(root)
    table = pd.read_csv(
        os.path.join(main_dir, f"{feat_type}_resized_features_words_data.csv")
    )
    table["file_path"] = root_dir + "/" + table["file_path"]
    table.to_csv(os.path.join(main_dir, f"{feat_type}_resized_features_words_data.csv"))

    table = pd.read_csv(os.path.join(main_dir, "audio_words_data.csv"))
    table["file_path"] = root_dir + "/" + table["file_path"]
    table.to_csv(os.path.join(main_dir, "audio_words_data.csv"))

    table = pd.read_csv(os.path.join(main_dir, f"{feat_type}_distance_data.csv"))
    table["file_path"] = root_dir + "/" + table["file_path"]
    table.to_csv(os.path.join(main_dir, f"{feat_type}_distance_data.csv"))

    folds_making(
        os.path.join(main_dir, f"{feat_type}_distance_data.csv"),
        os.path.join(main_dir, "audio_words_data.csv"),
        os.path.join(main_dir, f"{feat_type}_resized_features_words_data.csv"),
        folds_path,
        feat_type,
        spk_index,
        labels,
    )
