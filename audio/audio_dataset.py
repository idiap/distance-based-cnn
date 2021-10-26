# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
# Defining acoustic datasets

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
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchaudio

torchaudio.set_audio_backend("sox_io")
import bisect
import sys

# from audio_utils import get_waveform, create_transform # In case of local run
from .audio_utils import get_waveform, create_transform


class AcousticDataset(Dataset):
    def __init__(self, extractor, dataloading_config, file_path, feat_config, **kwargs):
        """AcousticDataset initialization
        Args:
            extractor (FeatureExtractor): online feature extraction modules.
            should be None for offline model
            dataloading_config (dict): dataloading config dict
            file_path (str): path of csv file of utterance paths
            feat_config (dict): feature extraction config
        """
        self.extractor = extractor
        self.dataloading_config = dataloading_config["dataloader"]
        self.sample_length = self.dataloading_config["sequence_length"]
        self.feat_config = feat_config
        assert self.sample_length > 0, "only segmented inputs are implemented"
        self.overlap = 0.5  # by default sampling is done with 50% overlap between extracted segments
        self.root = file_path
        assert (not feat_config["Pairwise-Reps"]) & (
            not feat_config["Pairwise-Distance"]
        ), "Wrong acoustic dataset "
        if feat_config["feat_type"] != "AP":
            mode = "torchaudio"
        else:
            mode = "APCNN"
        self.frame_length = feat_config[mode]["frame_length"]
        self.frame_shift = feat_config[mode]["frame_shift"]
        if self.dataloading_config["online"]:
            self.sample_rate = self.extractor.sample_rate
            print(
                "[Dataset] - Sampling random segments with sample length:",
                self.sample_length / 1e3,
                "seconds",
            )
        else:
            self.sample_length = 1 + int(
                (self.sample_length - self.frame_length) / (self.frame_shift)
            )
            # computing number of frames for segment_length
            print(
                "[Dataset] - Sampling random segments with sample length:",
                self.sample_length,
                "frames",
            )

        table = pd.read_csv(os.path.join(file_path))
        # Remove utterances that are shorter than sample_length
        self.table = table[table.length >= self.sample_length]
        self.X = self.table["file_path"].tolist()  # All paths
        # All uttrs length
        X_lens = self.table["length"].tolist()
        self.label = self.table["label"].tolist()
        self.spkID = self.table["ID"].tolist()
        self.X_lens = np.array(X_lens) - self.sample_length
        # Effective length for sampling segments
        print("[Dataset] - Number of individual utterance instances:", len(self.X))

    def _sample(self, x, interval):
        """sampling from data for creating batches
        Args:
            x (tensor): either waveform data [1 X T] or acoustic features [D X T]
            interval (int): sampling start
        Returns:
            (tensor): sampled interval from x
        """
        if not self.dataloading_config["online"]:  # offline dataloader
            return x[:, interval : interval + self.sample_length]
        else:
            interval = int(interval * self.sample_rate // 1e3)
            return x[
                :,
                interval : interval + int(self.sample_length * self.sample_rate // 1e3),
            ]

    def _getindex(self, index):
        """compute the interval and utterance number based on the random index.
        By default sampling is done with 50% overlap between extracted segments
        Args:
            index (int): index

        Returns:
            (tuple): a tuple containing:
                - (int): utterance number (or path number)
                - (int): time interval for the utterance
        """
        index_unit = int(self.sample_length * self.overlap)  # shift size (ms or frames)
        Cumuints = np.cumsum((1 + self.X_lens // index_unit))
        Cumuints1 = np.append(Cumuints, 0)
        uttr_num = bisect.bisect_left(Cumuints, index + 1)
        interval = index - Cumuints1[uttr_num - 1]
        return uttr_num, int((interval) * index_unit)

    def __len__(self):
        """Computing total number of audio segments
        Returns:
            (int): total number of audio segments
        """
        return int(np.sum(1 + (self.X_lens // int(self.overlap * self.sample_length))))


class AcousticDatasetPReps(Dataset):
    def __init__(self, extractor, dataloading_config, file_path, feat_config, **kwargs):
        """AcousticDatasetPReps initialization
        Args:
            extractor (FeatureExtractor): online feature extraction modules.
            should be None for offline model
            dataloading_config (dict): dataloading config dict
            file_path (str): path of csv file of utterance paths
            feat_config (dict): feature extraction config
        """
        self.extractor = extractor
        self.dataloading_config = dataloading_config["dataloader"]
        self.feat_config = feat_config
        self.sample_length = dataloading_config["dataloader"]["sequence_length"]
        if dataloading_config["dataloader"]["online"]:
            self.sample_rate = self.extractor.sample_rate
        # Read file
        self.root = file_path
        assert (feat_config["Pairwise-Reps"]) & (
            not feat_config["Pairwise-Distance"]
        ), "Wrong acoustic dataset "
        self.table = pd.read_csv(os.path.join(file_path))
        self.X = self.table["file_path"].tolist()  # All paths
        # number all uttrs
        X_lens = self.table["length"].tolist()
        self.label = self.table["label"].tolist()
        self.spkID = self.table["ID"].tolist()
        ctrl_label_indx = [l == 0 for l in self.label]
        self.ctrlID = list(np.unique(np.array(self.spkID)[ctrl_label_indx]))
        self.X_lens = np.array(X_lens)
        print("[Dataset] - Number of individual utterance instances:", len(self.X))

    def _sample(self, x, index):
        """passing the utterances as it is, since they are already isolated words,
        therefore no segmentation is need
        Args:
            x (tensor): either waveform data [1 X T] or acoustic features [D X T]
        Returns:
            (tensor): same as input
        """
        return x

    def _getindex(self, index):
        """computes the test and reference utterance index based on the random index
        NOTE: it is important that words are saved in order for each speaker
        e.g., first all utterances from each speaker are saved, i.e.,
        spk1-w1, spk1-w2, spk1-w3, spk2-w1, spk2-w2, spk2-w3, ...

        Args:
            index (int): index
        Returns:
            (tuple): a tuple containing:
                - (int): test utterance index
                - (int): reference utterance index
        """
        test_spk = np.unique(self.spkID)
        (test_id, ref_id, word_num) = np.unravel_index(
            index,
            (len(test_spk), len(self.ctrlID), np.sum(self.X_lens) // len(test_spk)),
        )
        test_idx = self.spkID.index(test_spk[test_id]) + word_num
        # it is important that (number of) utterances for each speaker are the same
        ref_idx = self.spkID.index(test_spk[ref_id]) + word_num
        return test_idx, ref_idx  # get two indices for pairs of data

    def __len__(self):
        """Computing total number of pairs considering the reference speakers
        Returns:
            (int): total number of paired data
        """
        return int(np.sum(self.X_lens) * len(self.ctrlID))


class AcousticDatasetPDist(Dataset):
    def __init__(self, extractor, dataloading_config, file_path, feat_config, **kwargs):
        """AcousticDatasetPDist initialization
        Args:
            extractor (FeatureExtractor): online feature extraction modules.
            should be None for offline model
            dataloading_config (dict): dataloading config dict
            file_path (str): path of csv file of utterance paths
            feat_config (dict): feature extraction config
        """
        self.extractor = extractor
        self.dataloading_config = dataloading_config["dataloader"]
        self.sample_length = dataloading_config["dataloader"]["sequence_length"]
        if self.dataloading_config["online"]:
            self.sample_rate = self.extractor.sample_rate
            self._getindex = self._getindex_on
        else:
            self._getindex = self._getindex_off

        # Read file
        self.root = file_path
        self.feat_config = feat_config
        assert feat_config["Pairwise-Distance"], "Wrong acoustic dataset "
        self.table = pd.read_csv(os.path.join(file_path))
        self.X = self.table["file_path"].tolist()  # All paths
        # All uttrs length
        X_lens = self.table["length"].tolist()
        self.label = self.table["label"].tolist()
        self.spkID = self.table["ID"].tolist()
        ctrl_label_indx = [l == 0 for l in self.label]
        self.ctrlID = list(np.unique(np.array(self.spkID)[ctrl_label_indx]))
        self.X_lens = np.array(X_lens)
        print("[Dataset] - Number of individual utterance instances:", len(self.X))

    def _sample(self, x, index):
        """sampling the word index from the saved distance matrices
        Args:
            x (tensor): distance matrix for all words
        Returns:
            (tensor): distance matrix for the given word index
        """
        return x[:, :, index].T

    def _getindex_off(self, index):
        """computes the distance matrix index based on the random index.
        this is only for the offline mode where the distance matrices between
        each pairs of speakers for all words are saved.
        Args:
            index (int): index
        Returns:
            (tuple): a tuple containing:
                - (int): pairs index
                - (int): word index
        """
        Cumuints = np.cumsum(self.X_lens)
        Cumuints1 = np.append(Cumuints, 0)
        uttr_num = bisect.bisect_left(Cumuints, index + 1)
        w_num = index - Cumuints1[uttr_num - 1]
        return uttr_num, w_num

    def _getindex_on(self, index):
        """computes the indices of test and reference utterances to create the distance
        matrix on the random index.
        this is only for the online mode where we use the wav files of an utterance from
        pairs of speakers.
        NOTE: it is important that words are saved in order for each speaker
        e.g., first all utterances from each speaker are saved, i.e.,
        spk1-w1, spk1-w2, spk1-w3, spk2-w1, spk2-w2, spk2-w3, ...
        Args:
            index (int): index
        Returns:
            (tuple): a tuple containing:
                - (int): test utterance index
                - (int): reference index
        """
        test_spk = np.unique(self.spkID)
        (test_id, ref_id, word_num) = np.unravel_index(
            index,
            (len(test_spk), len(self.ctrlID), np.sum(self.X_lens) // len(test_spk)),
        )
        test_idx = self.spkID.index(test_spk[test_id]) + word_num
        ref_idx = self.spkID.index(test_spk[ref_id]) + word_num
        return test_idx, ref_idx  # get two indices for pairs of data

    def __len__(self):
        """Computing total number of pairs or distance matrices
        considering the reference speakers
        Returns:
            (int): total number of paired data or distance matrices
        """
        if self.dataloading_config["online"]:
            return int(np.sum(self.X_lens) * len(self.ctrlID))
        else:
            return int(np.sum(self.X_lens))


class OfflineAcousticDataset(AcousticDataset):
    """generating offline AcousticDataset"""

    def __init__(self, extractor, dataloading_config, file_path, feat_config, **kwargs):
        super(OfflineAcousticDataset, self).__init__(
            extractor, dataloading_config, file_path, feat_config, **kwargs
        )

    def _load_feat(self, npy_path):
        """
        Args:
            npy_path (str): path of numpy file of feature

        Returns:
            Tensor D X T: features
        """
        feat = np.load(npy_path, allow_pickle=True)
        # D X T feature (previously saved feature arrays should be of dimension D X T)
        return torch.from_numpy(feat)

    def getDimension(self):
        """getting dimension of input feature tensors
        Returns:
            int, int: freq dim, seq len
        """
        INTERVAL = 0
        freqlen, seqlen = self._sample(self._load_feat(self.X[0]), INTERVAL).shape
        return freqlen, seqlen

    def standardize(self, tensor):
        """Standardize input tensors
        Args:
            tensor (tensors)
        Returns:
            (tensors): standardized tensor
        """
        MIN = tensor.min()
        MAX = tensor.max()
        return (tensor - MIN) / (MAX - MIN)

    def _get_pair_item(self, index):
        """Load pairs of test/ref representations
        Args:
            index (int): sampling index

        Returns:
            (tuple): a tuple containing:
                - (tuple):
                    - (tensor): test representation
                    - (tensor): reference representation
                - (int): test speaker index
                - (int): test utterance label
        """
        test_uttr, ref_uttr = self._getindex(index)
        test_feat_file = self._load_feat(self.X[test_uttr])
        ref_feat_file = self._load_feat(self.X[ref_uttr])
        return (
            (self.standardize(test_feat_file), self.standardize(ref_feat_file)),
            self.spkID[test_uttr],
            self.label[test_uttr],
        )

    def _get_single_item(self, index):
        """Load single feature representation
        Args:
            index (int): sampling index

        Returns:
            (tuple): a tuple containing:
                - (tensor): feature representation
                - (int): speaker index
                - (int): utterance label
        """
        uttr_num, interval = self._getindex(index)
        feat_file = self.X[uttr_num]
        return (
            self.standardize(self._sample(self._load_feat(feat_file), interval)),
            self.spkID[uttr_num],
            self.label[uttr_num],
        )


class OnlineAcousticDataset(AcousticDataset):
    """generating online AcousticDataset"""

    def __init__(self, extractor, dataloading_config, file_path, feat_config, **kwargs):
        super(OnlineAcousticDataset, self).__init__(
            extractor, dataloading_config, file_path, feat_config, **kwargs
        )
        _, test_fs = get_waveform(self.X[0], normalization=False)
        assert (
            test_fs == extractor.sample_rate
        ), f"the setting for sampling frequency in config file {extractor.sample_rate} should be {test_fs}"

    def _load_feat(self, wavfile):
        """
        Args:
            wavfile (str): path of wav file

        Returns:
            (tensor): [1 X T] waveform
        """
        wav, _ = get_waveform(wavfile, normalization=False)
        return torch.from_numpy(wav).unsqueeze(0)  # 1 X T

    def _feature_extract(self, wavfile):
        """Extracting features from wavfile
        Args:
            wavfile (tensor): Tansor wavform [1 X T]
        Returns:
            features (tensor): [D X T] extracted features from waveform
        """
        return self.extractor(wavfile).t()  # D X T

    def _feature_dist_extract(self, wavfile1, wavfile2):
        """Extracting distance matrix from pairs of test/ref wavfiles
        Args:
            wavfile1 (tensor): test wavform [1 X T]
            wavfile2 (tensor): reference wavform [1 X T]
        Returns:
            (tensor): [resize X resize] distance matrix
        """
        return self.extractor(wavfile1, wavfile2)  # D X T

    def getDimension(self):
        """getting dimension feature data (offline/online modes)
        Returns:
            (tuple): a tuple containing:
                - (int): freq dim
                - (int): seq len
            NOTE: In case of distance matrices freq and seq len are just dimension
            of the matrix
        """
        INTERVAL = 0
        EXAMPLE_FEAT_SEQLEN = int(self.sample_length * 1e-3 * self.sample_rate * 2)
        pseudo_wav = torch.randn(1, EXAMPLE_FEAT_SEQLEN)
        if self.feat_config["Pairwise-Distance"]:
            freqlen, seqlen = self._feature_dist_extract(pseudo_wav, pseudo_wav).shape
        else:
            freqlen, seqlen = self._feature_extract(
                self._sample(pseudo_wav, INTERVAL)
            ).shape

        return freqlen, seqlen

    def standardize(self, tensor):
        """Standardize input tensors
        Args:
            tensor (tensors)
        Returns:
            (tensors): standardized tensor
        """
        MIN = tensor.min()
        MAX = tensor.max()
        return (tensor - MIN) / (MAX - MIN)

    def _get_single_item(self, index):
        """computes and loads single feature representation
        Args:
            index (int): sampling index

        Returns:
            (tuple): a tuple containing:
                - (tensor): feature representation
                - (int): speaker index
                - (int): utterance label
        """
        uttr_num, interval = self._getindex(index)
        wav_file = self.X[uttr_num]
        return (
            self.standardize(
                self._feature_extract(self._sample(self._load_feat(wav_file), interval))
            ),
            self.spkID[uttr_num],
            self.label[uttr_num],
        )

    def _get_single_dist_item(self, index):
        # compute acoustic feature
        test_uttr, ref_uttr = self._getindex(index)
        test_wav_file, ref_wav_file = self.X[test_uttr], self.X[ref_uttr]
        dist_feat = self._feature_dist_extract(
            self._load_feat(test_wav_file), self._load_feat(ref_wav_file)
        )
        return self.standardize(dist_feat), self.spkID[test_uttr], self.label[test_uttr]

    def _get_pair_item(self, index):
        """computes and loads pairs of test/ref representations
        Args:
            index (int): sampling index
        Returns:
            (tuple): a tuple containing:
                - (tuple):
                    - (tensor): test representation
                    - (tensor): reference representation
                - (int): test speaker index
                - (int): test utterance label
        """
        test_uttr, ref_uttr = self._getindex(index)
        test_wav_file, ref_wav_file = self.X[test_uttr], self.X[ref_uttr]
        test_feat = self._feature_extract(self._load_feat(test_wav_file))
        ref_feat = self._feature_extract(self._load_feat(ref_wav_file))
        return (
            (self.standardize(test_feat), self.standardize(ref_feat)),
            self.spkID[test_uttr],
            self.label[test_uttr],
        )


def get_dataset(runner_config, file_path, feat_config):
    """depending on the runner config, the right datasets will be returned
    this part is a bit messy, we create dynamic classes, should do it in a better and compact way

    Args:
        runner_config (dict): main training config file
        file_path (str): path of csv file of utterance paths
        feat_config (dict): feature extraction config

    Returns:
        (DatasetOffline or DatasetOnline)
    """
    data_set_mode = runner_config.get("dataloader").get("online")
    if feat_config["Pairwise-Distance"]:
        base = AcousticDatasetPDist
    elif feat_config["Pairwise-Reps"]:
        base = AcousticDatasetPReps
    else:
        base = AcousticDataset

    if data_set_mode:
        transforms = create_transform(
            feat_config, runner_config.get("dataloader").get("fs")
        )
        print(f"\n[online datasets]...")
        DatasetOnline = type(
            OnlineAcousticDataset.__name__,
            tuple([base]),
            OnlineAcousticDataset.__dict__.copy(),
        )

        def constructor(self, *args):
            super(self.__class__, self).__init__(*args)

        def getitemfunc(self):
            if feat_config["Pairwise-Distance"]:
                return self._get_single_dist_item
            if feat_config["Pairwise-Reps"]:
                return self._get_pair_item
            else:
                return self._get_single_item

        DatasetOnline.__init__ = constructor
        DatasetOnline.__getitem__ = getitemfunc(DatasetOnline)

        return DatasetOnline(transforms, runner_config, file_path, feat_config)
    else:  # offline mode
        DatasetOffline = type(
            OfflineAcousticDataset.__name__,
            tuple([base]),
            OfflineAcousticDataset.__dict__.copy(),
        )

        def constructor(self, *args):
            super(self.__class__, self).__init__(*args)

        def getitemfunc(self):

            if (feat_config["Pairwise-Reps"]) & (not feat_config["Pairwise-Distance"]):
                return self._get_pair_item
            else:
                return self._get_single_item

        DatasetOffline.__init__ = constructor
        DatasetOffline.__getitem__ = getitemfunc(DatasetOffline)
        print(f"\n[offline datasets]...")
        return DatasetOffline(None, runner_config, file_path, feat_config)


if __name__ == "__main__":
    #  checking the parity between offline and online feature extraction
    from pathlib import Path

    file = Path(__file__).resolve()
    parent, root, subroot = file.parent, file.parents[1], file.parents[2]
    sys.path.append(str(subroot))
    sys.path.append(str(root))
    os.chdir(root)

    file = Path(__file__).resolve()
    parent, root = file.parent, file.parents[1]
    sys.path.append(str(root))
    # Additionally remove the current file's directory from sys.path
    try:
        sys.path.remove(str(parent))
    except ValueError:  # Already removed
        pass
    from audio_utils import get_config_args, create_transform
    import matplotlib.pyplot as plt

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
        file_path_off = f"preprocess/dummy_database/folds/test_fold1_resized_words_{feat_type}_offline.csv"
    else:
        file_path_off = (
            f"preprocess/dummy_database/folds/test_fold1_{feat_type}_offline.csv"
        )

    dataset_on = get_dataset(dataloading_config, file_path_on, feat_config)
    dataset_off = get_dataset(dataloading_config_off, file_path_off, feat_config)

    freqdim, seqlen = dataset_on.getDimension()
    freqdim, seqlen = dataset_off.getDimension()

    transforms = create_transform(feat_config, 16000)

    # online testing
    assert dataloading_config["dataloader"][
        "online"
    ], 'set "online field" in upstream config to "True"'

    if feat_config["Pairwise-Distance"]:
        print("...Pairwise-Distance features")
    file_path = f"preprocess/dummy_database/folds/test_fold1_{feat_type}_offline.csv"
    print("input size:", freqdim, seqlen)
    np.random.seed(1)
    test_indx = np.random.choice(dataset_on.__len__(), 3, replace=False)
    if not isinstance(dataset_on.__getitem__(test_indx[0])[0], tuple):
        plt.figure(1)
        plt.title("online")
        plt.subplot(311)
        plt.imshow(
            dataset_on.__getitem__(test_indx[0])[0].detach().numpy(),
            aspect="auto",
            cmap="jet",
        )
        plt.title(
            f"SPK ID: {dataset_on.__getitem__(test_indx[0])[1]}, uttr: {test_indx[0]}"
        )
        plt.subplot(312)
        plt.imshow(
            dataset_on.__getitem__(test_indx[1])[0].detach().numpy(),
            aspect="auto",
            cmap="jet",
        )
        plt.title(
            f"SPK ID: {dataset_on.__getitem__(test_indx[1])[1]}, uttr: {test_indx[1]}"
        )
        plt.subplot(313)
        plt.imshow(
            dataset_on.__getitem__(test_indx[2])[0].detach().numpy(),
            aspect="auto",
            cmap="jet",
        )
        plt.title(
            f"SPK ID: {dataset_on.__getitem__(test_indx[2])[1]}, uttr: {test_indx[2]}"
        )
        plt.figure(2)
        plt.title("offline")
        plt.subplot(311)
        plt.imshow(
            dataset_off.__getitem__(test_indx[0])[0].detach().numpy(),
            aspect="auto",
            cmap="jet",
        )
        plt.title(
            f"SPK ID: {dataset_off.__getitem__(test_indx[0])[1]}, uttr: {test_indx[0]}"
        )
        plt.subplot(312)
        plt.imshow(
            dataset_off.__getitem__(test_indx[1])[0].detach().numpy(),
            aspect="auto",
            cmap="jet",
        )
        plt.title(
            f"SPK ID: {dataset_off.__getitem__(test_indx[1])[1]}, uttr: {test_indx[1]}"
        )
        plt.subplot(313)
        plt.imshow(
            dataset_off.__getitem__(test_indx[2])[0].detach().numpy(),
            aspect="auto",
            cmap="jet",
        )
        plt.title(
            f"SPK ID: {dataset_off.__getitem__(test_indx[2])[1]}, uttr: {test_indx[2]}"
        )
        plt.colorbar(orientation="horizontal")
        print(
            np.linalg.norm(
                dataset_on.__getitem__(test_indx[0])[0].detach().numpy()
                - dataset_off.__getitem__(test_indx[0])[0].detach().numpy()
            )
        )
        print(
            np.linalg.norm(
                dataset_on.__getitem__(test_indx[1])[0].detach().numpy()
                - dataset_off.__getitem__(test_indx[1])[0].detach().numpy()
            )
        )
        print(
            np.linalg.norm(
                dataset_on.__getitem__(test_indx[2])[0].detach().numpy()
                - dataset_off.__getitem__(test_indx[2])[0].detach().numpy()
            )
        )
    else:
        plt.figure(1)
        plt.title("online")
        plt.subplot(311)
        num = 0
        plt.imshow(
            dataset_on.__getitem__(test_indx[0])[0][num].detach().numpy(),
            aspect="auto",
            cmap="jet",
        )
        plt.title(
            f"SPK ID: {dataset_on.__getitem__(test_indx[0])[1]}, uttr: {test_indx[0]}"
        )
        plt.subplot(312)
        plt.imshow(
            dataset_on.__getitem__(test_indx[1])[0][num].detach().numpy(),
            aspect="auto",
            cmap="jet",
        )
        plt.title(
            f"SPK ID: {dataset_on.__getitem__(test_indx[1])[1]}, uttr: {test_indx[1]}"
        )
        plt.subplot(313)
        plt.imshow(
            dataset_on.__getitem__(test_indx[2])[0][num].detach().numpy(),
            aspect="auto",
            cmap="jet",
        )
        plt.title(
            f"SPK ID: {dataset_on.__getitem__(test_indx[2])[1]}, uttr: {test_indx[2]}"
        )
        plt.figure(2)
        plt.title("offline")
        plt.subplot(311)
        plt.imshow(
            dataset_off.__getitem__(test_indx[0])[0][num].detach().numpy(),
            aspect="auto",
            cmap="jet",
        )
        plt.title(
            f"SPK ID: {dataset_off.__getitem__(test_indx[0])[1]}, uttr: {test_indx[0]}"
        )
        plt.subplot(312)
        plt.imshow(
            dataset_off.__getitem__(test_indx[1])[0][num].detach().numpy(),
            aspect="auto",
            cmap="jet",
        )
        plt.title(
            f"SPK ID: {dataset_off.__getitem__(test_indx[1])[1]}, uttr: {test_indx[1]}"
        )
        plt.subplot(313)
        plt.imshow(
            dataset_off.__getitem__(test_indx[2])[0][num].detach().numpy(),
            aspect="auto",
            cmap="jet",
        )
        plt.title(
            f"SPK ID: {dataset_off.__getitem__(test_indx[2])[1]}, uttr: {test_indx[2]}"
        )
        plt.colorbar(orientation="horizontal")
        print(
            np.linalg.norm(
                dataset_on.__getitem__(test_indx[0])[0][num].detach().numpy()
                - dataset_off.__getitem__(test_indx[0])[0][num].detach().numpy()
            )
        )
        print(
            np.linalg.norm(
                dataset_on.__getitem__(test_indx[1])[0][num].detach().numpy()
                - dataset_off.__getitem__(test_indx[1])[0][num].detach().numpy()
            )
        )
        print(
            np.linalg.norm(
                dataset_on.__getitem__(test_indx[2])[0][num].detach().numpy()
                - dataset_off.__getitem__(test_indx[2])[0][num].detach().numpy()
            )
        )
