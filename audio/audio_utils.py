# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
# Defining audio feature extraction utils
# used some implementation ideas from https://github.com/s3prl/s3prl/blob/master/upstream/apc/audio.py
# XXX # fix the transpose for function with numpy inputs? save numpy with size T X D?! 

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

import os.path as op
import torch
import torch.nn as nn
from typing import BinaryIO, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import pairwise_distances
import yaml
import torchaudio.compliance.kaldi as ta_kaldi
from torchaudio import transforms
import sys


def get_waveform(
    path_or_fp: Union[str, BinaryIO], normalization=True
) -> Tuple[np.ndarray, int]:
    """Get the waveform and sample rate of a 16-bit mono-channel WAV or FLAC.
    adapted from https://github.com/pytorch/fairseq/blob/master/fairseq/data/audio/audio_utils.py
    Args:
        path_or_fp (str or BinaryIO): the path or file-like object
        normalization (bool): Normalize values to [-1, 1] (Default: True)
    Returns:
        (numpy.ndarray): [n,] waveform array, (int): sample rate

    """
    if isinstance(path_or_fp, str):
        ext = op.splitext(op.basename(path_or_fp))[1]
        if ext not in {".flac", ".wav"}:
            raise ValueError(f"Unsupported audio format: {ext}")

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Please install soundfile to load WAV/FLAC file")

    waveform, sample_rate = sf.read(path_or_fp, dtype="float32")
    waveform -= np.mean(waveform)
    waveform /= np.max(np.abs(waveform))
    if not normalization:
        waveform *= 2 ** 15  # denormalized to 16-bit signed integers
    return waveform, sample_rate


def _get_torch_feat(waveform, sample_rate, **config) -> Optional[np.ndarray]:
    """Extract features based on config dict, In case of Mel-bank,
    MFCC or spectrogram features TorchAudio is used
    while in case of articulatory features it should use saved CNN models
    (not released here).
    Args:
        waveform (numpy.ndarray): input waveform array
        sample_rate (int): sample rate
    Returns:
        features (numpy.ndarray): extracted features from waveform
    """
    try:
        import torch
        import torchaudio.compliance.kaldi as ta_kaldi

        feat_type = config["feat_type"]
        apply_cmvn = config["postprocess"].get("cmvn")
        apply_delta = config["postprocess"].get("delta")
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        if feat_type == "AP":
            if "AP_AllExtract" not in dir():
                from Phonological_posteriors_extraction.posterior_extractor import (
                    AP_AllExtract,
                )
            features = AP_AllExtract(
                waveform, sample_frequency=sample_rate
            )  # Tensor T X D
        else:
            extractor = eval(f"ta_kaldi.{feat_type}")
            features = extractor(
                waveform, **config["torchaudio"], sample_frequency=sample_rate
            )
        if apply_cmvn:
            eps = 1e-10  # x size T X D
            features = (features - features.mean(dim=0, keepdim=True)) / (
                eps + features.std(dim=0, keepdim=True)
            )
        if apply_delta > 0:
            order = 1
            feats = [features]
            for o in range(order):
                feat = feats[-1].transpose(0, 1).unsqueeze(0)
                Delta = transforms.ComputeDeltas(win_length=apply_delta)
                delta = Delta(feat)
                feats.append(delta.squeeze(0).transpose(0, 1))
            features = torch.cat(feats, dim=-1)
        return features.numpy()
    except ImportError:
        return None


def get_feat(path_or_fp: Union[str, BinaryIO], config_path: str) -> np.ndarray:
    """Compute features based on feature config file. Note that
    TorchAudio requires 16-bit signed integers as inputs and hence the
    waveform should not be normalized.

    Args:
        path_or_fp (Union[str, BinaryIO]): path of wav file
        config_path (str): path of feature extraction config
    Returns:
        (np.ndarray): extracted features
    """
    sound, sample_rate = get_waveform(path_or_fp, normalization=False)
    config = get_config_args(config_path)
    features = _get_torch_feat(sound, sample_rate, **config)
    if features is None:
        raise ImportError(
            "Please install pyKaldi or torchaudio to enable "
            "online filterbank feature extraction"
        )
    return features  # T X D


def get_distfeat(test_path_npy: str, ref_path_npy: str, config_path: str) -> np.ndarray:
    """computes (resized) distance matrices
    Args:
        test_path_npy (str): path of test feature numpy array
        ref_path_npy (str): path of reference feature numpy array
        config_path (str): path of feature extraction config

    Returns:
        (np.ndarray): (resized) distance matrix between test and reference
        feature data
    """
    test_features = np.load(test_path_npy, allow_pickle=True).T  # T x D
    ref_features = np.load(ref_path_npy, allow_pickle=True).T  # T x D
    config = get_config_args(config_path)
    resize = config["DistMatResize"]
    # assert (config['Pairwise-Distance']), 'Wrong feature extraction function'
    if config["feat_type"] == "AP":
        Dist = pairwise_distances(
            test_features ** 2, ref_features ** 2, metric=kl_divergence
        )  # some disparity of precision
        Dist = Dist.astype(test_features.dtype)
    else:
        Dist = pairwise_distances(test_features, ref_features, metric="euclidean")
    out_feat = Resizing(Dist, resize)
    return out_feat


def get_resizedfeat(test_path_npy: str, config_path: str) -> np.ndarray:
    """temporally resizing the input feature data to specified size given in config file
    Args:
        test_path_npy (str): path of the feature numpy array
        config_path (str): path of feature extraction config

    Returns:
        (np.ndarray): resized feature numpy array
    """
    test_features = np.load(test_path_npy, allow_pickle=True)  # D X T
    config = get_config_args(config_path)
    resize = config["DistMatResize"]
    # assert config['Pairwise-Reps'], 'Wrong feature extraction function'
    out_feat = Resizing(test_features, resize, Dist=False)
    return out_feat.T  # T X D


def get_config_args(cpath):
    """get contents of yaml file
    Args:
        cpath (str): yaml file
    Returns:
        (dict): Contents of yaml file
    """
    with open(cpath, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def kl_divergence(posterior_a, posterior_b):
    """computing KL divergence between two arrays of posteriors

    Args:
        posterior_a (np.ndarray): posterior a vector
        posterior_b (np.ndarray)): posterior b vector

    Returns:
        (float): KL divergence value
    """
    out = 0.5 * np.sum(posterior_a * np.log2(posterior_a / posterior_b)) + 0.5 * np.sum(
        posterior_b * np.log2(posterior_b / posterior_a)
    )
    if posterior_a.dtype == posterior_b.dtype == np.float32:
        dtype = np.float32
    else:
        dtype = float
    return out.astype(dtype)


def kl_divergence_pairwise(posterior_a_mat, posterior_b_mat):
    """computes pairwise Kl div from two tensors
    Args:
        posterior_a_mat (tensor): T1 X D
        posterior_b_mat (tensor): T2 X D
    Returns:
        (tensor): pairwise KL div between rows of two input tensors
    """
    assert torch.is_tensor(posterior_a_mat) and torch.is_tensor(
        posterior_a_mat
    ), "inputs should be tensors"
    a_mat = torch.sum(
        posterior_a_mat * (torch.log2(posterior_a_mat)), axis=1, keepdim=True
    ).expand(posterior_a_mat.shape[0], posterior_b_mat.shape[0])
    b_mat = torch.sum(
        posterior_b_mat * (torch.log2(posterior_b_mat)), axis=1, keepdim=True
    ).expand(posterior_b_mat.shape[0], posterior_a_mat.shape[0])
    return 0.5 * (
        a_mat
        - torch.matmul(posterior_a_mat, torch.log2(posterior_b_mat.T))
        + (b_mat - torch.matmul(posterior_b_mat, torch.log2(posterior_a_mat.T))).T
    )


def Upsampling(xSize, Final_size):
    """getting upsampled indices for the distance matrix
    Args:
        xSize (int): initial size
        Final_size (int): final size
    Returns:
        (tuple): a tuple containing:
            - (tensor): new indices
            - (tensor): old indice
    """
    xInd = np.arange(Final_size - xSize)
    lower_add = xInd[: len(xInd) // 2] + 1
    if len(lower_add) == 0:
        lower_add = [0]
    upper_add = xInd[len(xInd) // 2 :] + xSize
    Xslice = np.arange(lower_add[-1], upper_add[0])
    return Xslice, np.arange(xSize)


def Downsampling(xSize, Final_size):
    """getting downsampled indices for the distance matrix
    Args:
        xSize (int): initial size
        Final_size (int): final size
    Returns:
        (tuple): a tuple containing:
            - (tensor): new indices
            - (tensor): old indice
    """
    xInd = np.linspace(0, xSize, Final_size, endpoint=False, dtype=int)  # downsampling
    return np.arange(Final_size), xInd


def Resizing(img, Final_size, Dist=True):
    """Resizing the distance matrix
    Args:
        img (np.ndarray or tensor): input distance matrix or input feature representation
        Final_size (int): final size
        Dist (boolean): if True treats img as a distance matrix, otherwise it is a T X D representation array to be resized in T (temporal) dimension
    Returns:
        (np.ndarray or tensor): resized matrix
    """
    xSize, ySize = img.shape  # D X T
    if torch.is_tensor(img):
        if Dist:
            img_resized = torch.max(img) * torch.ones((Final_size, Final_size))
        else:
            img_resized = torch.max(img) * torch.ones((img.shape[0], Final_size))
    else:
        if Dist:
            img_resized = np.max(img) * np.ones(
                (Final_size, Final_size), dtype=img.dtype
            )
        else:
            img_resized = np.max(img) * np.ones(
                (img.shape[0], Final_size), dtype=img.dtype
            )

    Xslice_n = Xslice_o = np.arange(Final_size)
    Yslice_n = Yslice_o = np.arange(Final_size)

    if ySize < Final_size:
        Yslice_n, Yslice_o = Upsampling(ySize, Final_size)
    elif ySize > Final_size:
        Yslice_n, Yslice_o = Downsampling(ySize, Final_size)
    if Dist:  # resizing both dimensions for input distance matrices
        if xSize < Final_size:
            Xslice_n, Xslice_o = Upsampling(xSize, Final_size)
        elif ySize > Final_size:
            Xslice_n, Xslice_o = Downsampling(xSize, Final_size)
        img_resized[
            Xslice_n[0] : Xslice_n[-1] + 1, Yslice_n[0] : Yslice_n[-1] + 1
        ] = img[Xslice_o, :][:, Yslice_o]
    else:
        img_resized[:, Yslice_n[0] : Yslice_n[-1] + 1] = img[:, Yslice_o]

    return img_resized


class CMVN(nn.Module):
    """Feature normalization module"""

    def __init__(self, eps=1e-10):
        super(CMVN, self).__init__()
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x (tensor): input feature
        Returns:
            (tensor): normalized feature
        """
        x = (x - x.mean(dim=0, keepdim=True)) / (self.eps + x.std(dim=0, keepdim=True))
        return x


class Delta(nn.Module):
    """Computing delta representation of features"""

    def __init__(self, order=1, **kwargs):
        super(Delta, self).__init__()
        self.order = order
        self.compute_delta = transforms.ComputeDeltas(**kwargs)

    def forward(self, x):
        """
        Args:
            x (tensor):  input tensor (feat_seqlen, feat_dim)  [T X D]
        Returns:
            x (tensor): [T X D1] concatenated features input features and
            its deltas according to order number
        """
        feats = [x]
        for o in range(self.order):
            feat = feats[-1].transpose(0, 1).unsqueeze(0)
            delta = self.compute_delta(feat)
            feats.append(delta.squeeze(0).transpose(0, 1))
        x = torch.cat(feats, dim=-1)
        return x


class ExtractAudioFeature(nn.Module):
    """first level audio feature extraction module"""

    def __init__(self, mode="spectrogram", sample_rate=16000, **kwargs):
        super(ExtractAudioFeature, self).__init__()
        assert (
            (mode == "fbank")
            | (mode == "spectrogram")
            | (mode == "mfcc")
            | (mode == "AP")
        ), "only spectrogram, fbank, mfcc, and AP are implemented"
        self.mode = mode
        if mode != "AP":
            self.extract_fn = eval(f"ta_kaldi.{self.mode}")
        else:
            from Phonological_posteriors_extraction.posterior_extractor import (
                AP_AllExtract,
            )

            self.extract_fn = AP_AllExtract

        self.sample_rate = sample_rate
        self.kwargs = kwargs

    def forward(self, waveform):
        """feature computation module
        Args:
            waveform (tensor): input waveform [1 X T]
        Returns:
            (tensor):  features (feat_seqlen, feat_dim) [T X D]
        """
        x = self.extract_fn(
            waveform.view(1, -1), sample_frequency=self.sample_rate, **self.kwargs
        )
        return x


class FeatureExtractor(nn.Module):
    """Full audio feature extraction considering normalization and delta computation"""

    def __init__(
        self,
        mode="spectrogram",
        sample_rate=16000,
        apply_cmvn=True,
        apply_delta=0,
        **kwargs,
    ):
        """feature extractor initialization
        Args:
            mode (str, optional): type of features (spectrogram, fbank, and mfcc).
            Defaults to "spectrogram".
            sample_rate (int, optional): audio sampling rate. Defaults to 16000.
            apply_cmvn (bool, optional): if True applies feature normalization. Defaults to True.
            apply_delta (int, optional): length of delta window. Defaults to 0
            (0, e.g., no delta computation).
        """
        super(FeatureExtractor, self).__init__()
        # ToDo: Other surface representation
        self.sample_rate = sample_rate
        self.kwargs = kwargs
        self.apply_cmvn = apply_cmvn
        self.apply_delta = apply_delta
        transforms = [
            ExtractAudioFeature(mode=mode, sample_rate=self.sample_rate, **self.kwargs)
        ]

        if self.apply_cmvn:
            transforms.append(CMVN())
        if self.apply_delta > 0:
            transforms.append(Delta(win_length=apply_delta))
        self.extract_postprocess = nn.Sequential(*transforms)

    def forward(self, waveform):
        y = self.extract_postprocess(waveform)  # T x D
        return y


class FeaturePairedExtractor(nn.Module):
    """audio feature extraction, transforming pairs of waveforms to distance matrix
    or transform one waveform to (resized) audio features"""

    def __init__(
        self,
        mode="spectrogram",
        sample_rate=16000,
        apply_cmvn=True,
        apply_delta=0,
        resize=58,
        Dist=True,
        **kwargs,
    ):
        """feature extractor initialization
        Args:
            mode (str, optional): type of features (spectrogram, fbank, and mfcc).
            Defaults to "spectrogram".
            sample_rate (int, optional): audio sampling rate. Defaults to 16000.
            apply_cmvn (bool, optional): if True applies feature normalization.
            Defaults to True.
            apply_delta (int, optional): length of delta window. Defaults to 0
            (0, e.g., not delta computation).
            resize (int, optional): size of temporal dimension. Defaults to 58.
            Dist (bool, optional): If True computes distance matrix otherwise
            computes resized feature representation. Defaults to True.
        """
        super(FeaturePairedExtractor, self).__init__()
        self.sample_rate = sample_rate
        self.kwargs = kwargs
        self.apply_cmvn = apply_cmvn
        self.apply_delta = apply_delta
        self.mode = mode
        self.resize = resize
        self.Dist = Dist
        transforms = [
            ExtractAudioFeature(mode=mode, sample_rate=self.sample_rate, **self.kwargs)
        ]
        if self.apply_cmvn:
            transforms.append(CMVN())
        if self.apply_delta > 0:
            transforms.append(Delta(win_length=apply_delta))
        self.extract_postprocess = nn.Sequential(*transforms)

        if self.Dist:
            self.forward = self._forward_dist
        else:
            self.forward = self._forward

    def _forward_dist(self, test_waveform, ref_waveform):
        test = self.extract_postprocess(test_waveform)  # T x D
        ref = self.extract_postprocess(ref_waveform)  # T x D
        if self.mode == "AP":
            Dist = kl_divergence_pairwise(
                test ** 2, ref ** 2
            )  # log2 is already applied on AP features
        else:
            Dist = torch.cdist(test, ref, p=2)
        out_feat = Resizing(
            Dist, self.resize, Dist=self.Dist
        )  # out_feat of size [resize X resize]
        return out_feat

    def _forward(self, waveform):
        test = self.extract_postprocess(waveform)  # T X D
        out_feat = Resizing(test.T, self.resize, Dist=self.Dist)
        return out_feat.T  # [resize X D]


def create_transform(audio_config, fs):
    """create transform for wav file to be converted to feature domain
    Args:
        audio_config (dict): config file for acoustic feature extraction
        fs (int): wav sample rate
    Returns:
        (FeatureExtractor or FeaturePairedExtractor): feature extractor object
         to be operated on (one or pairs of) wav tensor(s) of size [1 X time]
    """
    feat_type = audio_config["feat_type"]
    torchaudio_parmas = audio_config["torchaudio"]
    resize = audio_config["DistMatResize"]
    apply_cmvn = audio_config["postprocess"]["cmvn"]
    apply_delta = audio_config["postprocess"]["delta"]
    if (audio_config["Pairwise-Distance"]) | (audio_config["Pairwise-Reps"]):
        Dist = audio_config["Pairwise-Distance"]
        transforms = FeaturePairedExtractor(
            mode=feat_type,
            sample_rate=fs,
            apply_cmvn=apply_cmvn,
            apply_delta=apply_delta,
            resize=resize,
            Dist=Dist,
            **torchaudio_parmas,
        )
    else:
        transforms = FeatureExtractor(
            mode=feat_type,
            sample_rate=fs,
            apply_cmvn=apply_cmvn,
            apply_delta=apply_delta,
            **torchaudio_parmas,
        )
    return transforms


#%%
if __name__ == "__main__":
    # checking parity between online and offline feature extraction
    # all online implementation yields T X D size but offline when load/save are of D X T
    # torch.set_default_tensor_type('torch.FloatTensor')
    import matplotlib.pyplot as plt
    wav_test_path = "../preprocess/dummy_database/audio_data_words/spk6-w2.wav"
    wav_ref_path = "../preprocess/dummy_database/audio_data_words/spk5-w2.wav"
    wav_test, fs = get_waveform(wav_test_path, normalization=False)
    wav_test_tensor = torch.from_numpy(wav_test).unsqueeze(0)
    wave_ref, fs = get_waveform(wav_ref_path, normalization=False)
    wav_ref_tensor = torch.from_numpy(wave_ref).unsqueeze(0)
    conf_feat_path = "../config/audio_config.yaml"
    cf = get_config_args(conf_feat_path)
    test_features = get_feat(wav_test_path, conf_feat_path).T  # should be D X T
    np.save("../preprocess/test/test.npy", test_features)
    ref_features = get_feat(wav_ref_path, conf_feat_path).T
    np.save("../preprocess/test/ref.npy", ref_features)
    if cf["Pairwise-Distance"]:
        features = get_distfeat(
            "../preprocess/test/test.npy", "../preprocess/test/ref.npy", conf_feat_path
        ).T
    elif cf["Pairwise-Reps"]:
        features = (
            get_resizedfeat("../preprocess/test/test.npy", conf_feat_path).T,
            get_resizedfeat("../preprocess/test/ref.npy", conf_feat_path).T,
        )
    else:
        features = get_feat(wav_test_path, conf_feat_path).T

    # this when saved should be, numpy of size D X T
    n = 1
    N = 5 if isinstance(features, tuple) else 3
    plt.subplot(eval(f"{N}1{n}"))
    plt.plot(wav_test)
    if isinstance(features, tuple):
        n += 1
        plt.subplot(eval(f"{N}1{n}"))
        plt.imshow(features[0], aspect="auto", cmap="jet")
        n += 1
        plt.subplot(eval(f"{N}1{n}"))
        plt.imshow(features[1], aspect="auto", cmap="jet")
    else:
        n += 1
        plt.subplot(eval(f"{N}1{n}"))
        plt.imshow(features, aspect="auto", cmap="jet")

    transforms = create_transform(cf, fs)

    if cf["Pairwise-Distance"]:
        feat = transforms(wav_test_tensor, wav_ref_tensor)
    elif cf["Pairwise-Reps"]:
        feat = (transforms(wav_test_tensor), transforms(wav_ref_tensor))
    else:
        feat = transforms(wav_test_tensor)
    if isinstance(feat, tuple):
        n += 1
        plt.subplot(eval(f"{N}1{n}"))
        plt.imshow(feat[0].T.detach().numpy(), aspect="auto", cmap="jet")
        n += 1
        plt.subplot(eval(f"{N}1{n}"))
        plt.imshow(feat[1].T.detach().numpy(), aspect="auto", cmap="jet")
        print(
            "error feat computations:",
            np.linalg.norm(feat[0].t().detach().numpy() - features[0]),
        )
        print(
            "error feat computations:",
            np.linalg.norm(feat[1].t().detach().numpy() - features[1]),
        )
    else:
        n += 1
        plt.subplot(eval(f"{N}1{n}"))
        plt.imshow(feat.T.detach().numpy(), aspect="auto", cmap="jet")
        print(
            "error feat computations:",
            np.linalg.norm(feat.t().detach().numpy() - features),
        )
# %%
