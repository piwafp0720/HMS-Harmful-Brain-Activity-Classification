import os
import shutil

os.environ["OMP_NUM_THREADS"] = "1"  # OpenMPで使用するスレッド数を制限
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLASで使用するスレッド数を制限
os.environ["MKL_NUM_THREADS"] = "1"  # MKLで使用するスレッド数を制限

from pathlib import Path

import cv2
import librosa
import numpy as np
import pandas as pd
import pywt
from general_utils import tqdm_joblib
from joblib import Parallel, delayed
from tqdm import tqdm


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet="haar", level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (
        pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:]
    )

    ret = pywt.waverec(coeff, wavelet, mode="per")
    return ret


def fill_nans(x):
    m = np.nanmean(x)
    if np.isnan(x).mean() < 1:
        x = np.nan_to_num(x, nan=m)
    else:
        x[:] = 0
    return x


def to_spec(
    x,
    n_hop,
    n_fft,
    fmax,
    win_length,
    use_wavelet,
    audio_transforms,
):
    x = fill_nans(x)

    if use_wavelet:
        x = denoise(x, wavelet=use_wavelet)

    if audio_transforms is not None:
        x = audio_transforms(x)

    # RAW SPECTROGRAM
    spec = librosa.stft(
        y=x,
        hop_length=len(x) // n_hop,
        n_fft=n_fft,
        win_length=win_length,
    )

    spec = np.abs(spec) ** 2

    width = (spec.shape[1] // 32) * 32
    spec = spec.astype(np.float32)[:, :width]

    if fmax == 20:
        spec = spec[:100, :]
    elif fmax == 70:
        spec = spec[:350, :]
    elif fmax == 100:
        spec = spec[:500, :]
    elif fmax is None:
        pass

    spec = cv2.resize(
        spec,
        (256, 128),
        interpolation=cv2.INTER_LINEAR,
    )

    return spec


def to_mel_spec(
    x,
    n_hop,
    n_fft,
    n_mels,
    fmin,
    fmax,
    win_length,
    use_wavelet,
    audio_transforms,
):
    x = fill_nans(x)

    if use_wavelet:
        x = denoise(x, wavelet=use_wavelet)

    if audio_transforms is not None:
        x = audio_transforms(x)

    # RAW SPECTROGRAM
    mel_spec = librosa.feature.melspectrogram(
        y=x,
        sr=200,
        hop_length=len(x) // n_hop,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        win_length=win_length,
    )

    # LOG TRANSFORM
    width = (mel_spec.shape[1] // 32) * 32
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[
        :, :width
    ]

    # STANDARDIZE TO -1 TO 1
    mel_spec_db = (mel_spec_db + 40) / 40
    return mel_spec_db


def make_eeg_specs(
    df,
    index,
    spec_type,
    window_sec,
    eeg_electrode_solo,
    eeg_electrode_combination,
    freq_dict,
    n_hop,
    n_fft,
    n_mels,
    win_length,
    audio_transforms,
    use_wavelet,
    p_eeg_specs,
    p_output_root,
):
    row = df.iloc[index]
    eeg_id = row.eeg_id
    parquet_path = p_eeg_specs / f"{eeg_id}.parquet"
    eeg_org = pd.read_parquet(parquet_path)
    offset_sec = row.eeg_label_offset_seconds
    label_id = row.label_id
    sr = 200  # [Hz]

    eeg_specs = {}
    eeg_specs_neighbor = {}
    middle = int((offset_sec + 25) * sr)
    start = middle - int(window_sec / 2 * sr)
    end = middle + int(window_sec / 2 * sr)
    eeg = eeg_org.iloc[start:end]

    for column in eeg_electrode_solo:
        for freq_name, freqs in freq_dict.items():
            fmin, fmax = freqs
            x = eeg[column].values
            if spec_type == "mel_spectrogram":
                spec = to_mel_spec(
                    x,
                    n_hop,
                    n_fft,
                    n_mels,
                    fmin,
                    fmax,
                    win_length,
                    use_wavelet,
                    audio_transforms,
                )
            elif spec_type == "stft":
                spec = to_spec(
                    x,
                    n_hop,
                    n_fft,
                    fmax,
                    win_length,
                    use_wavelet,
                    audio_transforms,
                )
            spec = spec.astype("float32")
            key = f"{column}_{freq_name}" if freq_name != "" else column
            eeg_specs[key] = spec

    for i_spec, (spec_name, columns) in enumerate(
        eeg_electrode_combination.items()
    ):
        # VARIABLE TO HOLD SPECTROGRAM
        if spec_type == "mel_spectrogram":
            img = np.zeros((n_mels, n_hop), dtype="float32")
        elif spec_type == "stft":
            # img = np.zeros((n_fft // 2 + 1, n_hop), dtype="float32")
            img = np.zeros((128, 256), dtype="float32")
        for freq_name, freqs in freq_dict.items():
            fmin, fmax = freqs
            for i in range(len(columns) - 1):
                # COMPUTE PAIR DIFFERENCES
                x = eeg[columns[i]].values - eeg[columns[i + 1]].values
                if spec_type == "mel_spectrogram":
                    spec = to_mel_spec(
                        x,
                        n_hop,
                        n_fft,
                        n_mels,
                        fmin,
                        fmax,
                        win_length,
                        use_wavelet,
                        audio_transforms,
                    )
                elif spec_type == "stft":
                    spec = to_spec(
                        x,
                        n_hop,
                        n_fft,
                        fmax,
                        win_length,
                        use_wavelet,
                        audio_transforms,
                    )
                img += spec
                spec_name_neighbor = f"{columns[i]}-{columns[i + 1]}"
                key = (
                    f"{spec_name_neighbor}_{freq_name}"
                    if freq_name != ""
                    else spec_name_neighbor
                )
                eeg_specs_neighbor[key] = spec
            # AVERAGE THE 4 MONTAGE DIFFERENCES
            img /= 4.0
            key = f"{spec_name}_{freq_name}" if freq_name != "" else spec_name
            eeg_specs[key] = img
    eeg_specs |= eeg_specs_neighbor
    np.save(p_output_root / f"{label_id}.npy", eeg_specs)


if __name__ == "__main__":
    freq_dict = {
        "": [0, 20],
    }
    csv_path = Path(
        "data/generated/fold/fold.csv",
        index=False,
    )
    spec_type = "stft"
    n_hop = 256
    n_fft = 1024
    n_mels = 128
    window_sec = 50
    win_length = None
    # fmt: off
    eeg_electrode_solo = []
    # fmt: on
    eeg_electrode_combination = {
        "LL": ["Fp1", "F7", "T3", "T5", "O1"],
        "LP": ["Fp1", "F3", "C3", "P3", "O1"],
        "RP": ["Fp2", "F8", "T4", "T6", "O2"],
        "RR": ["Fp2", "F4", "C4", "P4", "O2"],
    }
    p_eeg_specs = Path("data/train_eegs")
    audio_transforms = None
    use_wavelet = None
    p_output_root = Path(f"data/generated/eeg_specs/offset")

    p_output_root.mkdir(exist_ok=True, parents=True)

    file_path_myself = Path(__file__)
    shutil.copy(file_path_myself, p_output_root / file_path_myself.name)

    df = pd.read_csv(csv_path)

    total = len(df)
    with tqdm_joblib(total=total):
        Parallel(n_jobs=-1, backend="multiprocessing", verbose=0)(
            delayed(make_eeg_specs)(
                df,
                index,
                spec_type,
                window_sec,
                eeg_electrode_solo,
                eeg_electrode_combination,
                freq_dict,
                n_hop,
                n_fft,
                n_mels,
                win_length,
                audio_transforms,
                use_wavelet,
                p_eeg_specs,
                p_output_root,
            )
            for index in range(total)
        )

    # for index in tqdm(range(total), total=total):
    #     make_eeg_specs(
    #         df,
    #         index,
    #         spec_type,
    #         window_sec,
    #         eeg_electrode_solo,
    #         eeg_electrode_combination,
    #         freq_dict,
    #         n_hop,
    #         n_fft,
    #         n_mels,
    #         win_length,
    #         audio_transforms,
    #         use_wavelet,
    #         p_eeg_specs,
    #         p_output_root,
    #     )
