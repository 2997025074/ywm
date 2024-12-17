import numpy as np
import pywt
from tqdm.auto import tqdm


class CWT_Preprocessor:
    def __init__(
            self, dataset: np.lib.npyio.NpzFile, band: np.ndarray, wavelet: str = 'cmor1.5-1.0',
    ) -> None:
        self.dataset = dataset
        self.band = band
        self.wavelet = wavelet

        if 'data' not in self.dataset or 'tr' not in self.dataset:
            raise KeyError("The dataset must contain 'data' and 'tr' keys.")

    def _get_CWT(self, in_seq: np.ndarray, tr: float) -> np.ndarray:
        fs = 1 / tr
        freq = self.band / fs
        scale = pywt.frequency2scale(self.wavelet, freq)
        cwt_data, _ = pywt.cwt(in_seq, scale, self.wavelet, axis=0)
        cwt_data = cwt_data.transpose(1, 0, 2)
        return np.array(cwt_data)

    def get_CWT_dataset(self) -> np.ndarray:
        all_cwt = []

        for i in tqdm(range(len(self.dataset['data']))):
            data_raw = self.dataset['data'][i]
            t_r = self.dataset['tr'][i]
            cwt_data = self._get_CWT(data_raw, t_r)  # [Time, n_band, ROIs]
            all_cwt.append(cwt_data)

        return np.array(all_cwt, dtype=np.complex64)


if __name__ == '__main__':
    raw_path = 'path_to_your_dataset.npz' # TODO: Replace with the full path to your .npz dataset file
    dataset = np.load(raw_path, allow_pickle=True)

    cwt_preprocessor = CWT_Preprocessor(
        dataset=dataset,
        band=np.linspace(0.01, 0.1, 5),
        wavelet='cmor1.5-1.0'
    )
    cwt_results = cwt_preprocessor.get_CWT_dataset()

    print("CWT processing complete. Result shape:", cwt_results.shape)
  
