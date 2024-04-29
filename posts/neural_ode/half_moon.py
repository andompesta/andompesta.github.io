import numpy as np
from torch.utils.data import Dataset


class HalfMoonDataset(Dataset):
    def __init__(
        self,
        num_samples,
        random_state=42,
        noise_std=1e-4,
        shuffle: bool = True,
    ):
        super(HalfMoonDataset).__init__()
        self.num_samples = num_samples
        self.random_state = random_state
        self.noise_std = noise_std
        self.shuffle = shuffle

        num_samples_out = self.num_samples // 2
        num_samples_in = num_samples - num_samples_out
        theta_out = np.linspace(0, np.pi, num_samples_out)
        theta_in = np.linspace(0, np.pi, num_samples_in)
        outer_circ_x = np.cos(theta_out)
        outer_circ_y = np.sin(theta_out)
        inner_circ_x = 1 - np.cos(theta_in)
        inner_circ_y = 1 - np.sin(theta_in) - 0.5

        dataset = np.vstack(
            [
                np.append(outer_circ_x, inner_circ_x),
                np.append(outer_circ_y, inner_circ_y),
            ]
        ).T
        self._dataset = dataset.astype(np.float32)

        self._labels = np.hstack(
            [
                np.zeros(num_samples_out),
                np.ones(num_samples_in),
            ],
        ).astype(np.float32)

        if self.noise_std is not None:
            self._dataset += self.noise_std * np.random.rand(
                self.num_samples,
                2,
            )

    def __getitem__(self, idx):
        return dict(
            data=self._dataset[idx],
            label=self._labels[idx],
        )

    def __len__(self):
        return len(self._dataset)
