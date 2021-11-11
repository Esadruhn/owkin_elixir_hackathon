from pathlib import Path
import numpy as np

import substratools as tools

LABELS = {
    "normal": 0,
    "tumor": 1,
}


class Opener(tools.Opener):
    def get_X(self, folders):
        image_paths = list()
        for folder in folders:
            files = Path(folder).glob("*")
            image_paths.extend([str(f.resolve()) for f in files])

        return sorted(image_paths)

    def get_y(self, folders):
        image_labels = list()
        for folder in folders:
            files = Path(folder).glob("*")
            for file_ in files:
                label = file_.stem.split("_")[0].lower()
                image_labels.append((file_.resolve(), LABELS[label]))

        return [label for _, label in sorted(image_labels, key=lambda x: x[0])]

    def save_predictions(self, y_pred, path):
        np.save(path + ".npy", y_pred)
        Path(str(path) + ".npy").rename(str(path))

    def get_predictions(self, path):
        print("Loading preds")
        return np.load(path)

    def fake_X(self, n_samples=None):
        return None

    def fake_y(self, n_samples=None):
        return None
