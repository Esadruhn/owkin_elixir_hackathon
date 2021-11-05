from pathlib import Path
import numpy as np

import substratools as tools

LABELS = {
    'normal': 0,
    'tumor': 1,
}

class Opener(tools.Opener):
    def get_X(self, folders):
        image_paths = list()
        for folder in folders:
            files = Path(folder).glob('*')
            image_paths.extend(list(files))

        print("In opener - found", len(image_paths), "images")
        return image_paths

    def get_y(self, folders):
        image_labels = list()
        for folder in folders:
            files = Path(folder).glob('*')
            for file in files:
                label = file.stem.split('_')[0].lower()
                image_labels.append(LABELS[label])

        print("In opener - found", len(image_labels), "labels")
        return image_labels

    def save_predictions(self, y_pred, path):
        print("Saving preds")
        print("path")
        np.save(path + ".npy", y_pred)
        Path(str(path) + '.npy').rename(str(path))
        #os.rename(path + ".npy", path)

        print("Saving DONE")

    def get_predictions(self, path):
        print("Loading preds")
        return np.load(path)


    def fake_X(self, n_samples=None):
        return None

    def fake_y(self, n_samples=None):
        return None
