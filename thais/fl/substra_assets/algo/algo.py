from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import substratools as tools

import torch
from torch.nn.functional import relu
from torchvision.io import read_image

N_UPDATE = 5
BATCH_SIZE = 32
IMAGE_SIZE = (180, 180)

# If you change this value, change it
# in fl_example/main.py#L361 too
N_ROUNDS = 3

torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class TorchModel(torch.nn.Module):
    
    def __init__(
        self,
    ):
        # This is a very simple model, performs badly
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = torch.nn.Linear(16 * 53 * 53, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 1)
        
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.pool(relu(x))
        x = self.pool(relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()


class CamelyonDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, labels=None, transform=None):
        self.transform = transform

        self.labels = labels
        self.samples = data_path


    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = read_image(img_path).to(torch_device)
        image = (1.0 / 255.0) * image
        
        label = self.labels[idx] if self.labels is not None else None
        sample = image, label, img_path
        
        if self.transform:
            sample = self.transform(sample)

        if label is None:
            return image, img_path
        return sample

def make_model():
    model = TorchModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer

def generate_batch_indexes(index, n_rounds, n_update, batch_size):
    """Generates n_rounds*n_update batch of `batch_size` size.

    Args:
        index (list): Index of your dataset
        n_rounds (int): The total number of rounds in your strategy
        n_update (int): The number of batches, you will perform the train method at each
        step of the compute plan
        batch_size (int): Number of data points in a batch

    Returns:
        List[List[index]]: A 2D list where each embedded list is the list of the
        indexes for a batch
    """
    ### DO NOT CHANGE THE CODE OF THIS FUNCTION ###
    my_index = index
    np.random.seed(42)
    np.random.shuffle(my_index)

    n = len(my_index)
    total_batches = n_rounds * n_update
    batches_iloc = []
    batch_size = min(n, batch_size)
    k = 0

    for _ in range(total_batches):

        # It is needed to convert the array to list.
        # Otherwise, you store references during the for loop and everything
        # is computed at the end of the loop. So, you won't see the impact
        # of the shuffle operation
        batches_iloc.append(list(my_index[k * batch_size : (k + 1) * batch_size]))
        k += 1
        if n < (k + 1) * batch_size:
            np.random.shuffle(my_index)
            k = 0

    return batches_iloc


class Algo(tools.algo.Algo):
    def train(self, X, y, models, rank):
        """Train function of the algorithm.
        To train the algorithm on different batches on each step
        we generate the list of index for each node (the first time the
        algorithm is trained on it). We save this list and for each task
        read it from the self.compute_plan_path folder which is a place
        where you can store information locally.

        Args:
            X (List[Path]): Training features, list of paths to the samples
            y (List[str]): Target, list of labels
            models (List[model]): List of models from the previous step of the compute plan
            rank (int): The rank of the task in the compute plan

        Returns:
            [model]: The updated algorithm after the training for this task
        """
        compute_plan_path = Path(self.compute_plan_path)

        batch_indexer_path = compute_plan_path / "batches_loc_node.txt"
        if batch_indexer_path.is_file():
            print("reading batch indexer state")
            batches_loc = eval(batch_indexer_path.read_text())
        else:
            print("Genering batch indexer")
            batches_loc = generate_batch_indexes(
                index=list(range(len(X))),
                n_rounds=N_ROUNDS,
                n_update=N_UPDATE,
                batch_size=BATCH_SIZE,
            )
        if models:
            # Nth round: we get the model from the previous round
            assert len(models) == 1, f"Only one parent model expected {len(models)}"
            model, optimizer = models[0]
        else:
            # First round: we initialize the model
            model, optimizer = make_model()
            model.to(torch_device)

        criterion = torch.nn.BCEWithLogitsLoss()

        model.train()
        dataset = CamelyonDataset(data_path=X, labels=y)
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            [item for sublist in batches_loc for item in sublist],
            batch_size=BATCH_SIZE,
            drop_last=False,
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0)
        dataiter = iter(dataloader)
        for i in range(N_UPDATE):
            # One update = train on one batch
            # The list of samples that belong to the batch is given by batch_loc

            # Get the index of the samples in the batch
            batch_loc = batches_loc.pop()
            # Load the batch samples
            inputs, labels, data_paths = next(dataiter)
            
            labels = labels.to(torch_device).float()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Save the batch indexer
            batch_indexer_path.write_text(str(batches_loc))

        return model, optimizer

    def predict(self, X, model):
        # X: list of paths to the samples
        # return the predictions of the model on X, for calculating the AUC
        torch_model, _ = model
        torch_model.eval()

        dataset = CamelyonDataset(data_path=X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
        with torch.no_grad():
            
            images, data_paths = next(iter(dataloader))
            
            inputs = images.to(torch_device)
            predictions = torch.sigmoid(torch_model(inputs)).detach().cpu().numpy()

        # predictions should be a numpy array of shape (n_samples)
        return predictions

    def load_model(self, path):
        # Load the model from path
        checkpoint = torch.load(path)
    
        model, optimizer = make_model()
        model.load_state_dict(checkpoint['model'])
        
        model.to(torch_device)

        # Load the optimizer state dict after setting the model to GPU
        optimizer.load_state_dict(checkpoint['optimizer'])

        return model, optimizer

    def save_model(self, model, path):
        # Save the model to path
        # Careful, you need to save it exactly to 'path'
        # For example numpy adds '.npy' to the end of the file when you save it:
        #   np.save(path, *model)
        #   shutil.move(path + '.npy', path) # rename the file
        torch_model, optimizer = model
        checkpoint = {   
            "model": torch_model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        torch.save(checkpoint, path)
        # shutil.move(path + '.pth', path) # rename the file


if __name__ == "__main__":
    tools.algo.execute(Algo())
