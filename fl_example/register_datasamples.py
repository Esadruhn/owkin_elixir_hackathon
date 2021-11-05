# Copyright 2018 Owkin, inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import json
import tarfile
from typing import List
from pathlib import Path

import substra
from substra.sdk import DEBUG_OWNER

# Number of centers - 3
N_CENTERS = 3

root_path = Path(__file__).parent
asset_path = root_path / 'substra_assets'

data_path =  Path('/home/user/data')  / "substra"

"""
DATA FOLDER TREE STRUCTURE:
- node_A
    |_ train
        |_ data_sample_0
- node_B
    |_ train
        |_ data_sample_0
- node_C
    |_ train
        |_ data_sample_0
    |_ test
        |_ data_sample_0
        
The test data is on the center C.
"""

def _create_archive(folder, destination):
    """Create a tarfile archive

    Args:
        folder ([type]): folder to compress
        destination ([type]): destination archive
    """
    with tarfile.open(destination, "w:gz") as tar:
        tar.add(folder, recursive=True)


def register_data_samples(clients: List[substra.Client], key_path: Path):

    assert len(clients) == N_CENTERS, 'There must be one client per center'
    assert len(list(data_path.glob('*'))) == N_CENTERS, 'There must be one folder in the data folder per center'

    # Create the archive for the opener
    opener_archive_path = asset_path / 'dataset.tar.gz'
    _create_archive(asset_path/'dataset', opener_archive_path)

    data_keys = dict()

    for client, center_data_path in zip(list(clients.values()), list(data_path.glob('*'))):
        center_name = center_data_path.stem

        # Upload the data opener
        dataset_key = client.add_dataset(
            substra.sdk.schemas.DatasetSpec(
                name=center_name,
                data_opener=asset_path/'dataset'/'opener.py',
                type='histology images',
                metadata = {DEBUG_OWNER: center_name},
                description=asset_path/'dataset'/'description.md',
                permissions={
                    'public': True,
                    'authorized_ids': list(),
                }
            )
        )
        # Upload the train data samples
        assert (center_data_path / 'train').is_dir(), 'There must be a "train" folder in the center folder'
        train_data_sample_folders = list((center_data_path / 'train').glob('*'))
        center_keys = client.add_data_samples(
            data=substra.sdk.schemas.DataSampleSpec(
                paths=train_data_sample_folders,
                data_manager_keys=[dataset_key],
                test_only=False,
            ),
            local=True,
        )
        data_keys[center_name] = dict()
        data_keys[center_name]['train_data_samples'] = center_keys
        data_keys[center_name]['dataset'] = dataset_key

        # If the test folder exists, add the test data sample keys
        if (center_data_path / 'test').is_dir():
            train_data_sample_folders = list((center_data_path / 'test').glob('*'))
            center_keys = client.add_data_samples(
                data=substra.sdk.schemas.DataSampleSpec(
                    paths=train_data_sample_folders,
                    data_manager_keys=[dataset_key],
                    test_only=True,

                ),
                local=True,
            )

            data_keys[center_name]['test_data_samples'] = center_keys

    key_path.write_text(json.dumps(data_keys))
