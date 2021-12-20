# ===================================
# Federated Learning Cyclic strategy
# ===================================
#
# This file MUST be run from fl_example folder.
#
# You can find details of this dataset `here <../assets/dataset/description.md>`
#
# A cyclic strategy consists in training your algorithm on one node after the other.
#
# Some vocabulary:
# * a *round* represents the fact that the algorithm has been fit on each node
# * a *step* represents the training of the algorithm on one node
# * an *epoch* is one complete pass through the training data
# * a *batch* is a fixed sample size of a training data
# * an *update* is the number of times the algorithm will be trained on different
# batches at each step
#
# A basic cyclic strategy is to incrementally train your model on small batches of
# your datasets on one node after the other.
#
# The elements of our architectures are:
# * node-1: hosting training data samples
# * node-2: hosting training data samples
# * node-2: hosting the compute plan, the test data samples, the metric and the algorithm.
# All those elements (the compute plan, the test data samples, the metric and the algorithm)
# could be hosted on different nodes.
#
# One round will be implemented as follow:
# * train on node-1
# * test on node-1
# * train on node-2
# * test on node-2
#
# It is to be noted that each train step depends only on the latest train step
# meaning that training step and testing step can be run in parallel.
#
#
# Prerequisites
# -------------
#
#
# In order to run this example, you'll need to:
# * use Python >= 3.7
# * have [Docker](https://www.docker.com/) installed
# * install the `substra <../../README.md#install>`_ cli
# * install the `connect-tools <https://github.com/owkin/connect-tools` library
# * pull the `connect-tools
# <https://github.com/owkin/connect-tools#pull-from-public-docker-registry>`
# docker images
# * install the requirements: pip install -r requirements.txt
#
# You can either:
# * run this example in debug mode (no connection to a connect platform)
# * have access to a Connect platform
#
# If you have access to a connect platform, be sure to log your substra client to the nodes you
# will use.
#

import os
import json
import uuid
import zipfile
from tqdm import tqdm
from pathlib import Path
from typing import List

import substra
from substra.sdk.schemas import Permissions
from substra.sdk.schemas import MetricSpec
from substra.sdk.schemas import AlgoSpec
from substra.sdk.schemas import ComputePlanTraintupleSpec
from substra.sdk.schemas import ComputePlanTesttupleSpec
from substra.sdk.schemas import ComputePlanSpec

import register_datasamples


current_directory = Path(__file__)
assets_directory = current_directory.parent / "substra_assets"
algo_directory = assets_directory / "algo"
compute_plan_info_path = current_directory.parent / "compute_plan_info.json"
test_data_path = Path('/') / 'home' / 'user' / 'data' / 'test'


# Configuration of our connect platform
# --------------------------------------
#
# For this example, we decided to use two nodes, you will find a way to use three nodes at
# the end of the notebook.
# The default mode is `Debug` so everything runs locally.
# If you have access to a connect platform, be sure to configure your substra client and
# to login to the different nodes before executing the script and to set DEBUG to False in
# the cell bellow.
#
# If you deployed your own connect platform with the tutorial, you are all set. Otherwise be
# sure to change the **PROFILE_NAMES, NODES_IDS, ALGO_NODE_ID, ALGO_NODE_PROFILE**
# below:


DEBUG = False

# Change to 'docker' to test with one Docker container per task
# The 'docker' mode is much slower, but should be tested at
# least once before submitting to a deployed Connect platform (DEBUG = False)
# because it also tests that the Dockerfiles are correct
os.environ['DEBUG_SPAWNER'] = 'subprocess'

N_CENTERS = 3

PROFILE_NAMES = ["node_A", "node_B", "node_C"]
NODES_IDS = [
    "MyOrg1MSP",
    "MyOrg2MSP",
    "MyOrg3MSP",
]
ALGO_NODE_PROFILE = PROFILE_NAMES[0]
TEST_NODE = PROFILE_NAMES[2]

# Interaction with the platform
# -------------------------------
#
# In debug mode, there is no notion of nodes. To mimic it, we duplicate an original
# client for each of our profiles.

if DEBUG:
    client = substra.Client(debug=True)
    clients = {profile_name: client for profile_name in PROFILE_NAMES}
else:
    clients = {
        profile_name: substra.Client.from_config_file(profile_name)
        for profile_name in PROFILE_NAMES
    }

client = clients[ALGO_NODE_PROFILE]

# Generating the needed datasets
# ------------------------------
#
# There are multiple notions behind a dataset in substra which are explained`here
# <https://doc.substra.ai/concepts.html?highlight=data%20sample#dataset>`.
#
# Data Samples
# ^^^^^^^^^^^^
#
# We have three data samples :
# * train data samples on node-1
# * train data samples on node-2
# * test data samples on node-2
#
#
# Opener
# ^^^^^^
#
# In our case, the opener is the same for our train and test data samples. We will create
# a default dataset with the opener for simplicity reasons.
#
#
# Datasets
# ^^^^^^^^
#
# A dataset links one or multiple data samples to an opener for a training or a testing task.
# It also specifyes who can access those datasamples. On each node, we will create a dataset linked
# to our opener containing a train datasample and a test datasample (if it exists).
#
#
# Datasets Permissions
# ^^^^^^^^^^^^^^^^^^^^
#
# For detailed information about substra permissions, please read
# `this section <https://doc.substra.ai/permissions.html>`
# of the documentation.
#
# Do not worry about the permissions for now, we will detail the why and how at the
# end of this example.
#
#
# Debug mode specificities
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# As we said earlier, there is only one node in debug mode. But, we'll see later that our
# algorithm will save and update a file on each of our nodes at each step of our strategy.
# Adding a **DEBUG_OWNER** to our dataset metadata specifies to the algorithm working on
# this dataset that it can only access to the stored information when it works on this dataset.
# Otherwise the file is shared accross all task, which is not the case when working on a
# connect platform.

key_path = Path(__file__).parent / "data_sample_keys.json"

if DEBUG:

    key_path = Path(__file__).parent / "local_data_sample_keys.json"

    # If in debug mode, register the data samples
    # In deployed mode, the keys will already be on the platform and we give you directly the json
    register_datasamples.register_data_samples(clients, key_path=key_path)

print(client.list_dataset())
keys = json.loads(key_path.read_text())

# Creating the metric
# ----------------------
#
# You will find detailed information about the metric concept
# `here <https://doc.substra.ai/concepts.html#metric>`.
#
# In our case, we will use multiple metrics : an accuracy and a f1 score.
# As for the test dataset, we will host the metrics on the algorithm node i.e. node-2.

def register_metric(
    client: substra.Client,
    metric_folder: Path,
    metric_name: str,
    permissions: Permissions,
) -> str:
    """This function register a metric.
    In the specified folder, there must be a `metric.py`, a `description.md`
    and a `Dockerfile`.

    Args:
        client (substra.Client): the substra client for the metric registration
        metric_folder (Path): the path where all the needed files can be found
        metric_name (str): the associated name
        permissions (Permissions): the wanted permissions for the metric
    Returns:
        str: The registration key retrive by the client.
    """

    metric = MetricSpec(
        name=metric_name,
        description=metric_folder / "description.md",
        file=metric_folder / "metrics.zip",
        permissions=permissions,
    )

    METRICS_DOCKERFILE_FILES = [
        metric_folder / "metrics.py",
        metric_folder / "Dockerfile",
    ]

    archive_path = metric.file
    with zipfile.ZipFile(archive_path, "w") as z:
        for filepath in METRICS_DOCKERFILE_FILES:
            z.write(filepath, arcname=filepath.name)

    metric_key = client.add_metric(metric)

    return metric_key


auc_metric_key = register_metric(
    client,
    assets_directory / "metric",
    "auc",
    Permissions(public=True, authorized_ids=[]),
)
"""
keys[TEST_NODE]['accuracy'] = auc_metric_key

with open(key_path, "w") as f:
    json.dump(keys, f, indent=2)

tqdm.write("Assets keys have been saved to %s" % key_path.absolute())
"""

# SGD Algorithm
# -------------
#
# You will find detailed information about the Algorithm concept
# `here <https://doc.substra.ai/concepts.html#algo>`.
#
# The algorithm is implemented `here <./../assets/algo/algo.py>`_
#
# Storing information during the train and predict methods
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Our datasets are unbalanced in both volume and target distribution. To avoid over
# fitting on either of our nodes dataset's,
# we'll train our algorithm on little batches (32 samples) at each step.
#
# We created our batch thanks to a random draw without re-throw. A batch can be of a smaller
# size if there is not enough data remaining.
# After each epoch, we re-shuffled the dataset to generate new batches.
#
# In substra, you can't store this kind of information in your model. A workaround is to
# store it on each node.
#
# In the *train* and *predict* method of your Algo class (`here <./../assets/algo/algo.py>`_) you
# can access the *self.compute_plan_path* argument
# which points to a folder where you can write and read files.
#
# The batches indexes will be saved there so that we can access them at each step of our strategy.
#
# Using the algorithm from the previous step
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# As said earlier, our strategy relies on incremental learning. During the first step we
# will initialized an SGD regressor with the parameter **warm_start=True**. It allows
# incremental learning thanks to the **partial_fit** method. Moreover, we specify to
# the **partial_fit** method the possible values of our target as the third positional argument.
# Thus, even if all the occurrences of the target are not in the first batch, our algorithm
# will be able to take them into account latter on.
#
# The **train** method of the algo class, contains two extra arguments:
# * **models** : a list containing the resulting model of the latest training task.
# * **rank** : an integer which represents the the order of execution of our tasks (from 0 to n).


ALGO_DOCKERFILE_FILES: List[Path] = [
    algo_directory / "algo.py",
    algo_directory / "Dockerfile",
]


archive_path = algo_directory / "algo.zip"
with zipfile.ZipFile(archive_path, "w") as z:
    for filepath in ALGO_DOCKERFILE_FILES:
        z.write(filepath, arcname=filepath.name)

algo = AlgoSpec(
    name="Keras CNN",
    file=archive_path,
    description=algo_directory / "description.md",
    permissions={
        "public": True,
        "authorized_ids": list(),
    },
    category=substra.sdk.schemas.AlgoCategory.simple,
)

algo_key = client.add_algo(algo)

# Traintuple, testtuple and compute plan
# --------------------------------------
#
# For detailed information about those concepts please check the
# `doc <https://doc.substra.ai/concepts.html>`.
#
# Short story for our example :
# Each traintuple is identify by its traintuple_id. It defines a task where an algorithm
# (*algo_key*) is applied to train data samples (*train_data_sample_keys*) open thanks
# to a dataset (*data_manager_key*).
#
# A testtuple define the following task: applying an metric (**metric_key**) to the
# the result of the **predict** method of the model associated to the **traintuple_id**.
#
# The cyclic strategy is defined here.
#
# For each round, we want to train our algorithm on each node, and after each step, test
# the algorithm (on our test dataset hosted on node-2). This schema is defined thanks to
# the **in_models_ids** of each traintuple and the traintuple_id of the testtuples.
#
# For the traintuple, if nothing is specified int the **in_models_ids** field, it will be
# the first to be executed and the **models** parameter of the train methode of our
# algorithm will be empty. If we pass a traintuple_id to the **in_models_ids** parameter,
# the task associated to our traintuple will be executed right after the one linked to
# the traintuple_id and the **models** parameter will contain the resulting model of the task
# identified by the **in_models_ids** parameters.

# If you change this value, change it
# in fl_example/substra_assets/algo/algo.py too
N_ROUNDS = 5

traintuples = []
testtuples = []
previous_id = None

metric_keys = [auc_metric_key]

for idx in range(N_ROUNDS):
    # This is where you define training plan for each round
    # Here it is : train on node 1, test, train on node 2, test
    for node in PROFILE_NAMES:
        assets_keys = keys[node]
        traintuple = ComputePlanTraintupleSpec(
            algo_key=algo_key,
            data_manager_key=assets_keys["dataset"],
            train_data_sample_keys=assets_keys["train_data_samples"],
            traintuple_id=str(uuid.uuid4()),
            in_models_ids=[previous_id] if previous_id else [],
        )
        traintuples.append(traintuple)
        previous_id = traintuple.traintuple_id

    # Remove or put one testtuple every n epochs to make it faster
    testtuple = ComputePlanTesttupleSpec(
        metric_keys=metric_keys,
        traintuple_id=previous_id,
        test_data_sample_keys=keys[TEST_NODE]["test_data_samples"],
        data_manager_key=keys[TEST_NODE]["dataset"],
        metadata={
            'round': idx,
        }
    )

    testtuples.append(testtuple)

last_traintuple = traintuple
compute_plan_spec = ComputePlanSpec(
    traintuples=traintuples,
    testtuples=testtuples,
    clean_models=True,
)

print("Adding compute plan")
compute_plan = client.add_compute_plan(compute_plan_spec)
print("Adding compute plan - done")

compute_plan_info = compute_plan_spec.dict(exclude_none=False, by_alias=True)
# print(compute_plan_info)

compute_plan_info.update({"key": compute_plan.key})
compute_plan_info_path.write_text(json.dumps(compute_plan_info))

tqdm.write(
    "Compute Plan keys have been saved to %s" % compute_plan_info_path.absolute()
)

# Check your compute plan progresses
# ----------------------------------
#
# You can  get the number of tasks done and the status (waiting, doing, failed, done)
# of your compute plan. This is particularly useful if the CP has failed.

client.get_compute_plan(compute_plan.key)

# Displaying Scores
# -----------------
#
# When you are connected to a remote connect instance, you can access your node to follow
# the evolution of your compute plan. For example:
import time

tqdm.write("Waiting for the compute plan to finish to get the performances.")

submitted_testtuples = client.list_testtuple(
    filters=[f'testtuple:compute_plan_key:{compute_plan_info["key"]}']
)

submitted_testtuples = sorted(submitted_testtuples, key=lambda x: x.rank)

for submitted_testtuple in tqdm(submitted_testtuples):
    while submitted_testtuple.test.perfs is None:
        time.sleep(0.5)
        submitted_testtuple = client.get_testtuple(submitted_testtuple.key)
    perfs = submitted_testtuple.test.perfs
    perfs["AUC"] = perfs.pop(auc_metric_key)

    tqdm.write("rank: %s, round: %s, node: %s, perf: %s" %
               (submitted_testtuple.rank,
                submitted_testtuple.metadata['round'],
                submitted_testtuple.worker,
                perfs)
    )

# Make predictions on the test set
# -----------------------------------

tqdm.write("Create the Kaggle submission file.")

from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd

# get the last traintuple of the compute plan
traintuple = client.get_traintuple(last_traintuple.traintuple_id)

# Download the associated model
model_path = Path(__file__).resolve().parents[1] / 'out'
model_filename = f'model_{traintuple.train.models[0].key}'
client.download_model_from_traintuple(traintuple.key, folder=model_path)

# SPECIFIC TO YOUR ALGO
# -----------------------
#
# Here we are loading the model that your algo produced
# so if you change the algo, for example if you use PyTorch,
# then you might need to change this part as well.

# Load the model and create predictions: this code depends on the algo code
model = keras.models.load_model(str(model_path / model_filename))

image_size = (180, 180)
batch_size = 32
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=test_data_path,
    labels="inferred",
    label_mode="binary",
    shuffle=False,
    seed=0,
    image_size=image_size,
    batch_size=batch_size,
)

predictions = np.array([])
labels = np.array([])
for x, y in test_ds:
    predictions = np.concatenate([predictions, model.predict(x).ravel()])
    labels = np.concatenate([labels, y.numpy().ravel()])
file_paths = test_ds.file_paths

df_submission = pd.DataFrame(data={
    'file_paths': file_paths,
    'predictions': predictions
})
df_submission["file_paths"] = df_submission["file_paths"].apply(
    lambda x: x.replace("/home/user/data/test", "/data/challenges_data/test"))

submission_filepath = Path(
    __file__).resolve().parents[1] / 'out' / 'df_submission.csv'
df_submission.to_csv(submission_filepath, index=False)

print(df_submission.head())

tqdm.write(f"Created the file at {submission_filepath}")

# Delete the downloaded model
(model_path / model_filename).unlink()

# Permission
# -----------
#
# You can fin a detailed documentation of the permission concept
# `here <https://doc.substra.ai/permissions.html?highlight=permissions>`
#
# As you may have noticed, for our example permissions are defined for:
# * datasets
# * objectivs
# * algorithms
#
# Permissions of traintuples are inherited and defined as the intersections of
# the permissions between:
# * the algorithm
# * the dataset
# * the permissions of the parent task
#
# Permissions of testtuples are inherited and defined as the intersections of
# the permissions between:
# * the algorithm
# * the metric
# * the dataset
# * the permissions of the parent task
#
#
# At first sight, we could think that each train dataset must be accessible from
# his node, and the algorithm's node each train dataset must be accessible from
# his node, the metric's node and the algorithm's node
#
# But as for traintuples, the permissions inherit from the parent task, each train
# dataset's node needs access to all the previous train dataset's nodes.
#
# For the testtuples it is quite different as a test task can't be a parent of an
# other task. Each dataset node involved in the training of the algorithm must grant
# access to the nodes's test dataset But the test dataset only need to be accessible
# by the node of both algorithm and metrics.
#
# In other word, it means that when a user puts data into substra platform, they decide
# which other person can access a model trained on his data.
#
#
# If you are using your own connect platform
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# * configure and connect your substra client to the third node
# * add the profile name and the node id to PROFILE_NAMES and NODES_IDS in this file.
#
