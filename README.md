# Elixir hackathon - project 30

Welcome to project 30 !

There is [a Kaggle challenge](https://www.kaggle.com/t/b2b76461caad4eeea06b7e0c3ff26712) for this hackathon.

- [Elixir hackathon - project 30](#elixir-hackathon---project-30)
  - [Dev environment](#dev-environment)
    - [Use a VM](#use-a-vm)
      - [Use jupyter notebooks](#use-jupyter-notebooks)
      - [**Advanced users**: define an SSH config](#advanced-users-define-an-ssh-config)
    - [Use your own computer](#use-your-own-computer)
  - [ML challenge](#ml-challenge)
  - [FL challenge](#fl-challenge)
    - [Owkin Connect](#owkin-connect)
    - [Challenge](#challenge)
    - [Suggested workflow](#suggested-workflow)
  - [Share the code](#share-the-code)
- [Connect platform](#connect-platform)
  - [URLs](#urls)
  - [Example](#example)

## Dev environment

For the development, we can either use your own machine or a VM we setup for you.

### Use a VM

The VMs technical specifications are:
- a P100 GPU
- 60Gb RAM

and software-wise:
- Debian 10
- Python 3.7

You can install any package you want on the VMs, they are yours for the duration of the hackathon
and will be deleted afterwards.

To connect to the VM, ask us for the IP and private SSH key. Then you can connect to the VM with the following command:
```sh
ssh user@VM_IP -i path/to/private/key
```

On the VM, there is already a virtual environment that you can use: `source /home/user/elixenv/bin/activate` with the dependencies installed.
You can also create your own virtual environment. In that case, you need to install the substra and substratools packages:

```sh
cd owkin_elixir_hackathon/dependencies
python3 -m pip install *.whl
```

The `/home/user/owkin_elixir_hackathon` directory is a clone of the Github repository: https://github.com/Esadruhn/owkin_elixir_hackathon

#### Use jupyter notebooks

1. Create a SSH tunnel when connecting to the VM

Add `-L localhost:10117:localhost:10117` to the ssh command:

`ssh user@VM_IP -i path/to/private/key -L localhost:10117:localhost:10117`

replace `VM_IP` by the IP of the VM, and the `path/to/private/key` by the path to the private key.

You can use any port, 10117 is an example.

2. Install Jupyter on the VM and create the kernel

When you use the jupyter notebook, you must be careful it uses the right Python: you want it to use the one from your virtual environment. For this,
you need to create what we call a **jupyter kernel**.
If you use the virtual environment that is already present:

 ```sh
source /home/user/elixenv/bin/activate
pip install jupyter
pip install ipykernel
python -m ipykernel install --user --name=elixirkernel
```

when you create a notebook, you’ll need to select this kernel.

3. Launch Jupyter on the right port

`jupyter notebook --no-browser --port=10117`

On your computer go to the URL `localhost:10117`

#### **Advanced users**: define an SSH config

For example in my ssh config file (~/.ssh/config) I add the following lines:

```yaml
Host elixir_instance_2
	HostName REPLACE_BY_THE_VM_IP
	User user
	ForwardAgent yes
	IdentityFile ~/.ssh/id_rsa_hackathon
	LocalForward 10117 127.0.0.1:10117
```
then I connect to the VM with `ssh elixir_instance_2`

### Use your own computer

If you use your own machine, you need the following:
- work on Linux or Mac, we do not support Windows
- have Docker installed and configured, be comfortable with Docker running as root
- Python 3.7 or higher

You need to install the substra and substratools packages:

```sh
git clone https://github.com/Esadruhn/owkin_elixir_hackathon.git
cd owkin_elixir_hackathon/dependencies
python3 -m pip install *.whl
```

You can download the challenge data from the Kaggle platform.

## ML challenge

Data: `/home/user/data/ML/train`

The data is in two folders, `target_0` and `target_1`. Images in the `target_0` folder have the label 0 (not tumoral), images
in the `target_1` folder have the label 1 (tumoral).

Once you have trained your model, you need to perform inference on the test data in `/home/user/data/ML/test`, create a CSV file and the submit the results to the [Kaggle platform](https://www.kaggle.com/t/b2b76461caad4eeea06b7e0c3ff26712)
To create the CSV file you can use the following code:

```
df_submission = pd.DataFrame(data={'file_paths': file_paths, 'predictions': predictions})
df_submission["file_paths"] = df_submission["file_paths"].apply(lambda x: x.replace("/home/user/data/test","/data/challenges_data/test"))
```
What the second line does is that it rewrites the filepath to be in the same format that Kaggle expects.

To transfer the submission file from the instance to your machine, you can do:
`scp -i path/to/private/key user@VM_IP:/home/user/path/to/submission/submission_file.csv .`

## FL challenge

### Owkin Connect
For the FL challenge, the Owkin Connect (also called Substra) software will be used. The software is made up of a python library and a deployed platform on the cloud.
Documentation for the software can be found in the [substra_materials](substra_materials) folder. In particular there is:
- a [api_reference](api_reference) folder with the API reference and the description of the main objects.
- a [quick start notebook](substra_materials/titanic_example/quick_start.ipynb) that shows a simple use of Connect on a single dataset (titanic dataset).
- the [concept file](substra_materials/concepts.md) that explains the main concepts of Substra.

### Challenge

For the FL challenge we consider three datasets, that are distribued on 3 nodes: A, B and C.
The train data used for the ML challenge correspond the to node A data. The node B and node C are new.

When you launch FL computations, you can run them locally (debug mode) or in the deployed mode. We advise you to start little experiments locally first. To use the Connect locally you need to put `DEBUG=True` in your main.py file.
An end to end example is provided in the [fl_example](fl_example) folder.

### Suggested workflow

- Get the latest updates from master

To avoid any conflict, you should put all the code you changed / created in your own folder.

```sh
git pull
git pull origin master
```

- Start by looking at the [quick start notebook](substra_materials/titanic_example/quick_start.ipynb). Understand it and run it in both debug and remote mode.
- Then look at the [fl_example](fl_example). Make sure you can run the `main.py` file in debug mode. Once it's working you can modify the `algo.py` file to use your own algorithm.

The only files you need to change are the [algo.py](./fl_example/substra_assets/algo/algo.py) for the algo code and the [Dockerfile](./fl_example/substra_assets/algo/Dockerfile) for the requirements (to install pytorch instead of tensorflow for example).

## Share the code

Create your own branch and a PR on this repository. Send us your Github username so that we can add you to the repo.

To create a new branch: `git checkout -b name_of_your_branch`, when you push for the first time git will tell you to add a --set-upstream argument, copy paste the command it gives you.

Afterwards, you can commit and push normally on your branch.

Git now requires an access token to be able to push your changes:

https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token


# Connect platform

DO NOT REGISTER THE DATA AGAIN ON THE CONNECT PLATFORM, IT IS ALREADY ON THE NODES.

Before running on the platform, run in debug mode - docker (and not subprocess) to check the Dockerfiles and algos are right.

There are 3 nodes:
- orga-1MSP
- orga-2MSP
- orga-3MSP

that represent 3 different hospitals, each with their own dataset.

Login to the backend from the CLI:

```sh
substra config --profile $PROFILE_NAME $NODE_BACKEND_URL
substra login --profile $PROFILE_NAME --username $NODE_USERNAME --password $NODE_PASSWORD
```

Define one profile name for each node, you'll have to give it every time you execute a command on that node. You only need to have access to
one node to register your model.

Then register your model and launch the training in Python scripts.

Use the frontend to visualize the assets on each node: go to the URL in your browser.

## URLs

|Node id            |Profile name |Frontend URL                               |Backend URL                                         |   |   |
|-------------------|-------------|-------------------------------------------|----------------------------------------------------|---|---|
|orga-1MSP          |node_A       |https://connect-org-1.1.elixir.owkin.biz   |https://substra-backend-org-1.1.elixir.owkin.biz    |   |   |
|orga-2MSP          |node_B       |https://connect-org-2.2.elixir.owkin.biz   |https://substra-backend-org-2.2.elixir.owkin.biz    |   |   |
|orga-3MSP          |node_C       |https://connect-org-3.3.elixir.owkin.biz   |https://substra-backend-org-3.3.elixir.owkin.biz    |   |   |

## Example

I login to the backend of the node 1 from the CLI:

```sh
substra config --profile node_A https://substra-backend-org-1.1.elixir.owkin.biz
substra login --profile node_A --username my_user_node_1 --password my_user_node_1
```

Then in a Python script I interact with the backend:

```python
import substra

client = substra.Client.from_config_file(profile_name='node_A')  # profile_name=$PROFILE_NAME from the CLI

# get assets from the platform
dataset = client.get_dataset(dataset_key)
# register assets to the platform
client.add_compute_plan(
    # ...
)
```

and on the frontend I visualize the assets and the progression of the compute plan