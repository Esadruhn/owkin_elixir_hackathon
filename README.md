# Elixir hackathon - project 30:

Welcome to project 30 !

There is [a Kaggle challenge](https://www.kaggle.com/c/biohackathon21-project-30/overview) for this hackathon.

- [Elixir hackathon - project 30:](#elixir-hackathon---project-30)
  - [Dev environment](#dev-environment)
    - [Use a VM](#use-a-vm)
      - [Use jupyter notebooks](#use-jupyter-notebooks)
      - [**Advanced users**: define an SSH config](#advanced-users-define-an-ssh-config)
    - [Use your own computer](#use-your-own-computer)
  - [ML challenge](#ml-challenge)
  - [Share the code](#share-the-code)

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

## Share the code

Create your own branch and a PR on this repository. Send us your Github username so that we can add you to the repo.

To create a new branch: `git switch -c name_of_your_branch`
