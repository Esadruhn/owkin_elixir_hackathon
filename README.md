# Elixir hackathon - project 30:

Welcome to project 30 !

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

You can download the challenge data from: TODO

## Challenge and FL example
