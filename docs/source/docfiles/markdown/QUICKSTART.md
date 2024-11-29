# Quickstart

This section shows how to setup a local FarmVibes.AI cluster and client on your
computer and execute a simple workflow. If you want to set up a remote
instance of FarmVibes.AI, please refer to our [remote setup guide](./AKS.md).

## Requirements

If you need to setup a new machine that fits all requirements detailed below and comes with all the
necessary software installed, follow the steps in this [document](./VM-SETUP.md)
to create it in Azure.

In order to run FarmVibes.AI cluster, you need the following:

* A Linux machine (Ubuntu 20.04 distro is highly recommended), with at least
16 GB of memory (32 GB, recommended), 4 CPU cores, and 512 GB of storage
(2 TB, recommended).

* The following software needs to be installed in the machine:

  * [Git](https://www.atlassian.com/git/tutorials/install-git#linux) to download
    the repository. If you already have access to the source code, then Git is
    not required.

  * [Docker](https://docs.docker.com/engine/install/ubuntu/). Make sure you can
    run the docker client without running `sudo` by adding your user account to
    the `docker` group (which might require a logout/login when adding oneself
    to the docker group).

  * [Curl](https://curl.se/). FarmVibes.AI installer requires curl to install
    additional software for FarmVibes.AI cluster management.

  * [Python 3.8+](https://www.python.org/downloads/). FarmVibes.AI provides
    a python client to simplify the consumption of results and parameters
    providing process.

For your assistance, we have a script that installs all the necessary dependencies in
your machine. [More information can be found below](#optional-installing-software-dependencies).

## Clone the repository

Choose a folder of your preference and clone the FarmVibes.AI repo.

```shell
git clone https://github.com/microsoft/farmvibes-ai.git
```

Observe you can clone FarmVibes.AI using HTTP or SSH (see [Cloning
Repos](https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories)).

## Optional: Installing software dependencies

A script that installs all the [required dependencies](#requirements) if they are not already installed. The script
assumes that your user has `sudo` permission on your computer and an Ubuntu installation. If this is
the case, all dependencies can be installed by running (from the root of the repository):

```shell
bash ./resources/vm/setup_farmvibes_ai_vm.sh
```

You might needed to restart your shell session once the script finishes.

## Install the FarmVibes.AI cluster

With python3.8+ and pip installed on your machine, please install
FarmVibes.AI `vibe_core` package, with the following command:

```shell
pip install ./src/vibe_core
```

With the core package installed, let's setup the FarmVibes.AI cluster. Please, make sure
to run this command in the project root folder. The installation in a local environment
may take up to 1 hour to complete.

```shell
farmvibes-ai local setup
```

When the installation process finishes, you should see a message similar the
following.

```shell
FarmVibes.AI REST API is running at http://192.168.49.2:30000
```

Note that the address `http://192.168.49.2:30000` depends on docker network
configuration and may be different on your setup.

For more information about the installation script, its options and arguments, make sure
to run it with the `--help` or `-h` flags:

```shell
farmvibes-ai local --help
```

## Check FarmVibes.AI Installation

If the installation was successful, you should be able to run the hello world test with:

```shell
python -m vibe_core.farmvibes_ai_hello_world
```

You should see an output listing the existing workflows on FarmVibes.AI and the
helloworld workflow output.

If you see the message `Successfully executed helloworld workflow.`, it means
that FarmVibes.AI and the python client are working properly.

For more information on how to execute workflows, please take a look at our
[client guide](./CLIENT.md). For information on any issues running the cluster, including on
how to re-start it after a machine reboot, take a look at our [troubleshoot guide](./TROUBLESHOOTING.md).
If you do not find the information you are looking for, please reach out to the team by opening
an issue on our [GitHub repository](https://github.com/microsoft/farmvibes-ai/issues) or browsing
through our [known issues](https://github.com/microsoft/farmvibes-ai/labels/known%20issues).
