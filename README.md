# ICESAT2 Track Analysis

## Installation for Developers

Prerequisites:
- A POSIX-compatible system (Linux or macOS)
- Python 3.9 (run `python --version` to check that your version of python is correct)
- MPI (e.g. from `brew install open-mpi` on macOS)
- HDF5 (e.g. from `brew install hdf5` on macOS)

> [!IMPORTANT]  
> Windows is not supported for development work – use [WSL](https://learn.microsoft.com/en-us/windows/wsl/) on Windows hosts

Installation:
- Clone the repository:
  - Navigate to https://github.com/brown-ccv/icesat2-tracks
  - Click the "<> Code" button and select a method to clone the repository, then follow the prompts
- Open a shell (bash, zsh) in the repository working directory
- Create a new virtual environment named `.venv`:
  ```shell
  python -m venv .venv
  ```
- Activate the environment
    ```shell
    source ".venv/bin/activate"
    ```
- Upgrade pip
  ```shell
  pip install --upgrade pip
  ```
- Install or update the environment with the dependencies for this project:
  ```shell
  pip install --upgrade --editable ".[dev]"
  ```
  > You may need to set the value of the `HDF5_DIR` environment variable to install some of the dependencies, especially when installing on macOS. 
  > 
  > For Apple Silicon (M-Series) CPUs:
  > ```shell
  > export HDF5_DIR="/opt/homebrew/opt/hdf5"
  > pip install --upgrade --editable ".[dev]"
  > ```
  >
  > For Intel CPUs:
  > ```shell
  > export HDF5_DIR="/usr/local/opt/hdf5"
  > pip install --upgrade --editable ".[dev]"
  > ```

- Check the module `icesat2_tracks` is available by loading the module:
  ```shell
  python -c "import icesat2_tracks; print(icesat2_tracks.__version__)"
  ```

## Installing on Oscar (Deprecated)

If any of these commands fail, check the conda configuration (listed below) before retrying for possible fixes.

Load a conda module.

```shell
module load miniconda/23.1.0
```

Follow any prompts from the module load command, like running the following line:
```shell
source /gpfs/runtime/opt/miniconda/23.1.0/etc/profile.d/conda.sh
```

Create a new environment using:
```shell
conda create --name "2021-icesat2-tracks"
```

Activate the environment using:
```shell
conda activate "2021-icesat2-tracks"
```

Install or update the packages in the environment with those listed in the `environment.yml` file using:
```shell
conda env update --file environment.yml
```

(You can create and install dependencies in the environment in a single command using:
```shell
conda env create --name "2021-icesat2-tracks" --file environment.yml
```
... but this has more steps and is thus more likely to fail. Since the installation step takes a long period of time, it is recommended to use the separate commands instead.)

## Conda Configuration

Conda draws its configuration from multiple places, and will behave differently when the configuration is different, even when using the same `environment.yml` file.

#### `.condarc`

The `.condarc` file in your home directory sets your conda configuration. If the file doesn't exist, you can create it with:
```shell
touch ~/.condarc
```

#### `pkgs_dirs`

`pkgs_dirs` is the location where conda downloads package files from registries like `anaconda.org`. 

If you use the defaults, when trying to install packages you may get a warning like:
```
WARNING conda.lock:touch(51): Failed to create lock, do not run conda in parallel processes [errno 13]
...
ERROR   Could not open file /gpfs/runtime/opt/miniconda/4.12.0/pkgs/cache/b63425f9.json
```

In this case, you might be trying to download packages to the global directory `/gpfs/runtime/opt/miniconda/4.12.0/pkgs` where you have no write-permissions, rather than your home directory where you have write-permissions.

View the conda configuration:
```shell
conda config --show
```

Check that the `pkgs_dirs` setting points to a location in your home directory:
```yaml
pkgs_dirs:
  - /users/username/anaconda/pkg
```

If it doesn't, update this using:
```shell
conda config --add pkgs_dirs ~/anaconda/pkg
```

(Use `--remove` instead of `--add` to remove an entry.)

#### `envs_dirs`

`envs_dirs` is the location where there is a separate directory per environment containing the installed packages.

View the conda configuration:
```shell
conda config --show
```

Check that the `envs_dirs` setting to a location in your home directory:
```yaml
envs_dirs:
  - /users/username/anaconda/envs
```

... and update this using:
```shell
conda config --add envs_dirs ~/anaconda/envs
```

(Use `--remove` instead of `--add` to remove an entry.)

Always re-check the configuration after running the `conda config` command. 

#### Environment Variables

Note that modules (like `miniconda/23.1.0`) set environment variables like `CONDA_ENVS_PATH` which override the conda config. 

You might view the conda config and see the following entries:
```yaml
envs_dirs:
  - /users/username/anaconda
  - /users/username/anaconda/envs
```

If you try to run 
```shell
conda config --remove envs_dirs ~/anaconda
```
... you'll get a warning:
```
'envs_dirs': '/users/username/anaconda' is not in the 'envs_dirs' key of the config file
```

... and find that the value is still there when you rerun `conda config --show`:
```yaml
envs_dirs:
  - /users/username/anaconda     # <-- still here!
  - /users/username/anaconda/envs
```

The value might have been silently set by the `module load` command using an environment variable. 

Check for environment variables by running:
```shell
$ printenv | grep ^CONDA_
CONDA_SHLVL=0
CONDA_EXE=/gpfs/runtime/opt/miniconda/23.1.0/bin/conda
CONDA_ENVS_PATH=~/anaconda  # <- this is the offending variable
CONDA_PYTHON_EXE=/gpfs/runtime/opt/miniconda/23.1.0/bin/python
```

To unset a value like `CONDA_ENVS_PATH` use:
```shell
unset CONDA_ENVS_PATH
```

... then check that rerun `conda config --show` no longer shows the has modified the conda config to match the values you wanted:
```yaml
envs_dirs:
  - /users/username/anaconda/envs
```
