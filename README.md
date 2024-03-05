# ICESAT2 Track Analysis

- [ICESAT2 Track Analysis](#icesat2-track-analysis)
  - [Installation for Developers](#installation-for-developers)
  - [Installing on Oscar (Deprecated)](#installing-on-oscar-deprecated)
  - [Conda Configuration (Deprecated)](#conda-configuration-deprecated)
      - [`.condarc`](#condarc)
      - [`pkgs_dirs`](#pkgs_dirs)
      - [`envs_dirs`](#envs_dirs)
      - [Environment Variables](#environment-variables)
  - [Command line interface](#command-line-interface)
  - [Sample workflow](#sample-workflow)

## Installation for Developers

Prerequisites:
- A POSIX-compatible system (Linux or macOS)
- Python 3.9 (run `python --version` to check that your version of python is correct)
- MPI (e.g. from `brew install open-mpi` on macOS)
- HDF5 (e.g. from `brew install hdf5` on macOS)

> [!IMPORTANT]  
> Windows is not supported for development work – use [WSL](https://learn.microsoft.com/en-us/windows/wsl/) on Windows hosts

Installation:
> [!NOTE]
> For testing purposes this repository uses Git Large File Storage (LFS) to handle large data files. If you want to clone the repository with the LFS files, make sure you have Git LFS installed on your system. You can download it from [here](https://git-lfs.github.com/). After installing, you can clone the repository as usual with `git clone`. Git LFS files will be downloaded automatically. If you've already cloned the repository, you can download the LFS files with `git lfs pull`.


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

## Conda Configuration (Deprecated)

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

## Command line interface

The `icesat2waves` package comes with a command-line interface (CLI) that facilitates interaction with the package directly from your terminal. This can be particularly useful for scripting and automation. You can access the help documentation for the CLI by running the following command:

```shell
icesat2waves --help
```

As suggested in the help, to run a specific command run `icesat2waves [OPTIONS] COMMAND [ARGS]...`.  To view help on running a command, run `icesat2waves COMMAND --help`. For example, to get help about the `load-file` command, you may issue `icesat2waves load-file --help` to get the following output:

```shell
(.venv) $ icesat2waves load-file --help
Usage: icesat2waves load-file [OPTIONS]

  Open an ICEsat2 tbeam_stats.pyrack, apply filters and corrections, and
  output smoothed photon heights on a regular grid in an .nc file.

Options:
  --track-name TEXT         [required]
  --batch-key TEXT          [required]
  --id-flag / --no-id-flag  [default: id-flag]
  --output-dir TEXT         [required]
  --verbose / --no-verbose  [default: no-verbose]
  --help                    Show this message and exit.

```

## Sample workflow
Below is a sample workflow that leverages the included CLI.
1. **Load single file**
```shell
icesat2waves load-file --track-name 20190502052058_05180312_005_01 --batch-key SH_testSLsinglefile2 --output-dir ./output
```


2. **Make spectra from downloaded data**
```shell
icesat2waves make-spectra --track-name SH_20190502_05180312 --batch-key SH_testSLsinglefile2 --output-dir ./output
```

3. **Plot spectra**
```shell
icesat2waves plot-spectra --track-name SH_20190502_05180312 --batch-key SH_testSLsinglefile2 --output-dir ./output
```


4. **Build IOWAGA priors**
```shell
icesat2waves make-iowaga-threads-prior --track-name SH_20190502_05180312 --batch-key SH_testSLsinglefile2 --output-dir ./output
```


5. **Build angles**

```shell
icesat2waves make-b04-angle --track-name SH_20190502_05180312 --batch-key SH_testSLsinglefile2 --output-dir ./output
```



6. **Define and plot angles**
```shell
icesat2waves define-angle --track-name SH_20190502_05180312 --batch-key SH_testSLsinglefile2 --output-dir ./output
```


7. **Make corrections and separations**
```shell
icesat2waves correct-separate --track-name SH_20190502_05180312 --batch-key SH_testSLsinglefile2 --output-dir ./output
```
