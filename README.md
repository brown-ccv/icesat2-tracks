# ICESAT2 Track Analysis

- [ICESAT2 Track Analysis](#icesat2-track-analysis)
  - [Installation for Developers](#installation-for-developers)
  - [Command line interface](#command-line-interface)
  - [Sample workflow](#sample-workflow)

## Installation for Developers

Prerequisites:
- A POSIX-compatible system (Linux or macOS)
- Python 3.10 (run `python --version` to check that your version of python is correct)
- MPI (e.g. from `brew install open-mpi` on macOS)
- HDF5 (e.g. from `brew install hdf5` on macOS)

> [!IMPORTANT]  
> Windows is not supported for development work – use [WSL](https://learn.microsoft.com/en-us/windows/wsl/) on Windows hosts

Installation:
> [!NOTE]
> For testing purposes this repository uses Git Large File Storage (LFS) to handle large data files. If you want to clone the repository with the LFS files, make sure you have Git LFS installed on your system. You can download it from [here](https://git-lfs.github.com/). After installing, you can clone the repository as usual with `git clone`. Git LFS files will be downloaded automatically. If you've already cloned the repository, you can download the LFS files with `git lfs pull`.


- Clone the repository:
  - Navigate to https://github.com/brown-ccv/icesat2waves
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

- Check the module `icesat2waves` is available by loading the module:
  ```shell
  python -c "import icesat2waves; print(icesat2waves.__version__)"
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
