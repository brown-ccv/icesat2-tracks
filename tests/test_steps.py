#!/usr/bin/env python
"""
This module contains a test suite for the following commands:
      - name: Step 1 B01_SL_load_single_file
        cmd: load-file --track-name 20190502052058_05180312_005_01 --batch-key SH_testSLsinglefile2 --output-dir ./work
      - name: second step make_spectra
        cmd: make-spectra --track-name SH_20190502_05180312 --batch-key SH_testSLsinglefile2 --output-dir ./work
      - name: third step plot_spectra
        cmd: plot-spectra --track-name SH_20190502_05180312 --batch-key SH_testSLsinglefile2 --output-dir ./work
      - name: fourth step IOWAGA threads 
        cmd:  make-iowaga-threads-prior --track-name SH_20190502_05180312 --batch-key SH_testSLsinglefile2 --output-dir ./work
      - name: fifth step B04_angle
        cmd: make-b04-angle --track-name SH_20190502_05180312 --batch-key SH_testSLsinglefile2 --output-dir ./work
      - name: sixth step B04_define_angle
        cmd: define-angle --track-name SH_20190502_05180312 --batch-key SH_testSLsinglefile2 --output-dir ./work
      - name: seventh step B06_correct_separate
        cmd: correct-separate --track-name SH_20190502_05180312 --batch-key SH_testSLsinglefile2 --output-dir ./work

To this end, it sets up a temporary directory within the `tests/` directory with subdirectories containing the required input data for each step. The tests are run in parallel using the `xdist` plugin, with each worker having its own copy of the input data. This allows the tests to modify the input data without affecting other workers. The `setup_module` function (fixture) is responsible for creating the temporary directory and setting up the input data for each step. It also prepares the target directories for each step by extracting the necessary files from tarballs in the `tests/testdata` directory and organizing them in the appropriate directory structure.

After the tests are completed, the `teardown_module` function is called to clean up the temporary directory.

The metadata for each script is stored in the `scripts` list, and the paths to the files that should be produced by the scripts are stored in the `paths` list. The `run_test` function is used to run the scripts and check whether the expected files were produced. If the files were produced, the test passes. If not, the test fails.

With the input data in place, pytest runs each of test_stepX functions, which call the `run_test(script, paths_to_check)` to run the scripts and check whether the expected files were produced. If the files were produced, the test passes. If not, the test fails.

To create similar tests for other steps, you can use the `create_script` function to create the script and the `makepathlist` function to create the paths to the files that should be produced by the script. The `create_script` function takes the name of the script as an argument and returns a list containing the command to run the script using the `subprocess.run` function. The `makepathlist` function takes a directory and a list of file names as arguments and returns a list of paths to the files in the directory. If the test requires input data, you can use the `extract_tarball` function to extract the input data from a tarball and organize it in the appropriate directory structure within the `setup_module` fixture.
"""

from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import tarfile
from tempfile import mkdtemp


def checkpaths(paths):
    result = [Path(pth).is_file() for pth in paths]
    print("\n")
    # report which files are missing (in red) and which are present (in green)
    for pth, res in zip(paths, result):
        if res:
            print(f"\033[92m{pth} is present\033[0m")
        else:
            print(f"\033[91m{pth} is missing\033[0m")
    return all(result)


def get_all_filenames(directory):
    """
    Get a list of all file names in a directory.
    """
    return [p.name for p in Path(directory).rglob("*")]


def check_file_exists(directory, prefix, stepnum: int = 4):
    """
    This is needed because step 4 produces files with different names even when using the same input data.
    """
    # Get a list of all files in the directory
    files = get_all_filenames(targetdirs[str(stepnum)] / directory)
    # Check if there is a file with the specified prefix
    file_exists = any(file.startswith(prefix) for file in files)
    return file_exists


def delete_pdf_files(directory):
    files = [file for file in Path(directory).iterdir() if file.suffix == ".pdf"]
    delete_files(files)


def delete_files(file_paths):
    for file_path in file_paths:
        delete_file(file_path)


def delete_file(file_path):
    path = Path(file_path)
    if path.exists():
        path.unlink()


def getoutputdir(script):
    outputdir = script.index("--output-dir") + 1
    return script[outputdir]


def extract_tarball(outputdir, tarball_path):
    tar = tarfile.open(Path(tarball_path))
    tar.extractall(Path(outputdir), filter="data")
    tar.close()


def create_script(script_name, track_name="SH_20190502_05180312"):
    head = ["python"]
    tail = [
        "--track-name",
        track_name,
        "--batch-key",
        "SH_testSLsinglefile2",
        "--output-dir",
    ]
    base_path = "src/icesat2waves/analysis_db/"
    return head + [f"{base_path}{script_name}.py"] + tail


def run_test(script, paths, delete_paths=True, suppress_output=True):
    # configuration
    outputdir = getoutputdir(script)

    # update paths to check
    paths = [Path(outputdir, pth) for pth in paths]

    if delete_paths:
        delete_files(paths)

    kwargs = {"check": True}
    if suppress_output:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL

    # run the script
    subprocess.run(script, **kwargs)

    # run the tests
    result = checkpaths(paths)

    return result


def makepathlist(dir, files):
    return [Path(dir, f) for f in files]


# The `scriptx` variables are the scripts to be tested. The `pathsx` variables are the paths to the files that should be produced by the scripts. The scripts are run and the paths are checked to see whether the expected files were produced. If the files were produced, the test passes. If not, the test fails.

# Script 1 is different from the others because it doesn't have any input data and uses a different track name.
script1 = create_script(
    script_name="B01_SL_load_single_file", track_name="20190502052058_05180312_005_01"
)
paths1 = [
    "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/B01_track.png.png",
    "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/B01b_ATL06_corrected.png.png",
    "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/B01b_beam_statistics.png.png",
    "work/SH_testSLsinglefile2/A01b_ID/A01b_ID_SH_20190502_05180312.json",
    "work/SH_testSLsinglefile2/B01_regrid/SH_20190502_05180312_B01_binned.h5",
]

script_names_2_to_7 = [
    "B02_make_spectra_gFT",
    "B03_plot_spectra_ov",
    "A02c_IOWAGA_thredds_prior",
    "B04_angle",
    "B05_define_angle",
    "B06_correct_separate_var",
]

script2, script3, script4, script5, script6, script7 = [
    create_script(name) for name in script_names_2_to_7
]

# Define the paths for the second script
_root = "work/SH_testSLsinglefile2/B02_spectra/"
_paths2 = [
    "B02_SH_20190502_05180312_params.h5",
    "B02_SH_20190502_05180312_gFT_x.nc",
    "B02_SH_20190502_05180312_gFT_k.nc",
    "B02_SH_20190502_05180312_FFT.nc",
]
paths2 = makepathlist(_root, _paths2)

# Define the paths for the third script
_root = "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312"
_paths3 = [
    "B03_specs_L25000.0.png",
    "B03_specs_coord_check.png",
    "B03_success.json",
]
paths3 = makepathlist(_root, _paths3)


paths4 = [
    "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/A02_SH_2_hindcast_data.pdf",
    "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/A02_SH_2_hindcast_prior.pdf",
]  # deterministic paths
dir4, prefix4 = (
    "work/SH_testSLsinglefile2/A02_prior/",
    "A02_SH_20190502_05180312_hindcast",
)  # stochastic paths

# Define the paths for the fifth script
_plotroot = "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312"
_plotnames = [
    "B04_success.json",
    "B04_prior_angle.png",
    "B04_marginal_distributions.pdf",
    "B04_data_avail.pdf",
]
_workroot = "work/SH_testSLsinglefile2/B04_angle"
_worknames = [
    "B04_SH_20190502_05180312_res_table.h5",
    "B04_SH_20190502_05180312_marginals.nc",
]
paths5 = makepathlist(_plotroot, _plotnames) + makepathlist(_workroot, _worknames)

# Define the paths for the sixth script
base_path = "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312"
paths6 = [
    f"{base_path}/B05_angle/B05_weighted_marginals_x{i}.pdf"
    for i in range(612, 788, 25)
]
other_files = [
    f"{base_path}/B05_dir_ov.pdf",
    f"{base_path}/B05_success.json",
    "work/SH_testSLsinglefile2/B04_angle/B05_SH_20190502_05180312_angle_pdf.nc",
]
paths6.extend(other_files)


# Paths for the seventh script
root_plots = "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/"
root_work = "work/SH_testSLsinglefile2/"
b06_correction = f"{root_plots}B06_correction/"
b06_corrected_separated = f"{root_work}B06_corrected_separated/"

b06_correction_files = [
    "B06_angle_def.png",
    "SH_20190502_05180312_B06_atten_ov.pdf",
    "SH_20190502_05180312_B06_atten_ov.png",
    "SH_20190502_05180312_B06_atten_ov_simple.pdf",
    "SH_20190502_05180312_B06_atten_ov_simple.png",
]

b06_corrected_separated_files = [
    "B06_SH_20190502_05180312_B06_corrected_resid.h5",
    "B06_SH_20190502_05180312_binned_resid.h5",
    "B06_SH_20190502_05180312_gFT_k_corrected.nc",
    "B06_SH_20190502_05180312_gFT_x_corrected.nc",
]

paths7 = [f"{b06_correction}{file}" for file in b06_correction_files]
paths7.extend(
    f"{b06_corrected_separated}{file}" for file in b06_corrected_separated_files
)
paths7.append(f"{root_plots}B06_success.json")

# These are the scripts to be tested and appended with the target directories for each step
scripts = [script1, script2, script3, script4, script5, script6, script7]

targetdirs = (
    dict()
)  # to be populated in setup_module with the target dirs for each step
__outdir = []  # to be populated in setup_module with the temp dir for all steps


def setup_module():
    """
    Set up the module for testing.

    This function makes a temporary directory with subdirectories with the required input data for each step.

    ```shell
    $ tree -L 1 tests/tempdir
    tests/tempdir
    ├── step1
    ├── step2
    ├── step3
    ├── step4
    └── step5

    5 directories, 0 files
    ```

    When running in parallel using the `xdist` plugin, each worker will have its own copy of all the input data. This is necessary because the tests are run in parallel and the input data is modified by the tests. This way, the teardown function can delete the temporary directory for each worker without affecting the other workers.
    """

    homedir = Path(__file__).parent
    input_data_dir = homedir / "testdata"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    _outdir = mkdtemp(dir=homedir, suffix=timestamp)  # temp dir for all steps
    __outdir.append(_outdir)

    # make temp dirs for each step in _outdir
    # the one for step1 with no input data
    tmpstep1 = Path(_outdir) / "step1"
    tmpstep1.mkdir()
    script1.append(tmpstep1)

    # make temp dirs for steps 2,... with input data
    for tarball in input_data_dir.glob("*.tar.gz"):
        source_script, for_step_num = tarball.stem.split("-for-step-")
        for_step_num = for_step_num.split(".")[0]
        target_output_dir = Path(_outdir) / f"step{for_step_num}"
        targetdirs[for_step_num] = target_output_dir
        extract_tarball(target_output_dir, tarball)

        # Extracted files are in targetdir/script_name. Move them to its parent targetdir. Delete the script_name dir.
        parent = target_output_dir / "work"

        # Rename and update parent to targetdir / script_name
        new_parent = Path(target_output_dir, source_script)
        parent.rename(new_parent)

        for child in new_parent.iterdir():
            if child.is_dir():
                shutil.move(str(child), Path(child.parent).parent)
        shutil.rmtree(new_parent)

    # add the target dirs to the scripts
    for i, script in enumerate(scripts[1:], start=2):
        script.append(targetdirs[str(i)])

    # throw in tmpstep1 in targetdirs to clean up later
    targetdirs["1"] = tmpstep1


def teardown_module():
    """
    Clean up after testing is complete.
    """
    shutil.rmtree(__outdir[-1])

def test_step5():
    # Step 5: B04_angle.py ~ 9 min
    assert run_test(script5, paths5)


def test_step1():
    # Step 1: B01_SL_load_single_file.py ~ 2 minutes
    assert run_test(script1, paths1, delete_paths=False)  # passing


def test_step2():
    # Step 2: B02_make_spectra_gFT.py ~ 2 min
    assert run_test(script2, paths2)  # passing


def check_B03_freq_reconst_x():
    # The script3 produces plots in the `<output-dir>/plots/SH/<batch-key>/SH_<track-name>/B03_spectra` directory.

    # ```shell
    # (.venv) $ tree <output-dir>/plots/SH/<batch-key>/SH_<track-name>/B03_spectra
    # <output-dir>/plots/SH/<batch-key>/SH_<track-name>/B03_spectra
    # ├── B03_freq_reconst_x28.pdf
    # ├── B03_freq_reconst_x47.pdf
    # ├── B03_freq_reconst_x56.pdf
    # ├── B03_freq_reconst_x66.pdf
    # └── B03_freq_reconst_x75.pdf

    # 0 directories, 5 files
    # ```

    # File names in this directory all have the `B03_freq_reconst_x` prefix but with different numbers at the end. The numbers are not deterministic, so we can't check for specific file names. Instead, we can check that there are 5 pdf files in the directory.

    outputdir = getoutputdir(script3)
    directory = Path(
        outputdir, "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/B03_spectra/"
    )
    files = get_all_filenames(directory)

    # Check there are 5 pdf files
    return len([f for f in files if f.endswith("pdf")]) == 5


def test_step3():
    # Step 3: B03_plot_spectra_ov.py ~ 11 sec
    # This script has stochastic behavior, so the files produced don't always have the same names but the count of pdf files is constant for the test input data.
    t1 = run_test(script3, paths3)
    assert t1
    t2 = check_B03_freq_reconst_x()
    assert t2


def test_step4():
    # Step 4: A02c_IOWAGA_thredds_prior.py ~ 23 sec
    # check deterministic paths
    t1 = run_test(script4, paths4)
    assert t1
    # check stochastic paths
    t2 = check_file_exists(dir4, prefix4)
    assert t2


def test_step6():
    # Step 5: B05_define_angle.py ~
    assert run_test(script6, paths6)


def test_step7():
    # Step 7: B06_correct_separate_var.py ~
    assert run_test(script7, paths7)
