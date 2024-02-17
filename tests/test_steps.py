#!/usr/bin/env python
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
    Get a list of all file names in a directory
    """
    return [p.name for p in Path(directory).rglob("*")]


def check_file_exists(directory, prefix, stepnum: int = 4):
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


script1 = [
    "python",
    "src/icesat2_tracks/analysis_db/B01_SL_load_single_file.py",
    "--track-name",
    "20190502052058_05180312_005_01",
    "--batch-key",
    "SH_testSLsinglefile2",
    "--output-dir",
]
paths1 = [
    "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/B01_track.png.png",
    "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/B01b_ATL06_corrected.png.png",
    "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/B01b_beam_statistics.png.png",
    "work/SH_testSLsinglefile2/A01b_ID/A01b_ID_SH_20190502_05180312.json",
    "work/SH_testSLsinglefile2/B01_regrid/SH_20190502_05180312_B01_binned.h5",
]


script2 = [
    "python",
    "src/icesat2_tracks/analysis_db/B02_make_spectra_gFT.py",
    "--track-name",
    "SH_20190502_05180312",
    "--batch-key",
    "SH_testSLsinglefile2",
    "--output-dir",
]
paths2 = [
    "work/SH_testSLsinglefile2/B02_spectra/B02_SH_20190502_05180312_params.h5",
    "work/SH_testSLsinglefile2/B02_spectra/B02_SH_20190502_05180312_gFT_x.nc",
    "work/SH_testSLsinglefile2/B02_spectra/B02_SH_20190502_05180312_gFT_k.nc",
    "work/SH_testSLsinglefile2/B02_spectra/B02_SH_20190502_05180312_FFT.nc",
]

script3 = [
    "python",
    "src/icesat2_tracks/analysis_db/B03_plot_spectra_ov.py",
    "--track-name",
    "SH_20190502_05180312",
    "--batch-key",
    "SH_testSLsinglefile2",
    "--id-flag",
    "--output-dir",
]

_root = "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312"
_paths3 = [
    "B03_specs_L25000.0.png",
    "B03_specs_coord_check.png",
    "B03_success.json",
]
paths3 = makepathlist(_root, _paths3)

script4 = [
    "python",
    "src/icesat2_tracks/analysis_db/A02c_IOWAGA_thredds_prior.py",
    "--track-name",
    "SH_20190502_05180312",
    "--batch-key",
    "SH_testSLsinglefile2",
    "--id-flag",
    "--output-dir",
]
paths4 = [
    "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/A02_SH_2_hindcast_data.pdf",
    "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/A02_SH_2_hindcast_prior.pdf",
]
dir4, prefix4 = (
    "work/SH_testSLsinglefile2/A02_prior/",
    "A02_SH_20190502_05180312_hindcast",
)

# TODO: step 5
script5 = [
    "python",
    "src/icesat2_tracks/analysis_db/B04_angle.py",
    "--track-name",
    "SH_20190502_05180312",
    "--batch-key",
    "SH_testSLsinglefile2",
    "--id-flag",
    "--output-dir",
]
paths5 = [
    "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/B04_success.json",
    "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/B04_prior_angle.png",
    "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/B04_marginal_distributions.pdf",
    "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/B04_data_avail.pdf",
    "work/SH_testSLsinglefile2/B04_angle/B04_SH_20190502_05180312_res_table.h5",
    "work/SH_testSLsinglefile2/B04_angle/B04_SH_20190502_05180312_marginals.nc",
]

scripts = [script1, script2, script3, script4, script5]
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

    When running in parallel using the xdist plugin, each worker will have its own copy of all the input data. This is necessary because the tests are run in parallel and the input data is modified by the tests. This way, the teardown function can delete the temporary directory for each worker without affecting the other workers.
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
        print("Target Dir")
        print(target_output_dir)
        print("TarBall")
        print(tarball)
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


def test_step1():
    # Step 1: B01_SL_load_single_file.py ~ 2 minutes
    assert run_test(script1, paths1, delete_paths=False)  # passing


# TODO: for steps 2-5 after their respective prs are merged
def test_step2():
    # Step 2: B02_make_spectra_gFT.py ~ 2 min
    assert run_test(script2, paths2)  # passing


def check_B03_freq_reconst_x():
    outputdir = getoutputdir(script3)
    directory = Path(
        outputdir, "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/B03_spectra/"
    )
    print("directory")
    print(directory)
    files = get_all_filenames(directory)
    print("files")
    print(files)

    # Check there are 5 pdf files
    return len([f for f in files if f.endswith("pdf")]) == 5


def test_step3():
    # Step 3: B03_plot_spectra_ov.py ~ 11 sec
    # This script has stochastic behavior, so the files produced don't always have the same names but the count of pdf files is constant for the test input data.
    t1 = run_test(script3, paths3)
    t2 = check_B03_freq_reconst_x()
    assert t1
    assert t2


def test_step4():
    # Step 4: A02c_IOWAGA_thredds_prior.py ~ 23 sec
    t1 = run_test(script4, paths4)
    t2 = check_file_exists(dir4, prefix4)
    assert t1
    assert t2


def test_step5():
    # Step 5: B04_angle.py ~ 9 min
    assert run_test(script5, paths5)

if __name__ == "__main__":
    setup_module()
    test_step1()  # passing
    test_step2() # passing
    test_step3()  # passing
    test_step4()  # passing
    test_step5()
    teardown_module()
