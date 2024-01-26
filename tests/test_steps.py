#!/usr/bin/env python
import subprocess
import shutil
import tarfile
from pathlib import Path


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


def check_file_exists(directory, prefix):
    # Get a list of all files in the directory
    files = get_all_filenames(directory)
    # Check if there is a file with the specified prefix
    file_exists = any(file.startswith(prefix) for file in files)
    return file_exists


def delete_pdf_files(directory):
    path = Path(directory)
    for file in path.iterdir():
        if file.suffix == ".pdf":
            file.unlink()


def check_B03_freq_reconst_x():
    directory = Path(
        outputdir, "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312B03_spectra/"
    )
    files = get_all_filenames(directory)

    # Check there are 5 pdf files
    return len([f for f in files if f.endswith("pdf")]) == 5


def cleanup_directory(directory_path):
    path = Path(directory_path)
    if path.exists():
        shutil.rmtree(path)


def delete_file(file_path):
    path = Path(file_path)
    if path.exists():
        path.unlink()


def delete_files(file_paths):
    for file_path in file_paths:
        delete_file(file_path)


def getoutputdir(script):
    outputdir = script.index("--output-dir") + 1
    return script[outputdir]


def run_test(script, paths, delete_paths=True, suppress_output=True):
    outputdir = getoutputdir(script)
    paths = [Path(outputdir, pth) for pth in paths]
    if delete_paths:
        delete_files(paths)

    kwargs = {"check": True}
    if suppress_output:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL

        subprocess.run(script, **kwargs)
    return checkpaths(paths)


def makepathlist(dir, files):
    return [Path(dir, f) for f in files]


def setup_module():
    """
    Set up the module for testing.

    This function extracts the plots.tar.gz and work.tar.gz files in the tests directory to the home directory. These directories contain the expected input and output required/produced by the scripts being tested.
    """
    tarballs = ["plots.tar.gz", "work.tar.gz"]
    for tarball in tarballs:
        tar = tarfile.open(Path("tests", tarball))
        tar.extractall()
        tar.close()


# def teardown_module():
#     """
#     Clean up after testing is complete.
#     """
#     cleanup_directory("work")
#     cleanup_directory("plots")


# The scriptx variables are the scripts to be tested. The pathsx variables are the paths to the files that should be produced by the scripts. The scripts are run and the paths are checked to see if the files were produced. If the files were produced, the test passes. If not, the test fails.

outputdir = "tests/scratch"

script1 = [
    "python",
    "src/icesat2_tracks/analysis_db/B01_SL_load_single_file.py",
    "--track-name",
    "20190502052058_05180312_005_01",
    "--batch-key",
    "SH_testSLsinglefile2",
    "--output-dir",
    outputdir,
]
paths1 = [
    "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/B01_track.png.png",
    "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/B01b_ATL06_corrected.png.png",
    "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/B01b_beam_statistics.png.png",
    "work/SH_testSLsinglefile2/A01b_ID/A01b_ID_SH_20190502_05180312.json",
    "work/SH_testSLsinglefile2/B01_regrid/SH_20190502_05180312_B01_binned.h5",
]

# update script2 so it confoms to the new command line interface: python src/icesat2_tracks/analysis_db/B02_make_spectra_gFT.py --track-name SH_20190502_05180312 --batch-key SH_testSLsinglefile2 --output-dir ./scratch
script2 = [
    "python",
    "src/icesat2_tracks/analysis_db/B02_make_spectra_gFT.py",
    "--track-name",
    "SH_20190502_05180312",
    "--batch-key",
    "SH_testSLsinglefile2",
    "--output-dir",
    outputdir,  # update later to custom output directory in test function
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
    outputdir,  # update later to custom output directory in test function
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
    "SH_20190502_05180312",
    "SH_testSLsinglefile2",
    "True",
]
paths4 = [
    "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/A02_SH_2_hindcast_data.pdf",
    "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/A02_SH_2_hindcast_prior.pdf",
]
dir4, prefix4 = (
    "work/SH_testSLsinglefile2/A02_prior/",
    "A02_SH_20190502_05180312_hindcast",
)

script5 = [
    "python",
    "src/icesat2_tracks/analysis_db/B04_angle.py",
    "SH_20190502_05180312",
    "SH_testSLsinglefile2",
    "True",
]
paths5 = [
    "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/B04_success.json",
    "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/B04_prior_angle.png",
    "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/B04_marginal_distributions.pdf",
    "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312/B04_data_avail.pdf",
    "work/SH_testSLsinglefile2/B04_angle/B04_SH_20190502_05180312_res_table.h5",
    "work/SH_testSLsinglefile2/B04_angle/B04_SH_20190502_05180312_marginals.nc",
]


# def test_directories_and_files_step1():
#     # Step 1: B01_SL_load_single_file.py ~ 9 minutes
#     assert run_test(script1, paths1, delete_paths=False)  # passing


# def test_directories_and_files_step2():
#     # Step 2: B02_make_spectra_gFT.py ~ 2 min
#     assert run_test(script2, paths2)  # passing


def test_directories_and_files_step3():
    # Step 3: B03_plot_spectra_ov.py ~ 11 sec
    # This script has stochastic behavior, so the files produced don't always have the same names but the count of pdf files is constant for the test input data.
    pdfdirectory = (
        "scratch/plots/SH/SH_testSLsinglefile2/SH_20190502_05180312B03_spectra/"
    )
    delete_pdf_files(pdfdirectory)  # remove old files
    t1 = run_test(script3, paths3)
    t2 = check_B03_freq_reconst_x()
    assert t1
    assert t2


# def test_directories_and_files_step4():
#     # Step 4: A02c_IOWAGA_thredds_prior.py ~ 23 sec
#     t1 = run_test(script4, paths4)
#     t2 = check_file_exists(dir4, prefix4)
#     assert all([t1, t2])


# def test_directories_and_files_step5():
#     # Step 5: B04_angle.py ~ 9 min
#     assert run_test(script5, paths5)

if __name__ == "__main__":
    # setup_module()
    # test_directories_and_files_step1()
    # test_directories_and_files_step2()
    test_directories_and_files_step3()
    # test_directories_and_files_step4()
    # test_directories_and_files_step5()
    # teardown_module()
