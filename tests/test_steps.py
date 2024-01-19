import subprocess
import os
from pathlib import Path

# Need output files to stick around for testing so no teardown_module


def checkpaths(paths):
    return all([os.path.isfile(pth) for pth in paths])


def makepathlist(dir, files):
    return [Path(dir, f) for f in files]


script1 = [
    "python",
    "src/icesat2_tracks/analysis_db/B01_SL_load_single_file.py",
    "20190502052058_05180312_005_01",
    "SH_testSLsinglefile2",
    "True",
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
    "SH_20190502_05180312",
    "SH_testSLsinglefile2",
    "True",
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
    "SH_20190502_05180312",
    "SH_testSLsinglefile2",
    "True",
]
_dir = "plots/SH/SH_testSLsinglefile2/SH_20190502_05180312"
_paths3 = [
    "B03_specs_L25000.0.png",
    "B03_specs_coord_check.png",
    "B03_spectra/B03_freq_reconst_x28.pdf",
    "B03_spectra/B03_freq_reconst_x47.pdf",
    "B03_spectra/B03_freq_reconst_x56.pdf",
    "B03_spectra/B03_freq_reconst_x66.pdf",
    "B03_spectra/B03_freq_reconst_x75.pdf",
    "B03_success.json",
]
paths3 = makepathlist(_dir, _paths3)

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
    "work/SH_testSLsinglefile2/A02_prior/A02_SH_20190502_05180312.h5",
    "work/SH_testSLsinglefile2/A02_prior/A02_SH_20190502_05180312_hindcast_success.json",
]

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


def setup_module(module):
    # Step 1: B01_SL_load_single_file.py
    subprocess.run(script1, check=True)

    # Step 2: B02_make_spectra_gFT.py
    subprocess.run(script2, check=True)

    # Step 3: B03_plot_spectra_ov.py
    subprocess.run(script3, check=True)

    # Step 4: A02c_IOWAGA_thredds_prior.py
    subprocess.run(script4, check=True)

    # Step 5: B04_angle.py
    subprocess.run(script5, check=True)


def test_directories_and_files_step1():
    assert checkpaths(paths1)


# def test_directories_and_files_step2():
#    assert checkpaths(paths2)


def test_directories_and_files_step3():
    assert checkpaths(paths3)


def test_directories_and_files_step4():
    assert checkpaths(paths4)


def test_directories_and_files_step5():
    assert checkpaths(paths5)
