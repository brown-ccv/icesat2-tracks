#!/usr/bin/env python
"""
Main CLI for icesat2waves.
"""
import logging
from enum import Enum

from typer import Typer, Option
from typing_extensions import Annotated

from icesat2waves.analysis_db.B01_SL_load_single_file import (
    run_B01_SL_load_single_file as _loadfile,
)

from icesat2waves.analysis_db.B02_make_spectra_gFT import (
    run_B02_make_spectra_gFT as _makespectra,
)

from icesat2waves.analysis_db.B03_plot_spectra_ov import (
    run_B03_plot_spectra_ov as _plotspectra,
)

from icesat2waves.analysis_db.A02c_IOWAGA_thredds_prior import (
    run_A02c_IOWAGA_thredds_prior as _threddsprior,
)


from icesat2waves.analysis_db.B04_angle import run_B04_angle as _run_B04_angle

from icesat2waves.analysis_db.B05_define_angle import define_angle as _define_angle

from icesat2waves.analysis_db.B06_correct_separate_var import (
    run_B06_correct_separate_var as _run_correct_separate_var,
)


from icesat2waves.clitools import (
    validate_track_name,
    validate_batch_key,
    validate_output_dir,
    validate_track_name_steps_gt_1,
)

_logger = logging.getLogger(__name__)

app = Typer(add_completion=False)


class _LogLevel(str, Enum):
    QUIET = "quiet"
    WARNING = "warning"
    VERBOSE = "verbose"
    DEBUG = "debug"

    @property
    def level(self):
        return {
            _LogLevel.QUIET: logging.CRITICAL,
            _LogLevel.WARNING: logging.WARNING,
            _LogLevel.VERBOSE: logging.INFO,
            _LogLevel.DEBUG: logging.DEBUG,
        }[self.value]


@app.callback()
def main(
    log: Annotated[
        _LogLevel,
        Option(help="Set the log level"),
    ] = "warning"
):
    """Analyze data from the ICESAT2 satellite."""
    logging.basicConfig(level=log.level)


validate_track_name_gt_1_opt = Option(..., callback=validate_track_name_steps_gt_1)
validate_batch_key_opt = Option(..., callback=validate_batch_key)
validate_output_dir_opt = Option(..., callback=validate_output_dir)


def run_job(
    analysis_func,
    track_name: str,
    batch_key: str,
    ID_flag: bool = True,
    output_dir: str = validate_output_dir_opt,
):
    analysis_func(
        track_name=track_name,
        batch_key=batch_key,
        ID_flag=ID_flag,
        output_dir=output_dir,
    )


@app.command(help=_loadfile.__doc__)
def load_file(
    track_name: str = Option(..., callback=validate_track_name),
    batch_key: str = validate_batch_key_opt,
    ID_flag: bool = True,
    output_dir: str = validate_output_dir_opt,
):
    run_job(
        analysis_func=_loadfile,
        track_name=track_name,
        batch_key=batch_key,
        ID_flag=ID_flag,
        output_dir=output_dir,
    )


@app.command(help=_makespectra.__doc__)
def make_spectra(
    track_name: str = validate_track_name_gt_1_opt,
    batch_key: str = validate_batch_key_opt,
    ID_flag: bool = True,
    output_dir: str = validate_output_dir_opt,
):
    run_job(
        analysis_func=_makespectra,
        track_name=track_name,
        batch_key=batch_key,
        ID_flag=ID_flag,
        output_dir=output_dir,
    )


@app.command(help=_plotspectra.__doc__)
def plot_spectra(
    track_name: str = validate_track_name_gt_1_opt,
    batch_key: str = validate_batch_key_opt,
    ID_flag: bool = True,
    output_dir: str = validate_output_dir_opt,
):
    run_job(
        analysis_func=_plotspectra,
        track_name=track_name,
        batch_key=batch_key,
        ID_flag=ID_flag,
        output_dir=output_dir,
    )


@app.command(help=_threddsprior.__doc__)
def make_iowaga_threads_prior(  # TODO: revise naming @mochell
    track_name: str = validate_track_name_gt_1_opt,
    batch_key: str = validate_batch_key_opt,
    ID_flag: bool = True,
    output_dir: str = validate_output_dir_opt,
):
    run_job(
        analysis_func=_threddsprior,
        track_name=track_name,
        batch_key=batch_key,
        ID_flag=ID_flag,
        output_dir=output_dir,
    )


@app.command(help=_run_B04_angle.__doc__)
def make_b04_angle(  # TODO: revise naming @mochell
    track_name: str = validate_track_name_gt_1_opt,
    batch_key: str = validate_batch_key_opt,
    ID_flag: bool = True,
    output_dir: str = validate_output_dir_opt,
):
    run_job(
        analysis_func=_run_B04_angle,
        track_name=track_name,
        batch_key=batch_key,
        ID_flag=ID_flag,
        output_dir=output_dir,
    )


@app.command(help=_define_angle.__doc__)
def define_angle(
    track_name: str = validate_track_name_gt_1_opt,
    batch_key: str = validate_batch_key_opt,
    ID_flag: bool = True,
    output_dir: str = validate_output_dir_opt,
):
    run_job(
        analysis_func=_define_angle,
        track_name=track_name,
        batch_key=batch_key,
        ID_flag=ID_flag,
        output_dir=output_dir,
    )


@app.command(help=_run_correct_separate_var.__doc__)
def correct_separate(  # TODO: rename with a verb or something
    track_name: str = validate_track_name_gt_1_opt,
    batch_key: str = validate_batch_key_opt,
    ID_flag: bool = True,
    output_dir: str = validate_output_dir_opt,
):
    run_job(
        analysis_func=_run_correct_separate_var,
        track_name=track_name,
        batch_key=batch_key,
        ID_flag=ID_flag,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    app()
