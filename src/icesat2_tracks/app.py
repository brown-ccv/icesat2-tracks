#!/usr/bin/env python
"""
Main CLI for icesat2waves.
"""
from typer import Typer, Option
from icesat2_tracks.analysis_db.B01_SL_load_single_file import (
    run_B01_SL_load_single_file as _loadfile,
)

from icesat2_tracks.clitools import (
    validate_track_name,
    validate_batch_key,
    validate_output_dir,
    validate_track_name_steps_gt_1,
)


app = Typer(add_completion=False)
validate_track_name_gt_1_opt = Option(..., callback=validate_track_name_steps_gt_1)
validate_batch_key_opt = Option(..., callback=validate_batch_key)
validate_output_dir_opt = Option(None, callback=validate_output_dir)


def run_job(
    analysis_func,
    track_name: str,
    batch_key: str,
    ID_flag: bool = True,
    output_dir: str = validate_output_dir_opt,
):
    analysis_func(
        track_name,
        batch_key,
        ID_flag,
        output_dir,
    )


@app.command(help=_loadfile.__doc__)
def loadfile(
    track_name: str = Option(..., callback=validate_track_name),
    batch_key: str = validate_batch_key_opt,
    ID_flag: bool = True,
    output_dir: str = validate_output_dir_opt,
):
    run_job(_loadfile, track_name, batch_key, ID_flag, output_dir)


if __name__ == "__main__":
    app()