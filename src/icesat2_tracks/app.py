"""
Main CLI for icesat2waves.
"""
from typer import Typer

from icesat2_tracks.analysis_db import (
    B01_SL_load_single_file as step1,
    B02_make_spectra_gFT as step2,
    B03_plot_spectra_ov as step3,
    A02c_IOWAGA_thredds_prior as step4,
)


def create_main_app():
    app = Typer()
    app.add_typer(step1.app, name="load-file")
    app.add_typer(step2.app, name="make-spectra")
    app.add_typer(step3.app, name="plot-spectra")
    app.add_typer(step4.app, name="threads-prior")
    return app


def main():
    app = create_main_app()
    app()


if __name__ == "__main__":
    main()
