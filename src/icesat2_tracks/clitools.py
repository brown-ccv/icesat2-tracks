from pathlib import Path
import os
import sys
import re

from contextlib import contextmanager
import typer
from termcolor import colored


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# Callbacks for typer
def validate_pattern_wrapper(
    ctx: typer.Context,
    param: typer.CallbackParam,
    value: str,
    pattern: str,
    error_message: str,
) -> str:
    if not re.match(pattern, value):
        raise typer.BadParameter(error_message)
    return value


def validate_track_name(
    ctx: typer.Context, param: typer.CallbackParam, value: str
) -> str:
    pattern = r"\d{4}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])([01][0-9]|2[0-3])([0-5][0-9]){2}_\d{8}_\d{3}_\d{2}"
    error_message = "track_name must be in the format: YYYYMMDDHHMMSS_XXXXXXXX_XXX_XX"
    return validate_pattern_wrapper(
        ctx,
        param,
        value,
        pattern,
        error_message,
    )


def validate_batch_key(
    ctx: typer.Context, param: typer.CallbackParam, value: str
) -> str:
    pattern = r".*_.*"
    error_message = "batch_key must be in the format 'SH_testSLsinglefile2'"
    return validate_pattern_wrapper(
        ctx,
        param,
        value,
        pattern,
        error_message,
    )


def validate_output_dir(
    ctx: typer.Context, param: typer.CallbackParam, value: str
) -> str:
    path = Path(value).resolve()
    if not path.is_dir():
        raise typer.BadParameter(f"{path} does not exist")
    return str(path)


def echo(text: str, color: str = "green"):
    typer.echo(colored(text, color))


def echoparam(text: str, value, textcolor: str = "green", valuecolor: str = "white"):
    # add tab to text and center around the :
    text = "\t" + text
    text = f"{text:<12}"
    echo(f"{colored(text,textcolor)}: {colored(value, valuecolor)}")
