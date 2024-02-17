import os
import re
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

import typer
from termcolor import colored


@contextmanager
def suppress_stdout(verbose=False):
    if verbose:
        yield
    else:
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull):
                yield


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
    """
    Validate the track name `value` based on a specific pattern (see below).

    Args:
        ctx (typer.Context): The context in which the command is being invoked.
        param (typer.CallbackParam): The parameter that is being validated.
        value (str): The value of the parameter.

    Returns:
        str: The validated track name.

    Raises:
        click.exceptions.BadParameter: If the track name does not match the pattern.

    Pattern:
        YYYYMMDDHHMMSS_XXXXXXXX_XXX_XX
        where:
        YYYYMMDDHHMMSS is a timestamp,
        XXXXXXXX is an 8-digit number,
        XXX is a 3-digit number,
        XX is a 2-digit number.

    Example:
        >>> validate_track_name(None, None, '20220101123000_12345678_123_12')
        '20220101123000_12345678_123_12'
        >>> validate_track_name(None, None, '20221231115959_87654321_321_21')
        '20221231115959_87654321_321_21'
        >>> validate_track_name(None, None, '20220228235959_00000000_000_00')
        '20220228235959_00000000_000_00'

    Doctest:
            >>> validate_track_name(None, None, 'invalid_track_name')
            Traceback (most recent call last):
            ...
            click.exceptions.BadParameter: track_name must be in the format: YYYYMMDDHHMMSS_XXXXXXXX_XXX_XX
    """
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
    """
    Validate a batch key based on a specific pattern (see below).

    Args:
        ctx (typer.Context): The context in which the command is being invoked.
        param (typer.CallbackParam): The parameter that is being validated.
        value (str): The value of the parameter.

    Returns:
        str: The validated batch key.

    Raises:
        click.exceptions.BadParameter: If the batch key does not match the pattern.

    Pattern:
        .*_.*
        where:
        .* is any character (including none),
        _ is a literal underscore,
        .* is any character (including none).

    Example:
        >>> validate_batch_key(None, None, 'SH_testSLsinglefile2')
        'SH_testSLsinglefile2'
        >>> validate_batch_key(None, None, 'batch_key')
        'batch_key'
        >>> validate_batch_key(None, None, '_')
        '_'

    Doctest:
        >>> validate_batch_key(None, None, '')
        Traceback (most recent call last):
        ...
        click.exceptions.BadParameter: batch_key must be in the format 'SH_testSLsinglefile2'
        >>> validate_batch_key(None, None, 'badbatchkey')
        Traceback (most recent call last):
        ...
        click.exceptions.BadParameter: batch_key must be in the format 'SH_testSLsinglefile2'
    """
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


def report_input_parameters(heading: str = "** Input parameters:", **kwargs):
    echo(heading)
    for key in kwargs:
        if key != "args":
            echoparam(key, kwargs[key])


def update_paths_mconfig(output_dir, mconfig):
    workdir, plotsdir = [
        Path(output_dir, mconfig["paths"][key]) for key in ["work", "plot"]
    ]

    return workdir, plotsdir


def validate_track_name_steps_gt_1(
    ctx: typer.Context, param: typer.CallbackParam, value: str
) -> str:
    pattern = r".*_\d{4}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])_\d{8}"
    error_message = "track_name must be in the format: any_characters_YYYYMMDD_12345678"
    return validate_pattern_wrapper(
        ctx,
        param,
        value,
        pattern,
        error_message,
    )


def makeapp(f, name):
    """
    Make a typer app from a function.
    """
    app = typer.Typer(add_completion=False, name=name)
    app.command()(f)
    return app
