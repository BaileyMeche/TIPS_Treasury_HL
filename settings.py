"""Load project configurations from .env files.
Provides easy access to paths and credentials used in the project.
Meant to be used as an imported module.

If `settings.py` is run on its own, it will create the appropriate
directories.

For information about the rationale behind decouple and this module,
see https://pypi.org/project/python-decouple/

Note that decouple mentions that it will help to ensure that
the project has "only one configuration module to rule all your instances."
This is achieved by putting all the configuration into the `.env` file.
You can have different sets of variables for difference instances,
such as `.env.development` or `.env.production`. You would only
need to copy over the settings from one into `.env` to switch
over to the other configuration, for example.

"""
from pathlib import Path
from platform import system
from decouple import config as _config
from pandas import to_datetime
BLOOMBERG = False

def get_os():
    os_name = system()
    if os_name == "Windows":
        return "windows"
    elif os_name in ("Darwin", "Linux"):
        return "nix"
    else:
        return "unknown"

def if_relative_make_abs(path):
    """If a relative path is given, make it absolute, assuming
    that it is relative to the project root directory (BASE_DIR)

    Example
    -------
    ```
    >>> if_relative_make_abs(Path('_data'))
    WindowsPath('C:/Users/jdoe/GitRepositories/blank_project/_data')

    >>> if_relative_make_abs(Path("C:/Users/jdoe/GitRepositories/blank_project/_output"))
    WindowsPath('C:/Users/jdoe/GitRepositories/blank_project/_output')
    ```
    """
    path = Path(path)
    if path.is_absolute():
        abs_path = path.resolve()
    else:
        abs_path = (d["BASE_DIR"] / path).resolve()
    return abs_path

d = {}
d["OS_TYPE"] = get_os()

# Absolute path to root directory of the project
#d["BASE_DIR"] = Path(__file__).absolute().parent.parent
override = _config("LOCAL_BASE_DIR", default=None, cast=str)
if override:
    d["BASE_DIR"] = Path(override).resolve()
else:
    d["BASE_DIR"] = Path(__file__).absolute().parent.parent

# fmt: off
## Other .env variables
d["START_DATE"] = _config("START_DATE", default="1913-01-01", cast=to_datetime)
d["END_DATE"] = _config("END_DATE", default="2024-01-01", cast=to_datetime)


## Name of Stata Executable in path
if d["OS_TYPE"] == "windows":
    d["STATA_EXE"] = _config("STATA_EXE", default="StataMP-64.exe")
elif d["OS_TYPE"] == "nix":
    d["STATA_EXE"] = _config("STATA_EXE", default="stata-mp")
else:
    raise ValueError("Unknown OS type")

def create_dirs():
    ## If they don't exist, create the _data and _output directories
    d["DATA_DIR"].mkdir(parents=True, exist_ok=True)
    d["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)
    # (d["BASE_DIR"] / "_docs").mkdir(parents=True, exist_ok=True)

    #Equity spot
    d["TEMP_DIR"].mkdir(parents=True, exist_ok=True)
    d["INPUT_DIR"].mkdir(parents=True, exist_ok=True)
    # d["PUBLISH_DIR"].mkdir(parents=True, exist_ok=True)
    d["PROCESSED_DIR"].mkdir(parents=True, exist_ok=True)

    # If you'd like to ensure these log files exist (touch them):
    for log_filename in (
        "futures_processing.log",
        "ois_processing.log",
        "bloomberg_data_extraction.log",
    ):
        log_file_path = d["TEMP_DIR"] / log_filename
        log_file_path.touch(exist_ok=True)

    # CIP
    d["REPORTS_DIR"].mkdir(parents=True, exist_ok=True)

def config(*args, **kwargs):
    """
    Retrieve configuration variables. 
    Checks `d` first. If not found, falls back to .env via decouple.config.
    """
    key = args[0]
    default = kwargs.get("default", None)
    cast = kwargs.get("cast", None)
    if key in d:
        var = d[key]
        # If a default was passed but we already have a value in d, raise an error
        if default is not None:
            raise ValueError(f"Default for {key} already exists. Check your settings.py file.")
        if cast is not None:
            # If cast is requested, check that it wouldn't change the type
            if not isinstance(var, cast):
                # or if we want to actually recast:
                try:
                    new_var = cast(var)
                except Exception as e:
                    raise ValueError(f"Could not cast {key} to {cast}: {e}") from e
                if type(new_var) is not type(var):
                    raise ValueError(f"Type for {key} differs. Check your settings.py file.")
        return var
    else:
        return _config(*args, **kwargs)

if __name__ == "__main__":
    create_dirs()