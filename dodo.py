"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based

"""

#######################################
## Configuration and Helpers for PyDoit
#######################################
## Make sure the src folder is in the path
import sys

sys.path.insert(1, "./src/")

import shutil
from os import environ, getcwd, path
from pathlib import Path

from colorama import Fore, Style, init

## Custom reporter: Print PyDoit Text in Green
# This is helpful because some tasks write to sterr and pollute the output in
# the console. I don't want to mute this output, because this can sometimes
# cause issues when, for example, LaTeX hangs on an error and requires
# presses on the keyboard before continuing. However, I want to be able
# to easily see the task lines printed by PyDoit. I want them to stand out
# from among all the other lines printed to the console.
from doit.reporter import ConsoleReporter

from settings import config

try:
    in_slurm = environ["SLURM_JOB_ID"] is not None
except:
    in_slurm = False


class GreenReporter(ConsoleReporter):
    def write(self, stuff, **kwargs):
        doit_mark = stuff.split(" ")[0].ljust(2)
        task = " ".join(stuff.split(" ")[1:]).strip() + "\n"
        output = (
            Fore.GREEN
            + doit_mark
            + f" {path.basename(getcwd())}: "
            + task
            + Style.RESET_ALL
        )
        self.outstream.write(output)


if not in_slurm:
    DOIT_CONFIG = {
        "reporter": GreenReporter,
        # other config here...
        # "cleanforget": True, # Doit will forget about tasks that have been cleaned.
        "backend": "sqlite3",
        "dep_file": "./.doit-db.sqlite",
    }
else:
    DOIT_CONFIG = {"backend": "sqlite3", "dep_file": "./.doit-db.sqlite"}
init(autoreset=True)


BASE_DIR = config("BASE_DIR")
DATA_DIR = config("DATA_DIR")
MANUAL_DATA_DIR = config("MANUAL_DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")
OS_TYPE = config("OS_TYPE")
PUBLISH_DIR = config("PUBLISH_DIR")
USER = config("USER")

## Helpers for handling Jupyter Notebook tasks
# fmt: off
## Helper functions for automatic execution of Jupyter notebooks
environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
def jupyter_execute_notebook(notebook):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --log-level WARN --inplace ./src/{notebook}.ipynb"
def jupyter_to_html(notebook, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --log-level WARN --output-dir={output_dir} ./src/{notebook}.ipynb"
def jupyter_to_md(notebook, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f"jupytext --to markdown --log-level WARN --output-dir={output_dir} ./src/{notebook}.ipynb"
def jupyter_to_python(notebook, build_dir):
    """Convert a notebook to a python script"""
    return f"jupyter nbconvert --log-level WARN --to python ./src/{notebook}.ipynb --output _{notebook}.py --output-dir {build_dir}"
def jupyter_clear_output(notebook):
    return f"jupyter nbconvert --log-level WARN --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace ./src/{notebook}.ipynb"
# fmt: on


def copy_file(origin_path, destination_path, mkdir=True):
    """Create a Python action for copying a file."""

    def _copy_file():
        origin = Path(origin_path)
        dest = Path(destination_path)
        if mkdir:
            dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, dest)

    return _copy_file


##################################
## Begin rest of PyDoit tasks here
##################################


def task_config():
    """Create empty directories for data and output if they don't exist"""
    return {
        "actions": ["ipython ./src/settings.py"],
        "targets": [DATA_DIR, OUTPUT_DIR],
        "file_dep": ["./src/settings.py"],
        "clean": [],
    }

def task_pull_fed_yield_curve():
    """ """
    file_dep = [
        "./src/pull_fed_yield_curve.py",
    ]
    targets = [
        DATA_DIR / "fed_yield_curve_all.parquet",
        DATA_DIR / "fed_yield_curve.parquet",
    ]

    return {
        "actions": [
            "ipython ./src/pull_fed_yield_curve.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": [],
    }

def task_pull_fed_tips_yield_curve():
    """ """
    file_dep = [
        "./src/pull_fed_tips_yield_curve.py",
    ]
    targets = [
        DATA_DIR / "fed_tips_yield_curve.parquet",
    ]

    return {
        "actions": [
            "ipython ./src/pull_fed_tips_yield_curve.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": [],
    }

def task_pull_bloomberg_treasury_inflation_swaps():
    """Run pull_bloomberg_treasury_inflation_swaps only if treasury_inflation_swaps.csv is not present in OUTPUT_DIR."""
    from pathlib import Path  # ensure Path is available
    output_dir = Path(OUTPUT_DIR)

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        
    if not any(output_dir.iterdir()):
        # Only yield the nested task if the CSV file is not present.
        yield {
            "name": "run",
            "actions": ["ipython ./src/pull_bloomberg_treasury_inflation_swaps.py"],
            "file_dep": ["./src/pull_bloomberg_treasury_inflation_swaps.py"],
            "targets": [DATA_DIR / "treasury_inflation_swaps.parquet"],
            "clean": [],
        }
    else:
        print("treasury_inflation_swaps.csv exists in OUTPUT_DIR; skipping task_pull_bloomberg_treasury_inflation_swaps")



def task_compute_tips_treasury():
    """ """
    file_dep = [
        "./src/compute_tips_treasury.py",
    ]
    targets = [
        OUTPUT_DIR / "tips_treasury_implied_rf.parquet",
    ]

    return {
        "actions": [
            "ipython ./src/compute_tips_treasury.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": [],
    }

def task_generate_figures():
    """ """
    file_dep = [
        "./src/generate_figures.py",
        "./src/generate_latex_table.py",
    ]
    file_output = [
        "tips_treasury_spreads.png",
        "tips_treasury_summary_stats.csv",
        'tips_treasury_summary_table.tex'
    ]
    targets = [OUTPUT_DIR / file for file in file_output]

    return {
        "actions": [
            "ipython ./src/generate_figures.py",
            "ipython ./src/generate_latex_table.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": [],
    }



def task_planC_novendor():
    """Run the Plan C pipeline using synthetic data only."""
    output_dir = OUTPUT_DIR / 'planC'
    csv_files = [
        'trace_trades.csv',
        'trace_summary.csv',
        'treasury_yields.csv',
        'inflation_curve.csv',
        'discount_table.csv',
        'tips_index.csv',
        'nominal_cashflows.csv',
        'tips_cashflows.csv',
        'synthetic_basis.csv',
    ]
    file_dep = [
        Path('./src/planC/__init__.py'),
        Path('./src/planC/pipeline.py'),
        Path('./src/planC/wrds_trace.py'),
        Path('./src/planC/crsp_tsy.py'),
        Path('./src/planC/tips_index.py'),
        Path('./src/planC/infl_curve_public.py'),
        Path('./src/planC/discount_curves.py'),
        Path('./src/planC/cashflows_nominal.py'),
        Path('./src/planC/cashflows_tips.py'),
        Path('./src/planC/synthetic_nominal.py'),
    ]
    targets = [output_dir / name for name in csv_files]

    def _clean():
        shutil.rmtree(output_dir, ignore_errors=True)

    return {
        'actions': [
            f"PYTHONPATH=src python -m planC.pipeline --novendor --start 2023-01-01 --end 2023-01-05 --output-dir {output_dir}",
        ],
        'file_dep': file_dep,
        'targets': targets,
        'clean': [_clean],
    }


def task_planC_proof_of_concept():
    """Generate proof-of-concept artefacts for the Plan C workflow."""

    output_dir = OUTPUT_DIR / "planC_poc"
    report_path = Path("reports") / "planC_proof_of_concept.md"
    figure_path = output_dir / "basis_timeseries.png"
    summary_path = output_dir / "basis_summary.csv"

    file_dep = [
        Path("./src/planC/__init__.py"),
        Path("./src/planC/pipeline.py"),
        Path("./src/planC/proof_of_concept.py"),
        Path("./src/planC/wrds_trace.py"),
        Path("./src/planC/crsp_tsy.py"),
        Path("./src/planC/tips_index.py"),
        Path("./src/planC/infl_curve_public.py"),
        Path("./src/planC/discount_curves.py"),
        Path("./src/planC/cashflows_nominal.py"),
        Path("./src/planC/cashflows_tips.py"),
        Path("./src/planC/synthetic_nominal.py"),
    ]

    def _clean():
        shutil.rmtree(output_dir, ignore_errors=True)
        if report_path.exists():
            report_path.unlink()

    command = (
        "PYTHONPATH=src python -m planC.proof_of_concept "
        "--start 2023-01-01 --end 2023-01-31 "
        f"--output-dir {output_dir} --report {report_path}"
    )

    return {
        "actions": [command],
        "file_dep": file_dep,
        "targets": [figure_path, summary_path, report_path],
        "clean": [_clean],
    }



notebook_tasks = {
    "arb_replication.ipynb": {
        "file_dep": [
            "./src/generate_figures.py"
        ],
        "targets": [],
    }
}

def task_convert_notebooks_to_scripts():
    """Convert notebooks to script form to detect changes to source code rather
    than to the notebook's metadata.
    """
    build_dir = Path(OUTPUT_DIR)

    for notebook in notebook_tasks.keys():
        notebook_name = notebook.split(".")[0]
        yield {
            "name": notebook,
            "actions": [
                jupyter_clear_output(notebook_name),
                jupyter_to_python(notebook_name, build_dir),
            ],
            "file_dep": [Path("./src") / notebook],
            "targets": [OUTPUT_DIR / f"_{notebook_name}.py"],
            "clean": True,
            "verbosity": 0,
        }


# fmt: off
def task_run_notebooks():
    """Preps the notebooks for presentation format.
    Execute notebooks if the script version of it has been changed.
    """
    for notebook in notebook_tasks.keys():
        notebook_name = notebook.split(".")[0]
        yield {
            "name": notebook,
            "actions": [
                """python -c "import sys; from datetime import datetime; print(f'Start """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
                jupyter_execute_notebook(notebook_name),
                jupyter_to_html(notebook_name),
                copy_file(
                    Path("./src") / f"{notebook_name}.ipynb",
                    OUTPUT_DIR / f"{notebook_name}.ipynb",
                    mkdir=True,
                ),
                jupyter_clear_output(notebook_name),
                # jupyter_to_python(notebook_name, build_dir),
                """python -c "import sys; from datetime import datetime; print(f'End """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
            ],
            "file_dep": [
                OUTPUT_DIR / f"_{notebook_name}.py",
                *notebook_tasks[notebook]["file_dep"],
            ],
            "targets": [
                OUTPUT_DIR / f"{notebook_name}.html",
                OUTPUT_DIR / f"{notebook_name}.ipynb",
                *notebook_tasks[notebook]["targets"],
            ],
            "clean": True,
        }
# fmt: on


# ###############################################################
# ## Task below is for LaTeX compilation
# ###############################################################


def task_compile_latex_docs():
    """Compile the LaTeX documents to PDFs"""
    file_dep = [
        "./reports/report.tex",
        "./reports/my_article_header.sty",      # style 
        #"./reports/slides_example.tex",
        #`"./reports/my_beamer_header.sty",       # style
        "./reports/my_common_header.sty",       # style
        # "./reports/report_simple_example.tex",
        # "./reports/slides_simple_example.tex",
        "./src/generate_figures.py",
        "./src/generate_latex_table.py",
    ]
    targets = [
        "./reports/report.pdf",
        #"./reports/slides_example.pdf",
        # "./reports/report_simple_example.pdf",
        # "./reports/slides_simple_example.pdf",
    ]

    return {
        "actions": [
            # My custom LaTeX templates
            "latexmk -xelatex -halt-on-error -cd ./reports/report.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/report.tex",  # Clean
      ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }
#
# notebook_sphinx_pages = [
#     "./docs/notebooks/EX_" + notebook.split(".")[0] + ".html"
#     for notebook in notebook_tasks.keys()
# ]
# sphinx_targets = [
#     "./docs/index.html",
#     "./docs/myst_markdown_demos.html",
#     "./docs/apidocs/index.html",
#     *notebook_sphinx_pages,
# ]
#
# def task_compile_sphinx_docs():
#     """Compile Sphinx Docs"""
#     notebook_scripts = [
#         OUTPUT_DIR / ("_" + notebook.split(".")[0] + ".py")
#         for notebook in notebook_tasks.keys()
#     ]
#     file_dep = [
#         "./README.md",
#         "./pipeline.json",
#         *notebook_scripts,
#     ]
#
#     return {
#         "actions": [
#             "chartbook generate -f",
#         ],  # Use docs as build destination
#         # "actions": ["sphinx-build -M html ./docs/ ./docs/_build"], # Previous standard organization
#         "targets": sphinx_targets,
#         "file_dep": file_dep,
#         "task_dep": ["run_notebooks",],
#         "clean": True,
#     }