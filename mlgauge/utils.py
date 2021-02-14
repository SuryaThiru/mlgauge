import sys
import contextlib
from tqdm.contrib import DummyTqdmFile


@contextlib.contextmanager
def redirect_stdout():
    """redirect stdout and prevents from blocking the progress bars.py

    Reference: https://github.com/tqdm/tqdm#redirecting-writing
    """
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err


class colors:
    """colors for progress bars"""

    GREEN = "\033[92m"
    CYAN = "\033[96m"
    ENDC = "\033[0m"
