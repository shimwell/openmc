"""Console script that launches the openmc executable bundled in the wheel.

The executable is installed inside the package rather than on the PATH, so this
module is registered as a console script in pyproject.toml and passes its
arguments straight through to it. Declaring it as a console script rather than
shipping a plain script file matters on Windows, where the packaging tools
generate an openmc.exe launcher for a console script. A plain script file is
installed without any extension, which Windows can not execute, so neither the
command line nor openmc.run() can find it.
"""

import os
import subprocess
import sys
from pathlib import Path


def _executable_path() -> Path:
    """Returns the path of the openmc executable inside the installed package"""

    name = 'openmc.exe' if os.name == 'nt' else 'openmc'
    return Path(__file__).resolve().parent / 'core' / 'bin' / name


def main():
    executable = _executable_path()

    if not executable.is_file():
        sys.exit(f'The openmc executable was not found at {executable}')

    if os.name == 'nt':
        # os.execv does not replace the calling process on Windows, it returns
        # as soon as the child has been started. That would make callers such
        # as openmc.run() think the simulation had finished immediately, so the
        # child is waited on instead.
        sys.exit(subprocess.call([str(executable), *sys.argv[1:]]))

    os.execv(executable, [str(executable), *sys.argv[1:]])


if __name__ == '__main__':
    main()
