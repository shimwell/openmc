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


def _windows_environment() -> dict:
    """Returns an environment that lets the executable find its own DLLs.

    DLLs bundled into the wheel by the repair tool are placed in a folder
    beside the package, and are only added to the DLL search path of the
    Python process, by the code injected into the package for that purpose.
    The executable runs as a separate process which does not inherit any of
    that, so those folders are put on its PATH instead. Without this the
    executable exits without writing anything at all, as a Windows process
    that can not resolve a DLL never gets as far as running.
    """

    package_dir = Path(__file__).resolve().parent
    directories = [package_dir / 'core' / 'bin']
    directories += sorted(package_dir.parent.glob('openmc*.libs'))

    environment = os.environ.copy()
    on_path = [str(d) for d in directories if d.is_dir()]
    environment['PATH'] = os.pathsep.join([*on_path, environment.get('PATH', '')])
    return environment


def main():
    executable = _executable_path()

    if not executable.is_file():
        sys.exit(f'The openmc executable was not found at {executable}')

    if os.name == 'nt':
        # os.execv does not replace the calling process on Windows, it returns
        # as soon as the child has been started. That would make callers such
        # as openmc.run() think the simulation had finished immediately, so the
        # child is waited on instead.
        sys.exit(subprocess.call(
            [str(executable), *sys.argv[1:]], env=_windows_environment()
        ))

    os.execv(executable, [str(executable), *sys.argv[1:]])


if __name__ == '__main__':
    main()
