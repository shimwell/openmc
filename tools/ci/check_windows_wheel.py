"""Checks that an installed Windows wheel can actually run OpenMC.

Importing openmc and loading libopenmc.dll through ctypes passes even when the
openmc executable can not start, because the two resolve their dependencies
differently. ctypes loads libopenmc.dll with the DLL search path altered to
include its own folder, and any DLLs vendored by delvewheel are added to the
search path of the Python process by the package itself. A separate process
started from the executable gets neither of those, so a dependency that is only
reachable that way makes it exit before writing any output.

This reports the layout, the DLL each binary imports and whether that DLL can be
found, then runs the executable and the console script. Everything is printed
before anything is asserted so that a failure is diagnosable from the log.
"""

import ctypes
import os
import struct
import subprocess
import sys
from pathlib import Path

import openmc
import openmc.lib


def imported_dlls(binary: Path):
    """Returns the names of the DLLs that a PE binary imports"""

    data = binary.read_bytes()

    pe_offset = struct.unpack_from('<I', data, 0x3C)[0]
    if data[pe_offset:pe_offset + 4] != b'PE\0\0':
        raise ValueError(f'{binary} is not a PE binary')

    number_of_sections = struct.unpack_from('<H', data, pe_offset + 6)[0]
    size_of_optional_header = struct.unpack_from('<H', data, pe_offset + 20)[0]
    optional_header = pe_offset + 24

    # the import table is the second entry of the data directory, which sits
    # after the windows specific fields of the optional header
    magic = struct.unpack_from('<H', data, optional_header)[0]
    data_directory = optional_header + (112 if magic == 0x20B else 96)
    import_rva = struct.unpack_from('<I', data, data_directory + 8)[0]
    if import_rva == 0:
        return []

    sections = []
    section_table = optional_header + size_of_optional_header
    for index in range(number_of_sections):
        entry = section_table + index * 40
        virtual_address = struct.unpack_from('<I', data, entry + 12)[0]
        raw_size = struct.unpack_from('<I', data, entry + 16)[0]
        raw_pointer = struct.unpack_from('<I', data, entry + 20)[0]
        sections.append((virtual_address, raw_size, raw_pointer))

    def to_offset(rva):
        for virtual_address, raw_size, raw_pointer in sections:
            if virtual_address <= rva < virtual_address + raw_size:
                return raw_pointer + (rva - virtual_address)
        raise ValueError(f'could not map rva {rva:#x} in {binary}')

    names = []
    descriptor = to_offset(import_rva)
    while True:
        entry = data[descriptor:descriptor + 20]
        if len(entry) < 20 or entry == b'\0' * 20:
            break
        name_rva = struct.unpack_from('<I', entry, 12)[0]
        if name_rva:
            offset = to_offset(name_rva)
            names.append(data[offset:data.index(b'\0', offset)].decode())
        descriptor += 20

    return names


# The return types have to be declared. A handle is 64 bit and the ctypes
# default of c_int truncates it, which gives a non-zero but invalid handle and
# makes GetModuleFileNameW return an empty path for everything.
_kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
_kernel32.LoadLibraryExW.restype = ctypes.c_void_p
_kernel32.LoadLibraryExW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p,
                                     ctypes.c_uint32]
_kernel32.GetModuleFileNameW.restype = ctypes.c_uint32
_kernel32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p,
                                         ctypes.c_uint32]

DONT_RESOLVE_DLL_REFERENCES = 0x1


def find_dll(name: str, extra_dirs):
    """Returns where a DLL would be loaded from, or None if it is not found"""

    for directory in extra_dirs:
        candidate = Path(directory) / name
        if candidate.is_file():
            return candidate

    # DONT_RESOLVE_DLL_REFERENCES so that this reports whether the DLL itself
    # can be found rather than whether its own dependencies can be
    handle = _kernel32.LoadLibraryExW(name, None, DONT_RESOLVE_DLL_REFERENCES)
    if not handle:
        return None

    buffer = ctypes.create_unicode_buffer(32768)
    if _kernel32.GetModuleFileNameW(handle, buffer, len(buffer)) == 0:
        return Path(name)
    return Path(buffer.value)


def main():
    print(f'openmc {openmc.__version__} loaded {openmc.lib._filename}')

    package_dir = Path(openmc.__file__).resolve().parent
    bin_dir = package_dir / 'core' / 'bin'
    site_packages = package_dir.parent

    print(f'openmc package {package_dir}')
    print('entries alongside the package that could hold vendored DLLs:')
    for entry in sorted(site_packages.glob('openmc*')):
        if entry.is_dir():
            print(f'  {entry.name}')

    print(f'contents of {bin_dir}')
    for entry in sorted(bin_dir.iterdir()):
        print(f'  {entry.name}')

    # anything vendored by delvewheel lives in a .libs folder next to the
    # package and is only on the search path of the Python process
    vendored = [p for p in site_packages.glob('openmc*.libs') if p.is_dir()]
    if vendored:
        for folder in vendored:
            print(f'vendored DLLs in {folder.name}')
            for entry in sorted(folder.iterdir()):
                print(f'  {entry.name}')
    else:
        print('no openmc*.libs folder, so no DLLs were vendored into the wheel')

    # h5py bundles its own HDF5 and puts it on the DLL search path of the
    # process, which is enough to make libopenmc.dll import even when the wheel
    # is missing HDF5. Reported so that is not mistaken for a working wheel.
    for h5py_dll in sorted(site_packages.glob('h5py*/**/hdf5.dll')):
        print(f'h5py ships an HDF5 at {h5py_dll}')

    extra_dirs = [bin_dir, *vendored]

    missing = []
    for binary in (bin_dir / 'openmc.exe', bin_dir / 'libopenmc.dll'):
        print(f'DLLs imported by {binary.name}')
        for name in imported_dlls(binary):
            location = find_dll(name, extra_dirs)
            print(f'  {name} -> {location if location else "NOT FOUND"}')
            if location is None:
                missing.append((binary.name, name))

    print('running the executable directly')
    completed = subprocess.run(
        [str(bin_dir / 'openmc.exe'), '--version'],
        capture_output=True,
        text=True,
    )
    print(f'  exit code {completed.returncode} ({completed.returncode & 0xFFFFFFFF:#010x})')
    print(f'  stdout {completed.stdout.strip()!r}')
    print(f'  stderr {completed.stderr.strip()!r}')

    print('running the openmc console script')
    script = subprocess.run(['openmc', '--version'], capture_output=True, text=True)
    print(f'  exit code {script.returncode}')
    print(f'  stdout {script.stdout.strip()!r}')
    print(f'  stderr {script.stderr.strip()!r}')

    if completed.returncode != 0:
        # Not a failure on its own. The executable is not meant to be started
        # from its own folder, the console script is what puts the folders
        # holding the bundled DLLs on the PATH first.
        print('  the executable does not run on its own, which is expected if '
              'it needs DLLs that only the console script puts on the PATH')

    if missing:
        sys.exit(f'DLLs that could not be found anywhere: {missing}')
    if script.returncode != 0 or 'openmc version' not in script.stdout.lower():
        sys.exit('the openmc console script did not run OpenMC')

    print('the Windows wheel can run OpenMC')


if __name__ == '__main__':
    main()
