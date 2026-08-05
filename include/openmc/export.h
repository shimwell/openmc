//! \file export.h
//! Symbol visibility macro for data that crosses the library boundary.

#ifndef OPENMC_EXPORT_H
#define OPENMC_EXPORT_H

// On Windows, symbols are hidden unless exported. CMake's
// WINDOWS_EXPORT_ALL_SYMBOLS covers functions, but it does not emit global
// data, so any variable read from outside libopenmc has to be annotated by
// hand. Two consumers need this:
//
//   * openmc.lib, which reads 24 globals through ctypes in_dll, both directly
//     and through the _DLLGlobal descriptor in openmc/lib/core.py
//   * main.cpp and the C++ unit tests, which link against the import library
//
// Only the const ones actually need the annotation, since bindexplib emits
// mutable data but not read-only data, but annotating by need rather than by
// rule would leave a trap for whoever next marks one of these const.
//
// Everywhere other than MSVC this expands to nothing, so POSIX builds are
// unaffected. OPENMC_DLL is set by CMake on the libopenmc target and
// propagates to anything linking it, so a static MSVC build still gets a
// plain declaration.
#if defined(_MSC_VER) && defined(OPENMC_DLL)
#ifdef OPENMC_BUILDING_LIBOPENMC
#define OPENMC_API __declspec(dllexport)
#else
#define OPENMC_API __declspec(dllimport)
#endif
#else
#define OPENMC_API
#endif

#endif // OPENMC_EXPORT_H
