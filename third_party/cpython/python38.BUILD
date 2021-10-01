# Copyright 2021 Cory Paik. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" CPython build file for python3.8 adapted from @dbx_build_tools [1]_.

This build file was adapted from  @dbx_build_tools [1]_ to integrate with the
rest of the current Bazel rule sets, primarly `rules_python`.

Design Goals:
* Integration with the current Bazel rule sets, e.g. `rules_python`.
* Usage should be platform agnostic, even if the implementation only works on
  linux (which is currently the case). On other platforms we should fall back to
  the autodetecting toolchains.

Modifications:
* Add a header library :local_config_python_headers to be aliased by
  @local_config_python//:python_headers and used by @pybind11.
* Fix `:stdlib-zip` to create an initial stdlib zip with a downloaded python
  binary (@pip_py3_interpreter) instead or relying on the system's Python via
  the default `@rules_python` toolchain. In contrast to @dbx_build_tools, we
  have no "other" python toolchain, so referncing the `py_binary` is a self edge.
* Add missing headers to `:_blake2`.
* Configure `:_sqlite3` to use the sqlite version defiend by @org_tensorflow.
* Fix permission errors for intermetiate python-prime in `:stdlib-zip`.

Notes:
* This currently relies on a hard-coded python binary to be defined as
  @pip_py3_interpreter:interpreter. This is currently defined by a repository
  rule @com_github_corypaik_coda//tools/python:interpreter.bzl#py3_interpreter.
  In the future these will be migrated into one cohesive package.

References:
.. [1] https://github.com/dropbox/dbx_build_tools/blob/master/thirdparty/cpython/BUILD.python38

"""
# Welcome to the CPython BUILD file.

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@rules_pkg//:pkg.bzl", "pkg_tar")

# Everything within or depending on Python will need these flags.
PY_COPTS = [
    "-fwrapv",
    "-Iexternal/%s/Include" % (repository_name()[1:],),
    "-Iexternal/%s/Include/internal" % (repository_name()[1:],),
    "-I$(GENDIR)/external/%s/Include" % (repository_name()[1:],),
]

CORE_DEFINES = [
    '-DVERSION=\\"3.8\\"',
    '-DVPATH=\\"external/%s\\"' % (repository_name()[1:],),
    '-DPREFIX=\\"/usr\\"',
    '-DEXEC_PREFIX=\\"/usr\\"',
    '-DPYTHONPATH=\\":plat-linux:lib-tk:lib-old\\"',
    '-DPLATFORM=\\"linux\\"',
    '-DSOABI=\\"cpython-38-x86_64-linux-gnu\\"',
    '-DABIFLAGS=\\"\\"',
]

# Flags for building the core python executable (i.e., not a shared extension).
CORE_COPTS = PY_COPTS + CORE_DEFINES + [
    "-DPy_BUILD_CORE",
]

# Flags for building builtin extensions.
CORE_BUILTIN_COPTS = PY_COPTS + CORE_DEFINES + [
    "-DPy_BUILD_CORE_BUILTIN",
]

PUBLIC_PYTHON_HEADERS = [
    ":Include/pyconfig.h",
    "Include/abstract.h",
    "Include/asdl.h",
    "Include/ast.h",
    "Include/bitset.h",
    "Include/bltinmodule.h",
    "Include/boolobject.h",
    "Include/bytearrayobject.h",
    "Include/bytes_methods.h",
    "Include/bytesobject.h",
    "Include/cellobject.h",
    "Include/ceval.h",
    "Include/classobject.h",
    "Include/codecs.h",
    "Include/code.h",
    "Include/compile.h",
    "Include/complexobject.h",
    "Include/context.h",
    "Include/datetime.h",
    "Include/descrobject.h",
    "Include/dictobject.h",
    "Include/dtoa.h",
    "Include/dynamic_annotations.h",
    "Include/enumobject.h",
    "Include/errcode.h",
    "Include/eval.h",
    "Include/fileobject.h",
    "Include/fileutils.h",
    "Include/floatobject.h",
    "Include/frameobject.h",
    "Include/funcobject.h",
    "Include/genobject.h",
    "Include/graminit.h",
    "Include/grammar.h",
    "Include/import.h",
    "Include/interpreteridobject.h",
    "Include/intrcheck.h",
    "Include/iterobject.h",
    "Include/listobject.h",
    "Include/longintrepr.h",
    "Include/longobject.h",
    "Include/marshal.h",
    "Include/memoryobject.h",
    "Include/methodobject.h",
    "Include/modsupport.h",
    "Include/moduleobject.h",
    "Include/namespaceobject.h",
    "Include/node.h",
    "Include/object.h",
    "Include/objimpl.h",
    "Include/odictobject.h",
    "Include/opcode.h",
    "Include/osdefs.h",
    "Include/osmodule.h",
    "Include/parsetok.h",
    "Include/patchlevel.h",
    "Include/picklebufobject.h",
    "Include/pyarena.h",
    "Include/pycapsule.h",
    "Include/pyctype.h",
    "Include/py_curses.h",
    "Include/pydebug.h",
    "Include/pydtrace.h",
    "Include/pyerrors.h",
    "Include/pyexpat.h",
    "Include/pyfpe.h",
    "Include/pyhash.h",
    "Include/pylifecycle.h",
    "Include/pymacconfig.h",
    "Include/pymacro.h",
    "Include/pymath.h",
    "Include/pymem.h",
    "Include/pyport.h",
    "Include/pystate.h",
    "Include/pystrcmp.h",
    "Include/pystrhex.h",
    "Include/pystrtod.h",
    "Include/Python-ast.h",
    "Include/Python.h",
    "Include/pythonrun.h",
    "Include/pythread.h",
    "Include/pytime.h",
    "Include/rangeobject.h",
    "Include/setobject.h",
    "Include/sliceobject.h",
    "Include/structmember.h",
    "Include/structseq.h",
    "Include/symtable.h",
    "Include/sysmodule.h",
    "Include/token.h",
    "Include/traceback.h",
    "Include/tracemalloc.h",
    "Include/tupleobject.h",
    "Include/typeslots.h",
    "Include/ucnhash.h",
    "Include/unicodeobject.h",
    "Include/warnings.h",
    "Include/weakrefobject.h",
]

PUBLIC_CPYTHON_HEADERS = [
    "Include/cpython/abstract.h",
    "Include/cpython/dictobject.h",
    "Include/cpython/fileobject.h",
    "Include/cpython/initconfig.h",
    "Include/cpython/interpreteridobject.h",
    "Include/cpython/object.h",
    "Include/cpython/objimpl.h",
    "Include/cpython/pyerrors.h",
    "Include/cpython/pylifecycle.h",
    "Include/cpython/pymem.h",
    "Include/cpython/pystate.h",
    "Include/cpython/sysmodule.h",
    "Include/cpython/traceback.h",
    "Include/cpython/tupleobject.h",
    "Include/cpython/unicodeobject.h",
]

PRIVATE_PYTHON_HEADERS = [
    "Include/internal/pycore_accu.h",
    "Include/internal/pycore_atomic.h",
    "Include/internal/pycore_ceval.h",
    "Include/internal/pycore_code.h",
    "Include/internal/pycore_condvar.h",
    "Include/internal/pycore_context.h",
    "Include/internal/pycore_fileutils.h",
    "Include/internal/pycore_getopt.h",
    "Include/internal/pycore_gil.h",
    "Include/internal/pycore_hamt.h",
    "Include/internal/pycore_initconfig.h",
    "Include/internal/pycore_object.h",
    "Include/internal/pycore_pathconfig.h",
    "Include/internal/pycore_pyerrors.h",
    "Include/internal/pycore_pyhash.h",
    "Include/internal/pycore_pylifecycle.h",
    "Include/internal/pycore_pymem.h",
    "Include/internal/pycore_pystate.h",
    "Include/internal/pycore_traceback.h",
    "Include/internal/pycore_tupleobject.h",
    "Include/internal/pycore_warnings.h",
]

PYTHON_HEADERS = PUBLIC_PYTHON_HEADERS + PUBLIC_CPYTHON_HEADERS + PRIVATE_PYTHON_HEADERS

genrule(
    name = "include_public_headers",
    srcs = PUBLIC_PYTHON_HEADERS,
    outs = ["include/python3.8/" + h.partition("/")[2] for h in PUBLIC_PYTHON_HEADERS],
    cmd = "cp $(SRCS) $(RULEDIR)/include/python3.8/",
)

genrule(
    name = "include_public_cpython_headers",
    srcs = PUBLIC_CPYTHON_HEADERS,
    outs = ["include/python3.8/" + h.partition("/")[2] for h in PUBLIC_CPYTHON_HEADERS],
    cmd = "cp $(SRCS) $(RULEDIR)/include/python3.8/cpython",
)

genrule(
    name = "include_private_headers",
    srcs = PRIVATE_PYTHON_HEADERS,
    outs = ["include/python3.8/" + h.partition("/")[2] for h in PRIVATE_PYTHON_HEADERS],
    cmd = "cp $(SRCS) $(RULEDIR)/include/python3.8/internal/",
)

filegroup(
    name = "include_headers",
    srcs = [
        ":include_private_headers",
        ":include_public_cpython_headers",
        ":include_public_headers",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    hdrs = PYTHON_HEADERS,
    linkstatic = True,
)

# Corresponds to @local_config_python//:python_headers used by @pybind11.
cc_library(
    name = "local_config_python_headers",
    hdrs = PYTHON_HEADERS,
    includes = ["Include"],
    visibility = ["//visibility:public"],
)

# All of the builtin extensions that we need from the Modules/ directory.
# These need to be a separate target so we can set CORE_BUILTIN_COPTS.
cc_library(
    name = "builtin_extensions",
    srcs = [
        "Modules/_abc.c",
        "Modules/_asynciomodule.c",
        "Modules/_bisectmodule.c",
        "Modules/_bz2module.c",
        "Modules/_codecsmodule.c",
        "Modules/_collectionsmodule.c",
        "Modules/_contextvarsmodule.c",
        "Modules/_cryptmodule.c",
        "Modules/_csv.c",
        "Modules/_curses_panel.c",
        "Modules/_cursesmodule.c",
        "Modules/_datetimemodule.c",
        "Modules/_functoolsmodule.c",
        "Modules/_hashopenssl.c",
        "Modules/_heapqmodule.c",
        "Modules/_io/_iomodule.c",
        "Modules/_io/_iomodule.h",
        "Modules/_io/bufferedio.c",
        "Modules/_io/bytesio.c",
        "Modules/_io/clinic/_iomodule.c.h",
        "Modules/_io/clinic/bufferedio.c.h",
        "Modules/_io/clinic/bytesio.c.h",
        "Modules/_io/clinic/fileio.c.h",
        "Modules/_io/clinic/iobase.c.h",
        "Modules/_io/clinic/stringio.c.h",
        "Modules/_io/clinic/textio.c.h",
        "Modules/_io/clinic/winconsoleio.c.h",
        "Modules/_io/fileio.c",
        "Modules/_io/iobase.c",
        "Modules/_io/stringio.c",
        "Modules/_io/textio.c",
        "Modules/_io/winconsoleio.c",
        "Modules/_json.c",
        "Modules/_localemodule.c",
        "Modules/_lsprof.c",
        "Modules/_lzmamodule.c",
        "Modules/_math.c",
        "Modules/_math.h",
        "Modules/_multiprocessing/clinic/posixshmem.c.h",
        "Modules/_multiprocessing/multiprocessing.c",
        "Modules/_multiprocessing/multiprocessing.h",
        "Modules/_multiprocessing/posixshmem.c",
        "Modules/_multiprocessing/semaphore.c",
        "Modules/_opcode.c",
        "Modules/_operator.c",
        "Modules/_pickle.c",
        "Modules/_posixsubprocess.c",
        "Modules/_queuemodule.c",
        "Modules/_randommodule.c",
        "Modules/_sha3/clinic/sha3module.c.h",
        "Modules/_sha3/sha3module.c",
        "Modules/_sre.c",
        "Modules/_ssl.c",
        "Modules/_ssl_data.h",
        "Modules/_stat.c",
        "Modules/_struct.c",
        "Modules/_threadmodule.c",
        "Modules/_tracemalloc.c",
        "Modules/_weakref.c",
        "Modules/addrinfo.h",
        "Modules/arraymodule.c",
        "Modules/atexitmodule.c",
        "Modules/audioop.c",
        "Modules/binascii.c",
        "Modules/cjkcodecs/_codecs_cn.c",
        "Modules/cjkcodecs/_codecs_hk.c",
        "Modules/cjkcodecs/_codecs_iso2022.c",
        "Modules/cjkcodecs/_codecs_jp.c",
        "Modules/cjkcodecs/_codecs_kr.c",
        "Modules/cjkcodecs/_codecs_tw.c",
        "Modules/cjkcodecs/alg_jisx0201.h",
        "Modules/cjkcodecs/cjkcodecs.h",
        "Modules/cjkcodecs/clinic/multibytecodec.c.h",
        "Modules/cjkcodecs/emu_jisx0213_2000.h",
        "Modules/cjkcodecs/mappings_cn.h",
        "Modules/cjkcodecs/mappings_hk.h",
        "Modules/cjkcodecs/mappings_jisx0213_pair.h",
        "Modules/cjkcodecs/mappings_jp.h",
        "Modules/cjkcodecs/mappings_kr.h",
        "Modules/cjkcodecs/mappings_tw.h",
        "Modules/cjkcodecs/multibytecodec.c",
        "Modules/cjkcodecs/multibytecodec.h",
        "Modules/clinic/_abc.c.h",
        "Modules/clinic/_asynciomodule.c.h",
        "Modules/clinic/_bz2module.c.h",
        "Modules/clinic/_codecsmodule.c.h",
        "Modules/clinic/_collectionsmodule.c.h",
        "Modules/clinic/_contextvarsmodule.c.h",
        "Modules/clinic/_cryptmodule.c.h",
        "Modules/clinic/_curses_panel.c.h",
        "Modules/clinic/_cursesmodule.c.h",
        "Modules/clinic/_datetimemodule.c.h",
        "Modules/clinic/_dbmmodule.c.h",
        "Modules/clinic/_elementtree.c.h",
        "Modules/clinic/_gdbmmodule.c.h",
        "Modules/clinic/_hashopenssl.c.h",
        "Modules/clinic/_heapqmodule.c.h",
        "Modules/clinic/_lzmamodule.c.h",
        "Modules/clinic/_opcode.c.h",
        "Modules/clinic/_operator.c.h",
        "Modules/clinic/_pickle.c.h",
        "Modules/clinic/_queuemodule.c.h",
        "Modules/clinic/_randommodule.c.h",
        "Modules/clinic/_sre.c.h",
        "Modules/clinic/_ssl.c.h",
        "Modules/clinic/_statisticsmodule.c.h",
        "Modules/clinic/_struct.c.h",
        "Modules/clinic/_tkinter.c.h",
        "Modules/clinic/_tracemalloc.c.h",
        "Modules/clinic/_weakref.c.h",
        "Modules/clinic/_winapi.c.h",
        "Modules/clinic/arraymodule.c.h",
        "Modules/clinic/audioop.c.h",
        "Modules/clinic/binascii.c.h",
        "Modules/clinic/cmathmodule.c.h",
        "Modules/clinic/fcntlmodule.c.h",
        "Modules/clinic/gcmodule.c.h",
        "Modules/clinic/grpmodule.c.h",
        "Modules/clinic/itertoolsmodule.c.h",
        "Modules/clinic/mathmodule.c.h",
        "Modules/clinic/md5module.c.h",
        "Modules/clinic/posixmodule.c.h",
        "Modules/clinic/pwdmodule.c.h",
        "Modules/clinic/pyexpat.c.h",
        "Modules/clinic/resource.c.h",
        "Modules/clinic/selectmodule.c.h",
        "Modules/clinic/sha1module.c.h",
        "Modules/clinic/sha256module.c.h",
        "Modules/clinic/sha512module.c.h",
        "Modules/clinic/signalmodule.c.h",
        "Modules/clinic/spwdmodule.c.h",
        "Modules/clinic/symtablemodule.c.h",
        "Modules/clinic/unicodedata.c.h",
        "Modules/clinic/zlibmodule.c.h",
        "Modules/cmathmodule.c",
        "Modules/errnomodule.c",
        "Modules/faulthandler.c",
        "Modules/fcntlmodule.c",
        "Modules/gcmodule.c",
        "Modules/grpmodule.c",
        "Modules/hashtable.c",
        "Modules/hashtable.h",
        "Modules/itertoolsmodule.c",
        "Modules/mathmodule.c",
        "Modules/mmapmodule.c",
        "Modules/parsermodule.c",
        "Modules/posixmodule.c",
        "Modules/posixmodule.h",
        "Modules/pwdmodule.c",
        "Modules/pyexpat.c",
        "Modules/readline.c",
        "Modules/resource.c",
        "Modules/rotatingtree.c",
        "Modules/rotatingtree.h",
        "Modules/selectmodule.c",
        "Modules/signalmodule.c",
        "Modules/socketmodule.c",
        "Modules/socketmodule.h",
        "Modules/spwdmodule.c",
        "Modules/sre.h",
        "Modules/sre_constants.h",
        "Modules/sre_lib.h",
        "Modules/symtablemodule.c",
        "Modules/syslogmodule.c",
        "Modules/termios.c",
        "Modules/timemodule.c",
        "Modules/unicodedata.c",
        "Modules/unicodedata_db.h",
        "Modules/unicodename_db.h",
        "Modules/xxsubtype.c",
        "Modules/zlibmodule.c",
    ],
    copts = CORE_BUILTIN_COPTS,
    # for Modules/_ssl.c
    textual_hdrs = ["Modules/_ssl/debughelpers.c"],
    deps = [
        ":_blake2",
        ":_ctypes",
        ":_elementtree",
        ":_sqlite3",
        ":_ssl",
        ":kcp",
        "@net_zlib//:zlib",
        "@org_bzip_bzip2//:bz2",
        "@org_gnu_ncurses//:cursesw",
        "@org_gnu_ncurses//:panel",
        "@org_gnu_readline//:readline",
        "@org_openssl//:crypto_ssl",
        "@org_tukaani//:lzma",
    ],
)

cc_library(
    name = "kcp",
    srcs = [
        "Modules/_sha3/kcp/KeccakHash.h",
        "Modules/_sha3/kcp/KeccakP-1600-SnP.h",
        "Modules/_sha3/kcp/KeccakP-1600-SnP-opt32.h",
        "Modules/_sha3/kcp/KeccakP-1600-SnP-opt64.h",
        "Modules/_sha3/kcp/KeccakP-1600-opt64-config.h",
        "Modules/_sha3/kcp/KeccakSponge.h",
        "Modules/_sha3/kcp/KeccakSponge.inc",
        "Modules/_sha3/kcp/PlSnP-Fallback.inc",
        "Modules/_sha3/kcp/SnP-Relaned.h",
        "Modules/_sha3/kcp/align.h",
    ],
    textual_hdrs = [
        "Modules/_sha3/kcp/KeccakSponge.c",
        "Modules/_sha3/kcp/KeccakHash.c",
        "Modules/_sha3/kcp/KeccakP-1600-inplace32BI.c",
        "Modules/_sha3/kcp/KeccakP-1600-opt64.c",
        "Modules/_sha3/kcp/KeccakP-1600-64.macros",
        "Modules/_sha3/kcp/KeccakP-1600-unrolling.macros",
    ],
)

# The main python executable including the interpreter and stdlib extension
# modules. Note that any file that's exporting Python C API symbols needs to be
# directly in srcs.
cc_binary(
    name = "bin/python",
    srcs = [
        # $(info $$LIBRARY_OBJS is [${LIBRARY_OBJS}]) in Makefile, then translate .o -> .c/.h
        "Objects/abstract.c",
        "Objects/accu.c",
        "Objects/boolobject.c",
        "Objects/bytearrayobject.c",
        "Objects/bytes_methods.c",
        "Objects/bytesobject.c",
        "Objects/call.c",
        "Objects/capsule.c",
        "Objects/cellobject.c",
        "Objects/classobject.c",
        "Objects/clinic/bytearrayobject.c.h",
        "Objects/clinic/bytesobject.c.h",
        "Objects/clinic/codeobject.c.h",
        "Objects/clinic/complexobject.c.h",
        "Objects/clinic/descrobject.c.h",
        "Objects/clinic/dictobject.c.h",
        "Objects/clinic/enumobject.c.h",
        "Objects/clinic/floatobject.c.h",
        "Objects/clinic/funcobject.c.h",
        "Objects/clinic/listobject.c.h",
        "Objects/clinic/longobject.c.h",
        "Objects/clinic/memoryobject.c.h",
        "Objects/clinic/moduleobject.c.h",
        "Objects/clinic/odictobject.c.h",
        "Objects/clinic/structseq.c.h",
        "Objects/clinic/tupleobject.c.h",
        "Objects/clinic/typeobject.c.h",
        "Objects/clinic/unicodeobject.c.h",
        "Objects/codeobject.c",
        "Objects/complexobject.c",
        "Objects/descrobject.c",
        "Objects/dict-common.h",
        "Objects/dictobject.c",
        "Objects/enumobject.c",
        "Objects/exceptions.c",
        "Objects/fileobject.c",
        "Objects/floatobject.c",
        "Objects/frameobject.c",
        "Objects/funcobject.c",
        "Objects/genobject.c",
        "Objects/interpreteridobject.c",
        "Objects/iterobject.c",
        "Objects/listobject.c",
        "Objects/longobject.c",
        "Objects/memoryobject.c",
        "Objects/methodobject.c",
        "Objects/moduleobject.c",
        "Objects/namespaceobject.c",
        "Objects/object.c",
        "Objects/obmalloc.c",
        "Objects/odictobject.c",
        "Objects/picklebufobject.c",
        "Objects/rangeobject.c",
        "Objects/setobject.c",
        "Objects/sliceobject.c",
        "Objects/stringlib/asciilib.h",
        "Objects/stringlib/clinic/transmogrify.h.h",
        "Objects/stringlib/codecs.h",
        "Objects/stringlib/count.h",
        "Objects/stringlib/ctype.h",
        "Objects/stringlib/eq.h",
        "Objects/stringlib/fastsearch.h",
        "Objects/stringlib/find.h",
        "Objects/stringlib/find_max_char.h",
        "Objects/stringlib/join.h",
        "Objects/stringlib/localeutil.h",
        "Objects/stringlib/partition.h",
        "Objects/stringlib/replace.h",
        "Objects/stringlib/split.h",
        "Objects/stringlib/stringdefs.h",
        "Objects/stringlib/transmogrify.h",
        "Objects/stringlib/ucs1lib.h",
        "Objects/stringlib/ucs2lib.h",
        "Objects/stringlib/ucs4lib.h",
        "Objects/stringlib/undef.h",
        "Objects/stringlib/unicode_format.h",
        "Objects/stringlib/unicodedefs.h",
        "Objects/structseq.c",
        "Objects/tupleobject.c",
        "Objects/typeobject.c",
        "Objects/typeslots.inc",
        "Objects/unicodectype.c",
        "Objects/unicodeobject.c",
        "Objects/unicodetype_db.h",
        "Objects/weakrefobject.c",
        "Parser/acceler.c",
        "Parser/grammar1.c",
        "Parser/listnode.c",
        "Parser/myreadline.c",
        "Parser/node.c",
        "Parser/parser.c",
        "Parser/parser.h",
        "Parser/parsetok.c",
        "Parser/token.c",
        "Parser/tokenizer.c",
        "Parser/tokenizer.h",
        "Python/Python-ast.c",
        "Python/_warnings.c",
        "Python/asdl.c",
        "Python/ast.c",
        "Python/ast_opt.c",
        "Python/ast_unparse.c",
        "Python/bltinmodule.c",
        "Python/bootstrap_hash.c",
        "Python/ceval.c",
        "Python/ceval_gil.h",
        "Python/clinic/_warnings.c.h",
        "Python/clinic/bltinmodule.c.h",
        "Python/clinic/context.c.h",
        "Python/clinic/import.c.h",
        "Python/clinic/marshal.c.h",
        "Python/clinic/sysmodule.c.h",
        "Python/clinic/traceback.c.h",
        "Python/codecs.c",
        "Python/compile.c",
        "Python/condvar.h",
        "Python/context.c",
        "Python/dtoa.c",
        "Python/dynamic_annotations.c",
        "Python/dynload_shlib.c",
        "Python/errors.c",
        "Python/fileutils.c",
        "Python/formatter_unicode.c",
        "Python/frozen.c",
        "Python/frozenmain.c",
        "Python/future.c",
        "Python/getargs.c",
        "Python/getcompiler.c",
        "Python/getcopyright.c",
        "Python/getopt.c",
        "Python/getplatform.c",
        "Python/getversion.c",
        "Python/graminit.c",
        "Python/hamt.c",
        "Python/import.c",
        "Python/importdl.c",
        "Python/importdl.h",
        "Python/importlib.h",
        "Python/importlib_external.h",
        "Python/importlib_zipimport.h",
        "Python/initconfig.c",
        "Python/marshal.c",
        "Python/modsupport.c",
        "Python/mysnprintf.c",
        "Python/mystrtoul.c",
        "Python/opcode_targets.h",
        "Python/pathconfig.c",
        "Python/peephole.c",
        "Python/preconfig.c",
        "Python/pyarena.c",
        "Python/pyctype.c",
        "Python/pyfpe.c",
        "Python/pyhash.c",
        "Python/pylifecycle.c",
        "Python/pymath.c",
        "Python/pystate.c",
        "Python/pystrcmp.c",
        "Python/pystrhex.c",
        "Python/pystrtod.c",
        "Python/pythonrun.c",
        "Python/pytime.c",
        "Python/structmember.c",
        "Python/symtable.c",
        "Python/sysmodule.c",
        "Python/thread.c",
        "Python/thread_pthread.h",
        "Python/traceback.c",
        "Python/wordcode_helpers.h",
        # A few things in Modules/ that are not just for built in extensions
        "Modules/getpath.c",
        "Modules/main.c",
        # plus the main python target
        "Programs/python.c",
        # plus the modules configuration
        ":modules-config",
    ],
    copts = CORE_COPTS,
    linkopts = [
        "-pthread",
        "-ldl",
        "-lutil",
        "-lcrypt",
        "-lrt",
        "-Wl,--version-script=$(location @dbx_build_tools//thirdparty/cpython:symbols.lds)",
        "-Wl,--export-dynamic",
        "-lm",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":builtin_extensions",
        ":expat",
        ":headers",
        "@dbx_build_tools//thirdparty/cpython:buildinfo",
        "@dbx_build_tools//thirdparty/cpython:symbols.lds",
    ],
)

# Next are some extension modules that we statically link in but can't be
# directly in the main rule because they require special copts or dependencies.

cc_library(
    name = "_blake2",
    srcs = [
        "Modules/_blake2/blake2b_impl.c",
        "Modules/_blake2/blake2module.c",
        "Modules/_blake2/blake2ns.h",
        "Modules/_blake2/blake2s_impl.c",
        "Modules/_blake2/clinic/blake2b_impl.c.h",
        "Modules/_blake2/clinic/blake2s_impl.c.h",
        "Modules/_blake2/impl/blake2b-load-sse41.h",
        "Modules/_blake2/impl/blake2s-load-sse41.h",
        "Modules/hashlib.h",
    ],
    copts = CORE_BUILTIN_COPTS + ["-DBLAKE2_USE_SSE"],
    deps = [
        ":blake2",
        ":headers",
    ],
)

cc_library(
    name = "blake2",
    srcs = [
        "Modules/_blake2/impl/blake2-config.h",
        "Modules/_blake2/impl/blake2-impl.h",
        "Modules/_blake2/impl/blake2b-load-sse2.h",
        "Modules/_blake2/impl/blake2b-round.h",
        "Modules/_blake2/impl/blake2s-load-sse2.h",
        "Modules/_blake2/impl/blake2s-round.h",
    ],
    hdrs = ["Modules/_blake2/impl/blake2.h"],
    textual_hdrs = [
        "Modules/_blake2/impl/blake2b.c",
        "Modules/_blake2/impl/blake2b-ref.c",
        "Modules/_blake2/impl/blake2s.c",
        "Modules/_blake2/impl/blake2s-ref.c",
    ],
)

cc_library(
    name = "_sqlite3",
    srcs = [
        "Modules/_sqlite/cache.c",
        "Modules/_sqlite/cache.h",
        "Modules/_sqlite/connection.c",
        "Modules/_sqlite/connection.h",
        "Modules/_sqlite/cursor.c",
        "Modules/_sqlite/cursor.h",
        "Modules/_sqlite/microprotocols.c",
        "Modules/_sqlite/microprotocols.h",
        "Modules/_sqlite/module.c",
        "Modules/_sqlite/module.h",
        "Modules/_sqlite/prepare_protocol.c",
        "Modules/_sqlite/prepare_protocol.h",
        "Modules/_sqlite/row.c",
        "Modules/_sqlite/row.h",
        "Modules/_sqlite/statement.c",
        "Modules/_sqlite/statement.h",
        "Modules/_sqlite/util.c",
        "Modules/_sqlite/util.h",
    ],
    copts = CORE_BUILTIN_COPTS + [
        "-DSQLITE_OMIT_LOAD_EXTENSION",
        '-DMODULE_NAME=\\"sqlite3\\"',
    ],
    linkstatic = True,
    deps = [
        ":headers",
        "@org_sqlite",
    ],
)

cc_library(
    name = "_elementtree",
    srcs = [
        "Modules/_elementtree.c",
        "Modules/clinic/_elementtree.c.h",
    ],
    copts = CORE_BUILTIN_COPTS,
    deps = [
        ":expat",
        ":headers",
    ],
)

cc_library(
    name = "_ctypes",
    srcs = [
        "Modules/_ctypes/_ctypes.c",
        "Modules/_ctypes/callbacks.c",
        "Modules/_ctypes/callproc.c",
        "Modules/_ctypes/cfield.c",
        "Modules/_ctypes/ctypes.h",
        "Modules/_ctypes/ctypes_dlfcn.h",
        "Modules/_ctypes/stgdict.c",
    ],
    copts = CORE_BUILTIN_COPTS,
    deps = [
        ":headers",
        "@org_sourceware_libffi//:ffi",
    ],
)

cc_library(
    name = "_ssl",
    srcs = [
    ],
    copts = CORE_BUILTIN_COPTS,
    deps = [":headers"],
)

# Python's vendorized expat library.
cc_library(
    name = "expat",
    srcs = [
        "Modules/expat/ascii.h",
        "Modules/expat/asciitab.h",
        "Modules/expat/iasciitab.h",
        "Modules/expat/internal.h",
        "Modules/expat/latin1tab.h",
        "Modules/expat/nametab.h",
        "Modules/expat/siphash.h",
        "Modules/expat/utf8tab.h",
        "Modules/expat/xmlparse.c",
        "Modules/expat/xmlrole.c",
        "Modules/expat/xmlrole.h",
        "Modules/expat/xmltok.c",
        "Modules/expat/xmltok.h",
        "Modules/expat/xmltok_impl.h",
    ],
    hdrs = [
        "Modules/expat/expat.h",
        "Modules/expat/expat_external.h",
        "Modules/expat/pyexpatns.h",
    ],
    copts = [
        "-DHAVE_MEMMOVE",
        "-DXML_NS",
        "-DXML_DTD",
        "-DXML_CONTEXT_BYTES=1234",
        "-DBYTEORDER=1234",
        # This prevents expat from barfing because it can't detect any
        # randomness syscall. Python takes care of injecting entropy.
        "-DXML_POOR_ENTROPY",
    ],
    strip_include_prefix = "Modules/expat",
    deps = [
        ":expat-textual-hdrs",
    ],
)

cc_library(
    name = "expat-textual-hdrs",
    textual_hdrs = [
        "Modules/expat/xmltok_impl.c",
        "Modules/expat/xmltok_ns.c",
    ],
)

# We compile test extensions into shared objects, so they aren't included in our
# production build. It's also good to have a few shared extensions to verify we
# can load them.

cc_binary(
    name = "Lib/_ctypes_test.so",
    srcs = ["Modules/_ctypes/_ctypes_test.c"],
    copts = PY_COPTS,
    linkshared = True,
    deps = [":headers"],
)

cc_binary(
    name = "Lib/_testcapi.so",
    srcs = [
        "Modules/_testcapimodule.c",
        "Modules/testcapi_long.h",
    ],
    copts = PY_COPTS,
    linkshared = True,
    deps = [":headers"],
)

cc_binary(
    name = "Lib/xxsubtype.so",
    srcs = ["Modules/xxsubtype.c"],
    copts = PY_COPTS,
    linkshared = True,
    deps = [":headers"],
)

cc_binary(
    name = "Lib/_testimportmultiple.so",
    srcs = ["Modules/_testimportmultiple.c"],
    copts = PY_COPTS,
    linkshared = True,
    deps = [":headers"],
)

cc_binary(
    name = "Lib/_testmultiphase.so",
    srcs = ["Modules/_testmultiphase.c"],
    copts = PY_COPTS,
    linkshared = True,
    deps = [":headers"],
)

cc_binary(
    name = "Lib/_md5.so",
    srcs = [
        "Modules/clinic/md5module.c.h",
        "Modules/hashlib.h",
        "Modules/md5module.c",
    ],
    copts = PY_COPTS,
    linkshared = True,
    deps = [":headers"],
)

cc_binary(
    name = "Lib/_sha1.so",
    srcs = [
        "Modules/clinic/sha1module.c.h",
        "Modules/hashlib.h",
        "Modules/sha1module.c",
    ],
    copts = PY_COPTS,
    linkshared = True,
    deps = [":headers"],
)

cc_binary(
    name = "Lib/_sha256.so",
    srcs = [
        "Modules/clinic/sha256module.c.h",
        "Modules/hashlib.h",
        "Modules/sha256module.c",
    ],
    copts = PY_COPTS,
    linkshared = True,
    deps = [":headers"],
)

cc_binary(
    name = "Lib/_sha512.so",
    srcs = [
        "Modules/clinic/sha512module.c.h",
        "Modules/hashlib.h",
        "Modules/sha512module.c",
    ],
    copts = PY_COPTS,
    linkshared = True,
    deps = [":headers"],
)

# Static table of extension modules. All extension modules that are statically
# linked into the interpreter must be listed here, so that import can find them.
genrule(
    name = "modules-config",
    outs = ["Modules/config.c"],
    cmd = '''
cat <<'E_O_F' >$@
#include "Python.h"
extern PyObject* PyInit__abc(void);
extern PyObject* PyInit__ast(void);
extern PyObject* PyInit__asyncio(void);
extern PyObject* PyInit__bisect(void);
extern PyObject* PyInit__blake2(void);
extern PyObject* PyInit__bz2(void);
extern PyObject* PyInit__codecs(void);
extern PyObject* PyInit__codecs_cn(void);
extern PyObject* PyInit__codecs_hk(void);
extern PyObject* PyInit__codecs_iso2022(void);
extern PyObject* PyInit__codecs_jp(void);
extern PyObject* PyInit__codecs_kr(void);
extern PyObject* PyInit__codecs_tw(void);
extern PyObject* PyInit__collections(void);
extern PyObject* PyInit__contextvars(void);
extern PyObject* PyInit__crypt(void);
extern PyObject* PyInit__csv(void);
extern PyObject* PyInit__ctypes(void);
extern PyObject* PyInit__curses(void);
extern PyObject* PyInit__curses_panel(void);
extern PyObject* PyInit__datetime(void);
extern PyObject* PyInit__elementtree(void);
extern PyObject* PyInit__functools(void);
extern PyObject* PyInit__hashlib(void);
extern PyObject* PyInit__heapq(void);
extern PyObject* PyInit__io(void);
extern PyObject* PyInit__json(void);
extern PyObject* PyInit__locale(void);
extern PyObject* PyInit__lsprof(void);
extern PyObject* PyInit__lzma(void);
extern PyObject* PyInit__multibytecodec(void);
extern PyObject* PyInit__multiprocessing(void);
extern PyObject* PyInit__opcode(void);
extern PyObject* PyInit__operator(void);
extern PyObject* PyInit__pickle(void);
extern PyObject* PyInit__posixshmem(void);
extern PyObject* PyInit__posixsubprocess(void);
extern PyObject* PyInit__queue(void);
extern PyObject* PyInit__random(void);
extern PyObject* PyInit__sha3(void);
extern PyObject* PyInit__signal(void);
extern PyObject* PyInit__socket(void);
extern PyObject* PyInit__sqlite3(void);
extern PyObject* PyInit__sre(void);
extern PyObject* PyInit__ssl(void);
extern PyObject* PyInit__stat(void);
extern PyObject* PyInit__string(void);
extern PyObject* PyInit__struct(void);
extern PyObject* PyInit__symtable(void);
extern PyObject* PyInit__thread(void);
extern PyObject* PyInit__tracemalloc(void);
extern PyObject* PyInit__weakref(void);
extern PyObject* PyInit_array(void);
extern PyObject* PyInit_atexit(void);
extern PyObject* PyInit_audioop(void);
extern PyObject* PyInit_binascii(void);
extern PyObject* PyInit_cmath(void);
extern PyObject* PyInit_errno(void);
extern PyObject* PyInit_faulthandler(void);
extern PyObject* PyInit_fcntl(void);
extern PyObject* PyInit_gc(void);
extern PyObject* PyInit_grp(void);
extern PyObject* PyInit__imp(void);
extern PyObject* PyInit_itertools(void);
extern PyObject* PyInit_math(void);
extern PyObject* PyInit_mmap(void);
extern PyObject* PyInit_parser(void);
extern PyObject* PyInit_posix(void);
extern PyObject* PyInit_pwd(void);
extern PyObject* PyInit_pyexpat(void);
extern PyObject* PyInit_readline(void);
extern PyObject* PyInit_resource(void);
extern PyObject* PyInit_select(void);
extern PyObject* PyInit_spwd(void);
extern PyObject* PyInit_syslog(void);
extern PyObject* PyInit_termios(void);
extern PyObject* PyInit_time(void);
extern PyObject* PyInit_unicodedata(void);
extern PyObject* PyInit_zipimport(void);
extern PyObject* PyInit_zlib(void);
extern PyObject* PyMarshal_Init(void);
extern PyObject* _PyWarnings_Init(void);

struct _inittab _PyImport_Inittab[] = {
    {"_abc", PyInit__abc},
    {"_ast", PyInit__ast},
    {"_asyncio", PyInit__asyncio},
    {"_bisect", PyInit__bisect},
    {"_blake2", PyInit__blake2},
    {"_bz2", PyInit__bz2},
    {"_codecs", PyInit__codecs},
    {"_codecs_cn", PyInit__codecs_cn},
    {"_codecs_hk", PyInit__codecs_hk},
    {"_codecs_iso2022", PyInit__codecs_iso2022},
    {"_codecs_jp", PyInit__codecs_jp},
    {"_codecs_kr", PyInit__codecs_kr},
    {"_codecs_tw", PyInit__codecs_tw},
    {"_collections", PyInit__collections},
    {"_contextvars", PyInit__contextvars},
    {"_crypt", PyInit__crypt},
    {"_csv", PyInit__csv},
    {"_ctypes", PyInit__ctypes},
    {"_curses", PyInit__curses},
    {"_curses_panel", PyInit__curses_panel},
    {"_datetime", PyInit__datetime},
    {"_elementtree", PyInit__elementtree},
    {"_functools", PyInit__functools},
    {"_hashlib", PyInit__hashlib},
    {"_heapq", PyInit__heapq},
    {"_imp", PyInit__imp},
    {"_io", PyInit__io},
    {"_json", PyInit__json},
    {"_locale", PyInit__locale},
    {"_lsprof", PyInit__lsprof},
    {"_lzma", PyInit__lzma},
    {"_multibytecodec", PyInit__multibytecodec},
    {"_multiprocessing", PyInit__multiprocessing},
    {"_opcode", PyInit__opcode},
    {"_operator", PyInit__operator},
    {"_pickle", PyInit__pickle},
    {"_posixshmem", PyInit__posixshmem},
    {"_posixsubprocess", PyInit__posixsubprocess},
    {"_queue", PyInit__queue},
    {"_random", PyInit__random},
    {"_sha3", PyInit__sha3},
    {"_signal", PyInit__signal},
    {"_socket", PyInit__socket},
    {"_sqlite3", PyInit__sqlite3},
    {"_sre", PyInit__sre},
    {"_ssl", PyInit__ssl},
    {"_stat", PyInit__stat},
    {"_string", PyInit__string},
    {"_struct", PyInit__struct},
    {"_symtable", PyInit__symtable},
    {"_thread", PyInit__thread},
    {"_tracemalloc", PyInit__tracemalloc},
    {"_warnings", _PyWarnings_Init},
    {"_weakref", PyInit__weakref},
    {"array", PyInit_array},
    {"atexit", PyInit_atexit},
    {"audioop", PyInit_audioop},
    {"binascii", PyInit_binascii},
    {"builtins", NULL},
    {"cmath", PyInit_cmath},
    {"errno", PyInit_errno},
    {"faulthandler", PyInit_faulthandler},
    {"fcntl", PyInit_fcntl},
    {"gc", PyInit_gc},
    {"grp", PyInit_grp},
    {"itertools", PyInit_itertools},
    {"marshal", PyMarshal_Init},
    {"math", PyInit_math},
    {"mmap", PyInit_mmap},
    {"parser", PyInit_parser},
    {"posix", PyInit_posix},
    {"pwd", PyInit_pwd},
    {"pyexpat", PyInit_pyexpat},
    {"readline", PyInit_readline},
    {"resource", PyInit_resource},
    {"select", PyInit_select},
    {"spwd", PyInit_spwd},
    {"sys", NULL},
    {"syslog", PyInit_syslog},
    {"termios", PyInit_termios},
    {"time", PyInit_time},
    {"unicodedata", PyInit_unicodedata},
    {"zlib", PyInit_zlib},
    /* Sentinel */
    {0, 0}
};
E_O_F
''',
)

# Zipping of the stdlib.

genrule(
    name = "test-stdlib-zip",
    srcs = glob(["Lib/**"]) + [
        ":Lib/_ctypes_test.so",
        ":Lib/_md5.so",
        ":Lib/_sha1.so",
        ":Lib/_sha256.so",
        ":Lib/_sha512.so",
        ":Lib/_testcapi.so",
        ":Lib/_testimportmultiple.so",
        ":Lib/_testmultiphase.so",
        ":Lib/xxsubtype.so",
        ":sysconfigdata",
    ],
    outs = ["test-stdlib.zip"],
    cmd = "$(location @dbx_build_tools//thirdparty/cpython:zip-stdlib) sloppy $@ $(SRCS)",
    tools = ["@dbx_build_tools//thirdparty/cpython:zip-stdlib"],
)

filegroup(
    name = "production-stdlib",
    srcs = glob(
        ["Lib/**/*.py"],
        exclude = [
            "Lib/ensurepip/**",
            "Lib/idlelib/**",
            "Lib/lib-tk/**",
            "Lib/plat-*/**",
            "Lib/turtledemo/**",
            "**/test/**",
            "**/tests/**",
        ],
    ) + [":sysconfigdata"],
)

genrule(
    name = "stdlib-zip",
    srcs = [
        ":bin/python",
        ":production-stdlib",
        "Lib/lib2to3/Grammar.txt",
        "Lib/lib2to3/PatternGrammar.txt",
    ],
    outs = ["lib/python38.zip"],
    cmd = """
export ASAN_OPTIONS=detect_leaks=0
mkdir -p $(@D)/python3.8
touch $(@D)/python3.8/os.py
mkdir -p $(@D)/python3.8/lib-dynload
# First, create an initial stdlib zip with the standalone python binary.
$(location @pip_py3_interpreter//:interpreter) $(location @dbx_build_tools//thirdparty/cpython:zip_stdlib.py) sloppy $@ $(locations :production-stdlib)
# Then do the final zip with our python using the temporary stdlib.
cp $(location :bin/python) $(@D)/python-prime
mkdir -p tmp/Lib/lib2to3
gram=$$($(@D)/python-prime $(location @dbx_build_tools//thirdparty/cpython:gen_2to3_grammar.py) $(location Lib/lib2to3/Grammar.txt))
patgram=$$($(@D)/python-prime $(location @dbx_build_tools//thirdparty/cpython:gen_2to3_grammar.py) $(location Lib/lib2to3/PatternGrammar.txt))
$(@D)/python-prime $(location @dbx_build_tools//thirdparty/cpython:zip_stdlib.py) final $@ $(locations :production-stdlib) $$gram $$patgram
""",
    tools = [
        "@dbx_build_tools//thirdparty/cpython:gen_2to3_grammar.py",
        "@dbx_build_tools//thirdparty/cpython:zip_stdlib.py",
        "@pip_py3_interpreter//:interpreter",
    ],
)

# The final package.
_tar_files = {
    ":bin/python": "bin/python",
    ":stdlib-zip": "lib/python38.zip",
}

_tar_files.update({h: "include/python3.8/" + h.partition("/")[2] for h in PYTHON_HEADERS})

pkg_tar(
    name = "drte-python",
    empty_files = [
        # Sentinels for Python to find its prefix.
        "lib/python3.8/os.py",
        "lib/python3.8/lib-dynload/.SENTINEL",
    ],
    extension = "tar.xz",
    files = _tar_files,
    visibility = ["//visibility:public"],
)

genrule(
    name = "runtime_sentinels",
    outs = [
        "lib/python3.8/os.py",
        "lib/python3.8/lib-dynload/.SENTINEL",
    ],
    cmd = "touch $(OUTS)",
)

filegroup(
    name = "runtime",
    srcs = [
        "bin/python",
        "lib/python3.8/lib-dynload/.SENTINEL",
        "lib/python3.8/os.py",
        "lib/python38.zip",
    ],
    visibility = ["//visibility:public"],
)

# CPython dumps every variable in its Makefile into Lib/_sysconfigdata.py and
# then exposes the resulting dictionary as an API through the sysconfig and
# distutils.sysconfig modules. This is, of course, a terrible idea. What follows
# is a small subset of that experimentally shown to allow distutils to work. In
# practice, we override most of these things in vpip.
genrule(
    name = "sysconfigdata",
    outs = ["Lib/_sysconfigdata__linux_.py"],
    cmd = """
cat <<'E_O_F' >$@
build_time_vars = {
 'Py_ENABLE_SHARED': 0,
 'LIBDIR': '/usr/local/lib',
 'SO': '.so',
 'SHLIBS': '-lpthread -ldl  -lutil',
 'BINDIR': '/usr/local/bin',
 'VERSION': '3.8',
 'EXE': '',
 'exec_prefix': '/usr/local',
 'host': 'x86_64-pc-linux-gnu',
 'prefix': '/usr/local',
 'CC': 'gcc -pthread',
 'CCSHARED': '-fPIC',
 'OPT': '-fno-strict-aliasing -DNDEBUG -fwrapv -O3 -Wall -Wstrict-prototypes',
 'CFLAGS': '-fno-strict-aliasing -O2 -DNDEBUG -fwrapv -O3 -Wall -Wstrict-prototypes',
 'CFLAGSFORSHARED': '',
 'CXX': 'g++ -pthread',
 'LDSHARED': 'gcc -pthread -shared',
 'LDFLAGS': '',
 'AR': 'ar',
 'ARFLAGS': 'rc',
 'EXT_SUFFIX': '.cpython-38-x86_64-linux-gnu.so',
 'WITH_DOC_STRINGS': 1,
 'HAVE_GETRANDOM_SYSCALL': 1,
}
E_O_F
""",
)

genrule(
    name = "pyconfig-h",
    outs = ["Include/pyconfig.h"],
    cmd = '''
cat <<'E_O_F' >$@
#ifndef Py_PYCONFIG_H
#define Py_PYCONFIG_H
#define DOUBLE_IS_LITTLE_ENDIAN_IEEE754 1
#define ENABLE_IPV6 1
#define HAVE_ACCEPT4 1
#define HAVE_ACOSH 1
#define HAVE_ADDRINFO 1
#define HAVE_ALARM 1
#define HAVE_ALLOCA_H 1
#define HAVE_ASINH 1
#define HAVE_ASM_TYPES_H 1
#define HAVE_ATANH 1
#define HAVE_BIND_TEXTDOMAIN_CODESET 1
#define HAVE_BUILTIN_ATOMIC 1
#define HAVE_CHOWN 1
#define HAVE_CHROOT 1
#define HAVE_CLOCK 1
#define HAVE_CLOCK_GETRES 1
#define HAVE_CLOCK_GETTIME 1
#define HAVE_CLOCK_SETTIME 1
#define HAVE_COMPUTED_GOTOS 1
#define HAVE_CONFSTR 1
#define HAVE_COPYSIGN 1
#define HAVE_CRYPT_H 1
#define HAVE_CRYPT_R 1
#define HAVE_CTERMID 1
#define HAVE_CURSES_FILTER 1
#define HAVE_CURSES_H 1
#define HAVE_CURSES_HAS_KEY 1
#define HAVE_CURSES_IMMEDOK 1
#define HAVE_CURSES_IS_PAD 1
#define HAVE_CURSES_IS_TERM_RESIZED 1
#define HAVE_CURSES_RESIZETERM 1
#define HAVE_CURSES_RESIZE_TERM 1
#define HAVE_CURSES_SYNCOK 1
#define HAVE_CURSES_TYPEAHEAD 1
#define HAVE_CURSES_USE_ENV 1
#define HAVE_CURSES_WCHGAT 1
#define HAVE_DECL_ISFINITE 1
#define HAVE_DECL_ISINF 1
#define HAVE_DECL_ISNAN 1
#define HAVE_DECL_RTLD_DEEPBIND 1
#define HAVE_DECL_RTLD_GLOBAL 1
#define HAVE_DECL_RTLD_LAZY 1
#define HAVE_DECL_RTLD_LOCAL 1
#define HAVE_DECL_RTLD_MEMBER 0
#define HAVE_DECL_RTLD_NODELETE 1
#define HAVE_DECL_RTLD_NOLOAD 1
#define HAVE_DECL_RTLD_NOW 1
#define HAVE_DEVICE_MACROS 1
#define HAVE_DEV_PTMX 1
#define HAVE_DIRENT_D_TYPE 1
#define HAVE_DIRENT_H 1
#define HAVE_DIRFD 1
#define HAVE_DLFCN_H 1
#define HAVE_DLOPEN 1
#define HAVE_DUP2 1
#define HAVE_DUP3 1
#define HAVE_DYNAMIC_LOADING 1
#define HAVE_ENDIAN_H 1
#define HAVE_EPOLL 1
#define HAVE_EPOLL_CREATE1 1
#define HAVE_ERF 1
#define HAVE_ERFC 1
#define HAVE_ERRNO_H 1
#define HAVE_EXECV 1
#define HAVE_EXPM1 1
#define HAVE_FACCESSAT 1
#define HAVE_FCHDIR 1
#define HAVE_FCHMOD 1
#define HAVE_FCHMODAT 1
#define HAVE_FCHOWN 1
#define HAVE_FCHOWNAT 1
#define HAVE_FCNTL_H 1
#define HAVE_FDATASYNC 1
#define HAVE_FDOPENDIR 1
#define HAVE_FEXECVE 1
#define HAVE_FINITE 1
#define HAVE_FLOCK 1
#define HAVE_FORK 1
#define HAVE_FORKPTY 1
#define HAVE_FPATHCONF 1
#define HAVE_FSEEKO 1
#define HAVE_FSTATAT 1
#define HAVE_FSTATVFS 1
#define HAVE_FSYNC 1
#define HAVE_FTELLO 1
#define HAVE_FTIME 1
#define HAVE_FTRUNCATE 1
#define HAVE_FUTIMENS 1
#define HAVE_FUTIMES 1
#define HAVE_FUTIMESAT 1
#define HAVE_GAI_STRERROR 1
#define HAVE_GAMMA 1
#define HAVE_GCC_ASM_FOR_X64 1
#define HAVE_GCC_ASM_FOR_X87 1
#define HAVE_GCC_UINT128_T 1
#define HAVE_GETADDRINFO 1
#define HAVE_GETC_UNLOCKED 1
#define HAVE_GETENTROPY 1
#define HAVE_GETGRGID_R 1
#define HAVE_GETGRNAM_R 1
#define HAVE_GETGROUPLIST 1
#define HAVE_GETGROUPS 1
#define HAVE_GETHOSTBYNAME_R 1
#define HAVE_GETHOSTBYNAME_R_6_ARG 1
#define HAVE_GETITIMER 1
#define HAVE_GETLOADAVG 1
#define HAVE_GETLOGIN 1
#define HAVE_GETNAMEINFO 1
#define HAVE_GETPAGESIZE 1
#define HAVE_GETPEERNAME 1
#define HAVE_GETPGID 1
#define HAVE_GETPGRP 1
#define HAVE_GETPID 1
#define HAVE_GETPRIORITY 1
#define HAVE_GETPWENT 1
#define HAVE_GETPWNAM_R 1
#define HAVE_GETPWUID_R 1
#define HAVE_GETRANDOM_SYSCALL 1
#define HAVE_GETRESGID 1
#define HAVE_GETRESUID 1
#define HAVE_GETSID 1
#define HAVE_GETSPENT 1
#define HAVE_GETSPNAM 1
#define HAVE_GETTIMEOFDAY 1
#define HAVE_GETWD 1
#define HAVE_GRP_H 1
#define HAVE_HSTRERROR 1
#define HAVE_HTOLE64 1
#define HAVE_HYPOT 1
#define HAVE_IF_NAMEINDEX 1
#define HAVE_INET_ATON 1
#define HAVE_INET_PTON 1
#define HAVE_INITGROUPS 1
#define HAVE_INTTYPES_H 1
#define HAVE_KILL 1
#define HAVE_KILLPG 1
#define HAVE_LANGINFO_H 1
#define HAVE_LCHOWN 1
#define HAVE_LGAMMA 1
#define HAVE_LIBDL 1
#define HAVE_LIBINTL_H 1
#define HAVE_LIBREADLINE 1
#define HAVE_LINK 1
#define HAVE_LINKAT 1
#define HAVE_LINUX_CAN_BCM_H 1
#define HAVE_LINUX_CAN_H 1
#define HAVE_LINUX_CAN_RAW_FD_FRAMES 1
#define HAVE_LINUX_CAN_RAW_H 1
#define HAVE_LINUX_MEMFD_H 1
#define HAVE_LINUX_NETLINK_H 1
#define HAVE_LINUX_RANDOM_H 1
#define HAVE_LINUX_TIPC_H 1
#define HAVE_LINUX_VM_SOCKETS_H 1
#define HAVE_LOCKF 1
#define HAVE_LOG1P 1
#define HAVE_LOG2 1
#define HAVE_LONG_DOUBLE 1
#define HAVE_LSTAT 1
#define HAVE_LUTIMES 1
#define HAVE_MADVISE 1
#define HAVE_MAKEDEV 1
#define HAVE_MBRTOWC 1
#define HAVE_MEMORY_H 1
#define HAVE_MEMRCHR 1
#define HAVE_MKDIRAT 1
#define HAVE_MKFIFO 1
#define HAVE_MKFIFOAT 1
#define HAVE_MKNOD 1
#define HAVE_MKNODAT 1
#define HAVE_MKTIME 1
#define HAVE_MMAP 1
#define HAVE_MREMAP 1
#define HAVE_NCURSES_H 1
#define HAVE_NETPACKET_PACKET_H 1
#define HAVE_NET_IF_H 1
#define HAVE_NICE 1
#define HAVE_OPENAT 1
#define HAVE_OPENPTY 1
#define HAVE_PATHCONF 1
#define HAVE_PAUSE 1
#define HAVE_PIPE2 1
#define HAVE_POLL 1
#define HAVE_POLL_H 1
#define HAVE_POSIX_FADVISE 1
#define HAVE_POSIX_FALLOCATE 1
#define HAVE_POSIX_SPAWN 1
#define HAVE_POSIX_SPAWNP 1
#define HAVE_PREAD 1
#define HAVE_PREADV 1
#define HAVE_PRLIMIT 1
#define HAVE_PROTOTYPES 1
#define HAVE_PTHREAD_CONDATTR_SETCLOCK 1
#define HAVE_PTHREAD_GETCPUCLOCKID 1
#define HAVE_PTHREAD_H 1
#define HAVE_PTHREAD_KILL 1
#define HAVE_PTHREAD_SIGMASK 1
#define HAVE_PTY_H 1
#define HAVE_PUTENV 1
#define HAVE_PWRITE 1
#define HAVE_PWRITEV 1
#define HAVE_READLINK 1
#define HAVE_READLINKAT 1
#define HAVE_READV 1
#define HAVE_REALPATH 1
#define HAVE_RENAMEAT 1
#define HAVE_RL_APPEND_HISTORY 1
#define HAVE_RL_CATCH_SIGNAL 1
#define HAVE_RL_COMPLETION_APPEND_CHARACTER 1
#define HAVE_RL_COMPLETION_DISPLAY_MATCHES_HOOK 1
#define HAVE_RL_COMPLETION_MATCHES 1
#define HAVE_RL_COMPLETION_SUPPRESS_APPEND 1
#define HAVE_RL_PRE_INPUT_HOOK 1
#define HAVE_RL_RESIZE_TERMINAL 1
#define HAVE_ROUND 1
#define HAVE_SCHED_GET_PRIORITY_MAX 1
#define HAVE_SCHED_H 1
#define HAVE_SCHED_RR_GET_INTERVAL 1
#define HAVE_SCHED_SETAFFINITY 1
#define HAVE_SCHED_SETPARAM 1
#define HAVE_SCHED_SETSCHEDULER 1
#define HAVE_SEM_GETVALUE 1
#define HAVE_SEM_OPEN 1
#define HAVE_SEM_TIMEDWAIT 1
#define HAVE_SEM_UNLINK 1
#define HAVE_SENDFILE 1
#define HAVE_SETEGID 1
#define HAVE_SETEUID 1
#define HAVE_SETGID 1
#define HAVE_SETGROUPS 1
#define HAVE_SETHOSTNAME 1
#define HAVE_SETITIMER 1
#define HAVE_SETLOCALE 1
#define HAVE_SETPGID 1
#define HAVE_SETPGRP 1
#define HAVE_SETPRIORITY 1
#define HAVE_SETREGID 1
#define HAVE_SETRESGID 1
#define HAVE_SETRESUID 1
#define HAVE_SETREUID 1
#define HAVE_SETSID 1
#define HAVE_SETUID 1
#define HAVE_SETVBUF 1
#define HAVE_SHADOW_H 1
#define HAVE_SHM_OPEN 1
#define HAVE_SHM_UNLINK 1
#define HAVE_SIGACTION 1
#define HAVE_SIGALTSTACK 1
#define HAVE_SIGFILLSET 1
#define HAVE_SIGINFO_T_SI_BAND 1
#define HAVE_SIGINTERRUPT 1
#define HAVE_SIGNAL_H 1
#define HAVE_SIGPENDING 1
#define HAVE_SIGRELSE 1
#define HAVE_SIGTIMEDWAIT 1
#define HAVE_SIGWAIT 1
#define HAVE_SIGWAITINFO 1
#define HAVE_SNPRINTF 1
#define HAVE_SOCKADDR_ALG 1
#define HAVE_SOCKADDR_STORAGE 1
#define HAVE_SOCKETPAIR 1
#define HAVE_SPAWN_H 1
#define HAVE_SSIZE_T 1
#define HAVE_STATVFS 1
#define HAVE_STAT_TV_NSEC 1
#define HAVE_STDARG_PROTOTYPES 1
#define HAVE_STDINT_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STD_ATOMIC 1
#define HAVE_STRDUP 1
#define HAVE_STRFTIME 1
#define HAVE_STRINGS_H 1
#define HAVE_STRING_H 1
#define HAVE_STRSIGNAL 1
#define HAVE_STRUCT_PASSWD_PW_GECOS 1
#define HAVE_STRUCT_PASSWD_PW_PASSWD 1
#define HAVE_STRUCT_STAT_ST_BLKSIZE 1
#define HAVE_STRUCT_STAT_ST_BLOCKS 1
#define HAVE_STRUCT_STAT_ST_RDEV 1
#define HAVE_STRUCT_TM_TM_ZONE 1
#define HAVE_SYMLINK 1
#define HAVE_SYMLINKAT 1
#define HAVE_SYNC 1
#define HAVE_SYSCONF 1
#define HAVE_SYSEXITS_H 1
#define HAVE_SYS_EPOLL_H 1
#define HAVE_SYS_FILE_H 1
#define HAVE_SYS_IOCTL_H 1
#define HAVE_SYS_MMAN_H 1
#define HAVE_SYS_PARAM_H 1
#define HAVE_SYS_POLL_H 1
#define HAVE_SYS_RESOURCE_H 1
#define HAVE_SYS_SELECT_H 1
#define HAVE_SYS_SENDFILE_H 1
#define HAVE_SYS_SOCKET_H 1
#define HAVE_SYS_STATVFS_H 1
#define HAVE_SYS_STAT_H 1
#define HAVE_SYS_SYSCALL_H 1
#define HAVE_SYS_SYSMACROS_H 1
#define HAVE_SYS_TIMES_H 1
#define HAVE_SYS_TIME_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_SYS_UIO_H 1
#define HAVE_SYS_UN_H 1
#define HAVE_SYS_UTSNAME_H 1
#define HAVE_SYS_WAIT_H 1
#define HAVE_SYS_XATTR_H 1
#define HAVE_TCGETPGRP 1
#define HAVE_TCSETPGRP 1
#define HAVE_TEMPNAM 1
#define HAVE_TERMIOS_H 1
#define HAVE_TERM_H 1
#define HAVE_TGAMMA 1
#define HAVE_TIMEGM 1
#define HAVE_TIMES 1
#define HAVE_TMPFILE 1
#define HAVE_TMPNAM 1
#define HAVE_TMPNAM_R 1
#define HAVE_TM_ZONE 1
#define HAVE_TRUNCATE 1
#define HAVE_UNAME 1
#define HAVE_UNISTD_H 1
#define HAVE_UNLINKAT 1
#define HAVE_UNSETENV 1
#define HAVE_UTIMENSAT 1
#define HAVE_UTIMES 1
#define HAVE_UTIME_H 1
#define HAVE_WAIT3 1
#define HAVE_WAIT4 1
#define HAVE_WAITID 1
#define HAVE_WAITPID 1
#define HAVE_WCHAR_H 1
#define HAVE_WCSCOLL 1
#define HAVE_WCSFTIME 1
#define HAVE_WCSXFRM 1
#define HAVE_WMEMCMP 1
#define HAVE_WORKING_TZSET 1
#define HAVE_WRITEV 1
#define HAVE_X509_VERIFY_PARAM_SET1_HOST 1
#define HAVE_ZLIB_COPY 1
#define MVWDELCH_IS_EXPRESSION 1
#define PTHREAD_KEY_T_IS_COMPATIBLE_WITH_INT 1
#define PTHREAD_SYSTEM_SCHED_SUPPORTED 1
#define PY_COERCE_C_LOCALE 1
#define PY_FORMAT_SIZE_T "z"
#define PY_SSL_DEFAULT_CIPHERS 1
#define RETSIGTYPE void
#define SHM_NEEDS_LIBRT 1
#define SIZEOF_DOUBLE 8
#define SIZEOF_FLOAT 4
#define SIZEOF_FPOS_T 16
#define SIZEOF_INT 4
#define SIZEOF_LONG 8
#define SIZEOF_LONG_DOUBLE 16
#define SIZEOF_LONG_LONG 8
#define SIZEOF_OFF_T 8
#define SIZEOF_PID_T 4
#define SIZEOF_PTHREAD_KEY_T 4
#define SIZEOF_PTHREAD_T 8
#define SIZEOF_SHORT 2
#define SIZEOF_SIZE_T 8
#define SIZEOF_TIME_T 8
#define SIZEOF_UINTPTR_T 8
#define SIZEOF_VOID_P 8
#define SIZEOF_WCHAR_T 4
#define SIZEOF__BOOL 1
#define STDC_HEADERS 1
#define SYS_SELECT_WITH_SYS_TIME 1
#define TIME_WITH_SYS_TIME 1
#ifndef _ALL_SOURCE
# define _ALL_SOURCE 1
#endif
#ifndef _GNU_SOURCE
# define _GNU_SOURCE 1
#endif
#ifndef _POSIX_PTHREAD_SEMANTICS
# define _POSIX_PTHREAD_SEMANTICS 1
#endif
#ifndef _TANDEM_SOURCE
# define _TANDEM_SOURCE 1
#endif
#ifndef __EXTENSIONS__
# define __EXTENSIONS__ 1
#endif
#define WINDOW_HAS_FLAGS 1
#define WITH_DOC_STRINGS 1
#define WITH_PYMALLOC 1
#if defined AC_APPLE_UNIVERSAL_BUILD
# if defined __BIG_ENDIAN__
#  define WORDS_BIGENDIAN 1
# endif
#else
# ifndef WORDS_BIGENDIAN
# endif
#endif
#define _DARWIN_C_SOURCE 1
#define _FILE_OFFSET_BITS 64
#define _GNU_SOURCE 1
#define _LARGEFILE_SOURCE 1
#define _NETBSD_SOURCE 1
#define _POSIX_C_SOURCE 200809L
#define _PYTHONFRAMEWORK ""
#define _XOPEN_SOURCE 700
#define _XOPEN_SOURCE_EXTENDED 1
#define __BSD_VISIBLE 1
#ifndef __CHAR_UNSIGNED__
#endif
#if defined(__USLC__) && defined(__SCO_VERSION__)
#define STRICT_SYSV_CURSES /* Don't use ncurses extensions */
#endif
#endif /*Py_PYCONFIG_H*/
E_O_F
''',
)