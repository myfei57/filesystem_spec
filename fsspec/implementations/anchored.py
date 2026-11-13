"""Anchored, strictly read-only local filesystem.

``AnchoredLocalFileSystem`` pins a root *directory file descriptor* at
construction time and performs every subsequent path resolution with
``openat2(2)`` and the atomic constraints

* ``RESOLVE_BENEATH``        - the resolved object must stay below the root fd;
* ``RESOLVE_NO_SYMLINKS``    - any symbolic link (including magic links) in
  the path makes the kernel fail the whole resolution;
* ``RESOLVE_NO_MAGICLINKS``  - magic links are rejected even on kernels that
  would treat them separately.

Because the kernel walks the whole path in one constrained operation, there
is no user-space "check then open" window in which a concurrent symlink swap
could redirect access outside the root.  The root is held open as an fd, so
renaming the root directory or replacing its original path with another
directory does not change what this instance reads.

The implementation is deliberately Linux-only.  On platforms or kernels that
cannot provide the required ``openat2`` semantics (the syscall itself landed
in 5.6, ``RESOLVE_NO_SYMLINKS`` in 5.12), construction raises
:class:`CapabilityError` instead of silently falling back to ``realpath`` /
string-based ``open``, which cannot offer the same guarantees.
"""
from __future__ import annotations

import ctypes
import errno
import os
import pickle
import platform
import stat
import sys
from functools import lru_cache

from ..spec import AbstractFileSystem
from ..utils import stringify_path

__all__ = [
    "AnchoredLocalFileSystem",
    "AnchoredFileSystemError",
    "CapabilityError",
    "UnsafePathError",
    "PathConstraintError",
    "PathEscapeError",
    "SymlinkDeniedError",
    "anchored_open_supported",
    "RESOLVE_NO_XDEV",
    "RESOLVE_NO_MAGICLINKS",
    "RESOLVE_BENEATH",
    "RESOLVE_IN_ROOT",
    "RESOLVE_NO_SYMLINKS",
]

# ---------------------------------------------------------------------------
# openat2(2) constants (see include/uapi/linux/openat2.h)
# ---------------------------------------------------------------------------

RESOLVE_NO_XDEV = 0x01
RESOLVE_NO_MAGICLINKS = 0x02
RESOLVE_BENEATH = 0x08
RESOLVE_IN_ROOT = 0x10
RESOLVE_NO_SYMLINKS = 0x20

#: Constraint mask applied to every path resolution performed by the fs.
ANCHORED_RESOLVE = (
    RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS
)

_AT_FDCWD = -100

# open(2) flags; numeric fallbacks keep the module importable on non-Linux
# platforms, where the capability probe short-circuits before they are used.
_O_RDONLY = os.O_RDONLY
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0o2000000)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0o200000)
_O_PATH = getattr(os, "O_PATH", 0o4000000)
_O_LARGEFILE = getattr(os, "O_LARGEFILE", 0)

# errno values that may be missing from the platform's errno module.
_ELOOP = getattr(errno, "ELOOP", 40)
_EXDEV = getattr(errno, "EXDEV", 18)

#: Per-architecture ``openat2`` syscall numbers.  glibc >= 2.36 also exposes a
#: direct ``openat2`` symbol, which is preferred when available.
_SYS_OPENAT2 = {
    "x86_64": 437,
    "aarch64": 441,
    "armv7l": 393,
    "armv6l": 393,
    "ppc64": 447,
    "ppc64le": 447,
    "s390x": 437,
    "riscv64": 437,
}


class _OpenHow(ctypes.Structure):
    """struct open_how { u64 flags; u64 mode; u64 resolve; }."""

    _fields_ = [
        ("flags", ctypes.c_uint64),
        ("mode", ctypes.c_uint64),
        ("resolve", ctypes.c_uint64),
    ]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AnchoredFileSystemError(OSError):
    """Base class for errors raised by the anchored filesystem."""


class CapabilityError(AnchoredFileSystemError):
    """The platform/kernel cannot provide the required openat2 constraints.

    Raised instead of falling back to a weaker string-based open that could
    not honor the anchoring guarantees.
    """


class UnsafePathError(AnchoredFileSystemError, ValueError):
    """The supplied relative path violates the input rules.

    Absolute paths, ``..`` components, NUL bytes and non-canonical empty/``.``
    components are rejected before any syscall is made.
    """


class PathConstraintError(AnchoredFileSystemError, PermissionError):
    """The kernel refused an atomic anchored resolution."""


class PathEscapeError(PathConstraintError):
    """Resolution would escape the root directory (``EXDEV``)."""


class SymlinkDeniedError(PathConstraintError):
    """Resolution crossed a symbolic or magic link (``ELOOP``)."""


# ---------------------------------------------------------------------------
# libc / openat2 plumbing
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _libc():
    return ctypes.CDLL(None, use_errno=True)


def _raw_openat2(dirfd, path, flags, resolve, mode=0):
    """Call openat2(2); raise OSError on failure, return the new fd otherwise.

    Only this function touches libc.  It performs *no* path pre-checks: the
    kernel resolves the whole path atomically under the given resolve mask.
    """
    libc = _libc()
    how = _OpenHow(flags=flags, mode=mode, resolve=resolve)
    size = ctypes.sizeof(_OpenHow)
    path_bytes = os.fsencode(path)

    direct = getattr(libc, "openat2", None)
    if direct is not None:
        direct.restype = ctypes.c_long
        rc = direct(
            ctypes.c_int(dirfd),
            ctypes.c_char_p(path_bytes),
            ctypes.byref(how),
            ctypes.c_size_t(size),
        )
    else:
        number = _SYS_OPENAT2.get(platform.machine())
        if number is None:
            raise CapabilityError(
                "openat2 syscall number is unknown for architecture "
                f"{platform.machine()!r}"
            )
        libc.syscall.restype = ctypes.c_long
        rc = libc.syscall(
            ctypes.c_long(number),
            ctypes.c_int(dirfd),
            ctypes.c_char_p(path_bytes),
            ctypes.byref(how),
            ctypes.c_size_t(size),
        )

    if rc < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), path)
    return int(rc)


@lru_cache(maxsize=1)
def anchored_open_supported():
    """Return True iff this platform can anchor resolutions via openat2.

    The result is probed, not assumed: a real constrained open is attempted
    and only a successful probe enables the filesystem.  Any failure
    (``ENOSYS`` for a missing syscall, ``EINVAL`` for an unknown resolve flag
    on older kernels, seccomp denial, unknown architecture, ...) means the
    capability is reported as unavailable.
    """
    if sys.platform != "linux":
        return False
    try:
        fd = _raw_openat2(
            _AT_FDCWD,
            ".",
            _O_RDONLY | _O_DIRECTORY | _O_CLOEXEC,
            ANCHORED_RESOLVE,
        )
    except (OSError, AttributeError, CapabilityError):
        return False
    try:
        os.close(fd)
    except OSError:
        pass
    return True


def _translate_open_error(exc, path):
    """Map kernel refusal errnos onto the explicit exception hierarchy."""
    if exc.errno == _ELOOP:
        raise SymlinkDeniedError(
            exc.errno,
            f"symbolic or magic link in {path!r} is not allowed by the "
            "anchored filesystem",
            path,
        ) from exc
    if exc.errno == _EXDEV:
        raise PathEscapeError(
            exc.errno,
            f"resolution of {path!r} would escape the anchored root",
            path,
        ) from exc
    raise exc


def _readonly(operation):
    """Build an explicit read-only rejection for a mutating entry point."""

    def denied(self, *args, **kwargs):
        raise PermissionError(
            f"{type(self).__name__} is strictly read-only: "
            f"{operation}() is not supported"
        )

    denied.__name__ = operation
    return denied


class AnchoredLocalFileSystem(AbstractFileSystem):
    """Read-only local filesystem anchored to an open root directory fd.

    All paths given to instances must be relative to the root.  They are
    resolved atomically with ``openat2`` ``RESOLVE_BENEATH |
    RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS`` against the directory fd
    saved at construction time, so:

    * absolute paths, ``..`` components and NUL bytes are rejected;
    * any symbolic or magic link in the path makes resolution fail, links can
      never be followed outside the tree;
    * renaming the root or replacing its original path with another directory
      does not redirect the instance - it keeps operating on the pinned
      directory inode;
    * there is no check-then-open race window.

    Already-open file objects are independent fds: moving the underlying
    inode after a file has been opened is outside the scope of the path
    resolution guarantee.

    Parameters
    ----------
    root: str or PathLike
        Root directory.  The path itself is opened with
        ``RESOLVE_NO_SYMLINKS`` so the pinned inode is exactly the named
        directory.  ``path`` / ``fo`` are accepted as aliases.
    """

    protocol = "anchored"
    root_marker = ""
    sep = "/"
    #: Instances pin a live fd and must never be reused from a string-keyed
    #: instance cache: every construction opens a fresh root descriptor.
    cachable = False
    local_file = True
    async_impl = False

    def __init__(self, root=None, path=None, fo=None, **storage_options):
        if not anchored_open_supported():
            raise CapabilityError(
                "AnchoredLocalFileSystem requires Linux openat2(2) with "
                "RESOLVE_BENEATH, RESOLVE_NO_SYMLINKS and "
                "RESOLVE_NO_MAGICLINKS (kernel >= 5.12). This platform or "
                "kernel does not provide the required atomic path-resolution "
                "constraints, and no weaker string-based fallback is offered."
            )

        candidates = [p for p in (root, path, fo) if p is not None]
        if len(candidates) != 1:
            raise TypeError(
                "exactly one of 'root', 'path' or 'fo' must be given"
            )
        root_path = os.path.abspath(stringify_path(candidates[0]))

        super().__init__(**storage_options)

        # Open the root without following any link in its own path.  The fd
        # (not the string path) is the only anchor used afterwards.
        try:
            root_fd = _raw_openat2(
                _AT_FDCWD,
                root_path,
                _O_RDONLY | _O_DIRECTORY | _O_CLOEXEC | _O_LARGEFILE,
                RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS,
            )
        except OSError as exc:
            _translate_open_error(exc, root_path)
            raise  # pragma: no cover - _translate_open_error always raises

        self._root_fd = root_fd
        self._closed = False
        # Diagnostic only; never consulted by path resolution.
        self.root_path = root_path

    # -- capability / introspection ----------------------------------------

    @classmethod
    def supported(cls):
        """Whether the required openat2 capability is available here."""
        return anchored_open_supported()

    @property
    def fsid(self):
        st = os.fstat(self._root_fd)
        return f"anchored:{st.st_dev}:{st.st_ino}"

    @property
    def root_fd(self):
        """The pinned root directory descriptor (diagnostic use)."""
        return self._root_fd

    @property
    def closed(self):
        """Whether the root descriptor has been released."""
        return self._closed

    def __repr__(self):
        state = "closed" if self._closed else f"fd={self._root_fd}"
        return f"{type(self).__name__}(root={self.root_path!r}, {state})"

    def _isfilestore(self):
        return True

    # -- lifecycle ----------------------------------------------------------

    def close(self):
        """Close the root descriptor.  Idempotent.

        Afterwards every new operation fails; file objects returned earlier
        own independent descriptors and follow their own lifecycle.
        """
        if self._closed:
            return
        self._closed = True
        fd = self._root_fd
        self._root_fd = -1
        try:
            os.close(fd)
        finally:
            self.invalidate_cache()

    def __getstate__(self):
        raise pickle.PicklingError(
            "AnchoredLocalFileSystem holds a live root directory file "
            "descriptor and cannot be pickled to another process"
        )

    def __reduce__(self):
        raise pickle.PicklingError(
            "AnchoredLocalFileSystem holds a live root directory file "
            "descriptor and cannot be pickled to another process"
        )

    # -- path handling ------------------------------------------------------

    @classmethod
    def _strip_protocol(cls, path):
        if isinstance(path, list):
            return [cls._strip_protocol(p) for p in path]
        path = stringify_path(path)
        for prefix in ("anchored://", "anchored:"):
            if path.startswith(prefix):
                return path[len(prefix) :]
        return path

    def _require_open(self):
        if self._closed:
            raise ValueError(
                f"{type(self).__name__} is closed; new operations are not allowed"
            )

    def _validate_relative(self, path):
        """Validate and normalise an input path; return a clean relative path.

        The kernel would reject most of these anyway, but validating here
        gives explicit, stable errors and guarantees no absolute/string path
        ever reaches a fallback open.
        """
        path = self._strip_protocol(path)
        if not isinstance(path, str):
            raise TypeError(
                f"path must be a string, got {type(path).__name__}"
            )
        if "\x00" in path:
            raise UnsafePathError(
                errno.EINVAL, "NUL byte in path is not allowed", path
            )
        if path.startswith("/"):
            raise UnsafePathError(
                errno.EINVAL,
                "absolute paths are not allowed; paths must be relative to "
                "the anchored root",
                path,
            )
        if path in ("", "."):
            return ""
        if path.endswith("/"):
            path = path.rstrip("/")
            if not path:
                return ""
        for part in path.split("/"):
            if part == "..":
                raise UnsafePathError(
                    errno.EINVAL,
                    "'..' path components are not allowed",
                    path,
                )
            if part in ("", "."):
                raise UnsafePathError(
                    errno.EINVAL,
                    "empty or '.' path components are not allowed",
                    path,
                )
        return path

    def _open_relative(self, path, flags):
        """Open a validated relative path against the root fd.

        Returns an fd owned by the caller, which must close it.  The root
        itself ("") is served by dup()ing the pinned descriptor, which also
        keeps ownership of the anchor with the filesystem.
        """
        self._require_open()
        rel = self._validate_relative(path)
        if rel == "":
            return os.dup(self._root_fd)
        try:
            return _raw_openat2(
                self._root_fd,
                rel,
                flags | _O_CLOEXEC | _O_LARGEFILE,
                ANCHORED_RESOLVE,
            )
        except OSError as exc:
            _translate_open_error(exc, rel)
            raise  # pragma: no cover - _translate_open_error always raises

    # -- read operations ----------------------------------------------------

    def _open(self, path, mode="rb", block_size=None, **kwargs):
        if mode != "rb":
            raise PermissionError(
                f"only binary read mode 'rb' is supported, got {mode!r}"
            )
        fd = self._open_relative(path, _O_RDONLY)
        try:
            # The buffered reader owns the independent fd and closes it on
            # its own close(); fs.close() does not touch it.
            return open(fd, mode="rb", closefd=True)
        except BaseException:  # pragma: no cover - open cannot really fail
            os.close(fd)
            raise

    @staticmethod
    def _stat_to_info(name, st):
        mode = st.st_mode
        if stat.S_ISDIR(mode):
            ftype = "directory"
        elif stat.S_ISREG(mode):
            ftype = "file"
        else:
            ftype = "other"
        return {
            "name": name,
            "size": st.st_size,
            "type": ftype,
            "created": getattr(st, "st_birthtime", st.st_ctime),
            "islink": False,
            "mode": st.st_mode,
            "uid": st.st_uid,
            "gid": st.st_gid,
            "mtime": st.st_mtime,
            "ino": st.st_ino,
            "nlink": st.st_nlink,
        }

    def info(self, path, **kwargs):
        """Stat an entry through an anchored, link-free resolution.

        An ``O_PATH`` descriptor is opened with the full constraint mask and
        fstat()ed: metadata never comes from a string-path ``stat`` that
        could follow an external link.
        """
        self._require_open()
        rel = self._validate_relative(path)
        fd = self._open_relative(rel, _O_PATH)
        try:
            st = os.fstat(fd)
        finally:
            os.close(fd)
        return self._stat_to_info(rel, st)

    def ls(self, path, detail=True, **kwargs):
        """List a directory without ever following links.

        Entries are fstatat()'d with ``AT_SYMLINK_NOFOLLOW`` relative to the
        anchored directory fd.  Symlink entries are reported from their own
        link metadata (type ``"symlink"``, no target information); opening or
        info()ing them still fails.  Entries that vanish between listing and
        stat are skipped (the explicit concurrent-disappearance policy), any
        other error is propagated, and external directory contents can never
        appear.
        """
        self._require_open()
        rel = self._validate_relative(path)
        try:
            dir_fd = self._open_relative(
                rel, _O_RDONLY | _O_DIRECTORY
            )
        except OSError as exc:
            if exc.errno == errno.ENOTDIR:
                info = self.info(rel)
                return [info] if detail else [rel]
            raise

        try:
            # os.scandir takes ownership of the fd handed to it, so hand it
            # a duplicate; the anchor descriptor ownership stays here.
            scan_fd = os.dup(dir_fd)
            try:
                entries = os.scandir(scan_fd)
            except BaseException:
                os.close(scan_fd)
                raise
            infos = []
            with entries:
                for entry in entries:
                    name = entry.name
                    try:
                        st = os.stat(
                            name,
                            dir_fd=dir_fd,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        # Concurrent disappearance: explicitly skip rather
                        # than retry or resolve anything by string path.
                        continue
                    child = name if rel == "" else f"{rel}/{name}"
                    if stat.S_ISLNK(st.st_mode):
                        info = self._stat_to_info(child, st)
                        info["type"] = "symlink"
                        info["islink"] = True
                        # No "destination": the target is never inspected.
                    else:
                        info = self._stat_to_info(child, st)
                    infos.append(info if detail else child)
        finally:
            os.close(dir_fd)
        return infos

    # -- explicit write/delete/rename rejection -----------------------------
    # Every mutating entry point fails loudly instead of accidentally
    # inheriting a string-path implementation from LocalFileSystem-like base
    # classes.  Generic aliases (makedir, mkdirs, cp, move, rename, delete,
    # upload, write_bytes, ...) dispatch through these canonical methods.

    touch = _readonly("touch")
    pipe = _readonly("pipe")
    pipe_file = _readonly("pipe_file")
    write_text = _readonly("write_text")
    write_bytes = _readonly("write_bytes")
    rm = _readonly("rm")
    rm_file = _readonly("rm_file")
    _rm = _readonly("_rm")
    rmdir = _readonly("rmdir")
    mkdir = _readonly("mkdir")
    makedirs = _readonly("makedirs")
    mv = _readonly("mv")
    move = _readonly("move")
    rename = _readonly("rename")
    delete = _readonly("delete")
    cp_file = _readonly("cp_file")
    copy = _readonly("copy")
    put = _readonly("put")
    put_file = _readonly("put_file")
    link = _readonly("link")
    symlink = _readonly("symlink")
    chmod = _readonly("chmod")
