from __future__ import annotations

import uuid
from errno import ENOTEMPTY

from ..spec import AbstractFileSystem
from .dirfs import _escapes_root

# Reserved directory, stored inside the upper filesystem, holding the overlay
# metadata (whiteouts, opaque markers, copy-up temp files). It never appears in
# the merged view and cannot be reached through ordinary overlay paths.
_META_ROOT = "/.fsspec-overlay"
_WHITEOUT = "whiteouts"
_OPAQUE = "opaque"
_TMP = "tmp"
# Suffixes keep marker files distinct from the parent directories created for
# markers of deeper paths (e.g. whiteouts/d.wh next to the whiteouts/d/ dir).
_WHITEOUT_SUFFIX = ".wh"
_OPAQUE_SUFFIX = ".op"


class OverlayFileSystem(AbstractFileSystem):
    """Synchronous overlay filesystem combining a read-only lower layer and
    an independent, writable upper layer.

    Reads are served from the upper layer first; a lower entry with the same
    name is shadowed once the upper gains an entry at that path. Directory
    listings merge the direct children of both layers, and any file/directory
    name clash is resolved in favour of the upper entry's identity, so
    ``ls`` and ``open`` always describe the same object.

    Deletions never touch the lower layer. Removing a lower-only path records
    a *whiteout* in the upper layer's reserved metadata space; removing a path
    that exists on both layers removes the upper copy and records a whiteout
    as well, so the lower copy cannot reappear. Deleting a directory and later
    recreating one with the same name marks the upper directory *opaque*,
    which keeps the old lower children hidden. Whiteouts and opaque markers
    are ordinary files stored on the upper filesystem, therefore they survive
    closing and recreating an :class:`OverlayFileSystem`; they are never kept
    in process memory only.

    Modifying a file that only exists on the lower layer with an ``r+`` or
    append open first performs a full copy-up into the upper layer. A failed
    copy-up never publishes a partial upper file and never writes to the
    lower layer. Brand new writes, and writes to files already on the upper
    layer, follow the upper filesystem's own contract; no cross-backend
    transaction is provided.

    Parameters
    ----------
    lower: AbstractFileSystem
        Synchronous filesystem treated as read-only. It must not be modified
        by anyone else while the overlay view is in use.
    upper: AbstractFileSystem
        Synchronous, independently writable filesystem. Whiteout and opaque
        metadata is persisted inside it.

    Notes
    -----
    The two layers must be different instances. The same logical path strings
    are used on both layers, so a layer rooted at a local directory is
    typically wrapped in :class:`~fsspec.implementations.dirfs.DirFileSystem`
    first. Cross-layer rename, asynchronous backends and stacks of more than
    two layers are not supported.
    """

    protocol = "overlay"
    root_marker = "/"

    def __init__(self, lower, upper, **storage_options):
        super().__init__(**storage_options)
        if lower is upper:
            raise ValueError(
                "lower and upper must be two different filesystem instances"
            )
        if getattr(lower, "async_impl", False) or getattr(
            upper, "async_impl", False
        ):
            raise ValueError(
                "OverlayFileSystem only supports synchronous filesystems"
            )
        self.lower = lower
        self.upper = upper
        # The persistent metadata space lives inside upper. Creating it here
        # also makes marker writes work on backends without implicit mkdir.
        self.upper.makedirs(self._b(f"{_META_ROOT}/{_WHITEOUT}"), exist_ok=True)
        self.upper.makedirs(self._b(f"{_META_ROOT}/{_OPAQUE}"), exist_ok=True)
        self.upper.makedirs(self._b(f"{_META_ROOT}/{_TMP}"), exist_ok=True)

    # ------------------------------------------------------------------
    # path handling
    # ------------------------------------------------------------------

    @classmethod
    def _strip_protocol(cls, path):
        p = super()._strip_protocol(path)
        if isinstance(p, list):
            return [cls._strip_protocol(item) for item in p]
        if p and not p.startswith("/"):
            p = "/" + p
        return p

    def _check(self, path):
        """Normalise a path and reject illegal ones before any backend call."""
        p = self._strip_protocol(path)
        if _escapes_root(p):
            raise ValueError(
                f"path {path!r} escapes the root of the overlay filesystem"
            )
        if p == _META_ROOT or p.startswith(_META_ROOT + "/"):
            raise PermissionError(
                f"path {p!r} is reserved for overlay metadata"
            )
        return p

    @staticmethod
    def _b(p):
        """Translate an overlay path into the path used on either backend."""
        if p == "/":
            return ""
        return p.removeprefix("/")

    @staticmethod
    def _child(parent, base):
        if parent == "/":
            return "/" + base
        return parent + "/" + base

    # ------------------------------------------------------------------
    # persistent metadata (whiteouts / opaque markers)
    # ------------------------------------------------------------------

    def _marker(self, kind, p):
        rel = p.lstrip("/")
        suffix = _WHITEOUT_SUFFIX if kind == _WHITEOUT else _OPAQUE_SUFFIX
        return f"{_META_ROOT}/{kind}/{rel}{suffix}"

    def _has_marker(self, kind, p):
        # isfile, not exists: a same-named directory must never count as a
        # marker on backends with implicit directories (e.g. MemoryFileSystem).
        return self.upper.isfile(self._b(self._marker(kind, p)))

    def _set_marker(self, kind, p):
        marker = self._marker(kind, p)
        self.upper.makedirs(self._b(self._parent(marker)), exist_ok=True)
        self.upper.pipe_file(self._b(marker), b"")

    def _clear_marker(self, kind, p):
        marker = self._b(self._marker(kind, p))
        if self.upper.isfile(marker):
            self.upper.rm_file(marker)

    def _is_whiteouted(self, p):
        return self._has_marker(_WHITEOUT, p)

    def _is_opaque(self, p):
        return self._has_marker(_OPAQUE, p)

    def _whiteout(self, p):
        self._set_marker(_WHITEOUT, p)

    def _materialize_dir(self, p):
        """p (and its ancestors) now exist as upper directories.

        A whiteout on any of them is converted into an opaque marker so the
        hidden lower entries stay hidden after the name is recreated.
        """
        cur = p
        while cur and cur != "/":
            if self._is_whiteouted(cur):
                self._clear_marker(_WHITEOUT, cur)
                self._set_marker(_OPAQUE, cur)
            cur = self._parent(cur)

    def _publish_file(self, p):
        """An upper file now occupies p.

        Its own whiteout is obsolete (the upper file shadows the lower one),
        and any whiteouted ancestor directory that was implicitly created
        becomes opaque.
        """
        self._clear_marker(_WHITEOUT, p)
        self._clear_marker(_OPAQUE, p)
        cur = self._parent(p)
        while cur and cur != "/":
            if self._is_whiteouted(cur):
                self._clear_marker(_WHITEOUT, cur)
                self._set_marker(_OPAQUE, cur)
            cur = self._parent(cur)

    # ------------------------------------------------------------------
    # inspection
    # ------------------------------------------------------------------

    def info(self, path, **kwargs):
        p = self._check(path)
        bp = self._b(p)
        if self.upper.exists(bp):
            info = dict(self.upper.info(bp, **kwargs))
            info["name"] = p
            return info
        if self._is_whiteouted(p):
            raise FileNotFoundError(p)
        if self.lower.exists(bp):
            info = dict(self.lower.info(bp, **kwargs))
            info["name"] = p
            return info
        raise FileNotFoundError(p)

    @staticmethod
    def _singleton(info, p, detail):
        if not detail:
            return [p]
        out = dict(info)
        out["name"] = p
        return [out]

    def ls(self, path, detail=True, **kwargs):
        p = self._check(path)
        bp = self._b(p)
        if self.upper.exists(bp):
            if self.upper.isfile(bp):
                return self._singleton(self.upper.info(bp), p, detail)
            return self._merge_dir(p, detail, **kwargs)
        # The name is absent from upper: a whiteout hides it entirely.
        if self._is_whiteouted(p) or self._is_opaque(p):
            raise FileNotFoundError(p)
        if self.lower.exists(bp):
            if self.lower.isfile(bp):
                return self._singleton(self.lower.info(bp), p, detail)
            return self._merge_dir(p, detail, **kwargs)
        raise FileNotFoundError(p)

    def _merge_dir(self, p, detail, **kwargs):
        bp = self._b(p)
        merged = {}
        if self.upper.isdir(bp):
            for entry in self.upper.ls(bp, detail=True, **kwargs):
                base = entry["name"].rstrip("/").rsplit("/", 1)[-1]
                name = self._child(p, base)
                # Never surface the reserved metadata space.
                if name == _META_ROOT or name.startswith(_META_ROOT + "/"):
                    continue
                out = dict(entry)
                out["name"] = name
                merged[base] = out
        # An opaque upper directory hides every lower child of that directory.
        if not self._is_opaque(p) and self.lower.isdir(bp):
            for entry in self.lower.ls(bp, detail=True, **kwargs):
                base = entry["name"].rstrip("/").rsplit("/", 1)[-1]
                name = self._child(p, base)
                # Upper identity wins clashes; whiteouts hide lower entries.
                if base in merged or self._is_whiteouted(name):
                    continue
                out = dict(entry)
                out["name"] = name
                merged[base] = out
        ordered = sorted(merged.values(), key=lambda entry: entry["name"])
        if detail:
            return ordered
        return [entry["name"] for entry in ordered]

    # ------------------------------------------------------------------
    # directory creation / removal
    # ------------------------------------------------------------------

    def mkdir(self, path, create_parents=True, **kwargs):
        p = self._check(path)
        if p == "/":
            return
        if self.exists(p):
            raise FileExistsError(p)
        self.upper.mkdir(
            self._b(p), create_parents=create_parents, **kwargs
        )
        # Recreating a whiteouted name must give an opaque directory.
        self._materialize_dir(p)

    def makedirs(self, path, exist_ok=False):
        try:
            self.mkdir(path, create_parents=True)
        except FileExistsError:
            if not exist_ok or self.isfile(path):
                raise

    def rm_file(self, path):
        p = self._check(path)
        if p == "/":
            raise ValueError("cannot delete the filesystem root")
        if not self.exists(p):
            raise FileNotFoundError(p)
        if self.isdir(p):
            raise IsADirectoryError(p)
        bp = self._b(p)
        if self.upper.isfile(bp):
            self.upper.rm_file(bp)
        # A lower copy (including one hidden behind an upper copy until now)
        # must stay hidden, so record a whiteout rather than touching lower.
        if self.lower.exists(bp):
            self._whiteout(p)
        self._clear_marker(_OPAQUE, p)

    def rmdir(self, path):
        p = self._check(path)
        if p == "/":
            raise ValueError("cannot delete the filesystem root")
        if not self.exists(p):
            raise FileNotFoundError(p)
        if self.isfile(p):
            raise NotADirectoryError(p)
        if self.ls(p, detail=False):
            raise OSError(ENOTEMPTY, "Directory not empty", p)
        bp = self._b(p)
        if self.upper.isdir(bp):
            self.upper.rmdir(bp)
        if self.lower.exists(bp):
            self._whiteout(p)
        self._clear_marker(_OPAQUE, p)

    def rm(self, path, recursive=False, maxdepth=None):
        if isinstance(path, str):
            checked = [self._check(path)]
        else:
            checked = [self._check(item) for item in path]
        if any(p == "/" for p in checked):
            raise ValueError("cannot delete the filesystem root")
        # expand_path/find are already whiteout- and opaque-aware; mirror the
        # MemoryFileSystem deletion loop so non-recursive rm on a non-empty
        # directory and maxdepth truncation keep their standard semantics.
        paths = self.expand_path(
            path, recursive=recursive, maxdepth=maxdepth
        )
        existed = {p for p in paths if self.exists(p)}
        for p in reversed(paths):
            if self.isfile(p):
                self.rm_file(p)
            elif not self.exists(p):
                if p not in existed:
                    raise FileNotFoundError(p)
                continue
            elif maxdepth is not None and self.ls(p, detail=False):
                # entries deeper than maxdepth keep this directory alive
                continue
            else:
                self.rmdir(p)

    # ------------------------------------------------------------------
    # file opening and copy-up
    # ------------------------------------------------------------------

    def _copy_up(self, p):
        """Fully copy a lower-only file into upper before modification."""
        if self._is_whiteouted(p):
            raise FileNotFoundError(p)
        if self.isdir(p):
            raise IsADirectoryError(p)
        bp = self._b(p)
        tmp = self._b(f"{_META_ROOT}/{_TMP}/{uuid.uuid4().hex}")
        try:
            # lower is only ever opened for reading
            data = self.lower.cat_file(bp)
            parent = self._parent(p)
            bparent = self._b(parent)
            if parent != "/" and not self.upper.exists(bparent):
                self.upper.makedirs(bparent, exist_ok=True)
            with self.upper.open(tmp, mode="wb") as f:
                f.write(data)
            # Intra-upper move; cross-layer rename is intentionally not used.
            self.upper.mv(tmp, bp)
        except Exception:
            # Never publish a half-written upper file, never touch lower.
            if self.upper.exists(tmp):
                try:
                    self.upper.rm(tmp, recursive=True)
                except OSError:
                    pass
            raise
        self._publish_file(p)

    def _open(self, path, mode="rb", block_size=None, **kwargs):
        p = self._check(path)
        bp = self._b(p)
        if mode in {"rb+", "wb+", "ab+", "xb+"}:
            mode = mode[0] + "+b"

        if "x" in mode:
            if self.exists(p):
                raise FileExistsError(p)
            f = self.upper.open(
                bp, mode=mode, block_size=block_size, **kwargs
            )
            self._publish_file(p)
            return f

        if "w" in mode:
            # Brand new writes follow the upper contract verbatim.
            f = self.upper.open(
                bp, mode=mode, block_size=block_size, **kwargs
            )
            self._publish_file(p)
            return f

        if "a" in mode:
            if self.upper.isfile(bp):
                # Already copied up: plain upper behaviour.
                return self.upper.open(
                    bp, mode=mode, block_size=block_size, **kwargs
                )
            if not self._is_whiteouted(p) and self.lower.isfile(bp):
                self._copy_up(p)
                return self.upper.open(
                    bp, mode=mode, block_size=block_size, **kwargs
                )
            # Append to a missing path creates it, like builtin open().
            f = self.upper.open(
                bp, mode=mode, block_size=block_size, **kwargs
            )
            self._publish_file(p)
            return f

        if "+" in mode:
            # r+ on a lower-only file requires a full copy-up first.
            if not self.upper.isfile(bp):
                self._copy_up(p)
            return self.upper.open(
                bp, mode=mode, block_size=block_size, **kwargs
            )

        # Plain read: upper wins, whiteouts and lower follow.
        if self.upper.isfile(bp):
            return self.upper.open(
                bp, mode="rb", block_size=block_size, **kwargs
            )
        if self._is_whiteouted(p):
            raise FileNotFoundError(p)
        # The merged identity may be a directory even when the lower layer
        # carries a file at the same name (an upper directory wins the clash);
        # open() must agree with ls()/info(), so reject before falling back.
        if self.isdir(p):
            raise IsADirectoryError(p)
        if self.lower.isfile(bp):
            return self.lower.open(
                bp, mode="rb", block_size=block_size, **kwargs
            )
        raise FileNotFoundError(p)

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------

    def invalidate_cache(self, path=None):
        super().invalidate_cache(path)
        if path is None:
            self.upper.invalidate_cache(None)
            self.lower.invalidate_cache(None)
        else:
            p = self._strip_protocol(path)
            self.upper.invalidate_cache(self._b(p))
            self.lower.invalidate_cache(self._b(p))

    def __repr__(self):
        return f"{self.__class__.__qualname__}(lower={self.lower!r}, upper={self.upper!r})"
