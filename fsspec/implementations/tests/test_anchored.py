"""Tests for :class:`AnchoredLocalFileSystem`.

The openat2-based guarantees are Linux-specific.  Every proprietary test is
gated on the *probed* capability with ``anchored_only``: on platforms without
the required semantics the tests skip instead of weakening the assertions.
The capability-error test, however, runs everywhere and checks that
unsupported platforms get an explicit error rather than a silent fallback.
"""

import gc
import os
import pickle
import sys
import threading

import pytest

import fsspec
from fsspec.implementations import anchored as anchored_mod
from fsspec.implementations.anchored import (
    ANCHORED_RESOLVE,
    AnchoredLocalFileSystem,
    CapabilityError,
    PathEscapeError,
    SymlinkDeniedError,
    UnsafePathError,
    anchored_open_supported,
)

anchored_only = pytest.mark.skipif(
    not anchored_open_supported(),
    reason=(
        "requires Linux openat2(2) with RESOLVE_BENEATH, "
        "RESOLVE_NO_SYMLINKS and RESOLVE_NO_MAGICLINKS"
    ),
)

DECOY = b"ANCHORED-DECOY-MUST-NEVER-BE-READ\n"


# ---------------------------------------------------------------------------
# Cross-platform capability handling
# ---------------------------------------------------------------------------


def test_capability_matches_platform():
    if sys.platform == "linux":
        # On Linux the probe itself must be a plain bool and must not raise.
        assert isinstance(anchored_open_supported(), bool)
    else:
        assert anchored_open_supported() is False
        assert AnchoredLocalFileSystem.supported() is False


@pytest.mark.skipif(
    anchored_open_supported(),
    reason="capability error is only observable where openat2 is unsupported",
)
def test_unsupported_platform_raises_capability_error(tmp_path):
    with pytest.raises(CapabilityError):
        AnchoredLocalFileSystem(tmp_path)
    with pytest.raises(CapabilityError):
        AnchoredLocalFileSystem(path=str(tmp_path))


# ---------------------------------------------------------------------------
# Linux-only behaviour
# ---------------------------------------------------------------------------


@pytest.fixture
def root(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "data").write_bytes(b"real-data\n")
    (tmp_path / "sub" / "nested").write_bytes(b"nested\n")
    return tmp_path


@pytest.fixture
def anchored_fs(root):
    fs = AnchoredLocalFileSystem(root)
    yield fs
    if not fs.closed:
        fs.close()


def _open_fds():
    return set(os.listdir("/proc/self/fd"))


@anchored_only
def test_basic_open_info_ls(root, anchored_fs):
    # rb open through the anchored root fd
    with anchored_fs.open("data") as f:
        assert f.read() == b"real-data\n"
        assert f.seek(5) == 5
        assert f.read() == b"data\n"

    info = anchored_fs.info("data")
    assert info["name"] == "data"
    assert info["type"] == "file"
    assert info["size"] == len(b"real-data\n")
    assert info["islink"] is False
    assert info["mode"] & 0o777

    root_info = anchored_fs.info("")
    assert root_info["type"] == "directory"
    assert anchored_fs.info(".") == root_info

    names = anchored_fs.ls("", detail=False)
    assert set(names) == {"data", "sub"}
    details = {e["name"]: e for e in anchored_fs.ls("", detail=True)}
    assert details["sub"]["type"] == "directory"
    assert details["data"]["type"] == "file"

    assert anchored_fs.ls("data", detail=False) == ["data"]
    single = anchored_fs.ls("data", detail=True)
    assert len(single) == 1 and single[0]["type"] == "file"

    with pytest.raises(FileNotFoundError):
        anchored_fs.info("does-not-exist")

    # trailing slash is normalised, root aliases work
    assert anchored_fs.info("sub/")["type"] == "directory"


@anchored_only
def test_inherited_read_only_operations(anchored_fs):
    assert anchored_fs.cat_file("data") == b"real-data\n"
    assert anchored_fs.read_bytes("data") == b"real-data\n"
    assert anchored_fs.read_text("data") == "real-data\n"
    assert anchored_fs.head(4) == b"real"
    assert anchored_fs.tail(1) == b"\n"
    assert anchored_fs.exists("data") is True
    assert anchored_fs.exists("missing") is False
    assert anchored_fs.isfile("data") is True
    assert anchored_fs.isdir("sub") is True
    assert anchored_fs.isdir("data") is False
    assert anchored_fs.isfile("sub") is False
    assert anchored_fs.size("data") == len(b"real-data\n")

    found = anchored_fs.find("")
    assert "data" in found and "sub/nested" in found
    assert set(anchored_fs.glob("sub/*")) == {"sub/nested"}
    walked = [p for p, _, _ in anchored_fs.walk("")]
    assert "" in walked and "sub" in walked
    assert anchored_fs.du("") >= len(b"real-data\n") + len(b"nested\n")
    assert isinstance(anchored_fs.checksum("data"), str)
    assert isinstance(anchored_fs.ukey("data"), str)
    assert anchored_fs.cat(["data"]) == {"data": b"real-data\n"}


@anchored_only
@pytest.mark.parametrize(
    "bad",
    [
        "/etc/passwd",
        "anchored:///etc/passwd",
        "../secret",
        "sub/../../secret",
        "a\x00b",
        "sub//nested",
        "sub/./nested",
    ],
)
def test_unsafe_inputs_rejected(anchored_fs, bad):
    with pytest.raises(UnsafePathError):
        anchored_fs.open(bad)
    with pytest.raises(UnsafePathError):
        anchored_fs.info(bad)
    with pytest.raises(UnsafePathError):
        anchored_fs.ls(bad)


@anchored_only
def test_leaf_symlink_rejected_and_never_followed(root, tmp_path, anchored_fs):
    decoy = tmp_path / "decoy"
    decoy.write_bytes(DECOY)
    os.symlink(decoy, root / "link")

    with pytest.raises(SymlinkDeniedError):
        anchored_fs.open("link")
    with pytest.raises(SymlinkDeniedError):
        anchored_fs.info("link")
    with pytest.raises(SymlinkDeniedError):
        anchored_fs.cat_file("link")

    # Listing uses fstatat(NOFOLLOW): the link appears, its target never does.
    entries = {e["name"]: e for e in anchored_fs.ls("", detail=True)}
    entry = entries["link"]
    assert entry["type"] == "symlink"
    assert entry["islink"] is True
    assert "destination" not in entry
    assert DECOY not in repr(entry).encode()

    # The decoy is unreachable through every read entry point.
    assert anchored_fs.exists("link") is False
    assert anchored_fs.isfile("link") is False


@anchored_only
def test_parent_symlink_rejected(root, tmp_path, anchored_fs):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_bytes(DECOY)
    os.symlink(outside, root / "sub" / "escape")

    with pytest.raises(SymlinkDeniedError):
        anchored_fs.open("sub/escape/secret")
    with pytest.raises(SymlinkDeniedError):
        anchored_fs.info("sub/escape/secret")
    with pytest.raises(SymlinkDeniedError):
        anchored_fs.ls("sub/escape")
    # The genuinely nested file remains readable.
    assert anchored_fs.cat_file("sub/nested") == b"nested\n"


@anchored_only
def test_magic_link_constraints_and_proc_link(root, anchored_fs):
    # The atomic mask explicitly rejects magic links and all symlinks.
    assert ANCHORED_RESOLVE & anchored_mod.RESOLVE_NO_MAGICLINKS
    assert ANCHORED_RESOLVE & anchored_mod.RESOLVE_NO_SYMLINKS
    assert ANCHORED_RESOLVE & anchored_mod.RESOLVE_BENEATH

    # Magic links only exist inside procfs; reaching one requires traversing a
    # symlink, which the anchored resolution must refuse atomically.
    proc_link = root / "proc-self"
    os.symlink("/proc/self", proc_link)
    with pytest.raises(SymlinkDeniedError):
        anchored_fs.open("proc-self/cmdline")
    with pytest.raises(SymlinkDeniedError):
        anchored_fs.info("proc-self/cmdline")


@anchored_only
def test_root_rename_keeps_anchor(root, tmp_path, anchored_fs):
    moved = tmp_path / "root-moved"
    os.rename(root, moved)

    # The pinned fd keeps serving the original inode under the new name.
    assert anchored_fs.cat_file("data") == b"real-data\n"
    assert set(anchored_fs.ls("", detail=False)) == {"data", "sub"}

    # A fresh directory recreated at the *old* path must not become the anchor.
    root.mkdir()
    (root / "decoy").write_bytes(DECOY)
    assert set(anchored_fs.ls("", detail=False)) == {"data", "sub"}
    assert not anchored_fs.exists("decoy")
    with pytest.raises(FileNotFoundError):
        anchored_fs.info("decoy")
    assert anchored_fs.cat_file("data") == b"real-data\n"


@anchored_only
def test_root_path_replaced_keeps_anchor(root, tmp_path, anchored_fs):
    stash = tmp_path / "stash"
    os.rename(root, stash)
    root.mkdir()
    (root / "decoy").write_bytes(DECOY)

    # Instance still reads the old tree, never the replacement directory.
    assert anchored_fs.cat_file("data") == b"real-data\n"
    assert anchored_fs.cat_file("sub/nested") == b"nested\n"
    assert not anchored_fs.exists("decoy")
    names = anchored_fs.ls("", detail=False)
    assert "decoy" not in names and set(names) == {"data", "sub"}


@anchored_only
def test_barrier_controlled_symlink_swap_race(root, tmp_path, anchored_fs):
    """A check/replace race controlled by barriers cannot leak the decoy.

    The attacker atomically swaps the real file for a symlink to an external
    decoy while the reader is guaranteed to be inside the open window.  Every
    read either returns the original bytes or is refused - decoy bytes must
    never be observed because openat2 resolves atomically.
    """
    data = root / "data"
    backup = root / "data.real"
    decoy = tmp_path / "decoy"
    decoy.write_bytes(DECOY)

    iterations = 30
    barrier = threading.Barrier(2)
    failures = []
    observed = set()

    def attacker():
        try:
            for _ in range(iterations):
                barrier.wait(timeout=10)  # reader ready: swap in symlink
                os.replace(data, backup)
                os.symlink(decoy, data)
                barrier.wait(timeout=10)  # symlink is on disk during read
                os.unlink(data)
                os.replace(backup, data)
                barrier.wait(timeout=10)  # tree restored for next round
        except Exception as exc:  # pragma: no cover - failure diagnostic
            failures.append(exc)

    # A normal, uncontended read is guaranteed to return the real bytes; the
    # swap loop below only needs to prove decoy bytes can never surface.
    assert anchored_fs.cat_file("data") == b"real-data\n"
    observed.add(b"real-data\n")

    thread = threading.Thread(target=attacker)
    thread.start()
    try:
        for _ in range(iterations):
            barrier.wait(timeout=10)
            try:
                payload = anchored_fs.cat_file("data")
            except OSError:
                payload = b"<resolution-refused>"
            barrier.wait(timeout=10)
            barrier.wait(timeout=10)
            observed.add(payload)
            assert DECOY not in payload
    finally:
        thread.join()

    assert failures == []
    assert b"real-data\n" in observed
    assert all(DECOY not in chunk for chunk in observed)
    assert data.read_bytes() == b"real-data\n"


@anchored_only
def test_close_is_idempotent_and_blocks_new_operations(anchored_fs):
    fd = anchored_fs.root_fd
    os.fstat(fd)  # still open
    anchored_fs.close()
    assert anchored_fs.closed is True
    anchored_fs.close()  # idempotent
    anchored_fs.close()
    with pytest.raises(OSError):
        os.fstat(fd)  # the held root fd has been released

    with pytest.raises(ValueError):
        anchored_fs.open("data")
    with pytest.raises(ValueError):
        anchored_fs.info("data")
    with pytest.raises(ValueError):
        anchored_fs.ls("")


@anchored_only
def test_returned_file_handle_has_independent_lifecycle(anchored_fs):
    handle = anchored_fs.open("data")
    anchored_fs.close()
    assert anchored_fs.closed is True
    # The independent fd keeps working and closes on its own.
    assert handle.read() == b"real-data\n"
    handle.close()
    assert handle.closed is True


@anchored_only
@pytest.mark.parametrize(
    "method,args",
    [
        ("touch", ("x",)),
        ("pipe", ("x", b"v")),
        ("pipe_file", ("x", b"v")),
        ("write_text", ("x", "v")),
        ("write_bytes", ("x", b"v")),
        ("rm", ("x",)),
        ("rm_file", ("x",)),
        ("rmdir", ("x",)),
        ("mkdir", ("x",)),
        ("makedirs", ("x",)),
        ("mv", ("a", "b")),
        ("move", ("a", "b")),
        ("rename", ("a", "b")),
        ("delete", ("x",)),
        ("cp_file", ("a", "b")),
        ("copy", ("a", "b")),
        ("put", ("a", "b")),
        ("put_file", ("a", "b")),
        ("link", ("a", "b")),
        ("symlink", ("a", "b")),
        ("chmod", ("a", 0o644)),
    ],
)
def test_write_entry_points_explicitly_rejected(anchored_fs, method, args):
    with pytest.raises(PermissionError):
        getattr(anchored_fs, method)(*args)


@anchored_only
@pytest.mark.parametrize("mode", ["wb", "ab", "xb", "r+b", "w", "a", "x"])
def test_open_write_modes_rejected(anchored_fs, mode):
    with pytest.raises(PermissionError):
        anchored_fs.open("data", mode=mode)


@anchored_only
def test_text_read_mode_is_rb_wrapper(anchored_fs):
    # Text read mode is a TextIOWrapper around the anchored rb open, while
    # text write mode is refused like every other write.
    with anchored_fs.open("data", mode="r") as f:
        assert f.read() == "real-data\n"
    with pytest.raises(PermissionError):
        anchored_fs.open("data", mode="w")


@anchored_only
def test_instances_are_not_cached(root):
    first = AnchoredLocalFileSystem(root)
    second = AnchoredLocalFileSystem(root)
    try:
        assert first is not second
        assert first.root_fd != second.root_fd
        assert AnchoredLocalFileSystem.cachable is False
        assert AnchoredLocalFileSystem._cache == {}
        # No cached instance can be handed back from the string token cache.
        with pytest.raises(TypeError):
            AnchoredLocalFileSystem.current()
    finally:
        first.close()
        second.close()


@anchored_only
def test_live_fd_cannot_be_pickled(anchored_fs):
    with pytest.raises(pickle.PicklingError):
        pickle.dumps(anchored_fs)
    with pytest.raises(pickle.PicklingError):
        pickle.dumps(anchored_fs, protocol=pickle.HIGHEST_PROTOCOL)


@anchored_only
def test_no_fd_leak_on_construction_and_resolution_failures(root, tmp_path):
    gc.collect()
    before = _open_fds()

    with pytest.raises(FileNotFoundError):
        AnchoredLocalFileSystem(tmp_path / "no-such-root")
    outside = tmp_path / "outside"
    outside.mkdir()
    link_root = tmp_path / "link-root"
    os.symlink(outside, link_root)
    with pytest.raises(OSError):
        AnchoredLocalFileSystem(link_root)  # symlink in the root path

    fs = AnchoredLocalFileSystem(root)
    with pytest.raises(UnsafePathError):
        fs.open("../outside/secret")
    with pytest.raises(UnsafePathError):
        fs.open("/etc/passwd")
    os.symlink(outside / "secret", root / "leaf")
    with pytest.raises(SymlinkDeniedError):
        fs.open("leaf")
    with pytest.raises(SymlinkDeniedError):
        fs.info("leaf")
    with pytest.raises(FileNotFoundError):
        fs.info("missing")
    fs.close()

    gc.collect()
    assert _open_fds() == before


@anchored_only
def test_ls_vanished_entry_is_skipped_without_leak(root, monkeypatch, anchored_fs):
    (root / "vanish").write_bytes(b"x")
    real_stat = os.stat

    def fake_stat(name, dir_fd=None, follow_symlinks=True):
        if name == "vanish":
            raise FileNotFoundError(2, "vanished concurrently", name)
        return real_stat(name, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    gc.collect()
    before = _open_fds()
    monkeypatch.setattr(anchored_mod.os, "stat", fake_stat)
    names = anchored_fs.ls("", detail=False)
    assert "vanish" not in names
    assert set(names) == {"data", "sub"}
    gc.collect()
    assert _open_fds() == before


@anchored_only
def test_ls_unexpected_stat_error_propagates_without_leak(
    root, monkeypatch, anchored_fs
):
    (root / "boom").write_bytes(b"x")
    real_stat = os.stat

    def fake_stat(name, dir_fd=None, follow_symlinks=True):
        if name == "boom":
            raise RuntimeError("stat blew up")
        return real_stat(name, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    gc.collect()
    before = _open_fds()
    monkeypatch.setattr(anchored_mod.os, "stat", fake_stat)
    with pytest.raises(RuntimeError):
        anchored_fs.ls("")
    # The fs is still fully usable after the failed traversal.
    assert anchored_fs.cat_file("data") == b"real-data\n"
    gc.collect()
    assert _open_fds() == before


@anchored_only
def test_filesystem_factory_and_registry(root):
    cls = fsspec.get_filesystem_class("anchored")
    assert cls is AnchoredLocalFileSystem
    fs = fsspec.filesystem("anchored", path=str(root))
    try:
        assert isinstance(fs, AnchoredLocalFileSystem)
        assert fs.cat_file("data") == b"real-data\n"
        # Factory must not hand out a cached instance either.
        assert fsspec.filesystem("anchored", path=str(root)) is not fs
    finally:
        fs.close()


@anchored_only
def test_path_escape_error_hierarchy(anchored_fs):
    # Defence-in-depth: kernel-level escape refusal has a dedicated type that
    # remains an OSError and a PermissionError.
    assert issubclass(PathEscapeError, PermissionError)
    assert issubclass(SymlinkDeniedError, PermissionError)
    assert issubclass(UnsafePathError, ValueError)
    assert issubclass(CapabilityError, OSError)
