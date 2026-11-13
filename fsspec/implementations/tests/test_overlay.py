import pytest

import fsspec
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.memory import MemoryFileSystem
from fsspec.implementations.overlay import (
    _META_ROOT,
    OverlayFileSystem,
)

WHITEOUT = ".fsspec-overlay/whiteouts"
OPAQUE = ".fsspec-overlay/opaque"
TMP = ".fsspec-overlay/tmp"


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------


@pytest.fixture(params=["memory", "local"])
def layers(request, tmp_path):
    """Paired, fully independent lower/upper filesystems."""
    if request.param == "memory":
        lower = MemoryFileSystem(global_store=False, skip_instance_cache=True)
        upper = MemoryFileSystem(global_store=False, skip_instance_cache=True)
        yield lower, upper
    else:
        ldir = tmp_path / "lower"
        udir = tmp_path / "upper"
        ldir.mkdir()
        udir.mkdir()
        lower = DirFileSystem(
            ldir.as_posix(), LocalFileSystem(), skip_instance_cache=True
        )
        upper = DirFileSystem(
            udir.as_posix(), LocalFileSystem(), skip_instance_cache=True
        )
        yield lower, upper


@pytest.fixture
def overlay(layers):
    lower, upper = layers
    return OverlayFileSystem(lower, upper, skip_instance_cache=True)


def reopen(layers):
    lower, upper = layers
    return OverlayFileSystem(lower, upper, skip_instance_cache=True)


def snapshot(fs):
    return {p: fs.cat_file(p) for p in fs.find("", detail=False)}


def whiteout_marker(rel_path):
    return f"{WHITEOUT}/{rel_path.lstrip('/')}.wh"


def opaque_marker(rel_path):
    return f"{OPAQUE}/{rel_path.lstrip('/')}.op"


class BoomError(RuntimeError):
    pass


def _boom(*args, **kwargs):
    raise BoomError()


# ----------------------------------------------------------------------
# construction
# ----------------------------------------------------------------------


def test_constructor_rejects_same_instance():
    fs = MemoryFileSystem(global_store=False, skip_instance_cache=True)
    with pytest.raises(ValueError, match="different filesystem instances"):
        OverlayFileSystem(fs, fs, skip_instance_cache=True)


def test_constructor_rejects_async_backend():
    class AsyncMemory(MemoryFileSystem):
        async_impl = True

    sync_fs = MemoryFileSystem(global_store=False, skip_instance_cache=True)
    async_fs = AsyncMemory(global_store=False, skip_instance_cache=True)
    with pytest.raises(ValueError, match="synchronous filesystems"):
        OverlayFileSystem(sync_fs, async_fs, skip_instance_cache=True)
    with pytest.raises(ValueError, match="synchronous filesystems"):
        OverlayFileSystem(async_fs, sync_fs, skip_instance_cache=True)


def test_registry_entry():
    assert fsspec.get_filesystem_class("overlay") is OverlayFileSystem


# ----------------------------------------------------------------------
# merged view: reads, listings, identity conflicts
# ----------------------------------------------------------------------


def test_read_prefers_upper_and_shadows_lower(overlay, layers):
    lower, upper = layers
    lower.pipe_file("lower_only", b"lower bytes")
    lower.pipe_file("both", b"lower version")
    upper.pipe_file("both", b"upper version")
    upper.pipe_file("upper_only", b"upper only")

    assert overlay.cat_file("/lower_only") == b"lower bytes"
    assert overlay.cat_file("/both") == b"upper version"
    assert overlay.cat_file("/upper_only") == b"upper only"

    assert overlay.info("/both")["type"] == "file"
    assert overlay.info("/both")["name"] == "/both"
    assert overlay.isfile("/lower_only")
    assert overlay.isfile("/upper_only")

    root = overlay.ls("/", detail=False)
    assert root == ["/both", "/lower_only", "/upper_only"]


def test_directory_listing_merges_direct_children(overlay, layers):
    lower, upper = layers
    lower.makedirs("d/sub", exist_ok=True)
    lower.pipe_file("d/lower_file", b"l")
    lower.pipe_file("d/sub/deep", b"deep")
    upper.makedirs("d", exist_ok=True)
    upper.pipe_file("d/upper_file", b"u")

    assert overlay.ls("/d", detail=False) == [
        "/d/lower_file",
        "/d/sub",
        "/d/upper_file",
    ]
    assert overlay.ls("/d/sub", detail=False) == ["/d/sub/deep"]
    assert overlay.isdir("/d")
    assert overlay.isdir("/d/sub")
    # every listed object is the same object info()/open() see
    for entry in overlay.ls("/d", detail=True):
        assert overlay.info(entry["name"])["type"] == entry["type"]
        assert overlay.info(entry["name"])["name"] == entry["name"]


def test_file_directory_name_conflict_takes_upper_identity(overlay, layers):
    lower, upper = layers
    # upper file vs lower directory of the same name
    lower.makedirs("clash/inner", exist_ok=True)
    lower.pipe_file("clash/inner/data", b"hidden")
    upper.pipe_file("clash", b"i am a file")
    # upper directory vs lower file of the same name
    lower.pipe_file("other", b"lower file")
    upper.makedirs("other", exist_ok=True)
    upper.pipe_file("other/inside", b"upper dir child")

    assert overlay.isfile("/clash")
    assert not overlay.isdir("/clash")
    assert overlay.cat_file("/clash") == b"i am a file"
    assert overlay.ls("/", detail=False) == ["/clash", "/other"]
    clash_entry = next(e for e in overlay.ls("/", detail=True) if e["name"] == "/clash")
    assert clash_entry["type"] == "file"

    assert overlay.isdir("/other")
    assert not overlay.isfile("/other")
    assert overlay.ls("/other", detail=False) == ["/other/inside"]
    # listing a file-like path behaves like the concrete filesystems
    assert overlay.ls("/clash", detail=False) == ["/clash"]
    with pytest.raises(IsADirectoryError):
        overlay.open("/other", "rb")


def test_find_walk_glob_see_merged_view_only(overlay, layers):
    lower, upper = layers
    lower.makedirs("tree/lower_dir", exist_ok=True)
    lower.pipe_file("tree/lower_dir/a", b"a")
    lower.pipe_file("tree/lower_only", b"l")
    upper.makedirs("tree", exist_ok=True)
    upper.pipe_file("tree/upper_only", b"u")

    found = overlay.find("/tree", detail=False)
    assert found == [
        "/tree/lower_dir/a",
        "/tree/lower_only",
        "/tree/upper_only",
    ]
    # walk() reports basenames in its file lists, like os.walk()
    walked = []
    for root, dirs, files in overlay.walk("/tree"):
        walked.extend(files)
    assert sorted(walked) == ["a", "lower_only", "upper_only"]
    assert overlay.glob("/tree/*") == [
        "/tree/lower_dir",
        "/tree/lower_only",
        "/tree/upper_only",
    ]


# ----------------------------------------------------------------------
# deletion: whiteouts
# ----------------------------------------------------------------------


def test_lower_rm_is_never_called(overlay, layers, monkeypatch):
    lower, upper = layers
    lower.pipe_file("f", b"lower")
    lower.makedirs("d", exist_ok=True)
    lower.pipe_file("d/g", b"g")

    for method in ("rm", "rm_file", "_rm", "rmdir"):
        monkeypatch.setattr(
            lower, method, lambda *a, **k: pytest.fail("lower was modified")
        )

    overlay.rm("/f")
    overlay.rm("/d", recursive=True)
    assert not overlay.exists("/f")
    assert not overlay.exists("/d")


def test_delete_lower_only_file_writes_persistent_whiteout(overlay, layers):
    lower, upper = layers
    lower.pipe_file("f", b"lower only")

    overlay.rm_file("/f")
    assert not overlay.exists("/f")
    with pytest.raises(FileNotFoundError):
        overlay.open("/f", "rb")
    # the marker really is persisted inside upper, not in process memory
    assert upper.isfile(whiteout_marker("f"))
    # lower itself was not touched
    assert lower.cat_file("f") == b"lower only"

    view = reopen(layers)
    assert not view.exists("/f")
    with pytest.raises(FileNotFoundError):
        view.open("/f", "rb")


def test_delete_path_on_both_layers_keeps_lower_hidden(overlay, layers):
    lower, upper = layers
    lower.pipe_file("f", b"lower")
    upper.pipe_file("f", b"upper")

    overlay.rm("/f")
    assert not overlay.exists("/f")
    assert not upper.isfile("f")
    assert upper.isfile(whiteout_marker("f"))
    assert lower.cat_file("f") == b"lower"

    view = reopen(layers)
    assert not view.exists("/f")
    assert lower.cat_file("f") == b"lower"


def test_upper_overwrite_then_delete(overlay, layers):
    lower, upper = layers
    lower.pipe_file("f", b"original lower")
    overlay.pipe_file("/f", b"overwritten")
    assert overlay.cat_file("/f") == b"overwritten"

    overlay.rm("/f")
    assert not overlay.exists("/f")
    assert lower.cat_file("f") == b"original lower"
    assert upper.isfile(whiteout_marker("f"))

    view = reopen(layers)
    assert not view.exists("/f")


def test_whiteout_is_cleared_when_name_is_recreated(overlay, layers):
    lower, upper = layers
    lower.pipe_file("f", b"lower")
    overlay.rm("/f")
    assert upper.isfile(whiteout_marker("f"))

    overlay.pipe_file("/f", b"brand new")
    assert overlay.cat_file("/f") == b"brand new"
    assert not upper.isfile(whiteout_marker("f"))
    # deleting the recreation hides the lower copy again
    overlay.rm("/f")
    assert not overlay.exists("/f")
    assert upper.isfile(whiteout_marker("f"))


def test_whiteout_hides_lower_directory_entries(overlay, layers):
    lower, upper = layers
    lower.makedirs("d/sub", exist_ok=True)
    lower.pipe_file("d/a", b"a")
    lower.pipe_file("d/sub/z", b"z")

    overlay.rm("/d/a")
    assert overlay.ls("/d", detail=False) == ["/d/sub"]
    overlay.rm("/d/sub/z")
    assert overlay.ls("/d/sub", detail=False) == []
    # the lower files are physically intact
    assert lower.cat_file("d/a") == b"a"
    assert lower.cat_file("d/sub/z") == b"z"

    view = reopen(layers)
    assert view.ls("/d", detail=False) == ["/d/sub"]
    assert view.ls("/d/sub", detail=False) == []


def test_rm_missing_path_raises(overlay):
    with pytest.raises(FileNotFoundError):
        overlay.rm("/does/not/exist")
    with pytest.raises(FileNotFoundError):
        overlay.rm_file("/nope")


def test_rm_accepts_lists_and_maxdepth(overlay, layers):
    lower, upper = layers
    lower.makedirs("d/sub", exist_ok=True)
    lower.pipe_file("d/a", b"a")
    lower.pipe_file("d/sub/b", b"b")
    lower.pipe_file("top", b"t")

    # maxdepth truncation: direct children go, deeper entries survive
    overlay.rm("/d", recursive=True, maxdepth=1)
    assert not overlay.exists("/d/a")
    assert overlay.isdir("/d/sub")
    assert overlay.exists("/d/sub/b")

    # batch deletion
    overlay.rm(["/top", "/d/sub/b"])
    assert not overlay.exists("/top")
    assert not overlay.exists("/d/sub/b")
    # lower bytes are untouched, whiteouts drive the hidden state
    assert lower.cat_file("top") == b"t"
    assert lower.cat_file("d/sub/b") == b"b"

    view = reopen(layers)
    assert not view.exists("/top")
    assert not view.exists("/d/a")
    assert not view.exists("/d/sub/b")
    assert view.isdir("/d/sub")


# ----------------------------------------------------------------------
# directory deletion, opaque semantics, non-empty rules
# ----------------------------------------------------------------------


def test_recursive_rm_and_recreate_is_opaque(overlay, layers):
    lower, upper = layers
    lower.makedirs("d/sub", exist_ok=True)
    lower.pipe_file("d/a", b"a")
    lower.pipe_file("d/sub/z", b"z")
    upper.makedirs("d", exist_ok=True)
    upper.pipe_file("d/u", b"u")

    overlay.rm("/d", recursive=True)
    assert not overlay.exists("/d")
    assert upper.isfile(whiteout_marker("d"))

    overlay.mkdir("/d")
    # opaque: none of the old lower children resurrect
    assert overlay.isdir("/d")
    assert overlay.ls("/d", detail=False) == []
    assert overlay.find("/d", detail=False) == []
    assert not overlay.exists("/d/a")
    assert not overlay.exists("/d/sub/z")
    # the whiteout became an opaque marker once the dir existed again
    assert not upper.isfile(whiteout_marker("d"))
    assert upper.isfile(opaque_marker("d"))
    # new content in the recreated dir works
    overlay.pipe_file("/d/new", b"n")
    assert overlay.cat_file("/d/new") == b"n"
    assert overlay.ls("/d", detail=False) == ["/d/new"]

    view = reopen(layers)
    assert view.ls("/d", detail=False) == ["/d/new"]
    assert not view.exists("/d/a")
    assert lower.cat_file("d/a") == b"a"


def test_non_recursive_rm_respects_non_empty_directories(overlay, layers):
    lower, upper = layers
    lower.makedirs("d", exist_ok=True)
    lower.pipe_file("d/a", b"a")

    with pytest.raises(OSError):
        overlay.rm("/d")
    with pytest.raises(OSError):
        overlay.rmdir("/d")
    # nothing was deleted and no whiteout leaked
    assert overlay.exists("/d")
    assert overlay.ls("/d", detail=False) == ["/d/a"]

    overlay.rm("/d", recursive=True)
    assert not overlay.exists("/d")


def test_rmdir_empty_lower_directory(overlay, layers):
    lower, upper = layers
    lower.makedirs("empty", exist_ok=True)

    overlay.rmdir("/empty")
    assert not overlay.exists("/empty")
    assert upper.isfile(whiteout_marker("empty"))
    assert lower.isdir("empty")

    view = reopen(layers)
    assert not view.exists("/empty")


def test_mkdir_existing_path_raises(overlay, layers):
    lower, upper = layers
    lower.pipe_file("f", b"x")
    lower.makedirs("d", exist_ok=True)
    with pytest.raises(FileExistsError):
        overlay.mkdir("/f")
    with pytest.raises(FileExistsError):
        overlay.mkdir("/d")
    # makedirs(exist_ok=True) tolerates an existing directory
    overlay.makedirs("/d", exist_ok=True)


def test_mkdir_recreate_under_whiteouted_ancestor(overlay, layers):
    lower, upper = layers
    lower.makedirs("a/b", exist_ok=True)
    lower.pipe_file("a/b/f", b"f")
    lower.pipe_file("a/other", b"o")

    overlay.rm("/a", recursive=True)
    # recreate a deep path; the resurrected ancestor must be opaque
    overlay.makedirs("/a/b", exist_ok=False)
    assert overlay.isdir("/a/b")
    assert overlay.ls("/a", detail=False) == ["/a/b"]
    assert overlay.ls("/a/b", detail=False) == []
    assert not overlay.exists("/a/other")
    assert upper.isfile(opaque_marker("a"))


def test_root_cannot_be_deleted(overlay, layers):
    lower, upper = layers
    lower.pipe_file("f", b"x")
    with pytest.raises((OSError, ValueError)):
        overlay.rm("/", recursive=True)
    assert overlay.exists("/f")


# ----------------------------------------------------------------------
# reserved metadata namespace
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [_META_ROOT, f"{_META_ROOT}/whiteouts", f"{_META_ROOT}/whiteouts/f.wh"],
)
def test_metadata_paths_are_forbidden(overlay, path):
    with pytest.raises(PermissionError):
        overlay.info(path)
    with pytest.raises(PermissionError):
        overlay.ls(path)
    with pytest.raises(PermissionError):
        overlay.open(path, "wb")
    with pytest.raises(PermissionError):
        overlay.pipe_file(path, b"x")
    with pytest.raises(PermissionError):
        overlay.mkdir(path)
    with pytest.raises(PermissionError):
        overlay.rm(path, recursive=True)


def test_metadata_never_listed(overlay, layers):
    lower, upper = layers
    lower.pipe_file("f", b"x")
    overlay.rm("/f")  # forces marker creation

    assert _META_ROOT not in overlay.ls("/", detail=False)
    assert all(
        not name.startswith(_META_ROOT) for name in overlay.find("", detail=False)
    )
    for root, dirs, files in overlay.walk(""):
        assert _META_ROOT not in dirs
        assert all(not f.startswith(_META_ROOT) for f in files)
    assert all(not name.startswith(_META_ROOT) for name in overlay.glob("**"))


# ----------------------------------------------------------------------
# path validation happens before any backend operation
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["..", "../", "../secret", "/..", "/../secret", "a/../../b", "foo/../.."],
)
def test_escape_paths_rejected_before_backend_calls(overlay, layers, monkeypatch, path):
    lower, upper = layers

    def fail(*args, **kwargs):
        raise AssertionError("backend must not be touched")

    for fs in (lower, upper):
        for method in ("info", "ls", "exists", "isfile", "isdir", "open"):
            monkeypatch.setattr(fs, method, fail)

    with pytest.raises(ValueError, match="escapes"):
        overlay.info(path)
    with pytest.raises(ValueError, match="escapes"):
        overlay.ls(path)
    with pytest.raises(ValueError, match="escapes"):
        overlay.open(path, "rb")
    with pytest.raises(ValueError, match="escapes"):
        overlay.mkdir(path)
    with pytest.raises(ValueError, match="escapes"):
        overlay.rm(path)


def test_internal_dotdot_that_stays_inside_root_is_allowed(overlay, layers):
    # only paths that actually escape the root are rejected; an internal
    # ".." segment must reach the backends as a normal path
    lower, upper = layers
    lower.makedirs("a", exist_ok=True)
    lower.pipe_file("a/f", b"x")
    try:
        overlay.info("/a/../a/f")
    except ValueError as e:
        pytest.fail(f"internal path rejected as an escape: {e}")
    except FileNotFoundError:
        # backends that treat ".." as a literal path part simply miss it
        pass


# ----------------------------------------------------------------------
# copy-up on modify
# ----------------------------------------------------------------------


def test_append_copy_up_lower_only_file(overlay, layers):
    lower, upper = layers
    lower.pipe_file("f", b"abc")

    with overlay.open("/f", "ab") as f:
        f.write(b"def")

    assert overlay.cat_file("/f") == b"abcdef"
    assert upper.isfile("f")
    # the lower copy is immutable from the overlay's point of view
    assert lower.cat_file("f") == b"abc"

    view = reopen(layers)
    assert view.cat_file("/f") == b"abcdef"
    # a second append goes straight to the existing upper file
    with view.open("/f", "ab") as f:
        f.write(b"!")
    assert view.cat_file("/f") == b"abcdef!"
    assert lower.cat_file("f") == b"abc"


def test_rplus_copy_up_lower_only_file(overlay, layers):
    lower, upper = layers
    lower.pipe_file("f", b"abcdef")

    with overlay.open("/f", "r+b") as f:
        f.seek(3)
        f.write(b"XYZ")

    assert overlay.cat_file("/f") == b"abcXYZ"
    assert upper.isfile("f")
    assert lower.cat_file("f") == b"abcdef"


def test_append_missing_file_creates_upper_file(overlay, layers):
    lower, upper = layers
    with overlay.open("/new", "ab") as f:
        f.write(b"data")
    assert overlay.cat_file("/new") == b"data"
    assert upper.isfile("new")
    assert not lower.exists("new")


def test_rplus_missing_file_raises(overlay):
    with pytest.raises(FileNotFoundError):
        overlay.open("/missing", "r+b")


def test_copy_up_failure_does_not_publish_half_file(overlay, layers, monkeypatch):
    lower, upper = layers
    lower.pipe_file("f", b"lower contents")

    with monkeypatch.context() as mp:
        mp.setattr(lower, "cat_file", _boom)
        with pytest.raises(BoomError):
            overlay.open("/f", "ab")

    # no upper file was published, lower is intact and still readable
    assert not upper.isfile("f")
    assert lower.cat_file("f") == b"lower contents"
    assert overlay.cat_file("/f") == b"lower contents"

    view = reopen(layers)
    assert view.cat_file("/f") == b"lower contents"
    assert not upper.isfile("f")


def test_copy_up_failure_at_move_cleans_temp(overlay, layers, monkeypatch):
    lower, upper = layers
    lower.pipe_file("f", b"lower contents")

    def boom(*args, **kwargs):
        raise BoomError()

    monkeypatch.setattr(upper, "mv", boom)
    with pytest.raises(BoomError):
        overlay.open("/f", "r+b")

    assert not upper.isfile("f")
    # the staging file is removed, no half file lingers in the metadata space
    assert upper.ls(TMP, detail=False) == []
    assert lower.cat_file("f") == b"lower contents"
    assert overlay.cat_file("/f") == b"lower contents"

    # after removing the fault, copy-up can succeed normally
    monkeypatch.undo()
    with overlay.open("/f", "ab") as f:
        f.write(b"!")
    assert overlay.cat_file("/f") == b"lower contents!"


def test_copy_up_failure_leaves_lower_bytes_unchanged(overlay, layers):
    lower, upper = layers
    lower.pipe_file("f", b"do not touch")
    before = snapshot(lower)

    original_cat = lower.cat_file

    def failing_cat(path, *a, **k):
        if path == overlay._b("/f"):
            raise BoomError()
        return original_cat(path, *a, **k)

    lower.cat_file = failing_cat
    try:
        with pytest.raises(BoomError):
            overlay.open("/f", "ab")
    finally:
        lower.cat_file = original_cat

    assert snapshot(lower) == before
    assert not upper.isfile("f")


# ----------------------------------------------------------------------
# persistence across overlay recreation
# ----------------------------------------------------------------------


def test_full_workflow_survives_reopen(overlay, layers):
    lower, upper = layers
    lower.makedirs("dir", exist_ok=True)
    lower.pipe_file("dir/keep", b"keep")
    lower.pipe_file("dir/gone", b"gone")
    lower.pipe_file("file", b"lower file")
    upper.makedirs("dir", exist_ok=True)
    upper.pipe_file("dir/upper", b"upper")

    overlay.rm("dir/gone")
    overlay.rm("file")
    with overlay.open("/dir/keep", "ab") as f:
        f.write(b"-edited")
    overlay.pipe_file("/fresh", b"new")

    view = reopen(layers)
    assert view.cat_file("/dir/keep") == b"keep-edited"
    assert view.cat_file("/dir/upper") == b"upper"
    assert view.cat_file("/fresh") == b"new"
    assert not view.exists("/dir/gone")
    assert not view.exists("/file")
    assert sorted(view.find("", detail=False)) == [
        "/dir/keep",
        "/dir/upper",
        "/fresh",
    ]


# ----------------------------------------------------------------------
# lower layer bytes are never mutated, on either backend kind
# ----------------------------------------------------------------------


def test_lower_bytes_never_change(overlay, layers, tmp_path, request):
    lower, upper = layers
    lower.makedirs("dir/sub", exist_ok=True)
    lower.pipe_file("dir/a", b"a")
    lower.pipe_file("dir/sub/b", b"bb")
    lower.pipe_file("file", b"lower")
    before = snapshot(lower)

    # exercise every mutating overlay operation
    overlay.pipe_file("/file", b"overwritten")
    overlay.rm("/dir/a")
    with overlay.open("/dir/sub/b", "ab") as f:
        f.write(b"xxx")
    overlay.rm("/dir", recursive=True)
    overlay.mkdir("/dir")
    overlay.pipe_file("/dir/new", b"n")
    try:
        overlay.open("/missing-lower", "r+b")
    except FileNotFoundError:
        pass

    assert snapshot(lower) == before

    if request.node.callspec.params["layers"] == "local":
        # additionally verify the bytes physically on disk
        ldir = tmp_path / "lower"
        on_disk = {
            p.relative_to(ldir).as_posix(): p.read_bytes()
            for p in ldir.rglob("*")
            if p.is_file()
        }
        assert on_disk == {
            "dir/a": b"a",
            "dir/sub/b": b"bb",
            "file": b"lower",
        }


def test_upper_holds_all_writes_and_metadata(overlay, layers):
    lower, upper = layers
    lower.pipe_file("lower_file", b"l")
    overlay.pipe_file("/new_file", b"n")
    overlay.rm("/lower_file")

    assert upper.isfile("new_file")
    assert upper.cat_file("new_file") == b"n"
    assert upper.isfile(whiteout_marker("lower_file"))
    # nothing was ever written to lower
    assert not upper.isfile("lower_file")
    # backends differ on a leading slash for root-relative names
    assert [p.lstrip("/") for p in lower.find("", detail=False)] == ["lower_file"]


def test_text_mode_roundtrip(overlay, layers):
    lower, upper = layers
    lower.pipe_file("f", "héllo".encode())
    with overlay.open("/f", "a", encoding="utf-8") as f:
        f.write(" wörld")
    assert overlay.read_text("/f", encoding="utf-8") == "héllo wörld"
    assert lower.cat_file("f") == "héllo".encode()


def test_meta_markers_use_backend_relative_paths(overlay, layers):
    lower, upper = layers
    lower.pipe_file("f", b"x")
    overlay.rm("/f")
    # markers live under the reserved, backend-relative metadata root
    marker = whiteout_marker("f")
    assert marker == ".fsspec-overlay/whiteouts/f.wh"
    assert upper.isfile(marker)
