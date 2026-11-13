import base64
import json
import os

import pytest

import fsspec
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.memory import MemoryFileSystem
from fsspec.implementations.reference import (
    ExtentReferenceError,
    LazyReferenceMapper,
    MultiExtentReference,
    ReferenceFileSystem,
    ReferenceNotReachable,
)
from fsspec.tests.conftest import data, reset_files, server, win  # noqa: F401


def test_simple(server):
    # The dictionary in refs may be dumped with a different separator
    # depending on whether json or ujson is imported
    from fsspec.implementations.reference import json as json_impl

    refs = {
        "a": b"data",
        "b": (server.realfile, 0, 5),
        "c": (server.realfile, 1, 5),
        "d": b"base64:aGVsbG8=",
        "e": {"key": "value"},
    }
    h = fsspec.filesystem("http")
    fs = fsspec.filesystem("reference", fo=refs, fs=h)

    assert fs.cat("a") == b"data"
    assert fs.cat("b") == data[:5]
    assert fs.cat("c") == data[1 : 1 + 5]
    assert fs.cat("d") == b"hello"
    assert fs.cat("e") == json_impl.dumps(refs["e"]).encode("utf-8")
    with fs.open("d", "rt") as f:
        assert f.read(2) == "he"


def test_open(m):
    from fsspec.implementations.reference import json as json_impl

    m.pipe("/data/0", data)
    refs = {
        "a": b"data",
        "b": ["memory://data/0"],
        "c": ("memory://data/0", 0, 5),
        "d": ("memory://data/0", 1, 5),
        "e": b"base64:aGVsbG8=",
        "f": {"key": "value"},
    }
    fs = fsspec.filesystem("reference", fo=refs, fs=m)

    with fs.open("a", "rb") as f:
        assert f.read() == b"data"

    with fs.open("b", "rb") as f:
        assert f.read() == data

    with fs.open("c", "rb") as f:
        assert f.read() == data[:5]
        assert not f.read()

    with fs.open("d", "rb") as f:
        assert f.read() == data[1:6]
        assert not f.read()

    with fs.open("e", "rb") as f:
        assert f.read() == b"hello"

    with fs.open("f", "rb") as f:
        assert f.read() == json_impl.dumps(refs["f"]).encode("utf-8")

    # check partial reads
    with fs.open("c", "rb") as f:
        assert f.read(2) == data[:2]
        f.seek(2, os.SEEK_CUR)
        assert f.read() == data[4:5]

    with fs.open("d", "rb") as f:
        assert f.read(2) == data[1:3]
        f.seek(1, os.SEEK_CUR)
        assert f.read() == data[4:6]


def test_simple_ver1(server):
    # The dictionary in refs may be dumped with a different separator
    # depending on whether json or ujson is imported
    from fsspec.implementations.reference import json as json_impl

    in_data = {
        "version": 1,
        "refs": {
            "a": b"data",
            "b": (server.realfile, 0, 5),
            "c": (server.realfile, 1, 5),
            "d": b"base64:aGVsbG8=",
            "e": {"key": "value"},
        },
    }
    h = fsspec.filesystem("http")
    fs = fsspec.filesystem("reference", fo=in_data, fs=h)

    assert fs.cat("a") == b"data"
    assert fs.cat("b") == data[:5]
    assert fs.cat("c") == data[1 : 1 + 5]
    assert fs.cat("d") == b"hello"
    assert fs.cat("e") == json_impl.dumps(in_data["refs"]["e"]).encode("utf-8")
    with fs.open("d", "rt") as f:
        assert f.read(2) == "he"


def test_target_options(m):
    m.pipe("data/0", b"hello")
    refs = {"a": ["memory://data/0"]}
    fn = "memory://refs.json.gz"
    with fsspec.open(fn, "wt", compression="gzip") as f:
        json.dump(refs, f)

    fs = fsspec.filesystem("reference", fo=fn, target_options={"compression": "gzip"})
    assert fs.cat("a") == b"hello"


def test_ls(server):
    refs = {"a": b"data", "b": (server.realfile, 0, 5), "c/d": (server.realfile, 1, 6)}
    h = fsspec.filesystem("http")
    fs = fsspec.filesystem("reference", fo=refs, fs=h)

    assert fs.ls("", detail=False) == ["a", "b", "c"]
    assert {"name": "c", "type": "directory", "size": 0} in fs.ls("", detail=True)
    assert fs.find("") == ["a", "b", "c/d"]
    assert fs.find("", withdirs=True) == ["a", "b", "c", "c/d"]
    assert fs.find("c", detail=True) == {
        "c/d": {"name": "c/d", "size": 6, "type": "file"}
    }


def test_nested_dirs_ls():
    # issue #1430
    refs = {"a": "A", "B/C/b": "B", "B/C/d": "d", "B/_": "_"}
    fs = fsspec.filesystem("reference", fo=refs)
    assert len(fs.ls("")) == 2
    assert {e["name"] for e in fs.ls("")} == {"a", "B"}
    assert len(fs.ls("B")) == 2
    assert {e["name"] for e in fs.ls("B")} == {"B/C", "B/_"}


def test_info(server):
    refs = {
        "a": b"data",
        "b": (server.realfile, 0, 5),
        "c/d": (server.realfile, 1, 6),
        "e": (server.realfile,),
    }
    h = fsspec.filesystem("http", headers={"give_length": "true", "head_ok": "true"})
    fs = fsspec.filesystem("reference", fo=refs, fs=h)
    assert fs.size("a") == 4
    assert fs.size("b") == 5
    assert fs.size("c/d") == 6
    assert fs.info("e")["size"] == len(data)


def test_size_of_whole_file_reference(m):
    m.pipe("/data/0", data)
    refs = {"a": ["memory://data/0"], "b": ("memory://data/0",), "c": b"data"}
    fs = fsspec.filesystem("reference", fo=refs, fs=m)
    assert fs.size("a") == len(data)
    assert fs.size("b") == len(data)
    assert fs.sizes(["a", "c"]) == [len(data), 4]
    assert fs.du("") == 2 * len(data) + 4


def test_size_of_inline_data_is_the_decoded_length():
    # an inline reference is stored in an encoded form ("base64:" or text),
    # so its size is the length cat() returns, not the length of the encoding
    binary = b"\x00\xff not utf-8 decodable"
    refs = {
        "a": "base64:" + base64.b64encode(binary).decode(),
        "b": b"base64:" + base64.b64encode(binary),
        "c": binary,
        "d": "héllo",
    }
    fs = fsspec.filesystem("reference", fo=refs)
    assert fs.cat("a") == binary
    assert fs.size("a") == len(binary)
    assert fs.sizes(["a", "b", "c"]) == 3 * [len(binary)]
    assert fs.size("d") == len("héllo".encode())
    assert {e["name"]: e["size"] for e in fs.ls("", detail=True)} == {
        "a": len(binary),
        "b": len(binary),
        "c": len(binary),
        "d": len("héllo".encode()),
    }
    assert fs.du("") == 3 * len(binary) + len("héllo".encode())


def test_mutable(server, m):
    refs = {
        "a": b"data",
        "b": (server.realfile, 0, 5),
        "c/d": (server.realfile, 1, 6),
        "e": (server.realfile,),
    }
    h = fsspec.filesystem("http", headers={"give_length": "true", "head_ok": "true"})
    fs = fsspec.filesystem("reference", fo=refs, fs=h)
    fs.rm("a")
    assert not fs.exists("a")

    bin_data = b"bin data"
    fs.pipe("aa", bin_data)
    assert fs.cat("aa") == bin_data

    fs.save_json("memory://refs.json")
    assert m.exists("refs.json")

    fs = fsspec.filesystem("reference", fo="memory://refs.json", remote_protocol="http")
    assert not fs.exists("a")
    assert fs.cat("aa") == bin_data


def test_put_get(tmpdir):
    d1 = f"{tmpdir}/d1"
    os.mkdir(d1)
    with open(f"{d1}/a", "wb") as f:
        f.write(b"1")
    with open(f"{d1}/b", "wb") as f:
        f.write(b"2")
    d2 = f"{tmpdir}/d2"

    fs = fsspec.filesystem("reference", fo={}, remote_protocol="file")
    fs.put(d1, "out", recursive=True)

    fs.get("out", d2, recursive=True)
    assert open(f"{d2}/a", "rb").read() == b"1"
    assert open(f"{d2}/b", "rb").read() == b"2"


def test_put_get_single(tmpdir):
    d1 = f"{tmpdir}/f1"
    d2 = f"{tmpdir}/f2"
    with open(d1, "wb") as f:
        f.write(b"1")

    # skip instance cache since this is the same kwargs as previous test
    fs = fsspec.filesystem(
        "reference", fo={}, remote_protocol="file", skip_instance_cache=True
    )
    fs.put_file(d1, "out")

    fs.get_file("out", d2)
    assert open(d2, "rb").read() == b"1"
    fs.pipe({"hi": b"data"})
    assert fs.cat("hi") == b"data"


def test_defaults(server):
    refs = {"a": b"data", "b": (None, 0, 5)}
    fs = fsspec.filesystem(
        "reference",
        fo=refs,
        target_protocol="http",
        target=server.realfile,
        remote_protocol="http",
    )

    assert fs.cat("a") == b"data"
    assert fs.cat("b") == data[:5]


jdata = """{
    "metadata": {
        ".zattrs": {
            "Conventions": "UGRID-0.9.0"
        },
        ".zgroup": {
            "zarr_format": 2
        },
        "adcirc_mesh/.zarray": {
            "chunks": [
                1
            ],
            "dtype": "<i4",
            "shape": [
                1
            ],
            "zarr_format": 2
        },
        "adcirc_mesh/.zattrs": {
            "_ARRAY_DIMENSIONS": [
                "mesh"
            ],
            "cf_role": "mesh_topology"
        },
        "adcirc_mesh/.zchunkstore": {
            "adcirc_mesh/0": {
                "offset": 8928,
                "size": 4
            },
            "source": {
                "array_name": "/adcirc_mesh",
                "uri": "https://url"
            }
        }
    },
    "zarr_consolidated_format": 1
}
"""


def test_spec1_expand():
    pytest.importorskip("jinja2")
    from fsspec.implementations.reference import json as json_impl

    in_data = {
        "version": 1,
        "templates": {"u": "server.domain/path", "f": "{{c}}"},
        "gen": [
            {
                "key": "gen_key{{i}}",
                "url": "http://{{u}}_{{i}}",
                "offset": "{{(i + 1) * 1000}}",
                "length": "1000",
                "dimensions": {"i": {"stop": 5}},
            },
            {
                "key": "gen_key{{i}}",
                "url": "http://{{u}}_{{i}}",
                "dimensions": {"i": {"start": 5, "stop": 7}},
            },
        ],
        "refs": {
            "key0": "data",
            "key1": ["http://target_url", 10000, 100],
            "key2": ["http://{{u}}", 10000, 100],
            "key3": ["http://{{f(c='text')}}", 10000, 100],
            "key4": ["http://target_url"],
            "key5": {"key": "value"},
        },
    }
    fs = fsspec.filesystem(
        "reference", fo=in_data, target_protocol="http", simple_templates=False
    )
    assert fs.references == {
        "key0": "data",
        "key1": ["http://target_url", 10000, 100],
        "key2": ["http://server.domain/path", 10000, 100],
        "key3": ["http://text", 10000, 100],
        "key4": ["http://target_url"],
        "key5": json_impl.dumps(in_data["refs"]["key5"]),
        "gen_key0": ["http://server.domain/path_0", 1000, 1000],
        "gen_key1": ["http://server.domain/path_1", 2000, 1000],
        "gen_key2": ["http://server.domain/path_2", 3000, 1000],
        "gen_key3": ["http://server.domain/path_3", 4000, 1000],
        "gen_key4": ["http://server.domain/path_4", 5000, 1000],
        "gen_key5": ["http://server.domain/path_5"],
        "gen_key6": ["http://server.domain/path_6"],
    }


def test_spec1_expand_simple():
    pytest.importorskip("jinja2")
    from fsspec.implementations.reference import json as json_impl

    in_data = {
        "version": 1,
        "templates": {"u": "server.domain/path"},
        "refs": {
            "key0": "base64:ZGF0YQ==",
            "key2": ["http://{{u}}", 10000, 100],
            "key4": ["http://target_url"],
            "key5": {"key": "value"},
        },
    }
    fs = fsspec.filesystem("reference", fo=in_data, target_protocol="http")
    assert fs.references["key2"] == ["http://server.domain/path", 10000, 100]
    fs = fsspec.filesystem(
        "reference",
        fo=in_data,
        target_protocol="http",
        template_overrides={"u": "not.org/p"},
    )
    assert fs.references["key2"] == ["http://not.org/p", 10000, 100]
    assert fs.cat("key0") == b"data"
    assert fs.cat("key5") == json_impl.dumps(in_data["refs"]["key5"]).encode("utf-8")


def test_spec1_gen_variants():
    pytest.importorskip("jinja2")
    with pytest.raises(ValueError):
        missing_length_spec = {
            "version": 1,
            "templates": {"u": "server.domain/path"},
            "gen": [
                {
                    "key": "gen_key{{i}}",
                    "url": "http://{{u}}_{{i}}",
                    "offset": "{{(i + 1) * 1000}}",
                    "dimensions": {"i": {"stop": 2}},
                },
            ],
        }
        fsspec.filesystem(
            "reference",
            fo=missing_length_spec,
            simple_templates=False,
            target_protocol="http",
        )

    with pytest.raises(ValueError):
        missing_offset_spec = {
            "version": 1,
            "templates": {"u": "server.domain/path"},
            "gen": [
                {
                    "key": "gen_key{{i}}",
                    "url": "http://{{u}}_{{i}}",
                    "length": "1000",
                    "dimensions": {"i": {"stop": 2}},
                },
            ],
        }
        fsspec.filesystem(
            "reference",
            simple_templates=False,
            fo=missing_offset_spec,
            target_protocol="http",
        )

    url_only_gen_spec = {
        "version": 1,
        "templates": {"u": "server.domain/path"},
        "gen": [
            {
                "key": "gen_key{{i}}",
                "url": "http://{{u}}_{{i}}",
                "dimensions": {"i": {"stop": 2}},
            },
        ],
    }

    fs = fsspec.filesystem(
        "reference",
        simple_templates=False,
        fo=url_only_gen_spec,
        target_protocol="http",
    )
    assert fs.references == {
        "gen_key0": ["http://server.domain/path_0"],
        "gen_key1": ["http://server.domain/path_1"],
    }


def test_empty():
    pytest.importorskip("jinja2")
    fs = fsspec.filesystem("reference", fo={"version": 1}, target_protocol="http")
    assert fs.references == {}


def test_get_sync(tmpdir):
    localfs = LocalFileSystem()

    real = tmpdir / "file"
    real.write_binary(b"0123456789")

    refs = {"a": b"data", "b": (str(real), 0, 5), "c/d": (str(real), 1, 6)}
    fs = fsspec.filesystem("reference", fo=refs, fs=localfs)

    fs.get("a", str(tmpdir / "a"))
    assert (tmpdir / "a").read_binary() == b"data"
    fs.get("b", str(tmpdir / "b"))
    assert (tmpdir / "b").read_binary() == b"01234"
    fs.get("c/d", str(tmpdir / "d"))
    assert (tmpdir / "d").read_binary() == b"123456"
    fs.get("c", str(tmpdir / "c"), recursive=True)
    assert (tmpdir / "c").isdir()
    assert (tmpdir / "c" / "d").read_binary() == b"123456"


@pytest.mark.parametrize(
    ("maxdepth", "expected"),
    [
        (1, {"shallow.txt": b"shallow"}),
        (2, {"shallow.txt": b"shallow", "deep/file.txt": b"deep"}),
        (
            None,
            {
                "shallow.txt": b"shallow",
                "deep/file.txt": b"deep",
                "deep/nested/file.txt": b"nested",
            },
        ),
    ],
)
def test_get_respects_maxdepth(tmp_path, maxdepth, expected):
    refs = {
        "dir/shallow.txt": b"shallow",
        "dir/deep/file.txt": b"deep",
        "dir/deep/nested/file.txt": b"nested",
    }
    fs = ReferenceFileSystem(fo=refs, skip_instance_cache=True)
    dest = tmp_path / "dest"

    fs.get("dir", str(dest), recursive=True, maxdepth=maxdepth)

    actual = {
        path.relative_to(dest).as_posix(): path.read_bytes()
        for path in dest.rglob("*")
        if path.is_file()
    }
    assert actual == expected


def test_get_maxdepth_skips_deeper_remote_references(tmp_path):
    refs = {
        "dir/shallow.txt": b"shallow",
        "dir/deep/file.txt": ("memory://maxdepth-regression-missing", 0, 4),
    }
    fs = ReferenceFileSystem(fo=refs, skip_instance_cache=True)
    dest = tmp_path / "dest"

    fs.get("dir", str(dest), recursive=True, maxdepth=1)

    assert (dest / "shallow.txt").read_bytes() == b"shallow"
    assert not (dest / "deep" / "file.txt").exists()


@pytest.mark.parametrize("maxdepth", [0, -1])
def test_get_rejects_invalid_maxdepth(tmp_path, maxdepth):
    fs = ReferenceFileSystem(
        fo={"dir/shallow.txt": b"shallow"}, skip_instance_cache=True
    )

    with pytest.raises(ValueError, match="maxdepth must be at least 1"):
        fs.get("dir", str(tmp_path / "dest"), recursive=True, maxdepth=maxdepth)


def test_get_does_not_write_above_destination(tmp_path):
    # Reference keys come from the spec, so a key holding ".." must not
    # place the copy above the destination the caller asked for.
    dest = tmp_path / "dest"
    dest.mkdir()
    outside = tmp_path / "escaped.txt"
    outside.write_bytes(b"ORIGINAL")

    refs = {
        "version": 1,
        "refs": {
            "dataset/../escaped.txt": b"ATTACKER",
            "dataset/normal.txt": b"ok",
        },
    }
    fs = fsspec.filesystem("reference", fo=refs, skip_instance_cache=True)
    with pytest.raises(ValueError, match="outside the destination"):
        fs.get("dataset", str(dest) + "/", recursive=True)

    assert outside.read_bytes() == b"ORIGINAL"
    assert not (dest / "normal.txt").exists()


def test_get_keeps_dotdot_inside_destination(tmp_path):
    # ".." that resolves within the destination stays a legitimate name.
    dest = tmp_path / "dest"
    dest.mkdir()
    refs = {
        "plain.txt": b"ok",
        "a/b/../inner.txt": b"inner",
    }
    fs = fsspec.filesystem("reference", fo=refs, skip_instance_cache=True)
    fs.get("", str(dest) + "/", recursive=True)

    assert (dest / "plain.txt").read_bytes() == b"ok"
    assert (dest / "a" / "inner.txt").read_bytes() == b"inner"


def test_multi_fs_provided(m, tmpdir):
    localfs = LocalFileSystem()

    real = tmpdir / "file"
    real.write_binary(b"0123456789")

    m.pipe("afile", b"hello")

    # local URLs are file:// by default
    refs = {
        "a": b"data",
        "b": (f"file://{real}", 0, 5),
        "c/d": (f"file://{real}", 1, 6),
        "c/e": ["memory://afile"],
    }

    fs = fsspec.filesystem("reference", fo=refs, fs={"file": localfs, "memory": m})
    assert fs.cat("c/e") == b"hello"
    assert fs.cat(["c/e", "a", "b"]) == {
        "a": b"data",
        "b": b"01234",
        "c/e": b"hello",
    }


def test_multi_fs_created(m, tmpdir):
    real = tmpdir / "file"
    real.write_binary(b"0123456789")

    m.pipe("afile", b"hello")

    # local URLs are file:// by default
    refs = {
        "a": b"data",
        "b": (f"file://{real}", 0, 5),
        "c/d": (f"file://{real}", 1, 6),
        "c/e": ["memory://afile"],
    }

    fs = fsspec.filesystem("reference", fo=refs, fs={"file": {}, "memory": {}})
    assert fs.cat("c/e") == b"hello"
    assert fs.cat(["c/e", "a", "b"]) == {
        "a": b"data",
        "b": b"01234",
        "c/e": b"hello",
    }


def test_missing_nonasync(m):
    zarr = pytest.importorskip("zarr")
    zarray = {
        "chunks": [1],
        "compressor": None,
        "dtype": "<f8",
        "fill_value": "NaN",
        "filters": [],
        "order": "C",
        "shape": [10],
        "zarr_format": 2,
    }
    refs = {".zarray": json.dumps(zarray)}

    a = zarr.open_array(
        "reference://", storage_options={"fo": refs, "remote_protocol": "memory"}
    )
    assert str(a[0]) == "nan"


def test_fss_has_defaults(m):
    fs = fsspec.filesystem("reference", fo={})
    assert None in fs.fss

    fs = fsspec.filesystem("reference", fo={}, remote_protocol="memory")
    assert fs.fss[None].protocol == "memory"
    assert fs.fss["memory"].protocol == "memory"

    fs = fsspec.filesystem("reference", fs=m, fo={})
    # Default behavior here wraps synchronous filesystems to enable the async API
    assert fs.fss[None].sync_fs is m

    fs = fsspec.filesystem("reference", fs={"memory": m}, fo={})
    assert fs.fss["memory"] is m
    assert fs.fss[None].protocol == ("file", "local")

    fs = fsspec.filesystem("reference", fs={None: m}, fo={})
    assert fs.fss[None] is m

    fs = fsspec.filesystem("reference", fo={"key": ["memory://a"]})
    assert fs.fss[None] == fs.fss["memory"]

    fs = fsspec.filesystem("reference", fo={"key": ["memory://a"], "blah": ["path"]})
    assert fs.fss[None] == fs.fss["memory"]


def test_merging(m):
    m.pipe("/a", b"test data")
    other = b"other test data"
    m.pipe("/b", other)
    fs = fsspec.filesystem(
        "reference",
        fo={
            "a": ["memory://a", 1, 1],
            "b": ["memory://a", 2, 1],
            "c": ["memory://b"],
            "d": ["memory://b", 4, 6],
        },
    )
    out = fs.cat(["a", "b", "c", "d"])
    assert out == {"a": b"e", "b": b"s", "c": other, "d": other[4:10]}


def test_cat_file_ranges(m):
    other = b"other test data"
    m.pipe("/b", other)

    fs = fsspec.filesystem(
        "reference",
        fo={
            "c": ["memory://b"],
            "d": ["memory://b", 4, 6],
        },
    )
    assert fs.cat_file("c") == other
    assert fs.cat_file("c", start=1) == other[1:]
    assert fs.cat_file("c", start=-5) == other[-5:]
    assert fs.cat_file("c", 1, -5) == other[1:-5]

    assert fs.cat_file("d") == other[4:10]
    assert fs.cat_file("d", start=1) == other[4:10][1:]
    assert fs.cat_file("d", start=-5) == other[4:10][-5:]
    assert fs.cat_file("d", 1, -3) == other[4:10][1:-3]

    # Offsets beyond either end of the part stay inside it, as with slicing.
    assert fs.cat_file("d", start=2, end=100) == other[4:10][2:100]
    assert fs.cat_file("d", start=-100) == other[4:10][-100:]
    assert fs.cat_file("d", start=100) == other[4:10][100:]
    assert fs.cat_file("d", end=-100) == other[4:10][:-100]


@pytest.mark.asyncio
async def test_async_cat_file_ranges():
    fsspec.get_filesystem_class("http").clear_instance_cache()
    fss = fsspec.filesystem("https", asynchronous=True)
    session = await fss.set_session()

    fs = fsspec.filesystem(
        "reference",
        fo={
            "version": 1,
            "refs": {
                "reference_time/0": [
                    "https://noaa-nwm-retro-v2-0-pds.s3.amazonaws.com/full_physics/2017/201704010000.CHRTOUT_DOMAIN1.comp",
                    39783,
                    12,
                ],
            },
        },
        fs={"https": fss},
        remote_protocol="https",
        asynchronous=True,
    )

    assert (
        await fs._cat_file("reference_time/0") == b"x^K0\xa9d\x04\x00\x03\x13\x01\x0f"
    )
    await session.close()


@pytest.mark.parametrize(
    "fo",
    [
        {
            "c": ["memory://b"],
            "d": ["memory://unknown", 4, 6],
        },
        {
            "c": ["memory://b"],
            "d": ["//unknown", 4, 6],
        },
    ],
    ids=["memory protocol", "mixed protocols: memory and unspecified"],
)
def test_cat_missing(m, fo):
    other = b"other test data"
    m.pipe("/b", other)
    fs = fsspec.filesystem(
        "reference",
        fo=fo,
    )
    with pytest.raises(FileNotFoundError):
        fs.cat("notafile")

    with pytest.raises(FileNotFoundError):
        fs.cat(["notone", "nottwo"])

    mapper = fs.get_mapper("")

    with pytest.raises(KeyError):
        mapper["notakey"]

    with pytest.raises(KeyError):
        mapper.getitems(["notone", "nottwo"])

    with pytest.raises(ReferenceNotReachable) as ex:
        fs.cat("d")
    assert ex.value.__cause__
    out = fs.cat("d", on_error="return")
    assert isinstance(out, ReferenceNotReachable)

    with pytest.raises(ReferenceNotReachable) as e:
        mapper["d"]
    assert '"d"' in str(e.value)
    assert "//unknown" in str(e.value)

    with pytest.raises(ReferenceNotReachable):
        mapper.getitems(["c", "d"])

    out = mapper.getitems(["c", "d"], on_error="return")
    assert isinstance(out["d"], ReferenceNotReachable)

    out = fs.cat(["notone", "c", "d"], on_error="return")
    assert isinstance(out["notone"], FileNotFoundError)
    assert out["c"] == other
    assert isinstance(out["d"], ReferenceNotReachable)

    out = mapper.getitems(["c", "d"], on_error="omit")
    assert list(out) == ["c"]


@pytest.mark.parametrize(
    "reference", [["memory://missing"], ["memory://missing", 1, 3]]
)
@pytest.mark.parametrize("path", ["missing", ["missing"]])
def test_cat_omit_unreachable(m, reference, path):
    fs = fsspec.filesystem("reference", fo={"missing": reference})

    assert fs.cat(path, on_error="omit") == {}


def test_cat_omit_unreachable_mixed(m):
    m.pipe("data", b"0123456789")
    fs = fsspec.filesystem(
        "reference",
        fo={
            "inline": b"inline data",
            "whole": ["memory://data"],
            "range": ["memory://data", 1, 3],
            "missing_whole": ["memory://missing"],
            "missing_range": ["memory://missing", 1, 3],
        },
    )

    assert fs.cat([*fs.references, "unknown"], on_error="omit") == {
        "inline": b"inline data",
        "whole": b"0123456789",
        "range": b"123",
    }


def test_df_single(m):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("fastparquet")
    data = b"data0data1data2"
    m.pipe({"data": data})
    df = pd.DataFrame(
        {
            "path": [None, "memory://data", "memory://data"],
            "offset": [0, 0, 4],
            "size": [0, 0, 4],
            "raw": [b"raw", None, None],
        }
    )
    df.to_parquet("memory://stuff/refs.0.parq")
    m.pipe(
        ".zmetadata",
        b"""{
    "metadata": {
        ".zgroup": {
            "zarr_format": 2
        },
        "stuff/.zarray": {
            "chunks": [1],
            "compressor": null,
            "dtype": "i8",
            "filters": null,
            "shape": [3],
            "zarr_format": 2
        }
    },
    "zarr_consolidated_format": 1,
    "record_size": 10
    }
    """,
    )
    fs = ReferenceFileSystem(fo="memory:///", remote_protocol="memory")
    allfiles = fs.find("")
    assert ".zmetadata" in allfiles
    assert ".zgroup" in allfiles
    assert "stuff/2" in allfiles

    assert fs.cat("stuff/0") == b"raw"
    assert fs.cat("stuff/1") == data
    assert fs.cat("stuff/2") == data[4:8]


def test_df_multi(m):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("fastparquet")
    data = b"data0data1data2"
    m.pipe({"data": data})
    df0 = pd.DataFrame(
        {
            "path": [None, "memory://data", "memory://data"],
            "offset": [0, 0, 4],
            "size": [0, 0, 4],
            "raw": [b"raw1", None, None],
        }
    )
    df0.to_parquet("memory://stuff/refs.0.parq")
    df1 = pd.DataFrame(
        {
            "path": [None, "memory://data", "memory://data"],
            "offset": [0, 0, 2],
            "size": [0, 0, 2],
            "raw": [b"raw2", None, None],
        }
    )
    df1.to_parquet("memory://stuff/refs.1.parq")
    m.pipe(
        ".zmetadata",
        b"""{
    "metadata": {
        ".zgroup": {
            "zarr_format": 2
        },
        "stuff/.zarray": {
            "chunks": [1],
            "compressor": null,
            "dtype": "i8",
            "filters": null,
            "shape": [6],
            "zarr_format": 2
        }
    },
    "zarr_consolidated_format": 1,
    "record_size": 3
    }
    """,
    )
    fs = ReferenceFileSystem(
        fo="memory:///", remote_protocol="memory", skip_instance_cache=True
    )
    allfiles = fs.find("")
    assert ".zmetadata" in allfiles
    assert ".zgroup" in allfiles
    assert "stuff/2" in allfiles
    assert "stuff/4" in allfiles

    assert fs.cat("stuff/0") == b"raw1"
    assert fs.cat("stuff/1") == data
    assert fs.cat("stuff/2") == data[4:8]
    assert fs.cat("stuff/3") == b"raw2"
    assert fs.cat("stuff/4") == data
    assert fs.cat("stuff/5") == data[2:4]


def test_mapping_getitems(m):
    m.pipe({"a": b"A", "b": b"B"})

    refs = {
        "a": ["a"],
        "b": ["b"],
    }
    h = fsspec.filesystem("memory")
    fs = fsspec.filesystem("reference", fo=refs, fs=h)
    mapping = fs.get_mapper("")
    assert mapping.getitems(["b", "a"]) == {"a": b"A", "b": b"B"}


def test_cached(m, tmpdir):
    fn = f"{tmpdir}/ref.json"

    m.pipe({"a": b"A", "b": b"B"})
    m.pipe("ref.json", b"""{"a": ["a"], "b": ["b"]}""")

    fs = fsspec.filesystem(
        "reference",
        fo="simplecache::memory://ref.json",
        fs=m,
        target_options={"cache_storage": str(tmpdir), "same_names": True},
    )
    assert fs.cat("a") == b"A"
    assert os.path.exists(fn)

    # truncate original file to show we are loading from the cached version
    m.pipe("ref.json", b"")
    fs = fsspec.filesystem(
        "reference",
        fo="simplecache::memory://ref.json",
        fs=m,
        target_options={"cache_storage": str(tmpdir), "same_names": True},
        skip_instance_cache=True,
    )
    assert fs.cat("a") == b"A"


@pytest.fixture()
def lazy_refs(m):
    zarr = pytest.importorskip("zarr")
    skip_zarr_2()
    l = LazyReferenceMapper.create("memory://refs.parquet", fs=m)
    g = zarr.open(
        "reference://",
        storage_options={"fo": "memory://refs.parquet", "remote_options": "memory"},
        zarr_format=2,
        mode="w",
    )
    g.create_array(name="data", shape=(100,), chunks=(10,), dtype="int64")
    g.store.fs.references.flush()
    return l


def test_append_parquet(lazy_refs, m):
    pytest.importorskip("kerchunk")
    with pytest.raises(KeyError):
        lazy_refs["data/0"]
    lazy_refs["data/0"] = b"data"
    assert lazy_refs["data/0"] == b"data"
    lazy_refs.flush()

    lazy2 = LazyReferenceMapper("memory://refs.parquet", fs=m)
    assert lazy2["data/0"] == b"data"
    with pytest.raises(KeyError):
        lazy_refs["data/1"]
    lazy2["data/1"] = b"Bdata"
    assert lazy2["data/1"] == b"Bdata"
    lazy2.flush()

    lazy2 = LazyReferenceMapper("memory://refs.parquet", fs=m)
    assert lazy2["data/0"] == b"data"
    assert lazy2["data/1"] == b"Bdata"
    lazy2["data/1"] = b"Adata"
    del lazy2["data/0"]
    assert lazy2["data/1"] == b"Adata"
    assert "data/0" not in lazy2
    lazy2.flush()

    lazy2 = LazyReferenceMapper("memory://refs.parquet", fs=m)
    with pytest.raises(KeyError):
        lazy2["data/0"]
    assert lazy2["data/1"] == b"Adata"


def skip_zarr_2():
    import zarr
    from packaging.version import parse

    if parse(zarr.__version__) < parse("3.0"):
        pytest.skip("Zarr 3 required")


@pytest.mark.parametrize("engine", ["fastparquet", "pyarrow"])
def test_deep_parq(m, engine):
    pytest.importorskip("kerchunk")
    zarr = pytest.importorskip("zarr")
    skip_zarr_2()
    kw = {"zarr_format": 2}

    lz = fsspec.implementations.reference.LazyReferenceMapper.create(
        "memory://out.parq",
        fs=m,
        engine=engine,
    )
    g = zarr.open_group(
        "reference://",
        mode="w",
        storage_options={"fo": "memory://out.parq", "remote_protocol": "memory"},
        **kw,
    )

    g2 = g.create_group("instant")
    arr = g2.create_array(name="one", shape=(3,), dtype="int64")
    arr[:] = [1, 2, 3]
    g.store.fs.references.flush()
    lz.flush()

    lz = fsspec.implementations.reference.LazyReferenceMapper(
        "memory://out.parq", fs=m, engine=engine
    )
    g = zarr.open_group(
        "reference://",
        storage_options={"fo": "memory://out.parq", "remote_protocol": "memory"},
        **kw,
    )
    assert g["instant"]["one"][:].tolist() == [1, 2, 3]
    assert sorted(_["name"] for _ in lz.ls("")) == [
        ".zattrs",
        ".zgroup",
        ".zmetadata",
        "instant",
    ]
    assert sorted(_["name"] for _ in lz.ls("instant")) == [
        "instant/.zattrs",
        "instant/.zgroup",
        "instant/one",
    ]

    assert sorted(_["name"] for _ in lz.ls("instant/one")) == [
        "instant/one/.zarray",
        "instant/one/.zattrs",
        "instant/one/0",
    ]


def test_parquet_no_data(m):
    zarr = pytest.importorskip("zarr")
    skip_zarr_2()
    fsspec.implementations.reference.LazyReferenceMapper.create(
        "memory://out.parq", fs=m
    )
    g = zarr.open_group(
        "reference://",
        storage_options={
            "fo": "memory://out.parq",
            "fs": m,
            "remote_protocol": "memory",
        },
        zarr_format=2,
        mode="w",
    )
    arr = g.create_array(
        name="one",
        dtype="int32",
        shape=(10,),
        chunks=(5,),
        compressor=None,
        fill_value=1,
    )
    g.store.fs.references.flush()

    assert (arr[:] == 1).all()


def test_parquet_no_references(m):
    zarr = pytest.importorskip("zarr")
    skip_zarr_2()
    lz = fsspec.implementations.reference.LazyReferenceMapper.create(
        "memory://out.parq", fs=m
    )

    g = zarr.open_group(
        "reference://",
        storage_options={
            "fo": "memory://out.parq",
            "fs": m,
            "remote_protocol": "memory",
        },
        zarr_format=2,
        mode="w",
    )
    arr = g.create_array(
        name="one",
        dtype="int32",
        shape=(),
        chunks=(),
        compressor=None,
        fill_value=1,
    )
    lz.flush()

    assert arr[...].tolist() == 1  #  scalar, equal to fill value


def test_no_default_jinja_execution():
    pytest.importorskip("jinja2")
    manifest = {
        "version": 1,
        "templates": {},
        "refs": {},
        "gen": [
            {"key": "{{ 1 / 0 }}", "url": "file:///dev/null", "dimensions": {"i": [0]}}
        ],
    }

    # no error - not evaluated
    fsspec.filesystem("reference", fo=manifest)

    with pytest.raises(ZeroDivisionError):
        fsspec.filesystem("reference", fo=manifest, simple_templates=False)


# ---------------------------------------------------------------------------
# Spec version 2: multi-extent virtual files with explicit size and holes
# ---------------------------------------------------------------------------


class RecordingMemoryFS(MemoryFileSystem):
    """Memory FS that records every physical range read it serves."""

    protocol = "memory"
    cachable = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requests = []
        self.short_read_paths = set()

    def cat_file(self, path, start=None, end=None, **kwargs):
        self.requests.append((path, start, end))
        data = super().cat_file(path, start=start, end=end, **kwargs)
        if path in self.short_read_paths and data:
            # emulate a backend that ends the response one byte early
            return data[:-1]
        return data

    def size(self, path):
        raise AssertionError(f"remote size() must not be requested for {path}")


class RecordingLocalFS(LocalFileSystem):
    cachable = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requests = []

    def cat_file(self, path, start=None, end=None, **kwargs):
        self.requests.append((path, start, end))
        return super().cat_file(path, start=start, end=end, **kwargs)


@pytest.fixture()
def rec():
    fs = RecordingMemoryFS()
    fs.store.clear()
    yield fs
    fs.store.clear()


def _independent_model(size, extents, sources):
    """Assemble the virtual file without using any fsspec reference logic."""
    buf = bytearray(size)
    for offset, length, url, source_offset in extents:
        if length:
            buf[offset : offset + length] = sources[url][
                source_offset : source_offset + length
            ]
    return bytes(buf)


V2_SOURCES = {
    "memory://a": bytes(range(256)),
    "memory://b": b"Z" * 100,
    "memory://c": bytes((i % 7) + 32 for i in range(50)),
}

# out-of-order on purpose: the reader must not rely on sorted input
V2_EXTENTS = [
    [30, 5, "memory://a", 200],
    [2, 5, "memory://a", 10],
    [15, 6, "memory://c", 0],
    [8, 0, "memory://a", 0],  # zero-length, must never be fetched
    [10, 4, "memory://b", 20],
    [38, 2, "memory://b", 0],
]
V2_SIZE = 40


def _v2_fs(rec, extents=V2_EXTENTS, size=V2_SIZE, refs=None, **kwargs):
    for url, payload in V2_SOURCES.items():
        rec.pipe(url, payload)
    doc = (
        refs
        if refs is not None
        else {
            "version": 2,
            "refs": {"v": {"size": size, "extents": extents}},
        }
    )
    return fsspec.filesystem("reference", fo=doc, fs=rec, **kwargs)


def test_v2_multi_source_matches_independent_model(rec):
    fs = _v2_fs(rec)
    model = _independent_model(V2_SIZE, V2_EXTENTS, V2_SOURCES)
    assert fs.cat_file("v") == model

    windows = [
        (None, None),
        (0, 1),
        (3, 17),
        (7, 12),
        (20, 35),
        (-10, None),
        (None, 5),
        (-15, -3),
        (100, 200),
        (5, 3),
        (-100, 100),
        (38, 40),
        (0, 0),
        (-1, -1),
        (None, -100),
    ]
    for start, end in windows:
        assert fs.cat_file("v", start=start, end=end) == model[start:end], (
            start,
            end,
        )


def test_v2_negative_slices_match_bytes_semantics(rec):
    fs = _v2_fs(rec)
    model = _independent_model(V2_SIZE, V2_EXTENTS, V2_SOURCES)
    for start, end in [
        (-1, None),
        (-5, -2),
        (None, -3),
        (-100, None),
        (2, -100),
        (-100, -90),
        (-V2_SIZE, None),
        (None, -V2_SIZE),
    ]:
        assert fs.cat_file("v", start=start, end=end) == model[start:end]


def test_v2_boundary_empty_reads(rec):
    fs = _v2_fs(rec)
    assert fs.cat_file("v", start=5, end=5) == b""
    assert fs.cat_file("v", start=100, end=200) == b""
    assert fs.cat_file("v", start=V2_SIZE, end=V2_SIZE + 10) == b""
    assert fs.cat_file("v", start=3, end=2) == b""
    assert fs.cat_file("v", start=0, end=0) == b""
    # empty reads never touch a source
    assert rec.requests == []


def test_v2_all_hole_file_uses_zero_requests(rec):
    fs = _v2_fs(
        rec,
        refs={"version": 2, "refs": {"h": {"size": 16, "extents": []}}},
    )
    assert fs.cat_file("h") == b"\x00" * 16
    assert fs.cat_file("h", start=3, end=9) == b"\x00" * 6
    assert fs.size("h") == 16
    assert rec.requests == []


def test_v2_none_extent_url_resolves_to_target(rec):
    rec.pipe("memory://default", b"0123456789")
    ref = MultiExtentReference(8, [[0, 3, None, 2], [5, 2, None, 7]])
    fs = fsspec.filesystem("reference", fo={"dir/v": ref}, target="memory://default")
    # protocol must be derivable even though every extent URL is None
    assert fs.cat_file("dir/v") == b"234\x00\x0078\x00"
    assert fs.ls("dir", detail=False) == ["dir/v"]


def test_v2_zero_length_extents_have_no_side_effects(rec):
    rec.pipe("memory://s", b"abcdef")
    ref = MultiExtentReference(
        6,
        [
            [2, 0, "memory://s", 0],
            [0, 0, "memory://s", 3],
            [5, 0, None, 0],
        ],
    )
    fs = fsspec.filesystem("reference", fo={"v": ref}, fs=rec)
    assert fs.cat_file("v") == b"\x00" * 6
    assert fs.cat_file("v", 1, 4) == b"\x00" * 3
    assert rec.requests == []


def test_v2_small_read_only_touches_intersecting_extent(rec):
    fs = _v2_fs(rec)
    rec.requests.clear()
    # logical [11, 13) sits inside the memory://b extent [10, 14)
    assert fs.cat_file("v", start=11, end=13) == (b"Z" * 2)
    requested = {path for path, _, _ in rec.requests}
    assert requested == {"memory://b"}
    # physical range must be source_offset + (logical - extent offset)
    assert rec.requests == [("memory://b", 21, 23)]


def test_v2_hole_read_needs_no_network(rec):
    fs = _v2_fs(rec)
    rec.requests.clear()
    # [7, 10) is entirely a hole: extents cover [2,7) and [10,14)
    assert fs.cat_file("v", start=7, end=10) == b"\x00" * 3
    assert rec.requests == []


def test_v2_merges_same_source_discontinuous_ranges(rec):
    rec.pipe("memory://s", bytes(range(20)))
    extents = [
        [0, 2, "memory://s", 0],
        [4, 2, "memory://s", 10],
        [8, 2, "memory://s", 18],
    ]
    model = _independent_model(10, extents, {"memory://s": bytes(range(20))})

    merged = _v2_fs(rec, extents=extents, size=10)
    assert merged.cat_file("v") == model
    # gaps (8 and 6) are below max_gap: one physical request, reassembled
    # back into virtual order
    assert rec.requests == [("memory://s", 0, 20)]

    rec2 = RecordingMemoryFS()
    unmerged = _v2_fs(rec2, extents=extents, size=10, max_gap=-1)
    assert unmerged.cat_file("v") == model
    assert sorted(rec2.requests) == sorted(
        [
            ("memory://s", 0, 2),
            ("memory://s", 10, 12),
            ("memory://s", 18, 20),
        ]
    )


def test_v2_repeated_source_ranges_restore_each_position(rec):
    source = b"abcdefghij"
    rec.pipe("memory://s", source)
    extents = [
        [0, 3, "memory://s", 0],
        [3, 3, "memory://s", 0],  # same physical range repeated
        [6, 3, "memory://s", 4],
    ]
    fs = _v2_fs(rec, extents=extents, size=10)
    # "abc" twice, then "efg", final byte is a hole
    assert fs.cat_file("v") == b"abcabcefg\x00"


def test_v2_different_protocols_are_never_merged(rec, tmp_path):
    local_file = tmp_path / "loc"
    local_file.write_bytes(b"L" * 20)
    local_url = local_file.resolve().as_uri()
    rec.pipe("memory://mem", b"M" * 20)
    local = RecordingLocalFS()

    extents = [
        [0, 4, local_url, 0],
        [4, 4, "memory://mem", 0],
        [8, 4, local_url, 4],
    ]
    doc = {"version": 2, "refs": {"v": {"size": 12, "extents": extents}}}
    fs = fsspec.filesystem("reference", fo=doc, fs={"file": local, "memory": rec})
    assert fs.cat_file("v") == b"LLLLMMMMLLLL"
    # same-URL local extents merge with each other; memory stays its own call
    assert local.requests == [(local_url, 0, 8)]
    assert rec.requests == [("memory://mem", 0, 4)]


def test_v2_open_seek_read_readinto_matches_cat(rec):
    fs = _v2_fs(rec)
    model = _independent_model(V2_SIZE, V2_EXTENTS, V2_SOURCES)

    with fs.open("v", block_size=3) as f:
        assert f.read(4) == model[:4]
        f.seek(7)
        # crosses the hole at [7, 10) into the memory://b extent
        assert f.read(6) == model[7:13]
        f.seek(0)
        buf = bytearray(10)
        n = f.readinto(buf)
        assert n == 10
        assert bytes(buf) == model[:10]
        f.seek(-4, os.SEEK_END)
        assert f.read() == model[-4:]
        f.seek(33)  # hole between [35, 38)
        assert f.read(5) == model[33:38]

    with fs.open("v") as f:
        assert f.read() == fs.cat_file("v") == model


def test_v2_info_and_ls_report_virtual_size_without_remote_calls(rec):
    fs = _v2_fs(rec)
    assert fs.info("v") == {"name": "v", "type": "file", "size": V2_SIZE}
    entries = {e["name"]: e for e in fs.ls("", detail=True)}
    assert entries["v"]["size"] == V2_SIZE
    assert fs.sizes(["v"]) == [V2_SIZE]
    assert fs.du("") == V2_SIZE
    # RecordingMemoryFS.size raises: none of the above may query the backend
    assert rec.requests == []


@pytest.mark.asyncio
async def test_v2_async_cat_matches_sync_and_model(rec):
    fs = _v2_fs(rec)
    model = _independent_model(V2_SIZE, V2_EXTENTS, V2_SOURCES)
    for start, end in [(None, None), (1, 9), (-8, -2), (12, 40), (0, 0)]:
        sync_out = fs.cat_file("v", start=start, end=end)
        async_out = await fs._cat_file("v", start=start, end=end)
        assert sync_out == async_out == model[start:end]


def test_v2_source_error_carries_full_context(rec):
    rec.pipe("memory://good", b"good-bytes")
    extents = [
        [0, 4, "memory://good", 0],
        [6, 4, "memory://missing", 0],
    ]
    fs = _v2_fs(rec, extents=extents, size=10)

    with pytest.raises(ExtentReferenceError) as exc_info:
        fs.cat_file("v")
    err = exc_info.value
    assert isinstance(err, ReferenceNotReachable)
    assert err.reference == "v"
    assert err.target == "memory://missing"
    assert err.logical_range == (0, 10)
    assert err.source_range == (0, 4)
    assert isinstance(err.__cause__, FileNotFoundError)
    message = str(err)
    assert "v" in message and "memory://missing" in message

    # holes still read as zeros and the healthy extent remains readable;
    # failures are never masked by zero-fill
    assert fs.cat_file("v", start=0, end=6) == b"good" + b"\x00\x00"

    out = fs.cat(["v"], on_error="return")
    assert isinstance(out["v"], ExtentReferenceError)
    assert fs.cat(["v"], on_error="omit") == {}


def test_v2_short_read_fails_instead_of_padding(rec):
    rec.pipe("memory://s", b"abcdefghij")
    rec.short_read_paths.add("memory://s")
    extents = [[0, 4, "memory://s", 2]]
    fs = _v2_fs(rec, extents=extents, size=6)
    with pytest.raises(ExtentReferenceError, match="short read"):
        fs.cat_file("v")


def test_v2_get_file_writes_assembled_bytes(rec, tmp_path):
    fs = _v2_fs(rec)
    model = _independent_model(V2_SIZE, V2_EXTENTS, V2_SOURCES)
    dest = tmp_path / "out"
    fs.get_file("v", str(dest))
    assert dest.read_bytes() == model


def test_v2_cat_mixed_with_classic_references(rec):
    rec.pipe("memory://s", b"0123456789")
    refs = {
        "classic": ("memory://s", 1, 3),
        "whole": ["memory://s"],
        "v2": MultiExtentReference(5, [[0, 2, "memory://s", 5]]),
        "inline": b"hi",
    }
    fs = fsspec.filesystem("reference", fo=refs, fs=rec)
    assert fs.cat(["classic", "whole", "v2", "inline"]) == {
        "classic": b"123",
        "whole": b"0123456789",
        "v2": b"56\x00\x00\x00",
        "inline": b"hi",
    }
    # mapper path exercises the same batch cat machinery
    assert fs.get_mapper("")["v2"] == b"56\x00\x00\x00"


def test_v2_keeps_version0_1_tuple_meaning(rec):
    rec.pipe("memory://s", b"0123456789")
    doc = {
        "version": 2,
        "refs": {
            "t": ["memory://s", 2, 4],
            "whole": ["memory://s"],
            "v": {"size": 2, "extents": [[0, 2, "memory://s", 6]]},
        },
    }
    fs = fsspec.filesystem("reference", fo=doc, fs=rec)
    # the 3-element tuple is still one (url, offset, size) reference
    assert fs.cat_file("t") == b"2345"
    assert fs.cat_file("whole") == b"0123456789"
    assert fs.cat_file("v") == b"67"
    assert fs.info("t")["size"] == 4


def test_v2_validation_rejects_bad_descriptors():
    url = "memory://s"
    bad = [
        (-1, []),
        (True, []),
        (4.0, []),
        (float("nan"), []),
        (float("inf"), []),
        (4, "not-a-list"),
        (4, [[0, 2, url, 0], [1, 2, url, 0]]),  # overlap
        (4, [[3, 2, url, 0]]),  # beyond size
        (4, [[-1, 1, url, 0]]),  # negative logical offset
        (4, [[0, -1, url, 0]]),  # negative length
        (4, [[0, 1, url, -1]]),  # negative source offset
        (4, [[0, True, url, 0]]),  # bool masquerading as int
        (4, [[False, 1, url, 0]]),
        (4, [[0, 1, url, True]]),
        (4, [[0, 1, 123, 0]]),  # url neither str nor None
        (4, [[0, 1, url]]),  # wrong arity
        (4, [[0, 1, url, 0, 9]]),  # wrong arity
        (4, [[0, 1.5, url, 0]]),  # float offset
        (4, [[0, 1, url, float("inf")]]),  # non-finite source offset
    ]
    for size, extents in bad:
        with pytest.raises((TypeError, ValueError)):
            MultiExtentReference(size, extents)


def test_v2_overlap_is_rejected_even_when_unsorted():
    extents = [
        [4, 2, "memory://s", 0],
        [0, 5, "memory://s", 0],
    ]
    with pytest.raises(ValueError, match="overlap"):
        MultiExtentReference(6, extents)


def test_v2_validates_before_any_source_access(rec):
    doc = {
        "version": 2,
        "refs": {"v": {"size": 4, "extents": [[0, 5, "memory://missing", 0]]}},
    }
    with pytest.raises(ValueError):
        fsspec.filesystem("reference", fo=doc, fs=rec)
    assert rec.requests == []


def test_v2_does_not_mutate_caller_extent_lists():
    extents = [
        [4, 1, "memory://s", 0],
        [0, 2, "memory://s", 0],
    ]
    snapshot = [list(item) for item in extents]
    ref = MultiExtentReference(6, extents)
    assert isinstance(ref.extents, tuple)
    assert extents == snapshot
    ref.to_dict()
    assert extents == snapshot

    rec = RecordingMemoryFS()
    rec.pipe("memory://s", b"abcdef")
    fs = fsspec.filesystem(
        "reference",
        fo={"version": 2, "refs": {"v": {"size": 6, "extents": extents}}},
        fs=rec,
    )
    fs.cat_file("v")
    assert extents == snapshot


def test_v2_save_json_roundtrip(rec):
    rec.pipe("memory://s", b"0123456789")
    extents = [[2, 3, "memory://s", 4], [8, 2, "memory://s", 0]]
    original = {"size": 10, "extents": [list(item) for item in extents]}
    doc = {
        "version": 2,
        "refs": {"v": original, "inline": b"data", "classic": ["memory://s", 1, 2]},
    }
    fs = fsspec.filesystem("reference", fo=doc, fs=rec)
    expected = fs.cat_file("v")
    fs.save_json("memory://out.json")

    raw = json.loads(rec.cat_file("memory://out.json"))
    assert raw["version"] == 2
    assert raw["refs"]["v"] == original
    assert raw["refs"]["classic"] == ["memory://s", 1, 2]

    reloaded = fsspec.filesystem(
        "reference", fo="memory://out.json", fs=RecordingMemoryFS()
    )
    assert isinstance(reloaded.references["v"], MultiExtentReference)
    assert reloaded.cat_file("v") == expected
    assert reloaded.cat_file("inline") == b"data"


def test_save_json_without_v2_stays_version_one(m):
    fs = fsspec.filesystem("reference", fo={"a": b"x"}, skip_instance_cache=True)
    fs.save_json("memory://refs.json")
    assert json.loads(m.cat_file("refs.json"))["version"] == 1


def test_v2_write_to_existing_extent_file_is_rejected(rec, tmp_path):
    rec.pipe("memory://good", b"abc")
    local = tmp_path / "in"
    local.write_bytes(b"local-bytes")
    ref = MultiExtentReference(4, [[0, 3, "memory://good", 0]])
    fs = fsspec.filesystem("reference", fo={"v": ref}, fs=rec)

    with pytest.raises(ValueError, match="version-2"):
        fs.pipe_file("v", b"new-inline")
    with pytest.raises(ValueError, match="version-2"):
        fs.put_file(str(local), "v")
    # the v2 descriptor is still intact and readable
    assert fs.cat_file("v") == b"abc\x00"

    # an explicit, validated replacement reference is the supported update
    fs.pipe_file("v", MultiExtentReference(2, [[0, 2, "memory://good", 1]]))
    assert fs.cat_file("v") == b"bc"

    with pytest.raises(ValueError, match="read-only"):
        fs.open("v", "wb")


@pytest.mark.asyncio
async def test_v2_async_write_is_rejected(rec):
    rec.pipe("memory://good", b"abc")
    fs = _v2_fs(
        rec,
        refs={
            "version": 2,
            "refs": {"v": {"size": 3, "extents": [[0, 3, "memory://good", 0]]}},
        },
    )
    with pytest.raises(ValueError, match="version-2"):
        await fs._pipe_file("v", b"no")
