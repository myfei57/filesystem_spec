"""Tests for the explicit ETag-consistency (``etag_lock``) read mode.

A local, fully controllable HTTP server stands in for a remote object store.
Its state (resource bytes, ETags, redirects and per-path faults) can be
mutated between two block reads of the same handle, which is exactly the
situation ``etag_lock`` is designed to make safe.
"""

import gzip
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

import fsspec
from fsspec.caching import AllBytes
from fsspec.implementations.http import (
    HTTPFile,
    HTTPResourceStaleError,
    _HTTPIdentity,
)

V1 = bytes((i * 7 + 3) % 256 for i in range(128))
V2 = bytes(255 - b for b in V1)  # same length, different bytes
ETAG1 = '"version-1"'
ETAG2 = '"version-2"'


class _ServerState:
    """Mutable server behaviour, shared with the test through the fixture."""

    def __init__(self):
        self.lock = threading.Lock()
        # path -> {"content": bytes, "etag": str, "weak": bool}
        self.resources = {}
        self.redirects = {}
        # path -> fault dict, e.g. {"force_200": True, "status": (503, 1)}
        self.faults = {}
        self.delay = {}
        self.requests = []
        self.aborted = {}
        self.bytes_written = {}

    def set_resource(self, path, content, etag=ETAG1, weak=False):
        with self.lock:
            self.resources[path] = {
                "content": content,
                "etag": etag,
                "weak": weak,
            }

    def resource(self, path):
        return self.resources[path]

    def etag_of(self, path):
        return self.resources[path]["etag"]

    def content_of(self, path):
        return self.resources[path]["content"]

    def add_fault(self, path, **fault):
        with self.lock:
            self.faults.setdefault(path, {}).update(fault)

    def clear_faults(self, path=None):
        with self.lock:
            if path is None:
                self.faults.clear()
            else:
                self.faults.pop(path, None)

    def enter_request(self, method, path, headers):
        # Logged on entry, so a cancelled request that never answers still
        # shows the request headers it sent.
        with self.lock:
            record = {
                "method": method,
                "path": path,
                "headers": dict(headers),
                "status": None,
            }
            self.requests.append(record)
            return record

    def gets(self, path=None):
        with self.lock:
            return [
                r
                for r in self.requests
                if r["method"] == "GET" and (path is None or r["path"] == path)
            ]

    def heads(self, path=None):
        with self.lock:
            return [
                r
                for r in self.requests
                if r["method"] == "HEAD" and (path is None or r["path"] == path)
            ]

    def wait_for_abort(self, path, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if self.aborted.get(path):
                    return True
            time.sleep(0.02)
        return False


class _EtagHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    @property
    def state(self):
        return self.server.state

    def _write(self, body):
        """Write in small chunks so an early client disconnect is observable."""
        written = 0
        try:
            for i in range(0, len(body), 4096):
                self.wfile.write(body[i : i + 4096])
                written += min(4096, len(body) - i)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            with self.state.lock:
                self.state.aborted[self.path] = self.state.aborted.get(self.path, 0) + 1
            self.close_connection = True
            return False
        finally:
            with self.state.lock:
                self.state.bytes_written[self.path] = (
                    self.state.bytes_written.get(self.path, 0) + written
                )
        return True

    def _respond(self, status, headers, body=b""):
        self._record["status"] = status
        sent = dict(headers)
        # HEAD replies carry the resource length with an empty body; let an
        # explicitly supplied Content-Length win instead of duplicating it.
        sent.setdefault("Content-Length", str(len(body)))
        self.send_response(status)
        for key, value in sent.items():
            self.send_header(key, value)
        self.end_headers()
        if body and self.command != "HEAD":
            self._write(body)

    def _etag_header(self, resource, faults):
        if faults.get("no_etag"):
            return None
        etag = resource["etag"]
        if resource["weak"] or faults.get("weak_etag"):
            etag = etag if etag.startswith(("W/", "w/")) else f"W/{etag}"
        return etag

    def do_HEAD(self):
        self._record = self.state.enter_request("HEAD", self.path, self.headers)
        resource = self.state.resources.get(self.path)
        if resource is None:
            self._respond(404, {}, b"")
            return
        headers = {
            "Accept-Ranges": "bytes",
            "ETag": resource["etag"],
            "Content-Length": str(len(resource["content"])),
        }
        self._respond(200, headers, b"")

    def do_GET(self):
        state = self.state
        self._record = state.enter_request("GET", self.path, self.headers)
        delay = state.delay.get(self.path)
        if delay:
            time.sleep(delay)

        with state.lock:
            if self.path in state.redirects:
                target = state.redirects[self.path]
                self._record["status"] = 302
                self.send_response(302)
                self.send_header("Location", target)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            resource = state.resources.get(self.path)
            if resource is None:
                self._respond(404, {}, b"not found")
                return
            content = resource["content"]
            etag = resource["etag"]
            faults = dict(state.faults.get(self.path, {}))

        # transient status fault with a budget, e.g. one 503 then recovery
        status_fault = faults.get("status")
        if status_fault is not None:
            code, budget = status_fault
            with state.lock:
                current = dict(state.faults.get(self.path, {}))
                left = current.get("status_left", budget)
                if left > 0:
                    current["status_left"] = left - 1
                    state.faults[self.path] = current
                    serve_fault = True
                else:
                    serve_fault = False
            if serve_fault:
                self._respond(code, {"Retry-After": "0"}, b"")
                return

        if_match = self.headers.get("If-Match")
        if (
            if_match is not None
            and not faults.get("ignore_if_match")
            and if_match != etag
        ):
            headers = {"ETag": etag, "Content-Length": "0"}
            self._respond(412, headers, b"")
            return

        range_header = self.headers.get("Range")
        force_200 = faults.get("force_200") or range_header is None
        start = end = None
        if range_header and not faults.get("force_200"):
            start, end = (
                int(x) for x in range_header.removeprefix("bytes=").split("-")
            )

        etag_header = self._etag_header(resource, faults)
        base_headers = {"Accept-Ranges": "bytes"}
        if etag_header is not None:
            base_headers["ETag"] = etag_header

        if force_200:
            # A server that ignores Range: whole body, status 200.
            self._respond(200, base_headers, content)
            return

        if start >= len(content):
            headers = {"Content-Range": f"bytes */{len(content)}"}
            if etag_header is not None:
                headers["ETag"] = etag_header
            self._respond(416, headers, b"")
            return

        end = min(end, len(content) - 1)
        body = content[start : end + 1]
        headers = dict(base_headers)
        headers["Content-Range"] = f"bytes {start}-{end}/{len(content)}"

        if faults.get("gzip"):
            body = gzip.compress(body, mtime=0)
            headers["Content-Encoding"] = "gzip"
        if faults.get("bad_range") == "total":
            headers["Content-Range"] = f"bytes {start}-{end}/{len(content) + 1}"
        elif faults.get("bad_range") == "start":
            headers["Content-Range"] = f"bytes {start + 1}-{end}/{len(content)}"
        if faults.get("short_body"):
            body = body[: max(1, len(body) // 2)]

        self._respond(206, headers, body)


@pytest.fixture
def etag_server():
    state = _ServerState()
    state.set_resource("/file", V1, ETAG1)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EtagHandler)
    server.state = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        yield SimpleNamespace(
            base=base,
            state=state,
            url=lambda path: base + path,
            set_resource=state.set_resource,
            resource=state.resource,
            etag_of=state.etag_of,
            content_of=state.content_of,
            add_fault=state.add_fault,
            clear_faults=state.clear_faults,
            gets=state.gets,
            heads=state.heads,
            wait_for_abort=state.wait_for_abort,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        assert not thread.is_alive()
        server.server_close()


def _fs(**kwargs):
    kwargs.setdefault("skip_instance_cache", True)
    kwargs.setdefault("retry_wait", 0)
    return fsspec.filesystem("http", **kwargs)


def _open_locked(server, path="/file", block_size=16, **kwargs):
    fs = kwargs.pop("fs", None) or _fs()
    f = fs.open(server.url(path), etag_lock=True, block_size=block_size, **kwargs)
    return fs, f


# ---------------------------------------------------------------------------
# happy path, request headers and caching
# ---------------------------------------------------------------------------


def test_locked_open_captures_identity_and_sends_if_match(etag_server):
    fs, f = _open_locked(etag_server)
    try:
        assert f.etag_lock is True
        assert f.stale is False
        assert f.stale_reason is None
        assert isinstance(f.identity, _HTTPIdentity)
        assert f.identity.etag == ETAG1
        assert f.identity.size == len(V1)
        assert f.identity.url == etag_server.url("/file")
        assert f.size == len(V1)
        assert f.details["ETag"] == ETAG1

        assert f.read(16) == V1[:16]
        f.seek(16)
        assert f.read(16) == V1[16:32]
    finally:
        f.close()

    gets = etag_server.gets()
    # the probe is a one-byte ranged GET that established the identity; it
    # cannot carry If-Match because the ETag is what it is discovering
    probe = gets[0]
    assert probe["headers"]["Range"] == "bytes=0-0"
    assert probe["status"] == 206
    assert probe["headers"]["Accept-Encoding"] == "identity"
    assert "If-Match" not in probe["headers"]
    assert all(rec["headers"].get("If-Match") == ETAG1 for rec in gets[1:])
    assert all(rec["headers"]["Accept-Encoding"] == "identity" for rec in gets)


def test_locked_cached_hit_makes_no_request(etag_server):
    fs, f = _open_locked(etag_server)
    try:
        assert f.read(16) == V1[:16]
        n = len(etag_server.gets())
        # pure cache hit: still the original version, still served locally
        f.seek(0)
        assert f.read(16) == V1[:16]
        assert len(etag_server.gets()) == n
    finally:
        f.close()


def test_locked_via_fsspec_open_keeps_custom_headers_and_readinto(etag_server):
    # storage options flow through the generic fsspec.open entry point
    with fsspec.open(
        etag_server.url("/file"),
        mode="rb",
        etag_lock=True,
        block_size=16,
        headers={"X-Test": "kept"},
        skip_instance_cache=True,
    ) as f:
        assert f.etag_lock is True
        buf = bytearray(16)
        assert f.readinto(buf) == 16
        assert bytes(buf) == V1[:16]
    # the caller's header reaches both the probe and the conditional ranges
    gets = etag_server.gets("/file")
    assert gets and all(r["headers"].get("X-Test") == "kept" for r in gets)


def test_locked_caller_size_does_not_skip_probe(etag_server):
    fs = _fs()
    f = fs.open(etag_server.url("/file"), etag_lock=True, block_size=16, size=len(V1))
    try:
        assert f.identity.etag == ETAG1
        assert any(r["headers"].get("Range") == "bytes=0-0" for r in etag_server.gets())
    finally:
        f.close()

    with pytest.raises(ValueError, match="size mismatch"):
        fs.open(
            etag_server.url("/file"),
            etag_lock=True,
            block_size=16,
            size=len(V1) + 1,
        )


def test_locked_block_size_zero_is_rejected(etag_server):
    fs = _fs()
    with pytest.raises(ValueError, match="random-access"):
        fs.open(etag_server.url("/file"), etag_lock=True, block_size=0)


def test_filesystem_level_etag_lock_default(etag_server):
    fs = _fs(etag_lock=True)
    f = fs.open(etag_server.url("/file"), block_size=16)
    try:
        assert f.etag_lock is True
        assert f.read(8) == V1[:8]
    finally:
        f.close()

    n_before = len(etag_server.gets("/file"))
    opted_out = fs.open(etag_server.url("/file"), etag_lock=False, block_size=16)
    try:
        assert opted_out.etag_lock is False
        opted_out.read(8)
        new_requests = etag_server.gets("/file")[n_before:]
        assert new_requests
        assert not any(r["headers"].get("If-Match") for r in new_requests)
    finally:
        opted_out.close()


# ---------------------------------------------------------------------------
# open-time rejection
# ---------------------------------------------------------------------------


def test_locked_open_rejects_weak_etag(etag_server):
    etag_server.set_resource("/weak", V1, ETAG1, weak=True)
    fs = _fs()
    with pytest.raises(ValueError, match="strong ETag"):
        fs.open(etag_server.url("/weak"), etag_lock=True, block_size=16)


def test_locked_open_rejects_missing_etag(etag_server):
    etag_server.add_fault("/file", no_etag=True)
    fs = _fs()
    with pytest.raises(ValueError, match="ETag"):
        fs.open(etag_server.url("/file"), etag_lock=True, block_size=16)


def test_locked_open_rejects_server_ignoring_range(etag_server):
    big = b"q" * (256 * 1024)
    etag_server.set_resource("/big", big, '"big"')
    etag_server.add_fault("/big", force_200=True)
    fs = _fs()
    before = len(etag_server.gets("/big"))
    with pytest.raises(ValueError, match="206"):
        fs.open(etag_server.url("/big"), etag_lock=True, block_size=16)
    # the rejected probe must not have downloaded the whole body
    assert len(etag_server.gets("/big")) == before + 1
    assert etag_server.wait_for_abort("/big")
    assert etag_server.state.bytes_written["/big"] < len(big)


def test_locked_open_rejects_compressed_representation(etag_server):
    etag_server.add_fault("/file", gzip=True)
    fs = _fs()
    with pytest.raises(ValueError, match="Content-Encoding"):
        fs.open(etag_server.url("/file"), etag_lock=True, block_size=16)


def test_locked_open_rejects_explicit_compression_request(etag_server):
    fs = _fs()
    with pytest.raises(ValueError, match="identity"):
        fs.open(
            etag_server.url("/file"),
            etag_lock=True,
            block_size=16,
            headers={"Accept-Encoding": "gzip"},
        )


def test_locked_open_empty_file(etag_server):
    etag_server.set_resource("/empty", b"", '"empty"')
    fs, f = _open_locked(etag_server, path="/empty")
    try:
        assert f.size == 0
        assert f.identity.etag == '"empty"'
        assert f.read() == b""
        assert f.stale is False
    finally:
        f.close()


# ---------------------------------------------------------------------------
# stale transitions
# ---------------------------------------------------------------------------


def test_same_length_change_returns_412_and_goes_stale(etag_server):
    fs, f = _open_locked(etag_server)
    try:
        assert f.read(16) == V1[:16]
        gets_before = len(etag_server.gets("/file"))

        etag_server.set_resource("/file", V2, ETAG2)  # same length, new version
        f.seek(32)
        with pytest.raises(HTTPResourceStaleError, match="412"):
            f.read(16)

        assert f.stale is True
        assert "412" in f.stale_reason
        # 412 is terminal: exactly one request, no retry, no new ETag fetch
        assert len(etag_server.gets("/file")) == gets_before + 1

        # cached hits stop delivering data once stale
        f.seek(0)
        with pytest.raises(HTTPResourceStaleError):
            f.read(16)

        # readinto fails without touching the caller's buffer
        buf = bytearray(b"Z" * 16)
        with pytest.raises(HTTPResourceStaleError):
            f.readinto(buf)
        assert bytes(buf) == b"Z" * 16
    finally:
        f.close()


def test_etag_change_without_server_enforcement_is_detected(etag_server):
    # The server does not honour If-Match; the client must compare ETags itself.
    etag_server.add_fault("/file", ignore_if_match=True)
    fs, f = _open_locked(etag_server)
    try:
        assert f.read(16) == V1[:16]
        gets_before = len(etag_server.gets("/file"))
        etag_server.set_resource("/file", V2, ETAG2)
        f.seek(32)
        with pytest.raises(HTTPResourceStaleError, match="ETag changed"):
            f.read(16)
        assert f.stale is True
        assert len(etag_server.gets("/file")) == gets_before + 1
    finally:
        f.close()


def test_redirect_to_same_etag_other_resource_is_stale(etag_server):
    etag_server.set_resource("/a", V1, ETAG1)
    etag_server.set_resource("/b", V2, ETAG1)  # same ETag value, other bytes
    fs, f = _open_locked(etag_server, path="/a")
    try:
        assert f.read(16) == V1[:16]
        etag_server.state.redirects["/a"] = "/b"
        f.seek(32)
        with pytest.raises(HTTPResourceStaleError, match="redirect"):
            f.read(16)
        assert f.stale is True
        # the unexpected redirect is visible in the request log: 302 then GET
        statuses = [(r["path"], r["status"]) for r in etag_server.state.requests]
        assert ("/a", 302) in statuses
        assert any(r["path"] == "/b" for r in etag_server.gets())
    finally:
        f.close()


def test_wrong_content_range_total_is_stale(etag_server):
    fs, f = _open_locked(etag_server)
    try:
        assert f.read(16) == V1[:16]
        etag_server.add_fault("/file", bad_range="total")
        f.seek(32)
        with pytest.raises(HTTPResourceStaleError, match="Content-Range"):
            f.read(16)
        assert f.stale is True
        f.seek(0)
        with pytest.raises(HTTPResourceStaleError):
            f.read(16)
    finally:
        f.close()


def test_wrong_content_range_start_is_stale(etag_server):
    fs, f = _open_locked(etag_server)
    try:
        assert f.read(16) == V1[:16]
        etag_server.add_fault("/file", bad_range="start")
        f.seek(32)
        with pytest.raises(HTTPResourceStaleError, match="Content-Range"):
            f.read(16)
        assert f.stale is True
    finally:
        f.close()


def test_unexpected_200_fails_before_downloading_the_whole_body(etag_server):
    big = b"q" * (256 * 1024)
    etag_server.set_resource("/big", big, '"big"')
    fs, f = _open_locked(etag_server, path="/big")
    try:
        assert f.read(16) == big[:16]
        etag_server.add_fault("/big", force_200=True)
        f.seek(64)
        with pytest.raises(HTTPResourceStaleError, match="206"):
            f.read(16)
        assert f.stale is True
        # no silent full-file fallback
        assert not isinstance(f.cache, AllBytes)
        assert f.size == len(big)
        # the client released the response without consuming the large body
        assert etag_server.wait_for_abort("/big")
        assert etag_server.state.bytes_written["/big"] < len(big)
    finally:
        f.close()


def test_short_body_is_stale(etag_server):
    fs, f = _open_locked(etag_server)
    try:
        assert f.read(16) == V1[:16]
        etag_server.add_fault("/file", short_body=True)
        f.seek(32)
        with pytest.raises(HTTPResourceStaleError, match="inconsistent length"):
            f.read(16)
        assert f.stale is True
        # the truncated block was never cached, and cached reads now fail too
        f.seek(0)
        with pytest.raises(HTTPResourceStaleError):
            f.read(16)
    finally:
        f.close()


def test_compressed_range_response_is_stale(etag_server):
    fs, f = _open_locked(etag_server)
    try:
        assert f.read(16) == V1[:16]
        etag_server.add_fault("/file", gzip=True)
        f.seek(32)
        with pytest.raises(HTTPResourceStaleError, match="compressed"):
            f.read(16)
        assert f.stale is True
    finally:
        f.close()


def test_weak_etag_on_later_response_is_stale(etag_server):
    fs, f = _open_locked(etag_server)
    try:
        f.read(16)
        etag_server.add_fault("/file", weak_etag=True)
        f.seek(32)
        with pytest.raises(HTTPResourceStaleError, match="ETag changed"):
            f.read(16)
        assert f.stale is True
    finally:
        f.close()


def test_stale_is_never_retried_even_with_broad_retry_config(etag_server):
    fs = _fs(retries=5, retry_exceptions=Exception)
    f = fs.open(etag_server.url("/file"), etag_lock=True, block_size=16)
    try:
        f.read(16)
        gets_before = len(etag_server.gets("/file"))
        etag_server.set_resource("/file", V2, ETAG2)
        f.seek(32)
        with pytest.raises(HTTPResourceStaleError):
            f.read(16)
        assert len(etag_server.gets("/file")) == gets_before + 1
    finally:
        f.close()


# ---------------------------------------------------------------------------
# retries for transient faults still work
# ---------------------------------------------------------------------------


def test_transient_503_is_retried_under_the_lock(etag_server):
    etag_server.add_fault("/file", status=(503, 1))
    fs, f = _open_locked(etag_server)
    try:
        assert f.read(16) == V1[:16]
    finally:
        f.close()
    gets = etag_server.gets("/file")
    # the probe consumes the 503 budget and retries into a 206
    assert gets[0]["status"] == 503
    assert gets[-1]["status"] == 206
    conditional = [r for r in gets if "If-Match" in r["headers"]]
    assert conditional and all(r["headers"]["If-Match"] == ETAG1 for r in conditional)


# ---------------------------------------------------------------------------
# cache isolation between versions and no stitching of mixed fragments
# ---------------------------------------------------------------------------


def test_two_version_handles_do_not_share_blocks(etag_server):
    fs = _fs()
    f1 = fs.open(etag_server.url("/file"), etag_lock=True, block_size=16)
    try:
        assert f1.read(16) == V1[:16]  # v1 bytes cached on f1

        etag_server.set_resource("/file", V2, ETAG2)
        f2 = fs.open(etag_server.url("/file"), etag_lock=True, block_size=16)
        try:
            assert f2.identity.etag == ETAG2
            assert f1.details["ETag"] != f2.details["ETag"]
            assert hash(f1) != hash(f2)
            assert f2.read(16) == V2[:16]

            # f1 may keep serving its own cached v1 bytes until it probes
            f1.seek(0)
            assert f1.read(16) == V1[:16]

            # an uncached range discovers the change and locks f1 down
            f1.seek(48)
            with pytest.raises(HTTPResourceStaleError):
                f1.read(16)
            assert f1.stale is True
            f1.seek(0)
            with pytest.raises(HTTPResourceStaleError):
                f1.read(16)

            # f2 is untouched and still serves the new version
            f2.seek(0)
            assert f2.read(16) == V2[:16]
        finally:
            f2.close()
    finally:
        f1.close()


def test_cross_cache_boundary_failure_returns_no_stitched_bytes(etag_server):
    fs = _fs()
    f = fs.open(
        etag_server.url("/file"),
        etag_lock=True,
        block_size=16,
        cache_type="none",
    )
    try:
        assert f.read(16) == V1[:16]
        assert f.tell() == 16
        etag_server.set_resource("/file", V2, ETAG2)
        with pytest.raises(HTTPResourceStaleError):
            f.read(32)  # spans an uncached boundary on the stale version
        # nothing was delivered and the position did not advance
        assert f.tell() == 16
        assert f.stale is True
    finally:
        f.close()


def test_fetch_all_is_conditional_and_validated(etag_server):
    fs, f = _open_locked(etag_server, block_size=16)
    try:
        assert f.read() == V1  # whole-file GET, file larger than one block
    finally:
        f.close()
    whole = [r for r in etag_server.gets("/file") if "Range" not in r["headers"]]
    assert whole and whole[0]["headers"]["If-Match"] == ETAG1

    # a handle opened on v1 must fail the whole-file fetch once v2 replaces it
    fs2, f2 = _open_locked(etag_server, block_size=16)
    try:
        etag_server.set_resource("/file", V2, ETAG2)
        with pytest.raises(HTTPResourceStaleError, match="412"):
            f2.read()
        assert f2.stale is True
    finally:
        f2.close()


# ---------------------------------------------------------------------------
# old default mode is unchanged on the same server
# ---------------------------------------------------------------------------


def test_default_mode_is_unchanged(etag_server):
    fs = _fs()
    f = fs.open(etag_server.url("/file"), block_size=16)
    try:
        assert f.etag_lock is False
        assert f.read(16) == V1[:16]
        assert not any(r["headers"].get("If-Match") for r in etag_server.gets())
    finally:
        f.close()

    # the historical whole-file fallback on an ignored Range still works
    etag_server.add_fault("/file", force_200=True)
    f2 = fs.open(etag_server.url("/file"), block_size=16)
    try:
        f2.seek(0)
        assert f2.read(16) == V1[:16]
    finally:
        f2.close()


def test_open_async_rejects_etag_lock(etag_server):
    import asyncio

    async def scenario():
        fs = fsspec.filesystem("http", asynchronous=True, skip_instance_cache=True)
        session = await fs.set_session()
        try:
            with pytest.raises(NotImplementedError, match="etag_lock"):
                await fs.open_async(etag_server.url("/file"), etag_lock=True)
        finally:
            await session.close()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# cancellation must not leak the response/connection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancelled_locked_request_releases_connection(etag_server):
    import asyncio

    fs = fsspec.filesystem("http", asynchronous=True, skip_instance_cache=True)
    session = await fs.set_session()
    try:
        url = etag_server.url("/file")
        identity = await fs._probe_resource_identity(url)
        f = HTTPFile(
            fs,
            url,
            session=session,
            block_size=16,
            size=identity.size,
            etag_lock=True,
            identity=identity,
            loop=fs.loop,
            asynchronous=True,
            cache_type="none",
        )
        etag_server.state.delay["/file"] = 0.5
        task = asyncio.create_task(f.async_fetch_range(32, 48))
        await asyncio.sleep(0.15)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # cancellation is not a version change ...
        assert f.stale is False
        # ... and the response was released, no connection stays leased
        assert len(session.connector._acquired) == 0
        # the in-flight request had carried the conditional headers
        cancelled = etag_server.gets("/file")[-1]
        assert cancelled["headers"]["If-Match"] == ETAG1
        assert cancelled["headers"]["Range"] == "bytes=32-47"

        # the handle and session remain usable after cancellation
        etag_server.state.delay.clear()
        out = await f.async_fetch_range(32, 48)
        assert out == V1[32:48]
        assert f.stale is False
    finally:
        await session.close()
