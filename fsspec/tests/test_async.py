import asyncio
import gc
import inspect
import io
import os
import subprocess
import sys
import textwrap
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import fsspec
import fsspec.asyn
from fsspec.asyn import _run_coros_in_chunks


def test_get_lock_is_thread_safe(monkeypatch):
    barrier = threading.Barrier(2)
    real_threading = fsspec.asyn.threading

    def make_lock():
        candidate = threading.Lock()
        barrier.wait()
        return candidate

    fsspec.asyn.reset_lock()
    try:
        monkeypatch.setattr(fsspec.asyn, "threading", SimpleNamespace(Lock=make_lock))
        with ThreadPoolExecutor(max_workers=2) as executor:
            locks = list(executor.map(lambda _: fsspec.asyn.get_lock(), range(2)))
    finally:
        monkeypatch.setattr(fsspec.asyn, "threading", real_threading)
        fsspec.asyn.reset_lock()

    assert locks[0] is locks[1]


@pytest.mark.parametrize("finalizing", [False, True])
def test_sync_stopped_io_thread(monkeypatch, finalizing):
    stopped_loop = asyncio.new_event_loop()
    try:
        monkeypatch.setattr(fsspec.asyn, "loop", [stopped_loop])
        monkeypatch.setattr(
            fsspec.asyn, "iothread", [SimpleNamespace(is_alive=lambda: False)]
        )
        monkeypatch.setattr(fsspec.asyn.sys, "is_finalizing", lambda: finalizing)

        def must_not_create_coroutine():
            pytest.fail("sync attempted to schedule work on a stopped IO thread")

        with pytest.raises(RuntimeError, match="IO thread has already stopped"):
            fsspec.asyn.sync(stopped_loop, must_not_create_coroutine)
    finally:
        stopped_loop.close()


def test_sync_supplied_loop_on_stopped_io_thread(monkeypatch):
    supplied_loop = asyncio.new_event_loop()
    try:
        monkeypatch.setattr(fsspec.asyn, "loop", [object()])
        monkeypatch.setattr(
            fsspec.asyn,
            "iothread",
            [SimpleNamespace(is_alive=lambda: False, ident=424242)],
        )
        # the supplied loop was being run by the fsspec IO thread
        supplied_loop._thread_id = 424242

        def must_not_create_coroutine():
            pytest.fail("sync attempted to schedule work on a stopped IO thread")

        with pytest.raises(RuntimeError, match="IO thread has already stopped"):
            fsspec.asyn.sync(supplied_loop, must_not_create_coroutine)
    finally:
        supplied_loop._thread_id = None
        supplied_loop.close()


@pytest.mark.parametrize("finalizing", [False, True])
def test_sync_custom_loop_ignores_stopped_io_thread(monkeypatch, finalizing):
    custom_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=custom_loop.run_forever)
    thread.start()
    try:
        monkeypatch.setattr(fsspec.asyn, "loop", [object()])
        monkeypatch.setattr(
            fsspec.asyn,
            "iothread",
            [SimpleNamespace(is_alive=lambda: False, ident=12345)],
        )
        monkeypatch.setattr(fsspec.asyn.sys, "is_finalizing", lambda: finalizing)

        async def work():
            return 42

        assert fsspec.asyn.sync(custom_loop, work, timeout=1) == 42
    finally:
        custom_loop.call_soon_threadsafe(custom_loop.stop)
        thread.join(timeout=5)
        custom_loop.close()


@pytest.mark.parametrize("finalizing", [False, True])
def test_sync_live_io_thread(monkeypatch, finalizing):
    live_loop = fsspec.asyn.get_loop()
    monkeypatch.setattr(fsspec.asyn.sys, "is_finalizing", lambda: finalizing)

    async def work():
        return 42

    assert fsspec.asyn.sync(live_loop, work, timeout=1) == 42


def _run_python(script, **kwargs):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    return subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(script)], env=env, **kwargs
    )


def test_io_loop_stopped_before_interpreter_shutdown():
    script = """
        import atexit
        import sys

        # Registered before fsspec is imported, so it runs after fsspec's hook.
        def report():
            print("alive", sys.modules["fsspec.asyn"].iothread[0].is_alive())

        atexit.register(report)

        import fsspec.asyn

        fsspec.asyn.get_loop()
    """
    proc = _run_python(script, stdout=subprocess.PIPE, text=True)
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 0
    assert out.strip() == "alive False"


def test_io_loop_blocked_at_exit_warns_and_waits():
    script = """
        import atexit
        import sys
        import threading
        import time

        # Registered before fsspec is imported, so it runs after fsspec's hook.
        def report():
            print("alive", sys.modules["fsspec.asyn"].iothread[0].is_alive())

        atexit.register(report)

        import fsspec.asyn

        fsspec.asyn._STOP_IO_LOOP_WARN_SECONDS = 0.2
        loop = fsspec.asyn.get_loop()
        blocking = threading.Event()

        def block():
            blocking.set()
            time.sleep(1)

        loop.call_soon_threadsafe(block)
        blocking.wait(10)
    """
    proc = _run_python(
        script, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0
    assert "fsspec IO loop is still running" in err
    assert out.strip() == "alive False"


@pytest.mark.skipif(sys.platform == "win32", reason="uses __cxa_atexit")
def test_late_uvloop_timer_does_not_crash_at_exit():
    pytest.importorskip("uvloop")
    # A C exit handler blocked in getchar() holds the process after
    # Py_Finalize; a timer on a still-running uvloop IO loop firing there
    # re-enters the finalized interpreter and segfaults (#2241).
    script = """
        import asyncio
        import ctypes
        import threading

        import uvloop

        import fsspec.asyn

        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        loop = fsspec.asyn.get_loop()
        process = ctypes.CDLL(None)
        process.__cxa_atexit(process.getchar, None, None)
        armed = threading.Event()

        def arm():
            loop.call_later(0.3, lambda: None)
            armed.set()

        loop.call_soon_threadsafe(arm)
        armed.wait(10)
        print("exiting", flush=True)
    """
    proc = _run_python(script, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert proc.stdout.readline() == b"exiting\n"
    time.sleep(1.5)  # well past the timer
    try:
        proc.stdin.write(b"\n")
        proc.stdin.close()
    except BrokenPipeError:
        pass  # the child already crashed
    assert proc.wait(timeout=60) == 0


def test_sync_methods():
    inst = fsspec.asyn.AsyncFileSystem()
    assert inspect.iscoroutinefunction(inst._info)
    assert hasattr(inst, "info")
    assert inst.info.__qualname__ == "AsyncFileSystem._info"
    assert not inspect.iscoroutinefunction(inst.info)


def test_async_gen():
    inst = fsspec.asyn.AsyncFileSystem()
    assert inst.walk is not fsspec.AbstractFileSystem.walk
    gen = inst.walk("path")  # does not raise yet
    with pytest.raises(NotImplementedError):
        list(gen)


def test_when_sync_methods_are_disabled():
    class TestFS(fsspec.asyn.AsyncFileSystem):
        mirror_sync_methods = False

    inst = TestFS()
    assert inspect.iscoroutinefunction(inst._info)
    assert not inspect.iscoroutinefunction(inst.info)
    assert inst.info.__qualname__ == "AbstractFileSystem.info"


def test_interrupt():
    loop = fsspec.asyn.get_loop()

    async def f():
        await asyncio.sleep(1000000)
        return True

    fut = asyncio.run_coroutine_threadsafe(f(), loop)
    time.sleep(0.01)  # task launches
    out = fsspec.asyn._dump_running_tasks(with_task=True)
    task = out[0]["task"]
    assert task.done() and fut.done()
    assert isinstance(fut.exception(), fsspec.asyn.FSSpecCoroutineCancel)


class _DummyAsyncKlass:
    def __init__(self):
        self.loop = fsspec.asyn.get_loop()

    async def _dummy_async_func(self):
        # Sleep 1 second function to test timeout
        await asyncio.sleep(1)
        return True

    async def _bad_multiple_sync(self):
        fsspec.asyn.sync_wrapper(_DummyAsyncKlass._dummy_async_func)(self)
        return True

    dummy_func = fsspec.asyn.sync_wrapper(_dummy_async_func)
    bad_multiple_sync_func = fsspec.asyn.sync_wrapper(_bad_multiple_sync)


def test_sync_wrapper_timeout_on_less_than_expected_wait_time_not_finish_function():
    test_obj = _DummyAsyncKlass()
    with pytest.raises(fsspec.FSTimeoutError):
        test_obj.dummy_func(timeout=0.1)


def test_sync_wrapper_timeout_on_more_than_expected_wait_time_will_finish_function():
    test_obj = _DummyAsyncKlass()
    assert test_obj.dummy_func(timeout=5)


def test_sync_wrapper_timeout_none_will_wait_func_finished():
    test_obj = _DummyAsyncKlass()
    assert test_obj.dummy_func(timeout=None)


def test_sync_wrapper_treat_timeout_0_as_none():
    test_obj = _DummyAsyncKlass()
    assert test_obj.dummy_func(timeout=0)


def test_sync_wrapper_bad_multiple_sync():
    test_obj = _DummyAsyncKlass()
    with pytest.raises(NotImplementedError):
        test_obj.bad_multiple_sync_func(timeout=5)


def test_run_coros_in_chunks(monkeypatch):
    total_running = 0

    async def runner():
        nonlocal total_running

        total_running += 1
        await asyncio.sleep(0)
        if total_running > 4:
            raise ValueError("More than 4 coroutines are running together")
        total_running -= 1
        return 1

    async def main(**kwargs):
        nonlocal total_running

        total_running = 0
        coros = [runner() for _ in range(32)]
        results = await _run_coros_in_chunks(coros, **kwargs)
        for result in results:
            if isinstance(result, Exception):
                raise result
        return results

    assert sum(asyncio.run(main(batch_size=4))) == 32

    with pytest.raises(ValueError):
        asyncio.run(main(batch_size=5))

    with pytest.raises(ValueError):
        asyncio.run(main(batch_size=-1))

    assert sum(asyncio.run(main(batch_size=4))) == 32

    monkeypatch.setitem(fsspec.config.conf, "gather_batch_size", 5)
    with pytest.raises(ValueError):
        asyncio.run(main())
    assert sum(asyncio.run(main(batch_size=4))) == 32  # override

    monkeypatch.setitem(fsspec.config.conf, "gather_batch_size", 4)
    assert sum(asyncio.run(main())) == 32  # override


def test_running_async():
    assert not fsspec.asyn.running_async()

    async def go():
        assert fsspec.asyn.running_async()

    asyncio.run(go())


class DummyAsyncFS(fsspec.asyn.AsyncFileSystem):
    _file_class = fsspec.asyn.AbstractAsyncStreamedFile

    async def _info(self, path, **kwargs):
        return {"name": "misc/foo.txt", "type": "file", "size": 100}

    async def open_async(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        return DummyAsyncStreamedFile(
            self,
            path,
            mode,
            block_size,
            autocommit,
            cache_options=cache_options,
            **kwargs,
        )


class DummyAsyncStreamedFile(fsspec.asyn.AbstractAsyncStreamedFile):
    def __init__(self, fs, path, mode, block_size, autocommit, **kwargs):
        super().__init__(fs, path, mode, block_size, autocommit, **kwargs)
        self.temp_buffer = io.BytesIO(b"foo-bar" * 20)

    async def _fetch_range(self, start, end):
        return self.temp_buffer.read(end - start)

    async def _initiate_upload(self):
        # Reinitialize for new uploads.
        self.temp_buffer = io.BytesIO()

    async def _upload_chunk(self, final=False):
        self.temp_buffer.write(self.buffer.getbuffer())

    async def get_data(self):
        return self.temp_buffer.getbuffer().tobytes()


@pytest.mark.asyncio
async def test_async_streamed_file_write():
    test_fs = DummyAsyncFS()
    streamed_file = await test_fs.open_async("misc/foo.txt", mode="wb")
    inp_data = b"foo-bar" * streamed_file.blocksize * 2
    await streamed_file.write(inp_data)
    assert streamed_file.loc == len(inp_data)
    await streamed_file.close()
    out_data = await streamed_file.get_data()
    assert out_data.count(b"foo-bar") == streamed_file.blocksize * 2


@pytest.mark.asyncio
async def test_async_streamed_file_read():
    test_fs = DummyAsyncFS()
    streamed_file = await test_fs.open_async("misc/foo.txt", mode="rb")
    assert (
        await streamed_file.read(7 * 3) + await streamed_file.read(7 * 18)
        == b"foo-bar" * 20
    )
    await streamed_file.close()


def test_rm_file_with_rm_implementation():
    class AsyncFSWithRm(fsspec.asyn.AsyncFileSystem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.removed_paths = []

        async def _rm(self, path, recursive=False, batch_size=None, **kwargs):
            if isinstance(path, str):
                path = [path]
            for p in path:
                self.removed_paths.append(p)
            return None

    fs = AsyncFSWithRm()
    fs.rm_file("test/file.txt")
    assert "test/file.txt" in fs.removed_paths


def test_rm_file_with_rm_file_implementation():
    class AsyncFSWithRmFile(fsspec.asyn.AsyncFileSystem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.removed_paths = []

        async def _rm_file(self, path, **kwargs):
            self.removed_paths.append(path)
            return None

    fs = AsyncFSWithRmFile()
    fs.rm_file("test/file.txt")
    assert "test/file.txt" in fs.removed_paths


def test_rm_file_without_implementation():
    fs = fsspec.asyn.AsyncFileSystem()
    with pytest.raises(NotImplementedError):
        fs.rm_file("test/file.txt")


class _MaxDepthFS(fsspec.asyn.AsyncFileSystem):
    # Mirrors the gcsfs/s3fs pattern: implements the primitives, inherits _rm.
    def __init__(self, files, **kwargs):
        super().__init__(**kwargs)
        self.files = set(files)
        self.removed_paths = []

    async def _ls(self, path, detail=True, **kwargs):
        path = path.rstrip("/")
        kids = {}
        for f in self.files:
            if not f.startswith(path + "/"):
                continue
            head = f[len(path) + 1 :].split("/")[0]
            full = f"{path}/{head}"
            kids[full] = {
                "name": full,
                "size": 0,
                "type": "file" if full in self.files else "directory",
            }
        return list(kids.values()) if detail else sorted(kids)

    async def _info(self, path, **kwargs):
        path = path.rstrip("/")
        typ = "file" if path in self.files else "directory"
        return {"name": path, "size": 0, "type": typ}

    async def _rm_file(self, path, **kwargs):
        self.removed_paths.append(path)
        self.files.discard(path)


def test_rm_honours_maxdepth():
    tree = ["/root/a.txt", "/root/d1/b.txt", "/root/d1/d2/c.txt"]
    fs = _MaxDepthFS(tree)
    fs.rm("/root", recursive=True, maxdepth=1)
    assert "/root/d1/b.txt" not in fs.removed_paths
    assert "/root/d1/d2/c.txt" not in fs.removed_paths


class _CatRangesFS(fsspec.asyn.AsyncFileSystem):
    # Mirrors the gcsfs/s3fs pattern: overrides _cat_file, inherits _cat_ranges.
    cachable = False
    store = {"ok": b"0123456789"}

    async def _cat_file(self, path, start=None, end=None, **kwargs):
        if path not in self.store:
            raise FileNotFoundError(path)
        return self.store[path][start:end]


def test_cat_ranges_on_error_return_keeps_exceptions_inline():
    # Default on_error="return": a failed range comes back as an exception value,
    # positionally aligned with the inputs (matches AbstractFileSystem.cat_ranges).
    fs = _CatRangesFS()
    out = fs.cat_ranges(["ok", "missing"], [0, 0], [4, 4])
    assert out[0] == b"0123"
    assert isinstance(out[1], FileNotFoundError)


def test_cat_ranges_on_error_raise_propagates():
    # on_error="raise" must raise the first error instead of returning it inline.
    fs = _CatRangesFS()
    with pytest.raises(FileNotFoundError):
        fs.cat_ranges(["ok", "missing"], [0, 0], [4, 4], on_error="raise")


# ---------------------------------------------------------------------------
# AsyncFileSystem._iter_ranges
# ---------------------------------------------------------------------------


class _EventRangeFS(fsspec.asyn.AsyncFileSystem):
    """Fake async backend whose ``_cat_file`` calls block on per-range events.

    Every call is recorded, so tests can assert how many ranges reached the
    backend, how many ran concurrently, and which ones were cancelled.
    """

    cachable = False
    protocol = "eventrange"

    def __init__(self, payloads=None, fail=(), **kwargs):
        super().__init__(**kwargs)
        self.calls = []
        self.cancelled = []
        self.active = 0
        self.max_active = 0
        self.events = {}
        self.payloads = dict(payloads or {})
        self.fail = set(fail)

    async def _cat_file(self, path, start=None, end=None, **kwargs):
        key = (path, start, end)
        self.calls.append(key)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            event = self.events.get(key)
            if event is not None:
                await event.wait()
            if key in self.fail:
                raise OSError(f"backend failure for {key}")
            if key in self.payloads:
                return self.payloads[key]
            return b"x" * (end - start)
        except asyncio.CancelledError:
            self.cancelled.append(key)
            raise
        finally:
            self.active -= 1


class _CountingIter:
    """Sync iterable recording how many items the consumer pulled."""

    def __init__(self, items):
        self.items = list(items)
        self.reads = 0

    def __iter__(self):
        for item in self.items:
            self.reads += 1
            yield item


class _CountingAsyncIter:
    """Async iterable recording how many items the consumer pulled."""

    def __init__(self, items):
        self.items = list(items)
        self.reads = 0
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self._i]
        self._i += 1
        self.reads += 1
        await asyncio.sleep(0)
        return item


def _no_pending_tasks():
    current = asyncio.current_task()
    return [t for t in asyncio.all_tasks() if t is not current and not t.done()]


@pytest.mark.asyncio
async def test_iter_ranges_ordered_identity_empty_and_duplicate():
    fs = _EventRangeFS()
    ranges = [
        ("f", 0, 3),
        ("f", 3, 3),  # empty range
        ("f", 0, 3),  # duplicate of the first range
        ("g", 1, 5),
    ]
    out = [
        item
        async for item in fs._iter_ranges(
            ranges, ordered=True, max_inflight_bytes=100, max_inflight_items=4
        )
    ]
    assert [i for i, _ in out] == [0, 1, 2, 3]
    assert out[0][1] == b"xxx"
    assert out[1][1] == b""
    assert out[2][1] == b"xxx"
    assert out[3][1] == b"xxxx"
    # the empty range is answered without touching the backend
    assert ("f", 3, 3) not in fs.calls
    assert len(fs.calls) == 3
    # duplicate ranges stay independent calls with their own identity
    assert fs.calls.count(("f", 0, 3)) == 2


@pytest.mark.asyncio
async def test_iter_ranges_unordered_completion_order():
    fs = _EventRangeFS()
    slow = asyncio.Event()
    fast = asyncio.Event()
    fs.events = {("f", 0, 4): slow, ("f", 4, 8): fast}

    gen = fs._iter_ranges(
        [("f", 0, 4), ("f", 4, 8)],
        max_inflight_bytes=100,
        max_inflight_items=2,
    )
    first = asyncio.ensure_future(gen.__anext__())
    await asyncio.sleep(0.05)
    assert not first.done()  # both backend calls are waiting on their events

    fast.set()
    index, data = await first
    # the fast, later input item is delivered first, but keeps its index
    assert (index, data) == (1, b"xxxx")

    slow.set()
    assert [item async for item in gen] == [(0, b"xxxx")]
    await gen.aclose()


@pytest.mark.asyncio
async def test_iter_ranges_ordered_slow_first_respects_byte_budget():
    fs = _EventRangeFS()
    block_first = asyncio.Event()
    fs.events = {("f", 0, 10): block_first}
    source = _CountingIter([("f", 10 * i, 10 * (i + 1)) for i in range(6)])

    gen = fs._iter_ranges(
        source, ordered=True, max_inflight_bytes=25, max_inflight_items=-1
    )
    first = asyncio.ensure_future(gen.__anext__())
    await asyncio.sleep(0.05)
    assert not first.done()
    # 25 bytes admit two 10-byte ranges; a third would need 30. The extra
    # pulled item is held, so only three input items have been read.
    assert len(fs.calls) == 2
    assert fs.max_active == 2
    assert source.reads == 3

    block_first.set()
    head = await first
    assert head == (0, b"x" * 10)
    out = [head] + [item async for item in gen]
    assert [i for i, _ in out] == list(range(6))
    assert all(data == b"x" * 10 for _, data in out)
    assert source.reads == 6
    await gen.aclose()


@pytest.mark.asyncio
async def test_iter_ranges_item_budget_lazy_input():
    fs = _EventRangeFS()
    block = asyncio.Event()
    fs.events = {("f", 0, 2): block}
    source = _CountingIter([("f", 2 * i, 2 * (i + 1)) for i in range(5)])

    gen = fs._iter_ranges(source, max_inflight_bytes=-1, max_inflight_items=2)
    first = asyncio.ensure_future(gen.__anext__())
    await asyncio.sleep(0.05)
    # the item budget caps how far the input iterator is consumed; the blocked
    # first item still lets the second one finish and get delivered
    assert source.reads == 2
    assert len(fs.calls) == 2
    index, _ = await first
    assert index == 1

    block.set()
    rest = [item async for item in gen]
    assert len(rest) == 4
    assert source.reads == 5
    await gen.aclose()


@pytest.mark.asyncio
async def test_iter_ranges_byte_budget_covers_buffered_results():
    # A completed but undelivered result must keep charging the byte budget:
    # limiting only running tasks would let the internal backlog grow without
    # bound while the slow first item blocks delivery.
    fs = _EventRangeFS()
    block_first = asyncio.Event()
    fs.events = {("f", 0, 10): block_first}
    source = _CountingIter([("f", 10 * i, 10 * (i + 1)) for i in range(5)])

    gen = fs._iter_ranges(source, max_inflight_bytes=25, max_inflight_items=-1)
    first = asyncio.ensure_future(gen.__anext__())
    await asyncio.sleep(0.05)
    assert len(fs.calls) == 2  # item 1 may be done and buffered, item 2 never starts

    block_first.set()
    index, _ = await first
    assert index == 1  # unordered: buffered fast item delivered first
    rest = [item async for item in gen]
    assert len(rest) == 4
    assert source.reads == 5
    await gen.aclose()


@pytest.mark.asyncio
async def test_iter_ranges_ordered_item_budget_no_unbounded_prefetch():
    fs = _EventRangeFS()
    block_first = asyncio.Event()
    fs.events = {("f", 0, 2): block_first}
    source = _CountingIter([("f", 2 * i, 2 * (i + 1)) for i in range(5)])

    gen = fs._iter_ranges(
        source, ordered=True, max_inflight_bytes=-1, max_inflight_items=2
    )
    first = asyncio.ensure_future(gen.__anext__())
    await asyncio.sleep(0.05)
    assert not first.done()
    # item 1 has finished and is buffered behind the blocked head, but the
    # item budget still stops the iterator from pulling any more input
    assert source.reads == 2
    assert len(fs.calls) == 2

    block_first.set()
    out = [await first] + [item async for item in gen]
    assert [i for i, _ in out] == [0, 1, 2, 3, 4]
    await gen.aclose()


@pytest.mark.asyncio
async def test_iter_ranges_forwards_kwargs_to_cat_file():
    fs = _EventRangeFS()
    seen = []

    async def spy(path, start=None, end=None, **kwargs):
        seen.append(kwargs)
        return b"z" * (end - start)

    fs._cat_file = spy
    async for _ in fs._iter_ranges(
        [("f", 0, 1), ("f", 1, 2)], max_inflight_bytes=10, block_size=42
    ):
        pass
    assert seen == [{"block_size": 42}, {"block_size": 42}]


@pytest.mark.asyncio
async def test_iter_ranges_empty_input_and_default_limits():
    fs = _EventRangeFS()
    assert [item async for item in fs._iter_ranges([])] == []
    assert fs.calls == []
    # default limits accept an ordinary stream
    assert [item async for item in fs._iter_ranges([("f", 0, 3)])] == [(0, b"xxx")]


@pytest.mark.asyncio
async def test_iter_ranges_oversized_rejected_without_backend_call():
    fs = _EventRangeFS()
    ranges = [("f", 0, 100), ("f", 0, 2)]

    gen = fs._iter_ranges(ranges, max_inflight_bytes=50, max_inflight_items=4)
    out = dict([item async for item in gen])
    assert isinstance(out[0], ValueError)
    assert "max_inflight_bytes" in str(out[0])
    assert out[1] == b"xx"
    assert fs.calls == [("f", 0, 2)]
    await gen.aclose()

    with pytest.raises(ValueError, match="max_inflight_bytes"):
        async for _ in fs._iter_ranges(ranges, max_inflight_bytes=50, on_error="raise"):
            pass


@pytest.mark.asyncio
async def test_iter_ranges_backend_error_return_inline():
    fs = _EventRangeFS(fail={("f", 0, 4)})
    out = dict(
        [
            item
            async for item in fs._iter_ranges(
                [("f", 0, 4), ("f", 4, 8)],
                max_inflight_bytes=100,
                max_inflight_items=2,
            )
        ]
    )
    assert isinstance(out[0], OSError)
    assert out[1] == b"xxxx"


@pytest.mark.asyncio
async def test_iter_ranges_on_error_raise_cancels_pending():
    fs = _EventRangeFS(fail={("f", 4, 8)})
    block = asyncio.Event()
    fs.events = {("f", 0, 4): block}
    gen = fs._iter_ranges(
        [("f", 0, 4), ("f", 4, 8)],
        max_inflight_bytes=100,
        max_inflight_items=2,
        on_error="raise",
    )
    with pytest.raises(OSError):
        async for _ in gen:
            pass
    # the still-blocked task is cancelled rather than leaked
    assert ("f", 0, 4) in fs.cancelled
    assert not _no_pending_tasks()


@pytest.mark.asyncio
async def test_iter_ranges_short_and_over_read_are_contract_errors():
    fs = _EventRangeFS(
        payloads={
            ("f", 0, 4): b"abc",  # short read
            ("g", 0, 4): b"abcde",  # over-long read
        }
    )
    out = dict(
        [
            item
            async for item in fs._iter_ranges(
                [("f", 0, 4), ("g", 0, 4)],
                max_inflight_bytes=100,
                max_inflight_items=2,
            )
        ]
    )
    assert isinstance(out[0], ValueError) and "short read" in str(out[0])
    assert isinstance(out[1], ValueError) and "over-long read" in str(out[1])
    # nothing is patched together from another range's data
    assert all(isinstance(value, ValueError) for value in out.values())


@pytest.mark.asyncio
async def test_iter_ranges_invalid_ranges_rejected_per_policy():
    fs = _EventRangeFS()
    ranges = [
        ("f", -1, 2),  # negative bound
        ("f", 5, 2),  # end before start
        ("f", 0, "3"),  # non-integer bound
        ("only-two",),  # wrong shape
        ("f", 0, 1),  # valid
    ]
    out = dict([item async for item in fs._iter_ranges(ranges, max_inflight_bytes=100)])
    for i in range(4):
        assert isinstance(out[i], (ValueError, TypeError))
    assert out[4] == b"x"
    assert fs.calls == [("f", 0, 1)]

    with pytest.raises((ValueError, TypeError)):
        async for _ in fs._iter_ranges(
            [("f", -1, 2)], max_inflight_bytes=100, on_error="raise"
        ):
            pass


@pytest.mark.asyncio
async def test_iter_ranges_aclose_cancels_and_stops_pulling():
    fs = _EventRangeFS()
    first_ready = asyncio.Event()
    second_blocked = asyncio.Event()
    fs.events = {("f", 0, 2): first_ready, ("f", 2, 4): second_blocked}
    source = _CountingIter([("f", 2 * i, 2 * (i + 1)) for i in range(5)])

    gen = fs._iter_ranges(source, max_inflight_bytes=-1, max_inflight_items=2)
    pending_next = asyncio.ensure_future(gen.__anext__())
    await asyncio.sleep(0.05)
    assert source.reads == 2
    assert len(fs.calls) == 2

    # deliver one result, then close early while the other task is blocked
    first_ready.set()
    index, _ = await pending_next
    assert index in (0, 1)
    await gen.aclose()

    # early close cancels backend work, stops reading input, leaks no tasks
    assert ("f", 2, 4) in fs.cancelled
    assert source.reads == 2
    assert not _no_pending_tasks()


@pytest.mark.asyncio
async def test_iter_ranges_cancellation_is_not_swallowed():
    fs = _EventRangeFS()
    block = asyncio.Event()
    fs.events = {("f", 0, 4): block}
    gen = fs._iter_ranges(
        [("f", 0, 4), ("f", 4, 8)],
        max_inflight_bytes=100,
        max_inflight_items=2,
    )

    async def consume():
        async for _ in gen:
            pass

    task = asyncio.ensure_future(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ("f", 0, 4) in fs.cancelled
    assert not _no_pending_tasks()


@pytest.mark.asyncio
async def test_iter_ranges_input_failure_cleans_up():
    fs = _EventRangeFS()
    block = asyncio.Event()
    fs.events = {("f", 0, 4): block}

    async def broken():
        yield ("f", 0, 4)
        await asyncio.sleep(0)  # let the backend task start and block
        raise RuntimeError("input blew up")

    gen = fs._iter_ranges(broken(), max_inflight_bytes=100, max_inflight_items=2)
    with pytest.raises(RuntimeError, match="input blew up"):
        async for _ in gen:
            pass
    assert ("f", 0, 4) in fs.cancelled
    assert not _no_pending_tasks()


@pytest.mark.asyncio
async def test_iter_ranges_sync_input_failure_leaks_no_tasks():
    fs = _EventRangeFS()

    def broken():
        yield ("f", 0, 4)
        raise RuntimeError("sync input blew up")

    # the failure may land before the scheduled backend task starts; it must
    # still be cancelled and awaited instead of warned about at GC time
    gen = fs._iter_ranges(broken(), max_inflight_bytes=100, max_inflight_items=2)
    with pytest.raises(RuntimeError, match="sync input blew up"):
        async for _ in gen:
            pass
    assert not _no_pending_tasks()


@pytest.mark.asyncio
async def test_iter_ranges_accepts_async_iterable():
    fs = _EventRangeFS()
    source = _CountingAsyncIter([("f", 0, 1), ("f", 1, 3)])
    out = [
        item
        async for item in fs._iter_ranges(source, ordered=True, max_inflight_bytes=100)
    ]
    assert out == [(0, b"x"), (1, b"xx")]
    assert source.reads == 2


@pytest.mark.asyncio
async def test_iter_ranges_rejects_bad_arguments():
    fs = _EventRangeFS()
    with pytest.raises(ValueError, match="on_error"):
        await fs._iter_ranges([], on_error="nope").__anext__()
    with pytest.raises(ValueError, match="max_inflight_items"):
        await fs._iter_ranges([], max_inflight_items=0).__anext__()
    with pytest.raises(ValueError, match="max_inflight_bytes"):
        await fs._iter_ranges([], max_inflight_bytes=0).__anext__()


def test_iter_ranges_sync_mirror():
    # mirror_sync_methods exposes the async generator as a blocking one;
    # keyword arguments are forwarded through the sync wrapper as well
    fs = _EventRangeFS()
    assert inspect.isgeneratorfunction(fs.iter_ranges)
    assert list(fs.iter_ranges([("f", 0, 2), ("f", 2, 4)], ordered=True)) == [
        (0, b"xx"),
        (1, b"xx"),
    ]
    assert dict(fs.iter_ranges([("f", 0, 2), ("f", 2, 4)])) == {
        0: b"xx",
        1: b"xx",
    }


# ---------------------------------------------------------------------------
# Tests for the prefix= hint that _glob passes to _find
# ---------------------------------------------------------------------------

_GLOB_PREFIX_FILES = [
    "data/2024/results.csv",
    "data/2024/report.txt",
    "data/2023/results.csv",
    "top_results.csv",
    "top_other.txt",
    "other/results.csv",
]


class _PrefixCapturingFS(fsspec.asyn.AsyncFileSystem):
    """Minimal AsyncFileSystem that records every _find call's kwargs.

    _find ignores the prefix hint and returns all files under *root* so that
    the client-side glob pattern-matching in _glob still works correctly.
    This simulates a "naive" backend that silently absorbs unknown kwargs.
    """

    protocol = "prefixmock"

    def __init__(self, fs_files=None, **kwargs):
        super().__init__(**kwargs)
        self.find_calls = []
        self.fs_files = fs_files if fs_files is not None else _GLOB_PREFIX_FILES

    async def _find(self, path, maxdepth=None, withdirs=False, detail=False, **kwargs):
        self.find_calls.append({"path": path, "kwargs": dict(kwargs)})
        root = (path.rstrip("/") + "/") if path else ""
        results = {
            f: {"name": f, "type": "file", "size": 0}
            for f in self.fs_files
            if f.startswith(root)
        }
        return results if detail else list(results)

    async def _info(self, path, **kwargs):
        for f in self.fs_files:
            if f == path:
                return {"name": f, "type": "file", "size": 0}
        raise FileNotFoundError(path)


@pytest.fixture
def prefix_fs():
    return _PrefixCapturingFS(skip_instance_cache=True)


@pytest.mark.parametrize(
    "pattern, expected_results, expected_prefix",
    [
        # root/stem* -> prefix="stem"
        (
            "data/2024/res*",
            ["data/2024/results.csv"],
            "res",
        ),
        # root/stem*.ext -> prefix="stem"
        (
            "data/2024/re*.txt",
            ["data/2024/report.txt"],
            "re",
        ),
        # stem* (no slash) -> prefix="stem"
        (
            "top_*",
            ["top_other.txt", "top_results.csv"],
            "top_",
        ),
        # ? wildcard: prefix is everything before the first ?
        (
            "top_r?sults.csv",
            ["top_results.csv"],
            "top_r",
        ),
        # [ wildcard: prefix is everything before the first [
        # re[rp]* translates to re[rp][^/]* so only report.txt matches (not results.csv)
        (
            "data/2024/re[rp]*",
            ["data/2024/report.txt"],
            "re",
        ),
        # root/* (wildcard immediately after /) -> empty prefix, NOT forwarded
        (
            "data/2024/*",
            ["data/2024/report.txt", "data/2024/results.csv"],
            None,
        ),
        # bare * (no prefix at all) -> NOT forwarded
        # * translates to [^/]+ so paths containing / are excluded
        (
            "*",
            ["top_other.txt", "top_results.csv"],
            None,
        ),
    ],
)
def test_glob_prefix_hint(prefix_fs, pattern, expected_results, expected_prefix):
    """_glob should extract the literal stem before the first wildcard and
    forward it as ``prefix=`` to ``_find``.  When the stem is empty the kwarg
    must not be forwarded at all so that backends that reject unknown kwargs
    are not broken.  The glob results must be correct regardless."""
    results = prefix_fs.glob(pattern)
    assert sorted(results) == sorted(expected_results)

    assert len(prefix_fs.find_calls) == 1
    forwarded = prefix_fs.find_calls[0]["kwargs"]

    if expected_prefix is None:
        assert "prefix" not in forwarded, (
            f"prefix= should not be forwarded for pattern {pattern!r}, "
            f"but got prefix={forwarded.get('prefix')!r}"
        )
    else:
        assert forwarded.get("prefix") == expected_prefix, (
            f"expected prefix={expected_prefix!r} for pattern {pattern!r}, "
            f"got {forwarded.get('prefix')!r}"
        )


@pytest.mark.asyncio
async def test_expand_path_special_characters_async():
    fs_files = [
        "bucket",
        "bucket/file[1].txt",
        "bucket/file?.txt",
        "bucket/normal.txt",
    ]
    fs = _PrefixCapturingFS(fs_files=fs_files)
    paths = await fs._expand_path("bucket", recursive=True)
    expected = [
        "bucket",
        "bucket/file[1].txt",
        "bucket/file?.txt",
        "bucket/normal.txt",
    ]
    assert sorted(paths) == sorted(expected)


@pytest.mark.asyncio
async def test_expand_path_with_magic_input_async():
    fs_files = [
        "bucket",
        "bucket/file[1].txt",
        "bucket/file?.txt",
        "bucket/normal.txt",
    ]
    fs = _PrefixCapturingFS(fs_files=fs_files)
    paths = await fs._expand_path("bucket/file*.txt", recursive=True)
    expected = [
        "bucket/file[1].txt",
        "bucket/file?.txt",
    ]
    assert sorted(paths) == sorted(expected)


@pytest.mark.asyncio
async def test_isfile_does_not_swallow_cancellation():
    # `_isdir` catches OSError; `_isfile` must not be broader than that to the point
    # of swallowing CancelledError, which does not derive from Exception.
    class CancellingFS(fsspec.asyn.AsyncFileSystem):
        async def _info(self, path, **kwargs):
            raise asyncio.CancelledError()

    fs = CancellingFS()
    with pytest.raises(asyncio.CancelledError):
        await fs._isdir("path")
    with pytest.raises(asyncio.CancelledError):
        await fs._isfile("path")


class UnreadableAsyncSubdirFS(fsspec.asyn.AsyncFileSystem):
    """The async counterpart of the unreadable-subdirectory tree."""

    protocol = "unreadableasyncsub"
    cachable = False
    mirror_sync_methods = False

    tree = {
        "": ["top"],
        "top": ["top/readable", "top/locked"],
        "top/readable": [],
    }

    async def _ls(self, path, detail=True, **kwargs):
        path = self._strip_protocol(path).rstrip("/")
        if path == "top/locked":
            raise PermissionError("no access to top/locked")
        entries = [
            {"name": name, "type": "directory", "size": 0} for name in self.tree[path]
        ]
        return entries if detail else [entry["name"] for entry in entries]


@pytest.mark.asyncio
async def test_async_walk_on_error_reaches_subdirectories():
    # _walk documents the same on_error contract as walk(), and had the same
    # omission in the recursive call.
    fs = UnreadableAsyncSubdirFS()
    # _walk yields the path it could not list, where walk() omits it. That
    # difference predates this and is left alone here; what matters is that the
    # last entry only appears at all when on_error is left at the default.
    omitted = [
        ("", ["top"], []),
        ("top", ["readable", "locked"], []),
        ("top/readable", [], []),
    ]
    yielded = omitted + [("top/locked", [], [])]

    assert [entry async for entry in fs._walk("")] == yielded

    with pytest.raises(PermissionError, match="no access to top/locked"):
        [entry async for entry in fs._walk("", on_error="raise")]

    handled = []
    assert [entry async for entry in fs._walk("", on_error=handled.append)] == yielded
    assert [str(exc) for exc in handled] == ["no access to top/locked"]


class _Boom(Exception):
    """An exception that can be weakly referenced, to tell when it is freed."""


class _AlwaysFailsFS(fsspec.asyn.AsyncFileSystem):
    protocol = "alwaysfails"
    cachable = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.errors = []

    def _fail(self, path):
        exc = _Boom(path)
        self.errors.append(weakref.ref(exc))
        return exc

    async def _pipe_file(self, path, value, mode="overwrite", **kwargs):
        raise self._fail(path)


@pytest.mark.skipif(
    sys.implementation.name != "cpython", reason="relies on reference counting"
)
def test_failed_call_is_freed_without_gc():
    # An exception raised by sync() has the frames it passed through in its
    # traceback. If any of them refers back to it, it can only be freed by the
    # cyclic garbage collector, and so can everything else those frames hold.
    fs = _AlwaysFailsFS(asynchronous=False)
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        try:
            fs.pipe({"a": b"1", "b": b"2"})
        except _Boom:
            pass
        else:
            raise AssertionError("expected the call to fail")
        # let the IO thread finish with the call too
        fsspec.asyn.sync(fs.loop, asyncio.sleep, 0)
        assert fs.errors
        assert all(ref() is None for ref in fs.errors), "exception kept alive"
    finally:
        if was_enabled:
            gc.enable()
