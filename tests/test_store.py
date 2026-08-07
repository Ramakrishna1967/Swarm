"""ContractStore: debounce, syntax-error fallback, and the settle gate (§3.2)."""

from __future__ import annotations

import time

from swarm.store import ContractStore


def test_extract_after_debounce(tmp_path):
    store = ContractStore(debounce_ms=30)
    root = str(tmp_path)
    (tmp_path / "a.py").write_text("def f(x):\n    return x\n", encoding="utf-8")
    store.touch("A", "a.py", root)
    time.sleep(0.2)
    fc = store.contract_for("A", "a.py")
    assert fc is not None
    assert len(fc.definitions) == 1


def test_syntax_error_keeps_last_good_and_marks_dirty(tmp_path):
    store = ContractStore(debounce_ms=30)
    root = str(tmp_path)
    good = tmp_path / "a.py"
    good.write_text("def f(x):\n    return x\n", encoding="utf-8")
    store.touch("A", "a.py", root)
    time.sleep(0.2)
    assert store.contract_for("A", "a.py") is not None

    good.write_text("def broken(:\n", encoding="utf-8")
    store.touch("A", "a.py", root)
    time.sleep(0.2)
    fc = store.contract_for("A", "a.py")
    assert fc is not None
    assert fc.dirty_unparseable is True
    # last good contract retained: the definition survives the bad edit
    assert any(k.qualname == "f" for k in fc.definitions)


def test_repo_for_cached(tmp_path):
    store = ContractStore()
    r1 = store.repo_for(str(tmp_path), "top")
    r2 = store.repo_for(str(tmp_path), "top")
    assert r1 is r2


def test_shutdown_cancels_timers(tmp_path):
    store = ContractStore(debounce_ms=1000)
    store.touch("A", "a.py", str(tmp_path))
    store.shutdown()  # must not raise


def test_contract_for_missing_returns_none(tmp_path):
    store = ContractStore(debounce_ms=30)
    assert store.contract_for("A", "nope.py") is None


def test_debounce_coalesces_burst(tmp_path):
    # rapid writes coalesce into one extraction that reflects the final text.
    store = ContractStore(debounce_ms=60)
    root = str(tmp_path)
    p = tmp_path / "a.py"
    for i in range(5):
        p.write_text(f"def f{i}(x):\n    return x\n", encoding="utf-8")
        store.touch("A", "a.py", root)
    import time
    time.sleep(0.3)
    fc = store.contract_for("A", "a.py")
    assert fc is not None
    assert any(k.qualname == "f4" for k in fc.definitions)
    assert all(k.qualname != "f0" for k in fc.definitions)


def test_missing_file_touch_is_quiet(tmp_path):
    store = ContractStore(debounce_ms=30)
    store.touch("A", "ghost.py", str(tmp_path))  # no file on disk
    import time
    time.sleep(0.15)
    assert store.contract_for("A", "ghost.py") is None


def test_on_change_callback_fires_once_per_flush(tmp_path):
    fired = []
    store = ContractStore(debounce_ms=40, on_change=fired.append)
    root = str(tmp_path)
    (tmp_path / "a.py").write_text("def f(x):\n    return x\n", encoding="utf-8")
    store.touch("A", "a.py", root)
    import time
    time.sleep(0.3)
    assert fired == ["A"]


def test_touch_after_delete_keeps_stale_contract(tmp_path):
    # a file removed from disk: extraction is skipped, the last contract stays
    # in the cache rather than crashing the flush.
    store = ContractStore(debounce_ms=30)
    root = str(tmp_path)
    p = tmp_path / "a.py"
    p.write_text("def f(x):\n    return x\n", encoding="utf-8")
    store.touch("A", "a.py", root)
    import time
    time.sleep(0.2)
    assert store.contract_for("A", "a.py") is not None
    p.unlink()
    store.touch("A", "a.py", root)
    time.sleep(0.2)
    # extraction returns None for a missing file; contract_for still sees the
    # last good contract from the cache.
    assert store.contract_for("A", "a.py") is not None
