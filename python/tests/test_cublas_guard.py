"""cuBLAS 可用性守卫的对外契约。

``resolve_cublas`` 决定 CUDA 默认路径是否走 cuBLAS，并在「显式开但扩展缺失」时回退 dlight
而非让 build 崩——Chamleon worker / PiZeroRunner / bench_kv 全部依赖这套三态语义。此处在
CPU 上验证纯逻辑契约（无需 GPU）。
"""

from __future__ import annotations

import warnings

import pytest


def test_non_cuda_target_always_disables_cublas(tvm):
    from mlc_vla.compile import resolve_cublas

    for req in (None, True, False):
        assert resolve_cublas(req, "c") is False
        assert resolve_cublas(req, "llvm") is False


def test_cublas_available_returns_bool(tvm):
    from mlc_vla.compile import cublas_available

    assert isinstance(cublas_available(), bool)


def test_cuda_auto_matches_availability(tvm):
    """``None``（自动）在 CUDA 上应等于探测结果，且恒为 bool。"""
    from mlc_vla.compile import cublas_available, resolve_cublas

    assert resolve_cublas(None, "cuda") == cublas_available()


def test_cuda_explicit_true_falls_back_without_extension(tvm):
    """``True`` 但扩展不可用时：告警并回退 False；可用时为 True。绝不抛异常。"""
    from mlc_vla.compile import cublas_available, resolve_cublas

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = resolve_cublas(True, "cuda")

    assert result == cublas_available()
    if not cublas_available():
        assert result is False
        assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_cuda_explicit_false_disables(tvm):
    from mlc_vla.compile import resolve_cublas

    assert resolve_cublas(False, "cuda") is False
