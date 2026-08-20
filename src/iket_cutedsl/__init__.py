# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Automatic source-level IKET profiling for NVIDIA CuTe DSL kernels."""

from .auto_ops import patch_cute_iket_ops

__all__ = ["patch_cute_iket_ops"]
__version__ = "0.1.0"
