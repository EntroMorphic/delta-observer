"""ctypes bridge to the native C/NEON Observer.D encoder."""

from __future__ import annotations

import ctypes
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch


class ObserverDNeon:
    """Native encoder for a trained ``Linear -> GELU -> Linear`` observer."""

    def __init__(self, rebuild: bool = False):
        directory = Path(__file__).resolve().parent
        build_directory = directory / "build"
        build_directory.mkdir(exist_ok=True)
        suffix = ".dylib" if platform.system() == "Darwin" else ".so"
        self.library_path = build_directory / f"libobserver_d_neon{suffix}"
        source = directory / "observer_d_neon.c"
        header = directory / "observer_d_neon.h"
        if rebuild or not self.library_path.exists() or (
            self.library_path.stat().st_mtime < max(source.stat().st_mtime, header.stat().st_mtime)
        ):
            self._compile(source)

        self.library = ctypes.CDLL(str(self.library_path))
        pointer = ctypes.POINTER(ctypes.c_float)
        self.library.observer_d_encode.argtypes = [
            pointer,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
        ]
        self.library.observer_d_encode.restype = ctypes.c_int
        self.library.observer_d_backend.argtypes = []
        self.library.observer_d_backend.restype = ctypes.c_char_p

    def _compile(self, source: Path) -> None:
        command = [
            "cc",
            "-O3",
            "-ffast-math",
            "-std=c11",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(self.library_path),
            "-lm",
        ]
        if platform.system() == "Darwin":
            command[command.index("-shared")] = "-dynamiclib"
        subprocess.run(command, check=True, capture_output=True, text=True)

    @property
    def backend(self) -> str:
        return self.library.observer_d_backend().decode("ascii")

    @staticmethod
    def _array(tensor: torch.Tensor) -> np.ndarray:
        return np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)

    def encode(self, features: torch.Tensor, encoder: torch.nn.Sequential) -> torch.Tensor:
        """Run a trained Observer.D encoder through C/NEON."""
        linear1 = encoder[0]
        linear2 = encoder[2]
        arrays = [
            self._array(features),
            self._array(linear1.weight),
            self._array(linear1.bias),
            self._array(linear2.weight),
            self._array(linear2.bias),
        ]
        inputs, weight1, bias1, weight2, bias2 = arrays
        output = np.empty((inputs.shape[0], weight2.shape[0]), dtype=np.float32)
        pointer = ctypes.POINTER(ctypes.c_float)
        status = self.library.observer_d_encode(
            inputs.ctypes.data_as(pointer),
            inputs.shape[0],
            inputs.shape[1],
            weight1.shape[0],
            weight2.shape[0],
            weight1.ctypes.data_as(pointer),
            bias1.ctypes.data_as(pointer),
            weight2.ctypes.data_as(pointer),
            bias2.ctypes.data_as(pointer),
            output.ctypes.data_as(pointer),
        )
        if status != 0:
            raise RuntimeError(f"Observer.D native encoder failed with status {status}")
        return torch.from_numpy(output)
