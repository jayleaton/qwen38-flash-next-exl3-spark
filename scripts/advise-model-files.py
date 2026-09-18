#!/usr/bin/env python3
"""posix_fadvise(DONTNEED) on every file in a model directory.

On GB10 (unified CPU+GPU memory) the page cache counts against GPU-allocatable
RAM. exllamav3's autosplit loader consults cudaMemGetInfo, which tracks MemFree
(not MemAvailable), so a freshly downloaded or recently read pack can make the
load fail with "Insufficient VRAM in split for model and cache".

Dropping the pages of the model files is enough and needs no privileges, so it
works from inside a container. (`drop_caches` is a host-global sysctl and is not
namespaced; that must be done on the host if needed.)
"""
import os
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: advise-model-files.py <model_dir>", file=sys.stderr)
        return 2
    root = sys.argv[1]
    n = 0
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            n += 1
        except OSError as e:
            print(f"warn: {path}: {e}", file=sys.stderr)
    print(f"advise-model-files: DONTNEED on {n} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
