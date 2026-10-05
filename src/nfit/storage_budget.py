"""Reserve temporary disk space across concurrent atomic scientific writes."""

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

_LOCK = RLock()
_RESERVED: dict[int, int] = {}


@contextmanager
def reserve_disk_space(directory, byte_count, *, operation="Saving scientific data"):
    """Reserve the replacement's peak footprint while retaining the old file.

    This checks free filesystem space; filesystem quotas can still reject a
    write. Atomic writers retain their original destination on either failure.
    """
    root = Path(directory)
    size = max(0, int(byte_count))
    device = os.stat(root).st_dev
    with _LOCK:
        free = shutil.disk_usage(root).free
        headroom = max(1024**3, free // 10)
        reserved = _RESERVED.get(device, 0)
        if size + reserved > max(0, free - headroom):
            raise OSError(f"{operation} needs {size / 1e9:.2f} GB of temporary disk space "
                          f"plus {headroom / 1e9:.2f} GB headroom; "
                          f"{max(0, free - reserved) / 1e9:.2f} GB is available in {root}.")
        _RESERVED[device] = reserved + size
    try:
        yield
    finally:
        with _LOCK:
            remaining = _RESERVED[device] - size
            if remaining:
                _RESERVED[device] = remaining
            else:
                _RESERVED.pop(device, None)
