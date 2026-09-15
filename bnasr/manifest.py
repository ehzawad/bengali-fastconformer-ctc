"""Atomic manifest publication.

Every source's resume check treats "manifest file exists" as "split already
complete", while write_manifest streams rows into that same final path. An
interrupted run therefore leaves a partial manifest that the next run accepts as
finished, silently dropping the rest of the corpus -- this cost 67% of the
OpenSLR-53 corpus once already, with no error raised.

Publishing through a temp file in the same directory and os.replace makes the
final path appear only once the manifest is complete.
"""
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_manifest(manifest_path):
    """Yield a writable handle whose content lands at manifest_path only on
    clean exit. On exception or process kill the final path never appears."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=manifest_path.parent, prefix=f".{manifest_path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, manifest_path)
    finally:
        tmp_path.unlink(missing_ok=True)
