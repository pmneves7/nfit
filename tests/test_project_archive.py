import zipfile

import pytest

from nfit.project_archive import open_project_artifact


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_open_project_artifact_is_bounded_and_seekable(tmp_path, compression):
    path = tmp_path / "project.nfit"
    member = "assets/binnings/cache/data.npz"
    payload = b"nested artifact payload"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, payload, compress_type=compression)
        archive.writestr("after.bin", b"must not be visible")

    with open_project_artifact(path, member) as stream:
        assert stream.seekable()
        assert stream.read(6) == payload[:6]
        stream.seek(-7, 2)
        assert stream.read() == payload[-7:]
        stream.seek(0)
        assert stream.read() == payload
