import pytest

from dstack._internal.core.backends.gcp.compute import _is_supported_gcp_instance

pytestmark = pytest.mark.windows


@pytest.mark.parametrize(
    "instance_type",
    ["g4-standard-6", "g4-standard-12", "g4-standard-24"],
)
def test_fractional_g4_instance_types_are_not_supported(instance_type: str) -> None:
    assert not _is_supported_gcp_instance(instance_type, "RTXPRO6000")


def test_full_g4_instance_type_is_supported() -> None:
    assert _is_supported_gcp_instance("g4-standard-48", "RTXPRO6000")
