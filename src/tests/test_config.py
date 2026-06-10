import pytest
from src.config import *

def test_config_exists():
    assert True


def test_configuration_values():
    assert isinstance(WATERMARK_THRESHOLD, int)
    assert WATERMARK_THRESHOLD > 0


def test_inpaint_radius():
    assert INPAINT_RADIUS > 0
