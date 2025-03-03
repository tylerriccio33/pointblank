import sys

import pytest
from unittest.mock import patch

from pointblank.validate import load_dataset
from pointblank.draft import DraftValidation


def test_draft_fail_no_chatlas():
    with patch.dict(sys.modules, {"chatlas": None}):
        with pytest.raises(ImportError):
            DraftValidation(data="data", model="model")


def test_draft_fail_invalid_provider():
    small_table = load_dataset(dataset="small_table")

    with pytest.raises(ValueError):
        DraftValidation(data=small_table, model="invalid:model")
