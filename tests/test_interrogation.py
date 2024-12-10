from pointblank._interrogation import _get_nw_closed_str


def test_get_nw_closed_str():

    assert _get_nw_closed_str(closed=(True, True)) == "both"
    assert _get_nw_closed_str(closed=(True, False)) == "left"
    assert _get_nw_closed_str(closed=(False, True)) == "right"
    assert _get_nw_closed_str(closed=(False, False)) == "none"
