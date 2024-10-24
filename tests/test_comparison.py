from pointblank.comparison import Comparator


def test_comparison_constructor():

    # Test the constructor
    comp = Comparator(x=[1, 2, 3, 4, 5], compare=3)

    assert comp.x == [1, 2, 3, 4, 5]
    assert comp.compare == [3, 3, 3, 3, 3]
