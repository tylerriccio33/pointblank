from pointblank.comparison import Comparator

def test_comparison_constructor():

    # Test the constructor
    comp = Comparator([1, 2, 3, 4, 5], 3)

    assert comp.values == [1, 2, 3, 4, 5]
    assert comp.value == 3
