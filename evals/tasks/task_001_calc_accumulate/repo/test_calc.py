from calc import calculate_total


class Item:
    def __init__(self, price):
        self.price = price


def test_calculate_total_single_item():
    assert calculate_total([Item(10)]) == 10


def test_calculate_total_multiple_items():
    assert calculate_total([Item(10), Item(20), Item(30)]) == 60


def test_calculate_total_empty_order():
    assert calculate_total([]) == 0
