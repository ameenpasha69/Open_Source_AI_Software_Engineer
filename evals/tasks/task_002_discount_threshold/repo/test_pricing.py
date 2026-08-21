from pricing import apply_bulk_discount


def test_no_discount_below_threshold():
    assert apply_bulk_discount(100, 5) == 100


def test_discount_above_threshold():
    assert apply_bulk_discount(100, 15) == 85.0


def test_discount_at_exact_threshold():
    assert apply_bulk_discount(100, 10) == 85.0
