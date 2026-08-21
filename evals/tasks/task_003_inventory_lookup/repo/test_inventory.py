from inventory import get_stock_level


def test_get_stock_level_known_product():
    assert get_stock_level("widget") == 42


def test_get_stock_level_zero_stock():
    assert get_stock_level("gizmo") == 0


def test_get_stock_level_unknown_product_returns_zero():
    assert get_stock_level("unknown-product") == 0
