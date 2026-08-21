_STOCK = {
    "widget": 42,
    "gadget": 17,
    "gizmo": 0,
}


def get_stock_level(product_name):
    """Return the stock level for a product, or 0 if it isn't tracked."""
    return _STOCK[product_name]
