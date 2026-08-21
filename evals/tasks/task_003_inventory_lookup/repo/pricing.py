_PRICES = {
    "widget": 9.99,
    "gadget": 19.99,
    "gizmo": 4.99,
}


def get_price(product_name):
    """Return the price for a product, or None if it isn't sold."""
    return _PRICES.get(product_name)
