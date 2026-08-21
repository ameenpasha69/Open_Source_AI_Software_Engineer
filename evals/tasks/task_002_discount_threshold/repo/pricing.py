def apply_bulk_discount(total, quantity):
    """Orders of 10 or more items get a 15% discount."""
    if quantity > 10:
        return total * 0.85
    return total
