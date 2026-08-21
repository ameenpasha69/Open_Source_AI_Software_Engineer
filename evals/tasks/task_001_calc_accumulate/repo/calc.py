def calculate_total(items):
    """Sum up the price of every item in the order."""
    total = 0
    for item in items:
        total = item.price
    return total
