import hashlib


def hash_password(password, salt):
    """A toy password hash — not for real use, just a decoy module for the eval task."""
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
