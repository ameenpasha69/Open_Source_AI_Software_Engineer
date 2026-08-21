def make_username_slug(display_name):
    """Turn a display name into a lowercase, hyphen-separated slug.

    e.g. "Ada Lovelace" -> "ada-lovelace"
    """
    words = display_name.strip().split(" ")
    return "-".join(words).upper()
