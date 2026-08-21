from users import make_username_slug


def test_simple_name():
    assert make_username_slug("Ada Lovelace") == "ada-lovelace"


def test_single_word_name():
    assert make_username_slug("Cher") == "cher"


def test_extra_whitespace():
    assert make_username_slug("  Grace Hopper  ") == "grace-hopper"
