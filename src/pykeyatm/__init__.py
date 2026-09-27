__version__ = "0.1.0"


def keyATM(*args, **kwargs):
    """Fit a keyword-assisted topic model."""
    from .python.keyATM import keyATM as _keyATM
    return _keyATM(*args, **kwargs)


def weightedLDA(*args, **kwargs):
    """Fit an LDA model using the package's supported count options."""
    from .python.keyATM import weightedLDA as _weightedLDA
    return _weightedLDA(*args, **kwargs)


def keyATM_read(*args, **kwargs):
    """Convert a document-term matrix to the input format used by PyKeyATM."""
    from .python.read import keyATM_read as _keyATM_read
    return _keyATM_read(*args, **kwargs)


__all__ = ["__version__", "keyATM", "keyATM_read", "weightedLDA"]
