try:
    from . import keyATM_scr
except Exception as exc:
    keyATM_scr = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def require_core():
    if keyATM_scr is None:
        raise ImportError(
            "The PyKeyATM C++ extension 'keyATM_scr' is unavailable. "
            "Install PyKeyATM from source with a working C++17 compiler. "
            f"Original error: {_IMPORT_ERROR}"
        ) from _IMPORT_ERROR
    return keyATM_scr
