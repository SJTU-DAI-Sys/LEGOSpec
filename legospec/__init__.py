"""LEGOSpec composable draft models."""


def __getattr__(name):
    if name in ("LegoSpecModel", "LegoSpecConfig"):
        from legospec.modeling.draft import LegoSpecModel, LegoSpecConfig

        return {"LegoSpecModel": LegoSpecModel, "LegoSpecConfig": LegoSpecConfig}[name]
    raise AttributeError(name)


__all__ = ["LegoSpecModel", "LegoSpecConfig"]
