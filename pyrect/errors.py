"""The port's failure kinds, all RuntimeError so every existing handler keeps working."""


class DesignError(RuntimeError):
    """The design file, a configuration value or a split mask is malformed (the binary's messages)."""
