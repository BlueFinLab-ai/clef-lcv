"""Ordered immutable-value storage with a revision for derived accounting.

Replacing, adding or removing a value invalidates derived storage totals.
Changing recency or non-tensor metadata does not. Retained tensor snapshots
must never be mutated in place; replace their bank value instead.
"""
from collections import OrderedDict


class RevisionBank(OrderedDict):
    def __init__(self, *args, **kwargs):
        self.revision = 0
        super().__init__()
        self.update(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.revision += 1

    def __delitem__(self, key):
        super().__delitem__(key)
        self.revision += 1

    def clear(self):
        if self:
            super().clear()
            self.revision += 1

    def pop(self, key, *default):
        if len(default) > 1:
            raise TypeError('pop expected at most 2 arguments')
        if key not in self:
            if default:
                return default[0]
            raise KeyError(key)
        value = self[key]
        del self[key]
        return value

    def popitem(self, last=True):
        value = super().popitem(last=last)
        self.revision += 1
        return value

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def __ior__(self, other):
        self.update(other)
        return self
