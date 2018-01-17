# -*- coding: utf-8 -*-

def alias(aliased_class):
    original = aliased_class.__dict__.copy()
    for alias, name in aliased_class._aliases.items():
        setattr(aliased_class, alias, original[name])
    return aliased_class
