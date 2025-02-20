from enum import Enum
from typing import Type


def enum_from_string(value: str, enum: Type[Enum]):
    enum_value = [evalue for evalue in enum if evalue.name.lower() == value.lower()]
    if len(enum_value) == 0:
        raise Exception("Not supported")
    return enum_value[0]