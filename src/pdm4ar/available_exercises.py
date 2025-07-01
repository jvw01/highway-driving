from pdm4ar.exercises_def.ex12.ex12 import get_exercise12
import frozendict
from pdm4ar.exercises_def.structures import Exercise
from typing import Mapping, Callable

available_exercises: Mapping[str, Callable[[], Exercise]] = frozendict.frozendict({"12": get_exercise12})
