from __future__ import annotations
from abc import abstractmethod
import tensorflow
from typing import (Protocol, Tuple)
from fable_modules.fable_library.seq import (length, head)
from fable_modules.fable_library.string import (to_console, interpolate, printf)
from fable_modules.fable_library.types import (Array, uint8)
from fable_modules.fable_library.util import IEnumerable

class IPhysicalDevice(Protocol):
    @property
    @abstractmethod
    def device_type(self) -> str:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class IConfig(Protocol):
    @abstractmethod
    def list_physical_devices(self, __arg0: str) -> Array[IPhysicalDevice]:
        ...


class IMnist(Protocol):
    @abstractmethod
    def load_data(self) -> Tuple[Tuple[IEnumerable[IEnumerable[IEnumerable[uint8]]], IEnumerable[uint8]], Tuple[IEnumerable[IEnumerable[IEnumerable[uint8]]], IEnumerable[uint8]]]:
        ...


class IDataSets(Protocol):
    @property
    @abstractmethod
    def mnist(self) -> IMnist:
        ...


class IKeras(Protocol):
    @property
    @abstractmethod
    def datasets(self) -> IDataSets:
        ...


class ITensorFlow(Protocol):
    @property
    @abstractmethod
    def config(self) -> IConfig:
        ...

    @property
    @abstractmethod
    def keras(self) -> IKeras:
        ...


tensorflow.config.list_physical_devices("CPU")

pattern_input_004037: Tuple[Tuple[IEnumerable[IEnumerable[IEnumerable[uint8]]], IEnumerable[uint8]], Tuple[IEnumerable[IEnumerable[IEnumerable[uint8]]], IEnumerable[uint8]]] = tensorflow.keras.datasets.mnist.load_data()

label_train: IEnumerable[uint8] = pattern_input_004037[0][1]

label_test: IEnumerable[uint8] = pattern_input_004037[1][1]

image_train: IEnumerable[IEnumerable[IEnumerable[uint8]]] = pattern_input_004037[0][0]

image_test: IEnumerable[IEnumerable[IEnumerable[uint8]]] = pattern_input_004037[1][0]

to_console(interpolate("Image count: %P()", [length(image_train)]))

arg: int = length(head(image_train)) or 0

to_console(printf("Line count: %A"))(arg)

arg: int = length(head(head(image_train))) or 0

to_console(printf("Column count: %A"))(arg)

