from __future__ import annotations
from abc import abstractmethod
from array import array
import tensorflow
from typing import (Protocol, Tuple, Any, List, Callable)
from fable_modules.fable_library.reflection import (TypeInfo, class_type, union_type)
from fable_modules.fable_library.types import (Array, Union)

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


def _expr0() -> TypeInfo:
    return class_type("Fable 4 Python Example.NDArray", None, NDArray)


class NDArray:
    ...

NDArray_reflection = _expr0

class IMnist(Protocol):
    @abstractmethod
    def load_data(self) -> Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]]:
        ...


class IDataSets(Protocol):
    @property
    @abstractmethod
    def mnist(self) -> IMnist:
        ...


def _expr1() -> TypeInfo:
    return union_type("Fable 4 Python Example.ITensors", [], ITensors, lambda: [[]])


class ITensors(Union):
    def __init__(self, tag: int, *fields: Any) -> None:
        super().__init__()
        self.tag: int = tag or 0
        self.fields: Array[Any] = list(fields)

    @staticmethod
    def cases() -> List[str]:
        return ["ITensors"]


ITensors_reflection = _expr1

class Layers(Protocol):
    @abstractmethod
    def Dense(self, units: int) -> Callable[[ITensors], ITensors]:
        ...


class IKeras(Protocol):
    @abstractmethod
    def Input(self, shape: Array[int]) -> ITensors:
        ...

    @property
    @abstractmethod
    def datasets(self) -> IDataSets:
        ...

    @property
    @abstractmethod
    def layers(self) -> Layers:
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

pattern_input_004063: Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]] = tensorflow.keras.datasets.mnist.load_data()

label_train: NDArray = pattern_input_004063[0][1]

label_test: NDArray = pattern_input_004063[1][1]

image_train: NDArray = pattern_input_004063[0][0]

image_test: NDArray = pattern_input_004063[1][0]

image_train_flat: NDArray = (image_train.reshape(60000, 784)) / 255

image_test_flat: NDArray = (image_test.reshape(10000, 784)) / 255

inputs: ITensors = tensorflow.keras.Input(shape = array("l", [784]))

dense: Callable[[ITensors], ITensors] = tensorflow.keras.layers.Dense(units = 10)

outputs: ITensors = dense(inputs)

