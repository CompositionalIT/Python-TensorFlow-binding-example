from __future__ import annotations
from abc import abstractmethod
from array import array
import tensorflow
from typing import (Protocol, Tuple, Any, List, Callable)
from fable_modules.fable_library.reflection import (TypeInfo, class_type, union_type)
from fable_modules.fable_library.string import (to_console, interpolate)
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


def _expr2() -> TypeInfo:
    return union_type("Fable 4 Python Example.ISummary", [], ISummary, lambda: [[]])


class ISummary(Union):
    def __init__(self, tag: int, *fields: Any) -> None:
        super().__init__()
        self.tag: int = tag or 0
        self.fields: Array[Any] = list(fields)

    @staticmethod
    def cases() -> List[str]:
        return ["ISummary"]


ISummary_reflection = _expr2

def _expr3() -> TypeInfo:
    return union_type("Fable 4 Python Example.IOptimizer", [], IOptimizer, lambda: [[]])


class IOptimizer(Union):
    def __init__(self, tag: int, *fields: Any) -> None:
        super().__init__()
        self.tag: int = tag or 0
        self.fields: Array[Any] = list(fields)

    @staticmethod
    def cases() -> List[str]:
        return ["IOptimizer"]


IOptimizer_reflection = _expr3

def _expr4() -> TypeInfo:
    return union_type("Fable 4 Python Example.ILoss", [], ILoss, lambda: [[]])


class ILoss(Union):
    def __init__(self, tag: int, *fields: Any) -> None:
        super().__init__()
        self.tag: int = tag or 0
        self.fields: Array[Any] = list(fields)

    @staticmethod
    def cases() -> List[str]:
        return ["ILoss"]


ILoss_reflection = _expr4

def _expr5() -> TypeInfo:
    return union_type("Fable 4 Python Example.IHistory", [], IHistory, lambda: [[]])


class IHistory(Union):
    def __init__(self, tag: int, *fields: Any) -> None:
        super().__init__()
        self.tag: int = tag or 0
        self.fields: Array[Any] = list(fields)

    @staticmethod
    def cases() -> List[str]:
        return ["IHistory"]


IHistory_reflection = _expr5

class IModel(Protocol):
    @abstractmethod
    def compile(self, optimizer: IOptimizer, loss: ILoss, metrics: str) -> Callable[[ITensors], ITensors]:
        ...

    @abstractmethod
    def evaluate(self, x: NDArray, y: NDArray, verbose: int) -> int:
        ...

    @abstractmethod
    def fit(self, x: NDArray, y: NDArray, epochs: int) -> IHistory:
        ...

    @abstractmethod
    def summary(self) -> ISummary:
        ...


class IKeras(Protocol):
    @abstractmethod
    def Input(self, shape: Array[int]) -> ITensors:
        ...

    @abstractmethod
    def Model(self, inputs: ITensors, outputs: ITensors, name: str) -> IModel:
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

pattern_input_004099: Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]] = tensorflow.keras.datasets.mnist.load_data()

label_train: NDArray = pattern_input_004099[0][1]

label_test: NDArray = pattern_input_004099[1][1]

image_train: NDArray = pattern_input_004099[0][0]

image_test: NDArray = pattern_input_004099[1][0]

to_console(interpolate("Image train shape: %P()", [image_train.shape]))

image_train_flat: NDArray = (image_train.reshape(60000, 784)) / 255

to_console(interpolate("Image train flat shape: %P()", [image_train_flat.shape]))

image_test_flat: NDArray = (image_test.reshape(10000, 784)) / 255

inputs: ITensors = tensorflow.keras.Input(shape = array("l", [784]))

dense: Callable[[ITensors], ITensors] = tensorflow.keras.layers.Dense(units = 10)

outputs: ITensors = dense(inputs)

model: IModel = tensorflow.keras.Model(inputs = inputs, outputs = outputs, name = "Digit_Recognition")

to_console(interpolate("Model summary: %P()", [model.summary()]))

model.compile(optimizer = tensorflow.keras.optimizers.SGD(0.1), loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(True), metrics = "accuracy")

model.fit(x = image_train_flat, y = label_train, epochs = 1)

model.evaluate(x = image_test_flat, y = label_test, verbose = 2)

