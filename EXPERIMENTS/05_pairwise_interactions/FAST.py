"""
FAST (Feature Association by Statistical Test) for pairwise interaction detection.

Reference:
Y. Lou, R. Caruana, J. Gehrke, and G. Hooker. Accurate intelligible models with 
pairwise interactions. In Proceedings of the 19th ACM SIGKDD International Conference 
on Knowledge Discovery and Data Mining (KDD), Chicago, IL, USA, 2013.

Python implementation converted from Java.
# package mltk.predictor.gam.interaction

- MathUtils: mltk.util.MathUtils;
- IntPair: mltk.util.tuple.IntPair;
- Pair: mltk.util.tuple.Pair;
- Element: mltk.util.Element;
- Type: mltk.core.Attribute.Type;
- Attribute: mltk.core.Attribute;
- BinnedAttribute: mltk.core.BinnedAttribute;
- NominalAttribute: mltk.core.NominalAttribute;
- NumericalAttribute: mltk.core.NumericalAttribute;
- Instance: mltk.core.Instance;
- Instances: mltk.core.Instances;
- InstancesReader: mltk.core.io.InstancesReader;
- Discretizer: mltk.core.processor.Discretizer;
- CHistogram: mltk.predictor.function.CHistogram;
- Histogram2D: mltk.predictor.function.Histogram2D;
- DoublePair: mltk.util.tuple.DoublePair;
- Bins: mltk.core.Bins;
"""

import threading
import argparse
import time
import struct
from dataclasses import dataclass
from typing import List, Optional, Any, TypeVar, Generic, Union, Tuple, Callable, Dict
import math
from functools import total_ordering
import numpy as np
import bisect
import pandas as pd

# =========================
# ---- mltk.*  helpers  -----
# =========================

class MathUtils:
    """
    Utility class for math functions.
    """

    EPSILON = 1e-8
    LOG2 = math.log(2)

    @staticmethod
    def equals(a: float, b: float) -> bool:
        """Returns True if two floats are equal within EPSILON."""
        return abs(a - b) < MathUtils.EPSILON

    @staticmethod
    def indicator(b: bool) -> int:
        """Returns 1 if input is True, otherwise 0."""
        return 1 if b else 0

    @staticmethod
    def is_first_better(a: float, b: float, is_larger_better: bool) -> bool:
        """Returns True if the first value is better."""
        return a > b if is_larger_better else a < b

    @staticmethod
    def is_integer(v: float) -> bool:
        """Returns True if the float is an integer."""
        return v % 1 == 0

    @staticmethod
    def is_zero(v: float) -> bool:
        """Returns True if the float is zero within EPSILON."""
        return abs(v) < MathUtils.EPSILON

    @staticmethod
    def sigmoid(a: float) -> float:
        """Returns the value of a sigmoid function."""
        return 1 / (1 + math.exp(-a))

    @staticmethod
    def sign(a) -> int:
        """Returns the sign of a number (-1, 0, 1)."""
        if a < 0:
            return -1
        elif a > 0:
            return 1
        else:
            return 0

    @staticmethod
    def divide(a: float, b: float, dv: float) -> float:
        """Performs division and returns default value if division by zero."""
        return dv if MathUtils.is_zero(b) else a / b

class DoublePair:
    """
    Class for <double, double> pair.
    Traduzione fedele da Java a Python.
    """

    def __init__(self, v1: float, v2: float):
        """
        Constructor.

        :param v1: the 1st double
        :param v2: the 2nd double
        """
        self.v1: float = v1
        self.v2: float = v2

    def __hash__(self) -> int:
        prime = 31
        result = 1

        # Simuliamo Double.doubleToLongBits di Java
        def double_to_long_bits(value: float) -> int:
            return struct.unpack(">Q", struct.pack(">d", value))[0]

        temp = double_to_long_bits(self.v1)
        result = prime * result + (int(temp ^ (temp >> 32)) & 0xFFFFFFFF)

        temp = double_to_long_bits(self.v2)
        result = prime * result + (int(temp ^ (temp >> 32)) & 0xFFFFFFFF)

        return result

    def __eq__(self, obj: object) -> bool:
        if self is obj:
            return True
        if obj is None or not isinstance(obj, DoublePair):
            return False

        def double_to_long_bits(value: float) -> int:
            return struct.unpack(">Q", struct.pack(">d", value))[0]

        if double_to_long_bits(self.v1) != double_to_long_bits(obj.v1):
            return False
        if double_to_long_bits(self.v2) != double_to_long_bits(obj.v2):
            return False

        return True

class Bins:
    """
    Class for bins. Each bin is defined as its upper bound and median.
    Traduzione fedele da Java a Python.
    """

    def __init__(self, boundaries: List[float] = None, medians: List[float] = None):
        if boundaries is None and medians is None:
            # protected Bins() in Java → costruttore vuoto
            self.boundaries: List[float] = []
            self.medians: List[float] = []
        else:
            if len(boundaries) != len(medians):
                raise ValueError("Boundary size doesn't match medians size")
            self.boundaries: List[float] = boundaries
            self.medians: List[float] = medians

    def size(self) -> int:
        """
        Returns the number of bins.
        """
        return len(self.boundaries)

    def getIndex(self, value: float) -> int:
        """
        Returns the bin index given a real value using binary search.
        """
        if value < self.boundaries[0]:
            return 0
        elif value >= self.boundaries[-1]:
            return len(self.boundaries) - 1
        else:
            return ArrayUtils.findInsertionPoint(self.boundaries, value)

    def getValue(self, index: int) -> float:
        """
        Returns the median of a bin.
        """
        return self.medians[index]

    def getBoundaries(self) -> List[float]:
        """
        Returns the upper bounds for each bin.
        """
        return self.boundaries

    def getMedians(self) -> List[float]:
        """
        Returns the medians for each bin.
        """
        return self.medians
    
@dataclass(frozen=True, order=True)
class IntPair:
    v1: int
    v2: int

    def __hash__(self):
        prime = 31
        result = 1
        result = prime * result + self.v1
        result = prime * result + self.v2
        return result

    def __eq__(self, other):
        if self is other:
            return True
        if other is None or not isinstance(other, IntPair):
            return False
        return self.v1 == other.v1 and self.v2 == other.v2


T1 = TypeVar("T1")
T2 = TypeVar("T2")

@dataclass(frozen=True)
class Pair(Generic[T1, T2]):
    v1: T1
    v2: T2

    def __hash__(self):
        prime = 31
        result = 1
        result = prime * result + (0 if self.v1 is None else hash(self.v1))
        result = prime * result + (0 if self.v2 is None else hash(self.v2))
        return result

    def __eq__(self, other):
        if self is other:
            return True
        if other is None or not isinstance(other, Pair):
            return False

        if self.v1 is None:
            if other.v1 is not None:
                return False
        else:
            if type(self.v1) is not type(other.v1):
                return False
            if self.v1 != other.v1:
                return False

        if self.v2 is None:
            if other.v2 is not None:
                return False
        else:
            if type(self.v2) is not type(other.v2):
                return False
            if self.v2 != other.v2:
                return False

        return True


class Element:
    def __init__(self, element, weight: float):
        """
        Costruisce un elemento con peso.

        :param element: L'elemento generico.
        :param weight: Il peso (float).
        """
        self.element = element
        self.weight = float(weight)

    def compare_to(self, other):
        """
        Confronta due Element in base al peso.
        Restituisce:
        - un numero negativo se self < other
        - zero se sono uguali
        - un numero positivo se self > other
        """
        return (self.weight > other.weight) - (self.weight < other.weight)

    def __lt__(self, other):
        return self.weight < other.weight

    def __eq__(self, other):
        return self.weight == other.weight

from abc import ABC, abstractmethod
from enum import Enum


class Attribute(ABC):
    class Type(Enum):
        NOMINAL = 1
        NUMERIC = 2
        BINNED = 3

    def __init__(self):
        self.type: Attribute.Type | None = None
        self.index: int = 0
        self.name: str | None = None

    def getType(self) -> "Attribute.Type":
        return self.type

    def getIndex(self) -> int:
        return self.index

    def setIndex(self, index: int) -> None:
        self.index = index

    def getName(self) -> str | None:
        return self.name

    def compareTo(self, att: "Attribute") -> int:
        return self.index - att.index

    def __hash__(self) -> int:
        prime = 31
        result = 1
        result = prime * result + self.index
        result = prime * result + (0 if self.name is None else hash(self.name))
        result = prime * result + (0 if self.type is None else hash(self.type))
        return result

    def __eq__(self, obj: object) -> bool:
        if self is obj:
            return True
        if obj is None or not isinstance(obj, Attribute):
            return False
        other: Attribute = obj
        if self.index != other.index:
            return False
        if self.name is None:
            if other.name is not None:
                return False
        elif self.name != other.name:
            return False
        if self.type != other.type:
            return False
        return True

    def __str__(self) -> str:
        return "" if self.name is None else self.name



class BinnedAttribute(Attribute):
    def __init__(self, name: str, numBins_or_bins, index: int = -1):
        super().__init__()
        self.name = name
        self.bins: Optional[Bins] = None
        if isinstance(numBins_or_bins, int):
            self.numBins = numBins_or_bins
            self.index = index
        elif isinstance(numBins_or_bins, Bins):
            self.numBins = numBins_or_bins.size()
            self.bins = numBins_or_bins
            self.index = index
        else:
            raise TypeError("Expected int or Bins for numBins_or_bins")
        self.type = Attribute.Type.BINNED

    def copy(self):
        copy = BinnedAttribute(self.name, self.numBins if self.bins is None else self.bins)
        copy.index = self.index
        return copy

    def getNumBins(self) -> int:
        return self.numBins

    def getBins(self) -> Optional[Bins]:
        return self.bins

    def __str__(self) -> str:
        if self.bins is None:
            return f"{self.name}: binned ({self.numBins})"
        else:
            return (f"{self.name}: binned ({self.bins.size()};"
                    f"{self.bins.boundaries};"
                    f"{self.bins.medians})")

    @staticmethod
    def parse(s: str):
        data = s.split(": ")
        start = data[1].index("(") + 1
        end = data[1].index(")")
        strs = data[1][start:end].split(";")
        numBins = int(strs[0])
        if len(strs) == 1:
            return BinnedAttribute(data[0], numBins)
        else:
            boundaries = ArrayUtils.parseDoubleArray(strs[1])
            medians = ArrayUtils.parseDoubleArray(strs[2])
            bins = Bins(boundaries, medians)
            return BinnedAttribute(data[0], bins)


class NominalAttribute(Attribute):
    def __init__(self, name: str, states: List[str], index: int = -1):
        super().__init__()
        self.name = name
        self.states = states
        self.index = index
        self.type = Attribute.Type.NOMINAL

    def copy(self):
        copy = NominalAttribute(self.name, self.states)
        copy.index = self.index
        return copy

    def getCardinality(self) -> int:
        return len(self.states)

    def getState(self, index: int) -> str:
        return self.states[index]

    def getStates(self) -> List[str]:
        return self.states

    def __str__(self) -> str:
        return f"{self.name}: {{" + ", ".join(self.states) + "}}"

    @staticmethod
    def parse(s: str):
        data = s.split(": ")
        start = data[1].index("{") + 1
        end = data[1].index("}")
        states = [st.strip() for st in data[1][start:end].split(",")]
        return NominalAttribute(data[0], states)


class NumericalAttribute(Attribute):
    def __init__(self, name: str, index: int = -1):
        super().__init__()
        self.name = name
        self.index = index
        self.type = Attribute.Type.NUMERIC

    def copy(self):
        copy = NumericalAttribute(self.name)
        copy.index = self.index
        return copy

    def __str__(self) -> str:
        return f"{self.name}: cont"

    @staticmethod
    def parse(s: str):
        data = s.split(": ")
        return NumericalAttribute(data[0])


class Vector(ABC):
    """
    Interface for vectors.
    """

    @abstractmethod
    def getValue(self, index: int) -> float:
        pass

    @abstractmethod
    def getValues(self, *indices: int) -> List[float]:
        pass

    @abstractmethod
    def setValue(self, index: int, value: float):
        pass

    @abstractmethod
    def setValueArray(self, indices: List[int], v: List[float]):
        pass

    @abstractmethod
    def isSparse(self) -> bool:
        pass

    @abstractmethod
    def copy(self) -> "Vector":
        pass


class DenseVector(Vector):
    """
    Class for dense vectors.
    """

    def __init__(self, values: List[float]):
        self.values = list(values)

    def getValue(self, index: int) -> float:
        return self.values[index]

    def getValues(self, *indices: int) -> List[float]:
        if not indices:
            return list(self.values)
        return [self.values[i] for i in indices]

    def setValue(self, index: int, value: float):
        self.values[index] = value

    def setValueArray(self, indices: List[int], v: List[float]):
        for i in range(len(indices)):
            self.values[indices[i]] = v[i]

    def copy(self) -> "DenseVector":
        return DenseVector(self.values.copy())

    def isSparse(self) -> bool:
        return False


class SparseVector(Vector):
    """
    Class for sparse vectors.
    """

    def __init__(self, indices: List[int], values: List[float]):
        self.indices = list(indices)
        self.values = list(values)

    def copy(self) -> "SparseVector":
        return SparseVector(self.indices.copy(), self.values.copy())

    def getValue(self, index: int) -> float:
        # binary search to match Arrays.binarySearch
        idx = bisect.bisect_left(self.indices, index)
        if idx < len(self.indices) and self.indices[idx] == index:
            return self.values[idx]
        else:
            return 0.0

    def getValues(self, *indices: int) -> List[float]:
        if not indices:
            return list(self.values)
        return [self.getValue(i) for i in indices]

    def getIndices(self) -> List[int]:
        return self.indices

    def setValue(self, index: int, value: float):
        idx = bisect.bisect_left(self.indices, index)
        if idx < len(self.indices) and self.indices[idx] == index:
            self.values[idx] = value
        else:
            raise NotImplementedError("setValue for new index not supported")

    def setValueArray(self, indices: List[int], v: List[float]):
        for i in range(len(indices)):
            self.setValue(indices[i], v[i])

    def isSparse(self) -> bool:
        return True


class Instance:
    def __init__(self, *args):
        """
        Emula i costruttori multipli di Java
        """
        self.vector: Optional[Vector] = None
        self.target: List[float] = [float("nan")]
        self.weight: float = 1.0

        if len(args) == 3 and isinstance(args[0], list) and all(isinstance(x, float) for x in args[0]):
            values, target, weight = args
            self.vector = DenseVector(values)
            self.target = [target]
            self.weight = weight
        elif len(args) == 4 and isinstance(args[0], list) and all(isinstance(x, int) for x in args[0]):
            indices, values, target, weight = args
            self.vector = SparseVector(indices, values)
            self.target = [target]
            self.weight = weight
        elif len(args) == 3 and isinstance(args[0], Vector):
            vector, target, weight = args
            self.vector = vector
            self.target = [target]
            self.weight = weight
        elif len(args) == 2 and isinstance(args[0], list) and all(isinstance(x, float) for x in args[0]):
            values, target = args
            self.vector = DenseVector(values)
            self.target = [target]
            self.weight = 1.0
        elif len(args) == 3 and isinstance(args[0], list) and all(isinstance(x, int) for x in args[0]):
            indices, values, target = args
            self.vector = SparseVector(indices, values)
            self.target = [target]
            self.weight = 1.0
        elif len(args) == 2 and isinstance(args[0], Vector):
            vector, target = args
            self.vector = vector
            self.target = [target]
            self.weight = 1.0
        elif len(args) == 1 and isinstance(args[0], list) and all(isinstance(x, float) for x in args[0]):
            values, = args
            self.vector = DenseVector(values)
            self.target = [float("nan")]
            self.weight = 1.0
        elif len(args) == 2 and isinstance(args[0], list) and all(isinstance(x, int) for x in args[0]):
            indices, values = args
            self.vector = SparseVector(indices, values)
            self.target = [float("nan")]
            self.weight = 1.0
        elif len(args) == 1 and isinstance(args[0], Vector):
            vector, = args
            self.vector = vector
            self.target = [float("nan")]
            self.weight = 1.0
        elif len(args) == 1 and isinstance(args[0], Instance):
            instance, = args
            self.vector = instance.vector
            self.target = instance.target
            self.weight = instance.weight

    def isSparse(self) -> bool:
        return self.vector.isSparse()

    def getValue(self, attIndex: Union[int, Attribute]) -> float:
        if isinstance(attIndex, Attribute):
            return self.vector.getValue(attIndex.getIndex())
        return self.vector.getValue(attIndex)

    def getValues(self, *attributes: int) -> List[float]:
        return self.vector.getValues(*attributes)

    def setValue(self, attIndex: Union[int, Attribute, List[int]], value: Union[float, List[float]]):
        if isinstance(attIndex, Attribute):
            self.vector.setValue(attIndex.getIndex(), value)
        elif isinstance(attIndex, list) and isinstance(value, list):
            for i in range(len(attIndex)):
                self.vector.setValue(attIndex[i], value[i])
        else:
            self.vector.setValue(attIndex, value)

    def copy(self) -> "Instance":
        copyVector = self.vector.copy()
        return Instance(copyVector, self.target[0], self.weight)

    def clone(self) -> "Instance":
        return Instance(self)

    def isMissing(self, attIndex: int) -> bool:
        return math.isnan(self.getValue(attIndex))

    def getVector(self) -> Vector:
        return self.vector

    def getWeight(self) -> float:
        return self.weight

    def setWeight(self, weight: float):
        self.weight = weight

    def getTarget(self) -> float:
        return self.target[0]

    def setTarget(self, target: float):
        self.target[0] = target

    def __str__(self) -> str:
        sb = []
        if self.isSparse():
            sb.append(str(self.getTarget()))
            sv: SparseVector = self.vector
            indices = sv.getIndices()
            values = sv.getValues()
            for i in range(len(indices)):
                sb.append(f" {indices[i]}:")
                sb.append(self._format(values[i]))
        else:
            values = self.getValues()
            sb.append(self._format(values[0]))
            for i in range(1, len(values)):
                sb.append("\t")
                sb.append(self._format(values[i]))
            if not math.isnan(self.getTarget()):
                sb.append("\t")
                sb.append(self._format(self.getTarget()))
        return "".join(sb)

    def _format(self, v: float) -> str:
        if MathUtils.isInteger(v):
            return str(int(v))
        else:
            return str(v)

import random
from typing import List, Iterator, Optional


class Random:
    instance: Optional["Random"] = None

    def __init__(self):
        self.rand = random.Random()

    @staticmethod
    def getInstance() -> "Random":
        if Random.instance is None:
            Random.instance = Random()
        return Random.instance

    def setSeed(self, seed: int) -> None:
        self.rand.seed(seed)

    def nextInt(self, n: Optional[int] = None) -> int:
        if n is None:
            # In Java, nextInt() without args gives any int32
            return self.rand.randint(-(2**31), 2**31 - 1)
        return self.rand.randrange(n)

    def nextDouble(self) -> float:
        return self.rand.random()

    def nextFloat(self) -> float:
        return float(self.rand.random())

    def nextGaussian(self) -> float:
        return self.rand.gauss(0.0, 1.0)

    def nextLong(self) -> int:
        # In Java: 64-bit signed
        return self.rand.randint(-(2**63), 2**63 - 1)

    def nextBoolean(self) -> bool:
        return self.rand.choice([True, False])

    def nextBytes(self, bytes_arr: bytearray) -> None:
        for i in range(len(bytes_arr)):
            bytes_arr[i] = self.rand.randint(0, 255)

    def getRandom(self) -> random.Random:
        return self.rand


class Instances:
    def __init__(self, 
                 attributes: List["Attribute"], 
                 targetAtt: Optional["Attribute"] = None, 
                 capacity: int = 1000):
        self.attributes: List["Attribute"] = attributes
        self.targetAtt: Optional["Attribute"] = targetAtt
        self.instances: List["Instance"] = []

    @classmethod
    def fromAttributes(cls, attributes: List["Attribute"]) -> "Instances":
        return cls(attributes, None)

    @classmethod
    def fromAttributesWithCapacity(cls, attributes: List["Attribute"], capacity: int) -> "Instances":
        return cls(attributes, None, capacity)

    @classmethod
    def fromInstances(cls, instances: "Instances") -> "Instances":
        copy = cls(instances.attributes, instances.targetAtt, len(instances.instances))
        copy.instances = list(instances.instances)
        return copy

    # ADD-ON AM
    def add(self, instance: "Instance") -> None:
        self.instances.append(instance)

    def get(self, index: int) -> "Instance":
        return self.instances[index]

    def getTargetAttribute(self) -> Optional["Attribute"]:
        return self.targetAtt

    def setTargetAttribute(self, targetAtt: "Attribute") -> None:
        self.targetAtt = targetAtt

    def __iter__(self) -> Iterator["Instance"]:
        return iter(self.instances)

    def size(self) -> int:
        return len(self.instances)

    def dimension(self) -> int:
        return len(self.attributes)

    def getAttributes(self, *indices: int) -> List["Attribute"]:
        if not indices:
            return self.attributes
        return [self.attributes[i] for i in indices]

    def setAttributes(self, attributes: List["Attribute"]) -> None:
        self.attributes = attributes

    def clear(self) -> None:
        self.instances.clear()

    def shuffle(self) -> None:
        rnd = Random.getInstance().getRandom()
        rnd.shuffle(self.instances)

    def shuffleWithRandom(self, rand: random.Random) -> None:
        rand.shuffle(self.instances)

    def copy(self) -> "Instances":
        attributes_copy = list(self.attributes)
        copy = Instances(attributes_copy, self.targetAtt, len(self.instances))
        for instance in self.instances:
            copy.add(instance.copy())
        return copy
    @staticmethod
    def ensure_numpy_cache(instances):
        if getattr(instances, "_np_ready", False):
            return
        n = instances.size()
        p = len(instances.getAttributes())
        bins = np.full((n, p), -1, dtype=np.int32)  # -1 = missing
        w = np.empty(n, dtype=np.float32)
        y = np.empty(n, dtype=np.float32)
        for i, row in enumerate(instances):
            w[i] = row.getWeight()
            y[i] = row.getTarget()
            for j in range(p):
                if not row.isMissing(j):
                    bins[i, j] = int(row.getValue(j))
        instances._np_bins = bins
        instances._np_w = w
        instances._np_y = y
        instances._np_ready = True
    
    @staticmethod
    def from_numpy(X: np.ndarray, feature_names: list[str]) -> "Instances":
        n_samples, n_features = X.shape

        # 1. Build attributes metadata (all numeric here)
        attributes = [NumericalAttribute(name, index=i) for i, name in enumerate(feature_names)]

        # 2. Build instances
        insts: list[Instance] = []
        for i in range(n_samples):
            row = X[i, :]
            values = row.astype(float).tolist()
            inst = Instance(values)  
            insts.append(inst)

        # 3. Create Instances object and attach data
        instances = Instances(attributes, None, n_samples)
        for inst in insts:
            instances.add(inst)

        return instances





class InstancesReader:

    @staticmethod
    def read(att_file: Optional[str], data_file: str, delimiter: str = r"\s+"):
        import re

        if att_file is not None:
            # In Java veniva usato AttributesReader.read(att_file)
            # Qui lo semplifichiamo: niente supporto esterno
            attributes = []  # Placeholder
            class_attr = None
            class_index = -1

            instances = Instances(attributes, class_attr)
            total_length = instances.dimension() + (1 if class_attr else 0)

            with open(data_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = re.split(delimiter, line)
                    instance = None
                    if len(data) >= 2 and ":" in data[1]:
                        instance = InstancesReader.parse_sparse_instance(data)
                    elif len(data) == total_length:
                        instance = InstancesReader.parse_dense_instance(data, class_index)
                    else:
                        print("Warning: Dense vector mismatch with attributes.")
                    if instance:
                        instances.add(instance)
            return instances

        else:
            attributes = []
            instances = Instances(attributes)
            total_length = -1
            attr_set = set()

            import re
            with open(data_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = re.split(delimiter, line)
                    instance = None
                    if len(data) >= 2 and ":" in data[1]:
                        instance = InstancesReader.parse_sparse_instance_with_set(data, attr_set)
                    else:
                        if total_length == -1:
                            total_length = len(data)
                        elif len(data) == total_length:
                            instance = InstancesReader.parse_dense_instance(data, -1)
                    if instance:
                        instances.add(instance)

            if total_length == -1:
                for att_index in sorted(attr_set):
                    attributes.append(NumericalAttribute(f"f{att_index}"))
                    attributes[-1].setIndex(att_index)
            else:
                for j in range(total_length):
                    attributes.append(NumericalAttribute(f"f{j}"))
                    attributes[-1].setIndex(j)

            InstancesReader.assign_target_attribute(instances)
            return instances

    @staticmethod
    def parse_dense_instance(data: List[str], class_index: int) -> Instance:
        class_value = float("nan")
        if class_index < 0:
            vector = [InstancesReader.parse_double(x) for x in data]
            return Instance(vector, class_value)
        else:
            vector = []
            for i, x in enumerate(data):
                val = InstancesReader.parse_double(x)
                if i == class_index:
                    class_value = val
                else:
                    vector.append(val)
            return Instance(vector, class_value)

    @staticmethod
    def parse_sparse_instance_with_set(data: List[str], attr_set: set) -> Instance:
        targetValue = float(data[0])
        indices = []
        values = []
        for pair in data[1:]:
            k, v = pair.split(":")
            indices.append(int(k))
            values.append(float(v))
            attr_set.add(int(k))
        return Instance((indices, values, targetValue))

    @staticmethod
    def parse_sparse_instance(data: List[str]) -> Instance:
        class_value = float(data[0])
        indices = []
        values = []
        for pair in data[1:]:
            k, v = pair.split(":")
            indices.append(int(k))
            values.append(float(v))
        return Instance((indices, values, class_value))

    @staticmethod
    def assign_target_attribute(instances: Instances):
        is_integer = all(float(i.getTarget()).is_integer() for i in instances)
        if is_integer:
            states = sorted({int(i.getTarget()) for i in instances})
            states_str = [str(v) for v in states]
            instances.setTargetAttribute(NominalAttribute("target", states_str))
        else:
            instances.setTargetAttribute(NumericalAttribute("target"))

    @staticmethod
    def parse_double(s: str) -> float:
        return float("nan") if s == "?" else float(s)

class ArrayUtils:
    """
    Utility per array (conversioni e funzioni varie).
    """

    @staticmethod
    def toIntArray(lst: List[int]) -> np.ndarray:
        return np.array(lst, dtype=int)

    @staticmethod
    def toDoubleArray(lst: List[float]) -> np.ndarray:
        return np.array(lst, dtype=float)

    @staticmethod
    def toIntArrayFromDouble(lst: List[float]) -> np.ndarray:
        return np.array(lst, dtype=int)

    @staticmethod
    def toString(a: List[float], start: int, end: int) -> str:
        return "[" + ", ".join(str(a[i]) for i in range(start, end)) + "]"

    @staticmethod
    def parseDoubleArray(s: Optional[str], delimiter: str = ",") -> Optional[np.ndarray]:
        if s is None or s.lower() == "null":
            return None
        data = s.strip()[1:-1].split(delimiter)
        return np.array([float(x.strip()) for x in data])

    @staticmethod
    def parseIntArray(s: Optional[str], delimiter: str = ",") -> Optional[np.ndarray]:
        if s is None or s.lower() == "null":
            return None
        data = s.strip()[1:-1].split(delimiter)
        return np.array([int(x.strip()) for x in data])

    @staticmethod
    def parseLongArray(s: Optional[str], delimiter: str = ",") -> Optional[np.ndarray]:
        if s is None or s.lower() == "null":
            return None
        data = s.strip()[1:-1].split(delimiter)
        return np.array([int(x.strip()) for x in data], dtype=np.int64)

    @staticmethod
    def isConstant(arr, begin: int, end: int, c) -> bool:
        return all(arr[i] == c for i in range(begin, end))

    @staticmethod
    def getMedian(a: List[float]) -> float:
        if len(a) == 0:
            return 0.0
        ary = sorted(a)
        mid = len(ary) // 2
        if len(ary) % 2 == 1:
            return ary[mid]
        else:
            return (ary[mid - 1] + ary[mid]) / 2.0

    @staticmethod
    def findInsertionPoint(a: List[float], key: float) -> int:
        """Trova l’indice di inserimento in un array ordinato (equivalente a Arrays.binarySearch in Java)."""
        return bisect.bisect_left(a, key)

class Discretizer:

    @staticmethod
    def computeBins_from_array(x: List[float], maxNumBins: int) -> Bins:
        lst = [Element(v, 1.0) for v in x if not math.isnan(v)]
        return Discretizer.computeBins_elements(lst, maxNumBins)

    @staticmethod
    def computeBins_elements(instances: List["Element"], maxNumBins: int) -> Bins:
        instances.sort(key=lambda e: e.element)
        stats: List[DoublePair] = []
        Discretizer.getStats(instances, stats)

        if len(stats) <= maxNumBins:
            a = [s.v1 for s in stats]
            return Bins(a, a)
        else:
            totalWeight = sum(s.v2 for s in stats)
            binSize = totalWeight / maxNumBins
            boundaryList = []
            medianList = []
            start = 0
            weight = 0

            for i, s in enumerate(stats):
                weight += s.v2
                totalWeight -= s.v2
                if weight >= binSize:
                    if i == start:
                        boundaryList.append(stats[start].v1)
                        medianList.append(stats[start].v1)
                        weight = 0
                        start = i + 1
                    else:
                        d1 = weight - binSize
                        d2 = s.v2 - d1
                        if d1 < d2:
                            boundaryList.append(s.v1)
                            medianList.append(
                                Discretizer.getMedian(stats, start, weight / 2)
                            )
                            start = i + 1
                            weight = 0
                        else:
                            weight -= s.v2
                            boundaryList.append(stats[i - 1].v1)
                            medianList.append(
                                Discretizer.getMedian(stats, start, weight / 2)
                            )
                            start = i
                            weight = s.v2

                    # 🚨 FIX: controlla per divisione per zero
                    if maxNumBins - len(boundaryList) <= 0:
                        break
                    binSize = (totalWeight + weight) / (maxNumBins - len(boundaryList))

                elif i == len(stats) - 1:
                    boundaryList.append(s.v1)
                    medianList.append(Discretizer.getMedian(stats, start, weight / 2))

            return Bins(boundaryList, medianList)


    @staticmethod
    def computeBins(instances: "Instances", attIndex: int, maxNumBins: int) -> Bins:
        """
        Compute bins from an Instances dataset given an attribute index.
        """
        values = [inst.getValue(attIndex) for inst in instances if not inst.isMissing(attIndex)]
        return Discretizer.computeBins_from_array(values, maxNumBins)

    @staticmethod
    def discretize(instances: "Instances", attIndex: int, bins_or_numBins):
        """
        Discretizes an attribute.
        - If bins_or_numBins is a Bins → use directly
        - If bins_or_numBins is an int → compute bins first
        """
        from typing import Union

        if isinstance(bins_or_numBins, int):
            bins = Discretizer.computeBins(instances, attIndex, bins_or_numBins)
        else:
            bins = bins_or_numBins  # assume it's already a Bins

        attribute = instances.getAttributes()[attIndex]
        binnedAttribute = BinnedAttribute(attribute.getName(), bins)
        binnedAttribute.setIndex(attribute.getIndex())
        instances.getAttributes()[attIndex] = binnedAttribute

        for instance in instances:
            if not instance.isMissing(attribute.getIndex()):
                v = bins.getIndex(instance.getValue(attribute.getIndex()))
                instance.setValue(attribute.getIndex(), v)


    @staticmethod
    def discretize_with_num_bins(instances: "Instances", attIndex: int, maxNumBins: int) -> None:
        """
        Discretizes an attribute with specified number of bins.

        :param instances: the dataset to discretize.
        :param attIndex: the attribute index.
        :param maxNumBins: the number of bins.
        """
        bins = Discretizer.computeBins(instances, attIndex, maxNumBins)
        Discretizer.discretize(instances, attIndex, bins)

    @staticmethod
    def getMedian(stats: List[DoublePair], start: int, midPoint: float) -> float:
        weight = 0
        for i in range(start, len(stats)):
            weight += stats[i].v2
            if weight >= midPoint:
                return stats[i].v1
        return stats[(start + len(stats)) // 2].v1

    @staticmethod
    def getStats(lst: List[Element], stats: List[DoublePair]):
        if not lst:
            return
        totalWeight = lst[0].weight
        lastValue = lst[0].element
        for element in lst[1:]:
            value = element.element
            weight = element.weight
            if value != lastValue:
                stats.append(DoublePair(lastValue, totalWeight))
                lastValue = value
                totalWeight = weight
            else:
                totalWeight += weight
        stats.append(DoublePair(lastValue, totalWeight))

class CHistogram:
    """
    Class for cumulative histograms.
    """

    def __init__(self, n: int):
        """
        Constructor.

        :param n: the size of this cumulative histogram.
        """
        self.sum = [0.0] * n
        self.count = [0.0] * n
        self.sum_on_mv = 0.0
        self.count_on_mv = 0.0

    def size(self) -> int:
        """
        Returns the size of this cumulative histogram.

        :return: the size of this cumulative histogram.
        """
        return len(self.sum)

    def has_missing_value(self) -> bool:
        """
        Returns True if missing values are present.

        :return: True if missing values are present.
        """
        return self.count_on_mv > 0


class Histogram2D:
    def __init__(self, n: int, m: int):
        self.resp = [[0.0 for _ in range(m)] for _ in range(n)]
        self.count = [[0.0 for _ in range(m)] for _ in range(n)]
        self.resp_on_mv1 = [0.0] * m
        self.count_on_mv1 = [0.0] * m
        self.resp_on_mv2 = [0.0] * n
        self.count_on_mv2 = [0.0] * n
        self.resp_on_mv12 = 0.0
        self.count_on_mv12 = 0.0

    class Table:
        def __init__(self, n: int, m: int):
            self.resp = [[[0.0] * 4 for _ in range(m)] for _ in range(n)]
            self.count = [[[0.0] * 4 for _ in range(m)] for _ in range(n)]
            self.resp_on_mv1 = [[0.0] * 2 for _ in range(m)]
            self.count_on_mv1 = [[0.0] * 2 for _ in range(m)]
            self.resp_on_mv2 = [[0.0] * 2 for _ in range(n)]
            self.count_on_mv2 = [[0.0] * 2 for _ in range(n)]
            self.resp_on_mv12 = 0.0
            self.count_on_mv12 = 0.0

    @staticmethod
    def compute_histogram2d(instances, f1: int, f2: int, hist2d: "Histogram2D"):
        for instance in instances:
            resp = instance.getTarget() * instance.getWeight()
            weight = instance.getWeight()
            if not instance.isMissing(f1) and not instance.isMissing(f2):
                idx1 = int(instance.getValue(f1))
                idx2 = int(instance.getValue(f2))
                hist2d.resp[idx1][idx2] += resp
                hist2d.count[idx1][idx2] += weight
            elif instance.isMissing(f1) and not instance.isMissing(f2):
                idx2 = int(instance.getValue(f2))
                hist2d.resp_on_mv1[idx2] += resp
                hist2d.count_on_mv1[idx2] += weight
            elif not instance.isMissing(f1) and instance.isMissing(f2):
                idx1 = int(instance.getValue(f1))
                hist2d.resp_on_mv2[idx1] += resp
                hist2d.count_on_mv2[idx1] += weight
            else:
                hist2d.resp_on_mv12 += resp
                hist2d.count_on_mv12 += weight

    @staticmethod
    def compute_table(hist2d: "Histogram2D", cHist1: CHistogram, cHist2: CHistogram) -> "Histogram2D.Table":
        table = Histogram2D.Table(len(hist2d.resp), len(hist2d.resp[0]))
        sum_ = 0.0
        count_ = 0.0

        # prima riga
        for j in range(len(hist2d.resp[0])):
            sum_ += hist2d.resp[0][j]
            table.resp[0][j][0] = sum_
            count_ += hist2d.count[0][j]
            table.count[0][j][0] = count_
            Histogram2D.fill_table(table, 0, j, cHist1, cHist2)

        # resto delle righe
        for i in range(1, len(hist2d.resp)):
            sum_ = 0.0
            count_ = 0.0
            for j in range(len(hist2d.resp[i])):
                sum_ += hist2d.resp[i][j]
                table.resp[i][j][0] = table.resp[i - 1][j][0] + sum_
                count_ += hist2d.count[i][j]
                table.count[i][j][0] = table.count[i - 1][j][0] + count_
                Histogram2D.fill_table(table, i, j, cHist1, cHist2)

        # marginali con missing su f1
        resp_on_mv1 = 0.0
        count_on_mv1 = 0.0
        for j in range(len(hist2d.resp_on_mv1)):
            resp_on_mv1 += hist2d.resp_on_mv1[j]
            count_on_mv1 += hist2d.count_on_mv1[j]
            table.resp_on_mv1[j][0] = resp_on_mv1
            table.resp_on_mv1[j][1] = cHist1.sum_on_mv - resp_on_mv1
            table.count_on_mv1[j][0] = count_on_mv1
            table.count_on_mv1[j][1] = cHist1.count_on_mv - count_on_mv1

        # marginali con missing su f2
        resp_on_mv2 = 0.0
        count_on_mv2 = 0.0
        for i in range(len(hist2d.resp_on_mv2)):
            resp_on_mv2 += hist2d.resp_on_mv2[i]
            count_on_mv2 += hist2d.count_on_mv2[i]
            table.resp_on_mv2[i][0] = resp_on_mv2
            table.resp_on_mv2[i][1] = cHist2.sum_on_mv - resp_on_mv2
            table.count_on_mv2[i][0] = count_on_mv2
            table.count_on_mv2[i][1] = cHist2.count_on_mv - count_on_mv2

        table.resp_on_mv12 = hist2d.resp_on_mv12
        table.count_on_mv12 = hist2d.count_on_mv12

        return table

    @staticmethod
    def fill_table(table: "Histogram2D.Table", i: int, j: int, cHist1: CHistogram, cHist2: CHistogram):
        count = table.count[i][j]
        resp = table.resp[i][j]
        resp[1] = cHist1.sum[i] - resp[0]
        resp[2] = cHist2.sum[j] - resp[0]
        resp[3] = cHist1.sum[cHist1.size() - 1] - cHist1.sum[i] - resp[2]

        count[1] = cHist1.count[i] - count[0]
        count[2] = cHist2.count[j] - count[0]
        count[3] = cHist1.count[cHist1.size() - 1] - cHist1.count[i] - count[2]

    def compute_chistogram(self) -> Tuple[CHistogram, CHistogram]:
        cHist1 = CHistogram(len(self.resp))
        cHist2 = CHistogram(len(self.resp[0]))

        # somma marginale
        for i in range(len(self.resp)):
            for j in range(len(self.resp[i])):
                cHist1.sum[i] += self.resp[i][j]
                cHist1.count[i] += self.count[i][j]
                cHist2.sum[j] += self.resp[i][j]
                cHist2.count[j] += self.count[i][j]

        # cumulative
        for i in range(1, cHist1.size()):
            cHist1.sum[i] += cHist1.sum[i - 1]
            cHist1.count[i] += cHist1.count[i - 1]

        for j in range(1, cHist2.size()):
            cHist2.sum[j] += cHist2.sum[j - 1]
            cHist2.count[j] += cHist2.count[j - 1]

        # missing values
        for j in range(len(self.resp_on_mv1)):
            cHist1.sum_on_mv += self.resp_on_mv1[j]
            cHist1.count_on_mv += self.count_on_mv1[j]
        cHist1.sum_on_mv += self.resp_on_mv12
        cHist1.count_on_mv += self.count_on_mv12

        for i in range(len(self.resp_on_mv2)):
            cHist2.sum_on_mv += self.resp_on_mv2[i]
            cHist2.count_on_mv += self.count_on_mv2[i]
        cHist2.sum_on_mv += self.resp_on_mv12
        cHist2.count_on_mv += self.count_on_mv12

        return cHist1, cHist2

# add-on AM
@dataclass
class Histogram2DNP:
    count: np.ndarray            # (s1, s2)
    resp: np.ndarray             # (s1, s2)
    count_mv1_col: np.ndarray    # (s2,)  # f1 missing, distribuzione su f2
    resp_mv1_col: np.ndarray     # (s2,)
    count_mv2_row: np.ndarray    # (s1,)  # f2 missing, distribuzione su f1
    resp_mv2_row: np.ndarray     # (s1,)
    count_mv12: float            # entrambi missing
    resp_mv12: float

# =========================
# ------- FAST.py ---------
# =========================

class FAST:

    class FASTThread(threading.Thread):

        def __init__(self, instances: Instances):
            super().__init__()
            self.pairs: List[Element] = []
            self.instances = instances

        def add(self, pair: Element) -> None:
            self.pairs.append(pair)

        def run(self) -> None:
            FAST.computeWeights_vectorized(self.instances, self.pairs)

    class Options:

        def __init__(self):
            self.attPath: Optional[str] = None     # -r
            self.datasetPath: Optional[str] = None # -d (required)
            self.residualPath: Optional[str] = None# -R (required)
            self.outputPath: Optional[str] = None  # -o (required)
            self.maxNumBins: int = 256             # -b
            self.numThreads: int = 1               # -p

    @staticmethod
    def main(args: List[str]) -> None:
        """
        Ranks pairwise interactions using FAST.

        Usage: mltk.predictor.gam.interaction.FAST
         -d  dataset path
         -R  residual path
         -o  output path
         [-r] attribute file path
         [-b] number of bins (default: 256)
         [-p] number of threads (default: 1)
        """
        opts = FAST.Options()
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("-r", dest="attPath", default=None)
        parser.add_argument("-d", dest="datasetPath", required=True)
        parser.add_argument("-R", dest="residualPath", required=True)
        parser.add_argument("-o", dest="outputPath", required=True)
        parser.add_argument("-b", dest="maxNumBins", type=int, default=256)
        parser.add_argument("-p", dest="numThreads", type=int, default=1)

        try:
            ns = parser.parse_args(args)
        except SystemExit:
            parser.print_usage()
            raise

        opts.attPath = ns.attPath
        opts.datasetPath = ns.datasetPath
        opts.residualPath = ns.residualPath
        opts.outputPath = ns.outputPath
        opts.maxNumBins = ns.maxNumBins
        opts.numThreads = ns.numThreads

        instances = InstancesReader.read(opts.attPath, opts.datasetPath)

        print("Reading residuals...")
        br = open(opts.residualPath, "r", buffering=65535)
        for i in range(instances.size()):
            line = br.readline()
            residual = float(line.strip())
            instance = instances.get(i)
            instance.setTarget(residual)
        br.close()

        attributes = instances.getAttributes()

        print("Discretizing attribute...")
        for i in range(len(attributes)):
            if attributes[i].getType() == Attribute.Type.NUMERIC:
                Discretizer.discretize(instances, i, opts.maxNumBins)

        print("Generating all pairs of attributes...")
        pairs: List[Element] = []
        for i in range(len(attributes)):
            for j in range(i + 1, len(attributes)):
                pairs.append(Element(IntPair(i, j), 0.0))

        print("Creating threads...")
        threads = [FAST.FASTThread(instances) for _ in range(opts.numThreads)]
        start = int(time.time() * 1000)
        for idx, pair in enumerate(pairs):
            threads[idx % len(threads)].add(pair)
        for t in threads:
            t.start()

        print("Running FAST...")
        for t in threads:
            t.join()
        end = int(time.time() * 1000)

        print("Sorting pairs...")
        # In Java: Collections.sort(pairs) using Element's natural ordering
        pairs.sort()  # dataclass(order=True) on Element

        print("Time: " + str((end - start) / 1000.0))

        out = open(opts.outputPath, "w")
        for i in range(len(pairs)):
            pair = pairs[i]
            out.write(f"{pair.element.v1}\t{pair.element.v2}\t{pair.weight}\n")
        out.flush()
        out.close()

    @staticmethod
    def computeWeights(instances: Instances, pairs: List[Element]) -> None:
        """
        Computes the weights of pairwise interactions.
        :param instances: the training set.
        :param pairs: the list of pairs to compute.
        """
        attributes = instances.getAttributes()
        used = [False] * len(attributes)
        for pair in pairs:
            f1 = pair.element.v1
            f2 = pair.element.v2
            used[f1] = True
            used[f2] = True

        cHist: List[Optional[CHistogram]] = [None] * len(attributes)
        for i in range(len(cHist)):
            if used[i]:
                t = attributes[i].getType()
                if t == Attribute.Type.BINNED:
                    binnedAtt: BinnedAttribute = attributes[i]  # type: ignore
                    cHist[i] = CHistogram(binnedAtt.getNumBins())
                elif t == Attribute.Type.NOMINAL:
                    nominalAtt: NominalAttribute = attributes[i]  # type: ignore
                    cHist[i] = CHistogram(nominalAtt.getCardinality())
                else:
                    # NUMERIC should have been discretized already
                    pass

        ySq = FAST.computeCHistograms_np(instances, used, cHist)  # type: ignore

        for pair in pairs:
            f1 = pair.element.v1
            f2 = pair.element.v2
            size1 = cHist[f1].size()  # type: ignore
            size2 = cHist[f2].size()  # type: ignore
            hist2d = FAST.compute_histogram2d_np(instances, f1, f2, size1, size2)
            FAST.computeWeight_np(pair, cHist, hist2d, ySq)

    @staticmethod
    def computeCHistograms(instances: Instances, used: List[bool], cHist: List[Optional[CHistogram]]) -> float:
        ySq = 0.0
        # compute histogram
        for instance in instances:
            resp = instance.getTarget()
            for j in range(len(instances.getAttributes())):
                if used[j]:
                    if not instance.isMissing(j):
                        idx = int(instance.getValue(j))
                        cHist[j].sum[idx] += resp * instance.getWeight()   # type: ignore
                        cHist[j].count[idx] += instance.getWeight()         # type: ignore
                    else:
                        cHist[j].sum_on_mv   += resp * instance.getWeight()     # type: ignore
                        cHist[j].count_on_mv += instance.getWeight()          # type: ignore
            ySq += resp * resp * instance.getWeight()
        # compute cumulative histogram
        for j in range(len(cHist)):
            if used[j]:
                for idx in range(1, cHist[j].size()):
                    cHist[j].sum[idx]   += cHist[j].sum[idx - 1]
                    cHist[j].count[idx] += cHist[j].count[idx - 1]         # type: ignore
        return ySq

    @staticmethod
    def computeCHistograms_np(instances, used, cHist) -> float:
        Instances.ensure_numpy_cache(instances)
        B = instances._np_bins
        y = instances._np_y
        w = instances._np_w
        ySq = float(np.sum((y*w)*y))

        for j, need in enumerate(used):
            if not need: 
                continue
            size = cHist[j].size()
            v = B[:, j]
            m = v >= 0
            idx = v[m].astype(np.int64)
            cw = np.bincount(idx, weights=w[m], minlength=size).astype(float)
            rw = np.bincount(idx, weights=(y[m]*w[m]), minlength=size).astype(float)

            # cumulative
            cHist[j].count = np.cumsum(cw).tolist()
            cHist[j].sum   = np.cumsum(rw).tolist()

            # missing
            mm = ~m
            cHist[j].count_on_mv = float(w[mm].sum())
            cHist[j].sum_on_mv   = float((y[mm]*w[mm]).sum())
        return ySq

    @staticmethod
    def computeWeight(pair: Element, cHist: List[Optional[CHistogram]], hist2d: Histogram2D, ySq: float) -> None:
        f1 = pair.element.v1
        f2 = pair.element.v2
        size1 = cHist[f1].size()  # type: ignore
        size2 = cHist[f2].size()  # type: ignore
        table = Histogram2D.compute_table(hist2d, cHist[f1], cHist[f2])  # type: ignore
        bestRSS = float("inf")
        predInt = [0.0] * 4
        predOnMV1 = [0.0] * 2
        predOnMV2 = [0.0] * 2
        predOnMV12 = MathUtils.divide(hist2d.resp_on_mv12, hist2d.count_on_mv12, 0.0)
        for v1 in range(0, size1 - 1):
            for v2 in range(0, size2 - 1):
                FAST.getPredictor(table, v1, v2, predInt, predOnMV1, predOnMV2)
                rss = FAST.getRSS(table, v1, v2, ySq, predInt, predOnMV1, predOnMV2, predOnMV12)
                if rss < bestRSS:
                    bestRSS = rss
        pair.weight = bestRSS

    @staticmethod
    def getPredictor(table: Any, v1: int, v2: int,
                     pred: List[float], predOnMV1: List[float], predOnMV2: List[float]) -> None:
        count = table.count[v1][v2]
        resp = table.resp[v1][v2]
        for i in range(len(pred)):
            pred[i] = MathUtils.divide(resp[i], count[i], 0.0)
        for i in range(len(predOnMV1)):
            predOnMV1[i] = MathUtils.divide(table.resp_on_mv1[v2][i], table.count_on_mv1[v2][i], 0.0)
        for i in range(len(predOnMV2)):
            predOnMV2[i] = MathUtils.divide(table.resp_on_mv2[v1][i], table.count_on_mv2[v1][i], 0.0)

    @staticmethod
    def getRSS(table: Any, v1: int, v2: int, ySq: float,
               pred: List[float], predOnMV1: List[float], predOnMV2: List[float], predOnMV12: float) -> float:
        count = table.count[v1][v2]
        resp = table.resp[v1][v2]
        respOnMV1 = table.resp_on_mv1[v2]
        countOnMV1 = table.count_on_mv1[v2]
        respOnMV2 = table.resp_on_mv2[v1]
        countOnMV2 = table.count_on_mv2[v1]
        rss = ySq
        # Compute main area
        t = 0.0
        for i in range(len(pred)):
            t += pred[i] * pred[i] * count[i]
        rss += t
        t = 0.0
        for i in range(len(pred)):
            t += pred[i] * resp[i]
        rss -= 2 * t
        # Compute on mv1
        t = 0.0
        for i in range(len(predOnMV1)):
            t += predOnMV1[i] * predOnMV1[i] * countOnMV1[i]
        rss += t
        t = 0.0
        for i in range(len(predOnMV1)):
            t += predOnMV1[i] * respOnMV1[i]
        rss -= 2 * t
        # Compute on mv2
        t = 0.0
        for i in range(len(predOnMV2)):
            t += predOnMV2[i] * predOnMV2[i] * countOnMV2[i]
        rss += t
        t = 0.0
        for i in range(len(predOnMV2)):
            t += predOnMV2[i] * respOnMV2[i]
        rss -= 2 * t
        return rss

    # ADD rispetto a FAST originale
    def computeWeight_np(pair, cHist, hist: Histogram2DNP, ySq: float) -> float:
        # dimensioni
        C = hist.count
        R = hist.resp
        s1, s2 = C.shape
        if s1 < 2 or s2 < 2:
            pair.weight = ySq  # nessuna soglia possibile
            return pair.weight

        # prefix 2D
        Cc = C.cumsum(0).cumsum(1)
        Rc = R.cumsum(0).cumsum(1)
        totalC = Cc[-1, -1]
        totalR = Rc[-1, -1]

        # tutte le soglie (v1 in [0..s1-2], v2 in [0..s2-2])
        C00 = Cc[:-1, :-1]
        R00 = Rc[:-1, :-1]
        C0_ = Cc[:-1, -1][:, None]
        R0_ = Rc[:-1, -1][:, None]
        C_0 = Cc[-1, :-1][None, :]
        R_0 = Rc[-1, :-1][None, :]

        # quattro quadranti
        C01 = C0_ - C00
        R01 = R0_ - R00
        C10 = C_0 - C00
        R10 = R_0 - R00
        C11 = totalC - C0_ - C_0 + C00
        R11 = totalR - R0_ - R_0 + R00

        C4 = np.stack([C00, C01, C10, C11], axis=-1)
        R4 = np.stack([R00, R01, R10, R11], axis=-1)
        pred4 = np.divide(R4, C4, out=np.zeros_like(R4), where=(C4 > 0))

        rss_main = (pred4**2 * C4).sum(axis=-1) - 2.0 * (pred4 * R4).sum(axis=-1)

        # missing su f1: dipende SOLO da v2 → broadcast su righe
        if hist.count_mv1_col.sum() > 0:
            cumC = np.cumsum(hist.count_mv1_col)
            cumR = np.cumsum(hist.resp_mv1_col)
            # soglie v2 in [0..s2-2] = lato sinistro incluso
            LC = cumC[:-1][None, :]  # (1, s2-1)
            LR = cumR[:-1][None, :]
            RC = cumC[-1] - LC
            RR = cumR[-1] - LR
            LP = np.divide(LR, LC, out=np.zeros_like(LR), where=(LC > 0))
            RP = np.divide(RR, RC, out=np.zeros_like(RC), where=(RC > 0))
            term_mv1 = (LP*LP*LC - 2.0*LP*LR) + (RP*RP*RC - 2.0*RP*RR)
            term_mv1 = np.broadcast_to(term_mv1, rss_main.shape)
        else:
            term_mv1 = 0.0

        # missing su f2: dipende SOLO da v1 → broadcast su colonne
        if hist.count_mv2_row.sum() > 0:
            cumC = np.cumsum(hist.count_mv2_row)
            cumR = np.cumsum(hist.resp_mv2_row)
            TC = cumC[:-1][:, None]  # (s1-1, 1)
            TR = cumR[:-1][:, None]
            BC = cumC[-1] - TC
            BR = cumR[-1] - TR
            TP = np.divide(TR, TC, out=np.zeros_like(TR), where=(TC > 0))
            BP = np.divide(BR, BC, out=np.zeros_like(BC), where=(BC > 0))
            term_mv2 = (TP*TP*TC - 2.0*TP*TR) + (BP*BP*BC - 2.0*BP*BR)
            term_mv2 = np.broadcast_to(term_mv2, rss_main.shape)
        else:
            term_mv2 = 0.0

        # entrambi missing: costante
        if hist.count_mv12 > 0:
            p12 = hist.resp_mv12 / hist.count_mv12
            term_mv12 = p12*p12*hist.count_mv12 - 2.0*p12*hist.resp_mv12
        else:
            term_mv12 = 0.0

        rss_grid = ySq + rss_main + term_mv1 + term_mv2 + term_mv12
        bestRSS = float(rss_grid.min())
        pair.weight = bestRSS
        return bestRSS
    
    def computeWeights_vectorized(instances, pairs):
        attributes = instances.getAttributes()
        used = [False] * len(attributes)
        for pair in pairs:
            used[pair.element.v1] = True
            used[pair.element.v2] = True

        cHist = [None] * len(attributes)
        for i, att in enumerate(attributes):
            if used[i]:
                t = att.getType()
                if t == Attribute.Type.BINNED:
                    cHist[i] = CHistogram(att.getNumBins())
                elif t == Attribute.Type.NOMINAL:
                    cHist[i] = CHistogram(att.getCardinality())

        ySq = FAST.computeCHistograms_np(instances, used, cHist)
        Instances.ensure_numpy_cache(instances)
        for pair in pairs:
            f1, f2 = pair.element.v1, pair.element.v2
            s1, s2 = cHist[f1].size(), cHist[f2].size()
            hist2d = FAST.compute_histogram2d_np(instances, f1, f2, s1, s2)
            FAST.computeWeight_np(pair, cHist, hist2d, ySq)

    @staticmethod
    def compute_histogram2d_np(instances, f1: int, f2: int, size1: int, size2: int) -> Histogram2DNP:
        B = instances._np_bins
        y = instances._np_y
        w = instances._np_w

        v1 = B[:, f1]
        v2 = B[:, f2]

        mask = (v1 >= 0) & (v2 >= 0)
        flat = (v1[mask].astype(np.int64) * size2 + v2[mask].astype(np.int64))
        count = np.bincount(flat, weights=w[mask], minlength=size1*size2).reshape(size1, size2).astype(np.float64)
        resp  = np.bincount(flat, weights=y[mask]*w[mask], minlength=size1*size2).reshape(size1, size2).astype(np.float64)

        # f1 missing, f2 osservato → colonna
        mv1 = (v1 < 0) & (v2 >= 0)
        count_mv1_col = np.bincount(v2[mv1], weights=w[mv1], minlength=size2).astype(np.float32)
        resp_mv1_col  = np.bincount(v2[mv1], weights=y[mv1]*w[mv1], minlength=size2).astype(np.float32)

        # f2 missing, f1 osservato → riga
        mv2 = (v2 < 0) & (v1 >= 0)
        count_mv2_row = np.bincount(v1[mv2], weights=w[mv2], minlength=size1).astype(np.float32)
        resp_mv2_row  = np.bincount(v1[mv2], weights=y[mv2]*w[mv2], minlength=size1).astype(np.float32)

        # entrambi missing
        mv12 = (v1 < 0) & (v2 < 0)
        count_mv12 = float(w[mv12].sum())
        resp_mv12  = float((y[mv12]*w[mv12]).sum())

        return Histogram2DNP(count, resp, count_mv1_col, resp_mv1_col,
                            count_mv2_row, resp_mv2_row, count_mv12, resp_mv12)

    @staticmethod
    def _normalize_cat_prefix(name: str) -> str:
        """Normalizza i prefissi cat_: 'cat__' -> 'cat_'."""
        return "cat_" + name[len("cat__"):] if name.startswith("cat__") else name

    @staticmethod
    def _extract_feature_names(X, feature_names):
        # 1) espliciti
        if feature_names is not None:
            return list(feature_names)
        # 2) pandas DataFrame
        if isinstance(X, pd.DataFrame):
            return list(X.columns)
        # 3) oggetti sklearn che espongono feature_names_in_
        if hasattr(X, "feature_names_in_"):
            return list(getattr(X, "feature_names_in_"))
        # 4) fallback anonimo
        return [f"f{i}" for i in range(X.shape[1])]
        
    @staticmethod
    def _cat_core(name: str) -> str:
        """Parte dopo 'cat_' una volta normalizzato."""
        s = FAST._normalize_cat_prefix(name)
        return s[len("cat_"):] if s.startswith("cat_") else s

    @staticmethod
    def _family(name: str) -> str:
        """
        Family per skip_same_family:
        - per OHE non collassate: 'cat_<col>' (es: 'cat_sex' da 'cat_sex_Male')
        - per collassate: il nome stesso ('cat_<col>')
        - per non-cat: il nome intero (ognuna la propria famiglia)
        """
        n = FAST._normalize_cat_prefix(name)
        if n.startswith("cat_"):
            core = FAST._cat_core(n)
            # se già collassata, core NON contiene '_'; se OHE, contiene '_'
            base = core.split("_", 1)[0] if "_" in core else core
            return f"cat_{base}"
        return n

    # -------- inferenza famiglie OHE col prefisso più lungo --------

    @staticmethod
    def _infer_families_by_longest_prefix(feature_names: List[str],
                                          only_prefix_cat: bool = True) -> Dict[str, List[int]]:
        """
        Raggruppa le feature OHE per 'famiglia' definita come il prefisso più lungo
        'cat_<prefix>' condiviso da >=2 feature. Evita fusioni eccessive (es. 'cat_c*').
        Le non-cat restano singleton.
        """
        cat_names: List[str] = []
        cat_indices: List[int] = []
        for j, nm in enumerate(feature_names):
            nn = FAST._normalize_cat_prefix(nm)
            if nn.startswith("cat_"):
                cat_indices.append(j)
                cat_names.append(nn)

        counts: Dict[str, int] = {}
        candidates_of: Dict[str, List[str]] = {}

        for nm in cat_names:
            core = FAST._cat_core(nm)
            cands: List[str] = []
            # trova tutte le posizioni '_' nella core
            for i, ch in enumerate(core):
                if ch == "_":
                    cand = "cat_" + core[:i]  # prefisso fino a quel '_'
                    if cand: 
                        cands.append(cand)
            # caso limite: core senza '_' (es. 'sex_Male') -> prefisso 'cat_sex'
            if "_" not in core and core:
                cands.append("cat_" + core)

            candidates_of[nm] = cands
            for c in cands:
                counts[c] = counts.get(c, 0) + 1

        # scegli, per ogni cat_name, il candidato più lungo con count>=2
        family_for_cat: Dict[str, str] = {}
        for nm in cat_names:
            cands = [c for c in candidates_of.get(nm, []) if counts.get(c, 0) >= 2]
            family_for_cat[nm] = (max(cands, key=len) if cands else f"__solo_{nm}")

        # costruisci gruppi (ordine di inserimento = ordine in feature_names)
        groups: Dict[str, List[int]] = {}
        for j, nm in enumerate(feature_names):
            nn = FAST._normalize_cat_prefix(nm)
            if only_prefix_cat and not nn.startswith("cat_"):
                key = f"__solo_{nm}"
            else:
                key = family_for_cat.get(nn, f"__solo_{nn}") if nn.startswith("cat_") else f"__solo_{nn}"
            groups.setdefault(key, []).append(j)
        return groups

    # ----------------- COLLASSO ONE-HOT -----------------

    @staticmethod
    def collapse_onehot_families(
        X: np.ndarray | pd.DataFrame,
        feature_names: List[str],
        family_fn: Callable[[str], str] | None = None,
        strict_onehot: bool = True,
        missing_value: int = -1,
        only_prefix_cat: bool = True,
    ) -> Tuple[np.ndarray, List[str], Dict]:
        """
        Collassa gruppi one-hot (stessa family) in UNA sola feature nominale con valori {0..K-1} e -1=missing.
        Ritorna: (X_new, names_new, meta) dove meta['collapsed'] mappa nuova_col -> {family, indices, categories}
        """
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        n, p = X_arr.shape

        # gruppi
        if family_fn is not None:
            groups: Dict[str, List[int]] = {}
            for j, name in enumerate(feature_names):
                nm = FAST._normalize_cat_prefix(name)
                key = family_fn(nm)
                if only_prefix_cat and not nm.startswith("cat_"):
                    key = f"__solo_{name}"
                groups.setdefault(key, []).append(j)
        else:
            groups = FAST._infer_families_by_longest_prefix(feature_names, only_prefix_cat=only_prefix_cat)

        keep_cols: List[np.ndarray] = []
        keep_names: List[str] = []
        meta: Dict[str, Dict] = {
            "collapsed": {},
            "kept": {},
            "summary": {
                "n_samples": n,
                "n_features_in": p,
                "n_groups": len(groups),
                "n_groups_collapsed": 0,
                "n_groups_kept": 0,
            },
        }

        for fam_key, idxs in groups.items():
            is_cat_group = fam_key.startswith("cat_")
            if (only_prefix_cat and not is_cat_group) or (len(idxs) == 1):
                for j in idxs:
                    keep_cols.append(X_arr[:, [j]])
                    keep_names.append(feature_names[j])
                    meta["kept"][feature_names[j]] = {"family": fam_key, "origin": [j]}
                meta["summary"]["n_groups_kept"] += 1
                continue

            cols = X_arr[:, idxs]
            uniq = np.unique(cols)
            if not np.all(np.isin(uniq, [0.0, 1.0])):
                # non è gruppo OHE pulito
                for j in idxs:
                    keep_cols.append(X_arr[:, [j]])
                    keep_names.append(feature_names[j])
                    meta["kept"][feature_names[j]] = {"family": fam_key, "origin": [j]}
                meta["summary"]["n_groups_kept"] += 1
                continue

            B = cols.astype(np.int8)  # 0/1
            s = B.sum(axis=1)

            if strict_onehot and np.any(s > 1):
                # multi-label -> non collassare
                for j in idxs:
                    keep_cols.append(X_arr[:, [j]])
                    keep_names.append(feature_names[j])
                    meta["kept"][feature_names[j]] = {"family": fam_key, "origin": [j]}
                meta["summary"]["n_groups_kept"] += 1
                continue

            # collassa: -1 = nessun 1; altrimenti indice della dummy attiva
            z = np.full(n, missing_value, dtype=np.int32)
            mask1 = (s == 1)
            if mask1.any():
                z[mask1] = np.argmax(B[mask1], axis=1).astype(np.int32)

            keep_cols.append(z.reshape(-1, 1).astype(float))
            new_name = fam_key  # 'cat_<prefix>'
            keep_names.append(new_name)
            meta["collapsed"][new_name] = {
                "family": fam_key,
                "origin": idxs,
                "categories": [feature_names[j] for j in idxs],
                "missing_value": missing_value,
            }
            meta["summary"]["n_groups_collapsed"] += 1

        X_new = np.hstack(keep_cols) if keep_cols else X_arr.copy()
        return X_new, keep_names, meta

    # ----------------- RUN -----------------

    @staticmethod
    def run(X: np.ndarray, residuals: np.ndarray,
            feature_names: List[str] | None = None,
            bins: int = 32, num_threads: int = 1,
            skip_same_family: bool = True,
            min_support: int = 50,
            max_cells: int = 1024,
            screen_top_m: int | None = 10,
            collapse_families: bool = True,
            strict_onehot: bool = True):
        num_threads = max(1, num_threads if num_threads is not None else -1)
        residuals = np.asarray(residuals, dtype=float)
        fn_raw = FAST._extract_feature_names(X, feature_names)
        
        # (A) collassa famiglie OHE (solo cat_)
        if collapse_families:
            Xc, fnc, meta = FAST.collapse_onehot_families(
                X, fn_raw, strict_onehot=strict_onehot,
                missing_value=-1, only_prefix_cat=True
            )
            X_use, fn = Xc, fnc
        else:
            X_use, fn = X, fn_raw

        n, p = X_use.shape
        feat = np.arange(p)
        fam = np.array([FAST._family(nm) for nm in fn])

        # (B) screening univariato
        if screen_top_m is not None and screen_top_m < p:
            r = residuals.reshape(-1, 1)
            Xc = X_use - X_use.mean(axis=0, keepdims=True)
            num = np.abs((Xc * r).sum(axis=0))
            den = np.sqrt((Xc**2).sum(axis=0) * (r.ravel()**2).sum())
            uni = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
            keep = np.argpartition(uni, -screen_top_m)[-screen_top_m:]
            keep.sort()
        else:
            keep = feat

        # (C) Instances + target (sostituisci con le tue classi)
        instances = Instances.from_numpy(X_use[:, keep], [fn[i] for i in keep])
        for i in range(instances.size()):
            instances.get(i).setTarget(float(residuals[i]))

        # (D) discretizza solo i numerici; le 'cat_*' collassate restano nominali
        attributes = instances.getAttributes()
        for j, name in enumerate([fn[i] for i in keep]):   # nomi effettivi passati a from_numpy
            if name.startswith("cat_"):
                # cardinalità = numero di valori distinti non-negativi
                vals = [int(instances.get(i).getValue(j)) for i in range(instances.size())]
                k = len({v for v in vals if v >= 0})
                attributes[j] = NominalAttribute(name, [str(t) for t in range(k)])
        instances.setAttributes(attributes)
        for j, att in enumerate(attributes):
            if att.getType() == Attribute.Type.NUMERIC:
                Discretizer.discretize(instances, j, bins)

        # (E) cardinalità
        def att_size(j):
            t = attributes[j].getType()
            if t == Attribute.Type.BINNED:
                return attributes[j].getNumBins()
            elif t == Attribute.Type.NOMINAL:
                return attributes[j].getCardinality()
            else:
                return 1

        sizes = [att_size(j) for j in range(len(attributes))]

        # (F) genera coppie evitando stessa family
        pairs: List[Element] = []
        dropped_same_family = dropped_cells = 0
        for a in range(len(attributes)):
            i = keep[a]
            for b in range(a + 1, len(attributes)):
                j = keep[b]
                if skip_same_family and fam[i] == fam[j]:
                    dropped_same_family += 1
                    continue
                if sizes[a] * sizes[b] > max_cells:
                    dropped_cells += 1
                    continue
                pairs.append(Element(IntPair(a, b), 0.0))

        # (G) threading
        threads = [FAST.FASTThread(instances) for _ in range(num_threads)]
        for idx, pair in enumerate(pairs):
            threads[idx % len(threads)].add(pair)
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        pairs.sort()
        original_pairs = [(keep[p.element.v1], keep[p.element.v2]) for p in pairs]
        return original_pairs, [p.weight for p in pairs]

if __name__ == "__main__":
    import sys
    FAST.main(sys.argv[1:])
