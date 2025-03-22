# Point3D - Memory-Efficient 3D Point Implementation

A high-performance, memory-efficient implementation of a 3D point class in Python using `struct` for binary packing and `__slots__` for memory optimization.

## Features

- Memory-efficient implementation using `__slots__` and `struct`
- Immutable points (frozen dataclass)
- Binary packing/unpacking support
- Type hints and comprehensive docstrings
- 100% test coverage
- Performance benchmarking suite

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from point3d import Point3d

# Create a point
point = Point3d(id=1, x=1.0, y=2.0, z=3.0)

# Pack to binary format
binary_data = point.pack()

# Unpack from binary format
restored_point = Point3d.unpack(binary_data)
```

## Memory Optimization

The implementation uses several techniques to minimize memory usage:
- `__slots__`: Prevents dynamic attribute creation
- `struct`: Efficiently packs coordinates into binary format
- `frozen=True`: Makes instances immutable
- Appropriate numeric types (uint64 for id, float32 for coordinates)

## Testing

Run the test suite with coverage:

```bash
python -m pytest test_point3d.py --cov=point3d
```

## Benchmarking

Run the performance benchmarks:

```bash
python benchmark_point3d.py
```

The benchmark measures:
1. Point creation performance
2. Binary packing performance
3. Combined pack/unpack operations

## Project Structure

- `point3d.py`: Core Point3D implementation
- `test_point3d.py`: Test suite with 100% coverage
- `benchmark_point3d.py`: Performance benchmarking
- `requirements.txt`: Project dependencies
