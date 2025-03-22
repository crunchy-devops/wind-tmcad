"""Benchmark module for Point3d class."""
import timeit
import random
from point3d import Point3d

def benchmark_creation(n=1000):
    """Benchmark point creation."""
    setup = """
from point3d import Point3d
import random
"""
    stmt = f"""
for i in range({n}):
    Point3d(id=i, x=random.random(), y=random.random(), z=random.random())
"""
    return timeit.timeit(stmt, setup=setup, number=100)

def benchmark_packing(n=1000):
    """Benchmark point packing/unpacking."""
    setup = """
from point3d import Point3d
import random
points = [Point3d(id=i, x=random.random(), y=random.random(), z=random.random()) 
         for i in range(1000)]
"""
    pack_stmt = """
for p in points:
    packed = p.pack()
"""
    unpack_stmt = """
for p in points:
    packed = p.pack()
    Point3d.unpack(packed)
"""
    
    pack_time = timeit.timeit(pack_stmt, setup=setup, number=100)
    pack_unpack_time = timeit.timeit(unpack_stmt, setup=setup, number=100)
    
    return pack_time, pack_unpack_time

if __name__ == '__main__':
    # Run benchmarks
    creation_time = benchmark_creation()
    pack_time, pack_unpack_time = benchmark_packing()
    
    print(f"Creation time (1000 points, 100 iterations): {creation_time:.4f} seconds")
    print(f"Pack time (1000 points, 100 iterations): {pack_time:.4f} seconds")
    print(f"Pack+Unpack time (1000 points, 100 iterations): {pack_unpack_time:.4f} seconds")
