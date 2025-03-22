"""Benchmark module for PointCloud class."""
import timeit
import random
import tempfile
import os
from point3d import Point3d
from point_cloud import PointCloud

def create_random_cloud(size: int) -> PointCloud:
    """Create a point cloud with random points."""
    cloud = PointCloud()
    for i in range(size):
        point = Point3d(id=i,
                       x=random.uniform(-1000, 1000),
                       y=random.uniform(-1000, 1000),
                       z=random.uniform(-1000, 1000))
        cloud.add_point(point)
    return cloud

def benchmark_creation(n=1000):
    """Benchmark point cloud creation."""
    setup = """
from point3d import Point3d
from point_cloud import PointCloud
import random

def create_cloud(size):
    cloud = PointCloud()
    for i in range(size):
        point = Point3d(id=i,
                       x=random.uniform(-1000, 1000),
                       y=random.uniform(-1000, 1000),
                       z=random.uniform(-1000, 1000))
        cloud.add_point(point)
    return cloud
"""
    stmt = f"cloud = create_cloud({n})"
    return timeit.timeit(stmt, setup=setup, number=10)

def benchmark_spatial_operations(n=1000):
    """Benchmark spatial operations."""
    setup = f"""
from point3d import Point3d
from point_cloud import PointCloud
import random

cloud = PointCloud()
for i in range({n}):
    point = Point3d(id=i,
                   x=random.uniform(-1000, 1000),
                   y=random.uniform(-1000, 1000),
                   z=random.uniform(-1000, 1000))
    cloud.add_point(point)
    
point_ids = list(cloud.points.keys())
test_id = random.choice(point_ids)
k = 10
"""
    
    nn_stmt = "cloud.find_nearest_neighbors(test_id, k)"
    dist_stmt = """
id1, id2 = random.sample(point_ids, 2)
cloud.distance(id1, id2)
"""
    slope_stmt = """
id1, id2 = random.sample(point_ids, 2)
try:
    cloud.slope_percentage(id1, id2)
except ZeroDivisionError:
    pass
"""
    bearing_stmt = """
id1, id2 = random.sample(point_ids, 2)
cloud.bearing_angle(id1, id2)
"""
    
    nn_time = timeit.timeit(nn_stmt, setup=setup, number=100)
    dist_time = timeit.timeit(dist_stmt, setup=setup, number=100)
    slope_time = timeit.timeit(slope_stmt, setup=setup, number=100)
    bearing_time = timeit.timeit(bearing_stmt, setup=setup, number=100)
    
    return nn_time, dist_time, slope_time, bearing_time

def benchmark_storage(n=1000):
    """Benchmark HDF5 storage operations."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tf:
        filename = tf.name.replace('\\', '/')  # Convert Windows path to forward slashes
    
    setup = rf"""
from point3d import Point3d
from point_cloud import PointCloud
import random

cloud = PointCloud()
for i in range({n}):
    point = Point3d(id=i,
                   x=random.uniform(-1000, 1000),
                   y=random.uniform(-1000, 1000),
                   z=random.uniform(-1000, 1000))
    cloud.add_point(point)

filename = "{filename}"
"""
    
    save_stmt = "cloud.save_to_hdf5(filename)"
    load_stmt = "loaded_cloud = PointCloud.load_from_hdf5(filename)"
    
    try:
        save_time = timeit.timeit(save_stmt, setup=setup, number=10)
        load_time = timeit.timeit(load_stmt, setup=setup, number=10)
    finally:
        os.unlink(filename)
    
    return save_time, load_time

if __name__ == '__main__':
    n_points = 10000
    
    print(f"\nBenchmarking with {n_points} points:")
    print("-" * 40)
    
    creation_time = benchmark_creation(n_points)
    print(f"Cloud creation (10 iterations): {creation_time:.4f} seconds")
    
    nn_time, dist_time, slope_time, bearing_time = benchmark_spatial_operations(n_points)
    print(f"\nSpatial operations (100 iterations each):")
    print(f"Nearest neighbors search: {nn_time:.4f} seconds")
    print(f"Distance calculation: {dist_time:.4f} seconds")
    print(f"Slope calculation: {slope_time:.4f} seconds")
    print(f"Bearing calculation: {bearing_time:.4f} seconds")
    
    save_time, load_time = benchmark_storage(n_points)
    print(f"\nStorage operations (10 iterations each):")
    print(f"HDF5 save: {save_time:.4f} seconds")
    print(f"HDF5 load: {load_time:.4f} seconds")
