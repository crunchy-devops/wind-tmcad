"""Tests for point cloud functionality."""
import pytest
from point_cloud import PointCloud
from point3d import Point3d
import os
import h5py

@pytest.fixture
def cloud():
    """Create a point cloud with sample points."""
    pc = PointCloud()
    p1 = Point3d(id=1, x=0.0, y=0.0, z=0.0)
    p2 = Point3d(id=2, x=3.0, y=4.0, z=0.0)
    p3 = Point3d(id=3, x=1.0, y=1.0, z=1.0)
    pc.add_point(p1)
    pc.add_point(p2)
    pc.add_point(p3)
    yield pc
    # Cleanup (if needed)
    pc = None

def test_add_point(cloud):
    """Test adding points."""
    p4 = Point3d(id=4, x=2.0, y=2.0, z=2.0)
    cloud.add_point(p4)
    assert len(cloud.points) == 4
    assert cloud.get_point(4) == p4

def test_add_duplicate_point(cloud):
    """Test adding point with duplicate ID."""
    p_dup = Point3d(id=1, x=5.0, y=5.0, z=5.0)
    with pytest.raises(ValueError):
        cloud.add_point(p_dup)

def test_remove_point(cloud):
    """Test removing points."""
    cloud.remove_point(1)
    assert len(cloud.points) == 2
    with pytest.raises(KeyError):
        cloud.get_point(1)

def test_distance(cloud):
    """Test distance calculation."""
    # Distance between (0,0,0) and (3,4,0) should be 5
    assert cloud.distance(1, 2) == 5.0

def test_slope_percentage(cloud):
    """Test slope percentage calculation."""
    # Points with 3,4,0 and 0,0,0 coordinates
    slope = cloud.slope_percentage(1, 2)
    # No vertical difference, so slope should be 0
    assert slope == 0.0

def test_bearing_angle(cloud):
    """Test bearing angle calculation."""
    # Point 2 is at (3,4,0) relative to (0,0,0)
    angle = cloud.bearing_angle(1, 2)
    # arctan(4/3) in degrees ≈ 53.13
    assert round(angle, 2) == 53.13

def test_nearest_neighbors(cloud):
    """Test finding nearest neighbors."""
    neighbors = cloud.find_nearest_neighbors(1, k=2)
    # Point 3 should be closer than Point 2
    assert len(neighbors) == 2
    assert neighbors[0][0] == 3  # First neighbor should be point 3
    assert neighbors[1][0] == 2  # Second neighbor should be point 2

def test_interpolate_z_idw(cloud):
    """Test IDW interpolation."""
    # Test point at (0.5, 0.5)
    z = cloud.interpolate_z_idw(0.5, 0.5)
    # Should be weighted average of surrounding points
    assert isinstance(z, float)

def test_interpolate_z_delaunay(cloud):
    """Test Delaunay interpolation."""
    # Add more points to create a valid triangulation
    p4 = Point3d(id=4, x=0.0, y=1.0, z=0.0)
    cloud.add_point(p4)
    # Compute triangulation
    cloud.compute_delaunay()
    # Test point at (0.5, 0.5)
    z = cloud.interpolate_z_delaunay(0.5, 0.5)
    # Should be interpolated from triangulation
    assert isinstance(z, float)

def test_hdf5_storage(cloud):
    """Test HDF5 file storage."""
    filename = 'test_cloud.h5'
    try:
        # Save to HDF5
        print(f"Original points: {cloud.points}")  # Debug print
        cloud.save_to_hdf5(filename)

        # Verify file exists and has content
        assert os.path.exists(filename), "HDF5 file was not created"
        print(f"File size: {os.path.getsize(filename)} bytes")  # Debug print

        # Inspect HDF5 file
        with h5py.File(filename, 'r') as f:
            print(f"File keys: {list(f.keys())}")
            points_data = f['points'][:]
            print(f"Points data shape: {points_data.shape}")
            print(f"Points data:")
            for point in points_data:
                print(f"Point: id={int(point[0])}, x={point[1]}, y={point[2]}, z={point[3]}")

        # Create new cloud and load
        new_cloud = PointCloud()
        try:
            new_cloud.load_from_hdf5(filename)
        except Exception as e:
            print(f"Error loading HDF5: {str(e)}")
            raise
        print(f"Loaded points: {new_cloud.points}")  # Debug print

        # Compare points
        assert len(new_cloud.points) == len(cloud.points)
        for point_id, point in cloud.points.items():
            assert point_id in new_cloud.points
            new_point = new_cloud.points[point_id]
            assert point.x == new_point.x
            assert point.y == new_point.y
            assert point.z == new_point.z

    finally:
        if os.path.exists(filename):
            os.remove(filename)
