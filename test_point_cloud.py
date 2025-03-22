"""Test module for PointCloud class."""
import unittest
import os
import math
import tempfile
from point3d import Point3d
from point_cloud import PointCloud

class TestPointCloud(unittest.TestCase):
    """Test cases for PointCloud class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cloud = PointCloud()
        self.p1 = Point3d(id=1, x=0.0, y=0.0, z=0.0)
        self.p2 = Point3d(id=2, x=3.0, y=4.0, z=0.0)
        self.p3 = Point3d(id=3, x=1.0, y=1.0, z=1.0)
        self.cloud.add_point(self.p1)
        self.cloud.add_point(self.p2)
        self.cloud.add_point(self.p3)
        
    def test_add_point(self):
        """Test adding points."""
        p4 = Point3d(id=4, x=2.0, y=2.0, z=2.0)
        self.cloud.add_point(p4)
        self.assertEqual(len(self.cloud.points), 4)
        self.assertEqual(self.cloud.get_point(4), p4)
        
    def test_add_duplicate_point(self):
        """Test adding point with duplicate ID."""
        p_dup = Point3d(id=1, x=5.0, y=5.0, z=5.0)
        with self.assertRaises(ValueError):
            self.cloud.add_point(p_dup)
            
    def test_remove_point(self):
        """Test removing points."""
        self.cloud.remove_point(1)
        self.assertEqual(len(self.cloud.points), 2)
        with self.assertRaises(KeyError):
            self.cloud.get_point(1)
            
    def test_distance(self):
        """Test distance calculation."""
        # Distance between (0,0,0) and (3,4,0) should be 5
        self.assertEqual(self.cloud.distance(1, 2), 5.0)
        
    def test_slope_percentage(self):
        """Test slope percentage calculation."""
        # Points with 3,4,0 and 0,0,0 coordinates
        slope = self.cloud.slope_percentage(1, 2)
        self.assertEqual(slope, 0.0)  # No vertical difference
        
        # Test with vertical difference
        p4 = Point3d(id=4, x=3.0, y=4.0, z=5.0)
        self.cloud.add_point(p4)
        slope = self.cloud.slope_percentage(1, 4)
        self.assertEqual(slope, 100.0)  # 45 degree angle = 100% slope
        
    def test_bearing_angle(self):
        """Test bearing angle calculation."""
        # Point 2 is at (3,4,0) relative to (0,0,0)
        angle = self.cloud.bearing_angle(1, 2)
        expected = math.degrees(math.atan2(4, 3))
        self.assertAlmostEqual(angle, (expected + 360) % 360)
        
    def test_nearest_neighbors(self):
        """Test nearest neighbor search."""
        neighbors = self.cloud.find_nearest_neighbors(1, 2)
        self.assertEqual(len(neighbors), 2)
        # Point 3 should be closer to point 1 than point 2
        self.assertEqual(neighbors[0][0], 3)
        self.assertEqual(neighbors[1][0], 2)
        
    def test_hdf5_storage(self):
        """Test HDF5 storage and loading."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tf:
            filename = tf.name
            
        try:
            # Save to HDF5
            self.cloud.save_to_hdf5(filename)
            
            # Load from HDF5
            loaded_cloud = PointCloud.load_from_hdf5(filename)
            
            # Verify points are the same
            self.assertEqual(len(loaded_cloud.points), len(self.cloud.points))
            for point_id, point in self.cloud.points.items():
                loaded_point = loaded_cloud.get_point(point_id)
                self.assertEqual(point.id, loaded_point.id)
                self.assertEqual(point.x, loaded_point.x)
                self.assertEqual(point.y, loaded_point.y)
                self.assertEqual(point.z, loaded_point.z)
        finally:
            os.unlink(filename)

if __name__ == '__main__':
    unittest.main()
