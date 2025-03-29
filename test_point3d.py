"""Test module for Point3d class."""
import unittest
import math
from point3d import Point3d

class TestPoint3d(unittest.TestCase):
    """Test cases for Point3d class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.point = Point3d(id=1, x=1.0, y=2.0, z=3.0)
    
    def test_initialization(self):
        """Test point initialization."""
        self.assertEqual(self.point.id, 1)
        self.assertEqual(self.point.x, 1.0)
        self.assertEqual(self.point.y, 2.0)
        self.assertEqual(self.point.z, 3.0)
    
    def test_invalid_id(self):
        """Test initialization with invalid id."""
        with self.assertRaises(ValueError):
            Point3d(id=-1, x=1.0, y=2.0, z=3.0)
        with self.assertRaises(ValueError):
            Point3d(id=0, x=1.0, y=2.0, z=3.0)
    
    def test_invalid_coordinates(self):
        """Test initialization with invalid coordinates."""
        with self.assertRaises(ValueError):
            Point3d(id=1, x=float('nan'), y=2.0, z=3.0)
        with self.assertRaises(ValueError):
            Point3d(id=1, x=1.0, y=float('inf'), z=3.0)
        with self.assertRaises(ValueError):
            Point3d(id=1, x=1.0, y=2.0, z=float('-inf'))
    
    def test_immutability(self):
        """Test that points are immutable."""
        with self.assertRaises(AttributeError):
            self.point.x = 5.0
        with self.assertRaises(AttributeError):
            self.point.y = 5.0
        with self.assertRaises(AttributeError):
            self.point.z = 5.0
        with self.assertRaises(AttributeError):
            self.point.id = 5
    
    def test_pack_unpack(self):
        """Test binary packing and unpacking."""
        packed = self.point.pack()
        unpacked = Point3d.unpack(packed)
        self.assertEqual(self.point, unpacked)
        
        # Test with extreme values
        extreme_point = Point3d(id=999999, x=-1e6, y=1e6, z=0.0)
        packed = extreme_point.pack()
        unpacked = Point3d.unpack(packed)
        self.assertEqual(extreme_point, unpacked)
    
    def test_repr(self):
        """Test string representation."""
        expected = "Point3d(id=1, x=1.0, y=2.0, z=3.0)"
        self.assertEqual(repr(self.point), expected)
        
        # Test with integer coordinates
        point = Point3d(id=2, x=1, y=2, z=3)
        expected = "Point3d(id=2, x=1.0, y=2.0, z=3.0)"
        self.assertEqual(repr(point), expected)
    
    def test_equality(self):
        """Test equality comparison."""
        same_point = Point3d(id=1, x=1.0, y=2.0, z=3.0)
        self.assertEqual(self.point, same_point)
        
        different_id = Point3d(id=2, x=1.0, y=2.0, z=3.0)
        self.assertNotEqual(self.point, different_id)
        
        different_coords = Point3d(id=1, x=1.1, y=2.0, z=3.0)
        self.assertNotEqual(self.point, different_coords)
    
    def test_hash(self):
        """Test hash function."""
        point_set = {self.point}
        same_point = Point3d(id=1, x=1.0, y=2.0, z=3.0)
        different_point = Point3d(id=2, x=1.0, y=2.0, z=3.0)
        
        self.assertIn(same_point, point_set)
        self.assertNotIn(different_point, point_set)
    
    def test_distance_to(self):
        """Test distance calculation."""
        other_point = Point3d(id=2, x=4.0, y=6.0, z=3.0)
        distance = self.point.distance_to(other_point)
        expected = math.sqrt(3**2 + 4**2)  # 3-4-5 triangle in xy plane, same z
        self.assertAlmostEqual(distance, expected)
        
        # Test distance to self
        self.assertEqual(self.point.distance_to(self.point), 0.0)

if __name__ == '__main__':
    unittest.main()
