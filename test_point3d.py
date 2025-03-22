"""Test module for Point3d class."""
import unittest
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
    
    def test_immutability(self):
        """Test that points are immutable."""
        with self.assertRaises(AttributeError):
            self.point.x = 5.0
    
    def test_pack_unpack(self):
        """Test binary packing and unpacking."""
        packed = self.point.pack()
        unpacked = Point3d.unpack(packed)
        self.assertEqual(self.point, unpacked)
    
    def test_repr(self):
        """Test string representation."""
        expected = "Point3d(id=1, x=1.0, y=2.0, z=3.0)"
        self.assertEqual(repr(self.point), expected)

if __name__ == '__main__':
    unittest.main()
