"""Test module for Point3d class."""
import pytest
import math
from point3d import Point3d

@pytest.fixture
def point():
    """Create a sample point for testing."""
    return Point3d(id=1, x=1.0, y=2.0, z=3.0)

def test_initialization(point):
    """Test point initialization."""
    assert point.id == 1
    assert point.x == 1.0
    assert point.y == 2.0
    assert point.z == 3.0

def test_invalid_id():
    """Test initialization with invalid id."""
    with pytest.raises(ValueError):
        Point3d(id=-1, x=1.0, y=2.0, z=3.0)
    with pytest.raises(ValueError):
        Point3d(id=0, x=1.0, y=2.0, z=3.0)

def test_invalid_coordinates():
    """Test initialization with invalid coordinates."""
    with pytest.raises(ValueError):
        Point3d(id=1, x=float('nan'), y=2.0, z=3.0)
    with pytest.raises(ValueError):
        Point3d(id=1, x=1.0, y=float('inf'), z=3.0)
    with pytest.raises(ValueError):
        Point3d(id=1, x=1.0, y=2.0, z=float('-inf'))

def test_immutability(point):
    """Test that points are immutable."""
    with pytest.raises(AttributeError):
        point.x = 5.0
    with pytest.raises(AttributeError):
        point.y = 5.0
    with pytest.raises(AttributeError):
        point.z = 5.0
    with pytest.raises(AttributeError):
        point.id = 5

def test_pack_unpack(point):
    """Test binary packing and unpacking."""
    packed = point.pack()
    unpacked = Point3d.unpack(packed)
    assert point == unpacked
    
    # Test with extreme values
    extreme_point = Point3d(id=999999, x=-1e6, y=1e6, z=0.0)
    packed = extreme_point.pack()
    unpacked = Point3d.unpack(packed)
    assert extreme_point == unpacked

def test_repr(point):
    """Test string representation."""
    expected = "Point3d(id=1, x=1.0, y=2.0, z=3.0)"
    assert repr(point) == expected
    
    # Test with integer coordinates
    point_int = Point3d(id=2, x=1, y=2, z=3)
    expected = "Point3d(id=2, x=1.0, y=2.0, z=3.0)"
    assert repr(point_int) == expected

def test_equality(point):
    """Test equality comparison."""
    same_point = Point3d(id=1, x=1.0, y=2.0, z=3.0)
    assert point == same_point
    
    different_id = Point3d(id=2, x=1.0, y=2.0, z=3.0)
    assert point != different_id
    
    different_coords = Point3d(id=1, x=1.1, y=2.0, z=3.0)
    assert point != different_coords

def test_hash(point):
    """Test hash function."""
    point_set = {point}
    same_point = Point3d(id=1, x=1.0, y=2.0, z=3.0)
    different_point = Point3d(id=2, x=1.0, y=2.0, z=3.0)
    
    assert same_point in point_set
    assert different_point not in point_set

def test_distance_to(point):
    """Test distance calculation."""
    other_point = Point3d(id=2, x=4.0, y=6.0, z=3.0)
    distance = point.distance_to(other_point)
    expected = math.sqrt(3**2 + 4**2)  # 3-4-5 triangle in xy plane, same z
    assert abs(distance - expected) < 1e-10
    
    # Test distance to self
    assert point.distance_to(point) == 0.0
