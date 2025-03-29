"""Module providing an optimized 3D point class using slots and struct."""
from dataclasses import dataclass
import struct
import math
from typing import ClassVar

__all__ = ['Point3d']

@dataclass(slots=True, frozen=True)
class Point3d:
    """A memory-efficient 3D point class using struct for packing coordinates.
    
    Attributes:
        id (int): Unique identifier for the point
        x (float): X coordinate
        y (float): Y coordinate
        z (float): Z coordinate
    """
    id: int
    x: float
    y: float
    z: float
    
    # Class variable to store the struct format
    _struct_format: ClassVar[str] = '=Qfff'  # Q for uint64, f for float32
    
    def __post_init__(self):
        """Validate the input values after initialization."""
        if not isinstance(self.id, int):
            raise ValueError("id must be an integer")
        if self.id <= 0:
            raise ValueError("id must be a positive integer")
            
        # Convert coordinates to float if they're integers
        object.__setattr__(self, 'x', float(self.x))
        object.__setattr__(self, 'y', float(self.y))
        object.__setattr__(self, 'z', float(self.z))
        
        # Check for invalid coordinates
        for coord in (self.x, self.y, self.z):
            if math.isnan(coord) or math.isinf(coord):
                raise ValueError("Coordinates must be finite numbers")
        
    def pack(self) -> bytes:
        """Pack the point coordinates into a binary string using struct.
        
        Returns:
            bytes: Binary representation of the point
        """
        return struct.pack(self._struct_format, self.id, self.x, self.y, self.z)
    
    @classmethod
    def unpack(cls, data: bytes) -> 'Point3d':
        """Create a Point3d instance from packed binary data.
        
        Args:
            data: Binary data containing packed point coordinates
            
        Returns:
            Point3d: New instance created from the binary data
        """
        id_, x, y, z = struct.unpack(cls._struct_format, data)
        return cls(id=id_, x=x, y=y, z=z)
        
    def distance_to(self, other: 'Point3d') -> float:
        """Calculate Euclidean distance to another point.
        
        Args:
            other: Point to calculate distance to
            
        Returns:
            float: Euclidean distance between the points
        """
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def __repr__(self) -> str:
        """Return string representation of the point."""
        return f"Point3d(id={self.id}, x={self.x}, y={self.y}, z={self.z})"
