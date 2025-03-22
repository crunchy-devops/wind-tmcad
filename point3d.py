"""Module providing an optimized 3D point class using slots and struct."""
from dataclasses import dataclass
import struct
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
        if not isinstance(self.id, int) or self.id < 0:
            raise ValueError("id must be a non-negative integer")
        
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
    
    def __repr__(self) -> str:
        """Return string representation of the point."""
        return f"Point3d(id={self.id}, x={self.x}, y={self.y}, z={self.z})"
