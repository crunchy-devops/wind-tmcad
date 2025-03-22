"""Module providing a memory-efficient Point Cloud implementation with spatial indexing."""
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from scipy.spatial import cKDTree
import h5py
import math
from point3d import Point3d

__all__ = ['PointCloud']

class PointCloud:
    """A memory-efficient point cloud implementation with spatial indexing.
    
    Uses KD-tree for spatial queries and HDF5 for efficient storage.
    
    Attributes:
        points (Dict[int, Point3d]): Dictionary mapping point IDs to Point3d objects
        _kdtree (cKDTree): Spatial index for efficient nearest neighbor queries
        _coords_array (np.ndarray): Numpy array of point coordinates for KD-tree
    """
    
    __slots__ = ['points', '_kdtree', '_coords_array']
    
    def __init__(self):
        """Initialize an empty point cloud."""
        self.points: Dict[int, Point3d] = {}
        self._kdtree: Optional[cKDTree] = None
        self._coords_array: Optional[np.ndarray] = None
        
    def add_point(self, point: Point3d) -> None:
        """Add a point to the cloud.
        
        Args:
            point: Point3d object to add
            
        Raises:
            ValueError: If point with same ID already exists
        """
        if point.id in self.points:
            raise ValueError(f"Point with ID {point.id} already exists")
        self.points[point.id] = point
        self._invalidate_index()
        
    def remove_point(self, point_id: int) -> None:
        """Remove a point from the cloud.
        
        Args:
            point_id: ID of point to remove
            
        Raises:
            KeyError: If point ID doesn't exist
        """
        del self.points[point_id]
        self._invalidate_index()
        
    def get_point(self, point_id: int) -> Point3d:
        """Get a point by its ID.
        
        Args:
            point_id: ID of point to retrieve
            
        Returns:
            Point3d object
            
        Raises:
            KeyError: If point ID doesn't exist
        """
        return self.points[point_id]
    
    def distance(self, id1: int, id2: int) -> float:
        """Calculate Euclidean distance between two points.
        
        Args:
            id1: ID of first point
            id2: ID of second point
            
        Returns:
            Distance in same units as point coordinates
            
        Raises:
            KeyError: If either point ID doesn't exist
        """
        p1 = self.points[id1]
        p2 = self.points[id2]
        return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)
    
    def slope_percentage(self, id1: int, id2: int) -> float:
        """Calculate slope percentage between two points.
        
        Args:
            id1: ID of first point
            id2: ID of second point
            
        Returns:
            Slope as a percentage
            
        Raises:
            KeyError: If either point ID doesn't exist
            ZeroDivisionError: If points have same x,y coordinates
        """
        p1 = self.points[id1]
        p2 = self.points[id2]
        
        horizontal_distance = math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
        if horizontal_distance == 0:
            raise ZeroDivisionError("Points have same x,y coordinates")
            
        vertical_distance = p2.z - p1.z
        return (vertical_distance / horizontal_distance) * 100
    
    def bearing_angle(self, id1: int, id2: int) -> float:
        """Calculate bearing angle between two points in degrees.
        
        Args:
            id1: ID of first point
            id2: ID of second point
            
        Returns:
            Bearing angle in degrees (0-360)
            
        Raises:
            KeyError: If either point ID doesn't exist
        """
        p1 = self.points[id1]
        p2 = self.points[id2]
        
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        
        angle = math.degrees(math.atan2(dy, dx))
        return (angle + 360) % 360
    
    def _build_index(self) -> None:
        """Build KD-tree spatial index from current points."""
        if not self.points:
            self._kdtree = None
            self._coords_array = None
            return
            
        coords = [(p.x, p.y, p.z) for p in self.points.values()]
        self._coords_array = np.array(coords)
        self._kdtree = cKDTree(self._coords_array)
    
    def _invalidate_index(self) -> None:
        """Invalidate spatial index when points are modified."""
        self._kdtree = None
        self._coords_array = None
    
    def find_nearest_neighbors(self, point_id: int, k: int) -> List[Tuple[int, float]]:
        """Find k nearest neighbors to a point.
        
        Args:
            point_id: ID of reference point
            k: Number of neighbors to find
            
        Returns:
            List of tuples (point_id, distance) for k nearest neighbors
            
        Raises:
            KeyError: If point ID doesn't exist
            ValueError: If k is larger than number of points
        """
        if k > len(self.points):
            raise ValueError(f"k ({k}) cannot be larger than number of points ({len(self.points)})")
            
        if self._kdtree is None:
            self._build_index()
            
        point = self.points[point_id]
        query_point = np.array([[point.x, point.y, point.z]])
        
        # k+1 because the point itself will be included
        distances, indices = self._kdtree.query(query_point, k=k+1)
        
        # Skip the first result (distance=0 to itself)
        point_ids = list(self.points.keys())
        return [(point_ids[i], d) for i, d in zip(indices[0][1:], distances[0][1:])]
    
    def save_to_hdf5(self, filename: str) -> None:
        """Save point cloud to HDF5 file format.
        
        Args:
            filename: Path to output HDF5 file
        """
        with h5py.File(filename, 'w') as f:
            # Store points as structured array
            dt = np.dtype([('id', np.uint64), ('x', np.float32), 
                         ('y', np.float32), ('z', np.float32)])
            data = np.array([(p.id, p.x, p.y, p.z) for p in self.points.values()], 
                          dtype=dt)
            f.create_dataset('points', data=data, compression='gzip', 
                           compression_opts=9)
    
    @classmethod
    def load_from_hdf5(cls, filename: str) -> 'PointCloud':
        """Load point cloud from HDF5 file format.
        
        Args:
            filename: Path to input HDF5 file
            
        Returns:
            New PointCloud instance
        """
        cloud = cls()
        with h5py.File(filename, 'r') as f:
            points = f['points'][:]
            for point in points:
                cloud.add_point(Point3d(id=int(point['id']), 
                                      x=float(point['x']),
                                      y=float(point['y']),
                                      z=float(point['z'])))
        return cloud
