"""Module providing a memory-efficient Point Cloud implementation with spatial indexing."""
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from scipy.spatial import cKDTree, Delaunay
from scipy.interpolate import griddata
import h5py
import math
from point3d import Point3d
import matplotlib.pyplot as plt

__all__ = ['PointCloud']

class PointCloud:
    """A memory-efficient point cloud implementation with spatial indexing."""
    
    __slots__ = ['points', '_kdtree', '_coords_array', '_triangulation']
    
    def __init__(self):
        """Initialize an empty point cloud."""
        self.points: Dict[int, Point3d] = {}
        self._kdtree: Optional[cKDTree] = None
        self._coords_array: Optional[np.ndarray] = None
        self._triangulation = None
        
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
        """Invalidate spatial index and triangulation."""
        self._kdtree = None
        self._coords_array = None
        self._triangulation = None
    
    def compute_delaunay(self) -> None:
        """Compute Delaunay triangulation of the point cloud.
        
        This creates a triangulation that can be used for interpolation.
        The triangulation is stored in the _triangulation attribute.
        """
        from scipy.spatial import Delaunay
        
        if not self.points:
            raise ValueError("Cannot compute Delaunay triangulation of empty point cloud")
        
        # Get 2D points (x,y coordinates only) for triangulation
        points_2d = np.array([[p.x, p.y] for p in self.points.values()])
        self._triangulation = Delaunay(points_2d)
    
    def interpolate_z_delaunay(self, x: float, y: float) -> float:
        """Interpolate Z value at given X,Y coordinates using Delaunay triangulation.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Interpolated Z value
            
        Raises:
            ValueError: If point is outside the triangulation or triangulation hasn't been computed
        """
        if self._triangulation is None:
            self.compute_delaunay()
            
        point = np.array([[x, y]])
        simplex = self._triangulation.find_simplex(point)
        
        if simplex < 0:
            raise ValueError("Point is outside the triangulation")
            
        # Get vertices of the triangle containing the point
        vertices = self._triangulation.simplices[simplex][0]  # Get first (and only) simplex
        points_list = list(self.points.values())
        triangle_points = [points_list[int(i)] for i in vertices]
        
        # Compute barycentric coordinates
        x1, y1 = triangle_points[0].x, triangle_points[0].y
        x2, y2 = triangle_points[1].x, triangle_points[1].y
        x3, y3 = triangle_points[2].x, triangle_points[2].y
        
        det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
        w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
        w3 = 1 - w1 - w2
        
        # Interpolate z using barycentric coordinates
        z = (w1 * triangle_points[0].z + 
             w2 * triangle_points[1].z + 
             w3 * triangle_points[2].z)
        
        return z
    
    def interpolate_z_idw(self, x: float, y: float, k: int = None, p: float = 2) -> float:
        """Interpolate Z value at given X,Y coordinates using Inverse Distance Weighting.
        
        Args:
            x: X coordinate
            y: Y coordinate
            k: Number of nearest neighbors to use (default: all points)
            p: Power parameter (higher values give more weight to closer points)
            
        Returns:
            Interpolated Z value
            
        Raises:
            ValueError: If point cloud is empty
        """
        if not self.points:
            raise ValueError("Cannot interpolate with empty point cloud")
            
        if k is None:
            k = len(self.points)
        else:
            k = min(k, len(self.points))
            
        # Ensure KD-tree is built
        if self._kdtree is None:
            self._build_index()
            
        # Find k nearest neighbors
        distances, indices = self._kdtree.query(np.array([[x, y, 0]]), k=k)
        distances = distances[0]  # Flatten array
        indices = indices[0]      # Flatten array
        
        # Handle exact matches
        if distances[0] == 0:
            return list(self.points.values())[indices[0]].z
            
        # Compute weights using inverse distance
        weights = 1.0 / (distances ** p)
        weights_sum = np.sum(weights)
        
        # Get z values of neighbors
        points_list = list(self.points.values())
        neighbors = [points_list[i] for i in indices]
        z_values = np.array([p.z for p in neighbors])
        
        # Compute weighted average
        z = np.sum(weights * z_values) / weights_sum
        
        return float(z)
    
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
    
    def get_contour_lines(self, intervals: List[float] = None) -> Dict[float, List[List[List[float]]]]:
        """Generate contour lines at specified intervals.
        
        Args:
            intervals: List of elevation intervals for contours.
                     If None, uses [0.25, 0.5, 1.0] meters.
        
        Returns:
            Dictionary mapping elevation levels to lists of contour paths.
            Each path is a list of [x, y] coordinates.
        """
        if not self.points:
            return {}
            
        if intervals is None:
            # Default intervals: 25cm, 50cm, and 1m
            intervals = [0.25, 0.5, 1.0]
        
        # Extract point coordinates
        point_list = list(self.points.values())
        x_coords = np.array([p.x for p in point_list])
        y_coords = np.array([p.y for p in point_list])
        z_values = np.array([p.z for p in point_list])
        
        # Calculate grid bounds with some padding
        x_min, x_max = float(x_coords.min()), float(x_coords.max())
        y_min, y_max = float(y_coords.min()), float(y_coords.max())
        padding = 0.05  # 5% padding
        x_pad = (x_max - x_min) * padding
        y_pad = (y_max - y_min) * padding
        
        # Create regular grid
        grid_size = 100  # number of points in each dimension
        xi = np.linspace(x_min - x_pad, x_max + x_pad, grid_size)
        yi = np.linspace(y_min - y_pad, y_max + y_pad, grid_size)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Interpolate Z values on the grid
        points = np.column_stack((x_coords, y_coords))
        zi_grid = griddata(points, z_values, (xi_grid, yi_grid), method='cubic')
        
        # Generate contour lines
        contours = {}
        z_min, z_max = float(z_values.min()), float(z_values.max())
        
        # Generate levels based on intervals
        all_levels = []
        for interval in intervals:
            # Generate levels from min to max at the current interval
            start_level = float(np.ceil(z_min / interval) * interval)
            end_level = float(np.floor(z_max / interval) * interval)
            levels = np.arange(start_level, end_level + interval, interval)
            all_levels.extend(levels.tolist())  # Convert to list
        
        # Remove duplicates and sort
        all_levels = sorted(set(all_levels))
        
        # Generate contours for each level
        plt.ioff()  # Turn off interactive mode
        for level in all_levels:
            # Generate contour using matplotlib
            plt.figure()
            cs = plt.contour(xi_grid, yi_grid, zi_grid, levels=[level])
            plt.close()
            
            # Extract contour coordinates and convert to lists
            contour_paths = []
            for path in cs.collections[0].get_paths():
                vertices = path.vertices
                if len(vertices) > 2:  # Only include contours with at least 3 points
                    # Convert vertices to a list of [x, y] coordinates
                    path_coords = [[float(x), float(y)] for x, y in vertices]
                    contour_paths.append(path_coords)
            
            if contour_paths:
                contours[float(level)] = contour_paths
        
        return contours
