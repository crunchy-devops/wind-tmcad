"""Module providing a memory-efficient Point Cloud implementation with spatial indexing."""
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from scipy.spatial import cKDTree, Delaunay, ConvexHull
from scipy.interpolate import griddata
import h5py
import math
from point3d import Point3d
import matplotlib.pyplot as plt
from matplotlib.path import Path
import logging
from shapely.geometry import LineString, Polygon, MultiLineString
from shapely.ops import unary_union, linemerge
from shapely.validation import explain_validity

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('point_cloud.log'),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)

__all__ = ['PointCloud']

class PointCloud:
    """A memory-efficient point cloud implementation with spatial indexing."""
    
    __slots__ = ['points', '_kdtree', '_coords_array', '_triangulation', '_convex_hull', '_boundary_polygon', 'break_lines']
    
    def __init__(self):
        """Initialize an empty point cloud."""
        self.points: Dict[int, Point3d] = {}
        self._kdtree: Optional[cKDTree] = None
        self._coords_array: Optional[np.ndarray] = None
        self._triangulation = None
        self._convex_hull = None
        self._boundary_polygon = None
        self.break_lines: List[Tuple[int, int]] = []  # List of (start_point_id, end_point_id) tuples
        
    def add_point(self, point: Point3d) -> None:
        """Add a point to the point cloud.
        
        Args:
            point: The Point3d object to add
            
        Raises:
            ValueError: If a point with the same ID already exists
        """
        if point.id in self.points:
            raise ValueError(f"Point with ID {point.id} already exists")
        self.points[point.id] = point
        # Reset cached data structures
        self._kdtree = None
        self._coords_array = None
        self._triangulation = None
        self._convex_hull = None
        self._boundary_polygon = None
        
    def _get_boundary_polygon(self) -> Polygon:
        """Get the boundary polygon of the point cloud."""
        if self._boundary_polygon is None:
            hull = self._get_convex_hull()
            vertices = hull.points[hull.vertices]
            # Close the polygon by adding the first point at the end
            vertices = np.vstack((vertices, vertices[0]))
            self._boundary_polygon = Polygon(vertices)
        return self._boundary_polygon
        
    def _validate_and_fix_contour(self, contour: LineString, level: float) -> Optional[LineString]:
        """Validate and fix a contour line."""
        if not contour.is_valid:
            logger.warning(f"Invalid contour at level {level}: {explain_validity(contour)}")
            return None
            
        # Ensure the contour is within the boundary
        boundary = self._get_boundary_polygon()
        if not contour.within(boundary):
            contour = contour.intersection(boundary)
            if contour.is_empty:
                return None
            if isinstance(contour, MultiLineString):
                # Merge multilinestring parts if possible
                contour = linemerge(contour)
                if isinstance(contour, MultiLineString):
                    # Take the longest part if we can't merge
                    contour = max(contour.geoms, key=lambda x: x.length)
                    
        # Ensure the contour is closed
        if not contour.is_ring:
            # Try to close the contour
            start = np.array(contour.coords[0])
            end = np.array(contour.coords[-1])
            if np.allclose(start, end, rtol=1e-5):
                # Points are already close enough, just make them exactly equal
                coords = list(contour.coords)
                coords[-1] = coords[0]
                contour = LineString(coords)
            else:
                # Add a closing segment
                coords = list(contour.coords)
                coords.append(coords[0])
                contour = LineString(coords)
                
        return contour
        
    def _simplify_contours(self, contours: List[LineString], tolerance: float = 0.1) -> List[LineString]:
        """Simplify contours while preserving topology."""
        simplified = []
        for contour in contours:
            # Simplify with a small tolerance to remove redundant points
            simple = contour.simplify(tolerance, preserve_topology=True)
            if simple.is_valid and not simple.is_empty:
                simplified.append(simple)
        return simplified
        
    def get_contour_lines(self, intervals: List[float] = None) -> Dict[float, List[List[List[float]]]]:
        """Generate contour lines at specified intervals."""
        logger.debug(f"Starting contour generation with intervals: {intervals}")  # Debug print
        
        if not self.points:
            logger.debug("No points in point cloud")  # Debug print
            return {}
            
        if intervals is None:
            intervals = [0.25, 0.5, 1.0]
            
        # Get point coordinates
        points = np.array([(p.x, p.y, p.z) for p in self.points.values()])
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        logger.debug(f"Point cloud stats: {len(points)} points")  # Debug print
        logger.debug(f"X range: [{x.min():.2f}, {x.max():.2f}]")  # Debug print
        logger.debug(f"Y range: [{y.min():.2f}, {y.max():.2f}]")  # Debug print
        logger.debug(f"Z range: [{z.min():.2f}, {z.max():.2f}]")  # Debug print
        
        # Get data bounds
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        z_min, z_max = z.min(), z.max()
        
        # Create a regular grid for interpolation
        grid_size = max(50, min(200, len(points) // 2))  # Adjust grid size based on point count
        xi = np.linspace(x_min, x_max, grid_size)
        yi = np.linspace(y_min, y_max, grid_size)
        xi, yi = np.meshgrid(xi, yi)
        logger.debug(f"Created interpolation grid of size {grid_size}x{grid_size}")  # Debug print
        
        # Interpolate z values on the grid
        try:
            zi = griddata((x, y), z, (xi, yi), method='cubic')
            logger.debug("Completed cubic interpolation")  # Debug print
        except Exception as e:
            logger.debug(f"Error in cubic interpolation: {str(e)}")  # Debug print
            return {}
            
        # Fill any NaN values using linear then nearest neighbor interpolation
        mask = np.isnan(zi)
        if mask.any():
            logger.debug(f"Filling {np.sum(mask)} NaN values with linear interpolation")  # Debug print
            zi_linear = griddata((x, y), z, (xi[mask], yi[mask]), method='linear')
            zi[mask] = zi_linear
            
            # Fill any remaining NaNs with nearest neighbor
            mask = np.isnan(zi)
            if mask.any():
                logger.debug(f"Filling {np.sum(mask)} remaining NaN values with nearest neighbor")  # Debug print
                zi[mask] = griddata((x, y), z, (xi[mask], yi[mask]), method='nearest')
        
        # Generate contours
        contours = {}
        for interval in intervals:
            min_level = math.floor(z_min / interval) * interval
            max_level = math.ceil(z_max / interval) * interval
            levels = np.arange(min_level, max_level + interval, interval)
            logger.debug(f"Generating contours for interval {interval}, levels: {levels}")  # Debug print
            
            if len(levels) == 0:
                logger.debug(f"No levels generated for interval {interval}")  # Debug print
                continue
                
            try:
                plt.figure()
                cs = plt.contour(xi, yi, zi, levels=levels)
                plt.close()
                
                for level_idx, level in enumerate(cs.levels):
                    level_contours = []
                    paths = cs.collections[level_idx].get_paths()
                    logger.debug(f"Found {len(paths)} paths for level {level}")  # Debug print
                    
                    # Convert matplotlib paths to Shapely geometries
                    shapely_contours = []
                    for path in paths:
                        vertices = path.vertices
                        if len(vertices) >= 3:
                            contour = LineString(vertices)
                            fixed_contour = self._validate_and_fix_contour(contour, level)
                            if fixed_contour is not None:
                                shapely_contours.append(fixed_contour)
                    
                    logger.debug(f"Created {len(shapely_contours)} valid Shapely contours")  # Debug print
                    
                    # Simplify and clean up contours
                    if shapely_contours:
                        # Merge overlapping contours
                        merged = linemerge(unary_union(shapely_contours))
                        if isinstance(merged, MultiLineString):
                            # Process each part separately
                            parts = list(merged.geoms)
                        else:
                            parts = [merged]
                            
                        # Simplify and validate each part
                        simplified = self._simplify_contours(parts)
                        logger.debug(f"Simplified to {len(simplified)} contours")  # Debug print
                        
                        # Convert back to coordinate lists
                        for contour in simplified:
                            coords = list(contour.coords)
                            if len(coords) >= 3:
                                level_contours.append(coords)
                    
                    if level_contours:
                        contours[float(level)] = level_contours
                        logger.debug(f"Added {len(level_contours)} contours for level {level}")  # Debug print
                        
            except Exception as e:
                logger.debug(f"Error generating contours at interval {interval}: {str(e)}")  # Debug print
                continue
        
        logger.debug(f"Finished contour generation, generated contours for {len(contours)} levels")  # Debug print
        return contours

    def _get_kdtree(self) -> cKDTree:
        """Get or create KD-tree for efficient nearest neighbor search.
        
        Returns:
            KD-tree of point cloud
        """
        if self._kdtree is None:
            # Create array of point coordinates
            if self._coords_array is None:
                points = list(self.points.values())
                self._coords_array = np.array([[p.x, p.y] for p in points])
            # Create KD-tree
            self._kdtree = cKDTree(self._coords_array)
        return self._kdtree

    def interpolate_z_idw(self, x: float, y: float, k: int = 5, p: int = 2) -> float:
        """Interpolate z value at (x,y) using Inverse Distance Weighting (IDW).
        
        Args:
            x: x-coordinate
            y: y-coordinate
            k: number of nearest neighbors to use
            p: power parameter for IDW
            
        Returns:
            Interpolated z value
            
        Raises:
            ValueError: If point is outside the convex hull of the point cloud
        """
        # Get k nearest neighbors
        distances, indices = self._get_kdtree().query([x, y], k=min(k, len(self.points)))
        
        # Check if point is too far from any existing points
        if np.min(distances) > 100:  # Arbitrary threshold
            raise ValueError("Point is too far from existing points")
            
        # Get z values of neighbors
        points_list = list(self.points.values())
        z_values = np.array([points_list[i].z for i in indices])
        
        # Handle exact matches
        if np.any(distances == 0):
            return z_values[distances == 0][0]
            
        # Calculate weights using inverse distance weighting
        weights = 1 / (distances ** p)
        weights /= np.sum(weights)  # Normalize weights
        
        # Calculate interpolated value
        return np.sum(weights * z_values)

    def interpolate_z_delaunay(self, x: float, y: float) -> float:
        """Interpolate z value at (x,y) using Delaunay triangulation.
        
        Args:
            x: x-coordinate
            y: y-coordinate
            
        Returns:
            Interpolated z value
            
        Raises:
            ValueError: If point is outside the triangulation
        """
        if self._triangulation is None:
            self.compute_delaunay()
            
        if self._triangulation is None:
            raise ValueError("No triangulation available")
            
        # Find the triangle containing the point
        simplex = self._triangulation.find_simplex(np.array([[x, y]]))
        if simplex < 0:
            raise ValueError("Point is outside the triangulation")
            
        # Get the vertices of the triangle
        vertices = self._triangulation.points[self._triangulation.simplices[simplex[0]]]
        points_array = np.array([[p.x, p.y] for p in self.points.values()])
        
        # Find the indices of the vertices in our points list
        vertex_indices = []
        for vertex in vertices:
            distances = np.sum((points_array - vertex) ** 2, axis=1)
            vertex_indices.append(np.argmin(distances))
            
        # Get the z values of the vertices
        points_list = list(self.points.values())
        z_values = np.array([points_list[i].z for i in vertex_indices])
        
        # Calculate barycentric coordinates
        b = self._triangulation.transform[simplex[0], :2].dot(np.array([x, y]) - self._triangulation.transform[simplex[0], 2])
        weights = np.append(b, 1 - b.sum())
        
        # Calculate interpolated value
        return np.sum(weights * z_values)
        
    def get_point(self, point_id: int) -> Point3d:
        """Get a point by its ID.
        
        Args:
            point_id: ID of the point to retrieve
            
        Returns:
            Point3d object with the given ID
            
        Raises:
            KeyError: If point with given ID does not exist
        """
        if point_id not in self.points:
            raise KeyError(f"Point with ID {point_id} does not exist")
        return self.points[point_id]
        
    def remove_point(self, point_id: int) -> None:
        """Remove a point from the point cloud.
        
        Args:
            point_id: ID of the point to remove
            
        Raises:
            KeyError: If point with given ID does not exist
        """
        if point_id not in self.points:
            raise KeyError(f"Point with ID {point_id} does not exist")
        del self.points[point_id]
        # Reset cached data structures
        self._kdtree = None
        self._coords_array = None
        self._triangulation = None
        self._convex_hull = None
        self._boundary_polygon = None
        
    def distance(self, point_id1: int, point_id2: int) -> float:
        """Calculate Euclidean distance between two points.
        
        Args:
            point_id1: ID of first point
            point_id2: ID of second point
            
        Returns:
            Distance between the points
            
        Raises:
            KeyError: If either point ID does not exist
        """
        p1 = self.get_point(point_id1)
        p2 = self.get_point(point_id2)
        return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)
        
    def bearing_angle(self, point_id1: int, point_id2: int) -> float:
        """Calculate bearing angle between two points.
        
        Args:
            point_id1: ID of first point
            point_id2: ID of second point
            
        Returns:
            Bearing angle in degrees (0-360)
            
        Raises:
            KeyError: If either point ID does not exist
        """
        p1 = self.get_point(point_id1)
        p2 = self.get_point(point_id2)
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        angle = math.degrees(math.atan2(dy, dx))
        return (angle + 360) % 360
        
    def find_nearest_neighbors(self, point_id: int, k: int) -> List[Tuple[int, float]]:
        """Find k nearest neighbors to a point.
        
        Args:
            point_id: ID of the point to find neighbors for
            k: Number of neighbors to find
            
        Returns:
            List of tuples (point_id, distance) of nearest neighbors
            
        Raises:
            KeyError: If point ID does not exist
            ValueError: If k is greater than number of points
        """
        if k > len(self.points) - 1:  # -1 because we exclude the query point
            raise ValueError(f"k ({k}) cannot be greater than number of points - 1 ({len(self.points) - 1})")
        point = self.get_point(point_id)
        distances, indices = self._get_kdtree().query([point.x, point.y], k=k+1)  # k+1 because point itself is included
        # Convert indices to point IDs and pair with distances, excluding the query point itself
        point_ids = list(self.points.keys())
        return [(point_ids[i], d) for i, d in zip(indices[1:], distances[1:])]
        
    def slope_percentage(self, point_id1: int, point_id2: int) -> float:
        """Calculate slope percentage between two points.
        
        Args:
            point_id1: ID of first point
            point_id2: ID of second point
            
        Returns:
            Slope percentage
            
        Raises:
            KeyError: If either point ID does not exist
        """
        p1 = self.get_point(point_id1)
        p2 = self.get_point(point_id2)
        dz = p2.z - p1.z
        dxy = math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
        if dxy == 0:
            return float('inf') if dz > 0 else float('-inf')
        return (dz / dxy) * 100
        
    def save_to_hdf5(self, filename: str) -> None:
        """Save point cloud to HDF5 file.
        
        Args:
            filename: Path to save file
        """
        with h5py.File(filename, 'w') as f:
            # Store points as a single dataset with id, x, y, z
            points_data = np.array([[p.id, p.x, p.y, p.z] for p in self.points.values()])
            f.create_dataset('points', data=points_data)
                
    def _load_from_hdf5(self, filename: str) -> None:
        """Load point cloud from HDF5 file.
        
        Args:
            filename: Path to load file
            
        Raises:
            IOError: If file cannot be read
        """
        self.points.clear()
        try:
            with h5py.File(filename, 'r') as f:
                logger.debug("HDF5 file opened successfully")  
                logger.debug(f"File keys: {list(f.keys())}")  
                points_data = f['points'][:]
                logger.debug(f"Loading {len(points_data)} points")  
                for point in points_data:
                    point_id = int(point[0])
                    x = float(point[1])
                    y = float(point[2])
                    z = float(point[3])
                    self.points[point_id] = Point3d(id=point_id, x=x, y=y, z=z)
                logger.debug(f"Loaded {len(self.points)} points")  
        except Exception as e:
            logger.error(f"Error loading HDF5 file: {str(e)}")  
            raise

    def compute_delaunay(self) -> None:
        """Compute Delaunay triangulation of points."""
        if not self.points:
            return
        if self._triangulation is None:
            points = np.array([[p.x, p.y] for p in self.points.values()])
            self._triangulation = Delaunay(points)
            
    def _get_convex_hull(self) -> ConvexHull:
        """Get or compute convex hull of points."""
        if self._convex_hull is None:
            points = np.array([[p.x, p.y] for p in self.points.values()])
            self._convex_hull = ConvexHull(points)
        return self._convex_hull

    @classmethod
    def load_from_hdf5(cls, filename: str) -> 'PointCloud':
        """Load point cloud from HDF5 file.
        
        Args:
            filename: Path to load file
            
        Returns:
            New PointCloud instance loaded from file
            
        Raises:
            IOError: If file cannot be read
        """
        cloud = cls()
        cloud._load_from_hdf5(filename)
        return cloud
