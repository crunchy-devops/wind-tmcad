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

__all__ = ['PointCloud']

class PointCloud:
    """A memory-efficient point cloud implementation with spatial indexing."""
    
    __slots__ = ['points', '_kdtree', '_coords_array', '_triangulation', '_convex_hull', '_boundary_polygon']
    
    def __init__(self):
        """Initialize an empty point cloud."""
        self.points: Dict[int, Point3d] = {}
        self._kdtree: Optional[cKDTree] = None
        self._coords_array: Optional[np.ndarray] = None
        self._triangulation = None
        self._convex_hull = None
        self._boundary_polygon = None
        
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
            logging.warning(f"Invalid contour at level {level}: {explain_validity(contour)}")
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
        if not self.points:
            return {}
            
        if intervals is None:
            intervals = [0.25, 0.5, 1.0]
            
        # Reset cached geometries
        self._convex_hull = None
        self._boundary_polygon = None
        
        # Extract point coordinates
        point_list = list(self.points.values())
        x_coords = np.array([p.x for p in point_list])
        y_coords = np.array([p.y for p in point_list])
        z_values = np.array([p.z for p in point_list])
        
        # Calculate grid bounds with padding
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        z_min, z_max = z_values.min(), z_values.max()
        
        padding = 0.05  # 5% padding
        x_pad = (x_max - x_min) * padding
        y_pad = (y_max - y_min) * padding
        
        # Create denser grid for better interpolation
        grid_size = int(np.sqrt(len(point_list)) * 4)  # Increased density further
        xi = np.linspace(x_min - x_pad, x_max + x_pad, grid_size)
        yi = np.linspace(y_min - y_pad, y_max + y_pad, grid_size)
        xi_mg, yi_mg = np.meshgrid(xi, yi)
        
        # Interpolate Z values using cubic interpolation
        zi = griddata((x_coords, y_coords), z_values, (xi_mg, yi_mg), method='cubic')
        
        # Fill NaN values using nearest neighbor
        mask = np.isnan(zi)
        if mask.any():
            zi[mask] = griddata(
                (x_coords, y_coords), z_values, (xi_mg[mask], yi_mg[mask]), 
                method='nearest'
            )
        
        # Generate contours
        contours = {}
        for interval in intervals:
            min_level = math.ceil(z_min / interval) * interval
            max_level = math.floor(z_max / interval) * interval
            levels = np.arange(min_level, max_level + interval, interval)
            
            if len(levels) == 0:
                continue
                
            try:
                plt.figure()
                cs = plt.contour(xi, yi, zi, levels=levels)
                plt.close()
                
                for level_idx, level in enumerate(cs.levels):
                    level_contours = []
                    paths = cs.collections[level_idx].get_paths()
                    
                    # Convert matplotlib paths to Shapely geometries
                    shapely_contours = []
                    for path in paths:
                        vertices = path.vertices
                        if len(vertices) >= 3:
                            contour = LineString(vertices)
                            fixed_contour = self._validate_and_fix_contour(contour, level)
                            if fixed_contour is not None:
                                shapely_contours.append(fixed_contour)
                    
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
                        
                        # Convert back to coordinate lists
                        for contour in simplified:
                            coords = list(contour.coords)
                            if len(coords) >= 3:
                                level_contours.append(coords)
                    
                    if level_contours:
                        contours[float(level)] = level_contours
                        
            except Exception as e:
                logging.warning(f"Error generating contours at interval {interval}: {str(e)}")
                continue
        
        return contours
