"""DXF processing module for converting DXF files to Point Cloud data."""
from typing import List, Dict, Set, Optional, Tuple
import ezdxf
import numpy as np
from point3d import Point3d

__all__ = ['DXFProcessor']

class DXFProcessor:
    """Processes DXF files and extracts point cloud data."""
    __slots__ = ['_dxf_path', '_doc', '_points', '_break_lines']

    def __init__(self, dxf_path: str):
        """Initialize DXF processor.
        
        Args:
            dxf_path: Path to the DXF file
        """
        self._dxf_path = dxf_path
        self._doc = ezdxf.readfile(dxf_path)
        self._points: Dict[int, Point3d] = {}
        self._break_lines: List[List[int]] = []

    def get_layers(self) -> List[str]:
        """Get all available layers in the DXF file.
        
        Returns:
            List of layer names
        """
        return [layer.dxf.name for layer in self._doc.layers]

    def extract_points_from_layer(self, layer_name: str) -> Dict[int, Point3d]:
        """Extract Point3d objects from TEXT entities in specified layer.
        
        Args:
            layer_name: Name of the layer to process
            
        Returns:
            Dictionary mapping point IDs to Point3d objects
            
        Raises:
            KeyError: If layer_name does not exist in the DXF file
        """
        # Check if layer exists
        if layer_name not in [layer.dxf.name for layer in self._doc.layers]:
            raise KeyError(f"Layer '{layer_name}' not found in DXF file")
            
        msp = self._doc.modelspace()
        point_id = 1
        points = {}  # Local dictionary to store points
        
        # Filter TEXT entities from specified layer
        for text in msp.query('TEXT[layer=="{}"]'.format(layer_name)):
            try:
                # Get position (x, y) from entity
                x, y = text.dxf.insert.x, text.dxf.insert.y
                # Get z value from text content
                z = float(text.dxf.text.strip())
                
                points[point_id] = Point3d(point_id, x, y, z)
                point_id += 1
            except (ValueError, AttributeError) as e:
                print(f"Warning: Skipping invalid point data: {e}")
                
        self._points.update(points)
        return points

    def add_break_line(self, point_ids: List[int]) -> bool:
        """Add a break line defined by a sequence of point IDs.
        
        Args:
            point_ids: List of point IDs defining the break line
            
        Returns:
            True if break line was added successfully, False otherwise
        """
        # Validate input
        if not point_ids or len(point_ids) < 2:
            return False
            
        # Validate all points exist
        if not all(pid in self._points for pid in point_ids):
            return False
            
        # Check if break line already exists
        if point_ids in self._break_lines:
            return False
            
        self._break_lines.append(point_ids)
        return True

    def get_break_lines(self) -> List[List[int]]:
        """Get all defined break lines.
        
        Returns:
            List of break lines, where each break line is a list of point IDs
        """
        return self._break_lines

    def get_point_cloud(self) -> np.ndarray:
        """Convert points to numpy array format.
        
        Returns:
            Nx3 numpy array of points
        """
        return np.array([[p.x, p.y, p.z] for p in self._points.values()])
