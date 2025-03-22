"""Tests for DXF processor module."""
import pytest
import numpy as np
from dxf_processor import DXFProcessor
from point3d import Point3d

def test_get_layers():
    """Test layer retrieval."""
    processor = DXFProcessor("data/plan-masse.dxf")
    layers = processor.get_layers()
    assert isinstance(layers, list)
    assert len(layers) > 0
    assert all(isinstance(layer, str) for layer in layers)

def test_extract_points_from_layer():
    """Test point extraction from z value TN layer."""
    processor = DXFProcessor("data/plan-masse.dxf")
    points = processor.extract_points_from_layer("z value TN")
    
    assert isinstance(points, dict)
    assert len(points) > 0
    
    # Test a few points
    for point_id, point in points.items():
        assert isinstance(point, Point3d)
        assert point.id == point_id
        assert isinstance(point.x, float)
        assert isinstance(point.y, float)
        assert isinstance(point.z, float)

def test_break_lines():
    """Test break line functionality."""
    processor = DXFProcessor("data/plan-masse.dxf")
    points = processor.extract_points_from_layer("z value TN")
    
    # Get first three point IDs
    point_ids = list(points.keys())[:3]
    
    # Add break line
    assert processor.add_break_line(point_ids)
    
    # Try to add invalid break line
    assert not processor.add_break_line([999999])  # Non-existent point ID
    
    # Get break lines
    break_lines = processor.get_break_lines()
    assert len(break_lines) == 1
    assert break_lines[0] == point_ids

def test_point_cloud_conversion():
    """Test conversion to numpy point cloud."""
    processor = DXFProcessor("data/plan-masse.dxf")
    processor.extract_points_from_layer("z value TN")
    
    point_cloud = processor.get_point_cloud()
    assert isinstance(point_cloud, np.ndarray)
    assert point_cloud.shape[1] == 3  # 3D points
    assert len(point_cloud) > 0
