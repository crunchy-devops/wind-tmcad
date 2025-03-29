"""Tests for DXF processor module."""
import os
import pytest
import numpy as np
from dxf_processor import DXFProcessor
from point3d import Point3d

@pytest.fixture
def processor():
    """Create a DXFProcessor instance for testing."""
    dxf_path = os.path.join("data", "project.dxf")
    if not os.path.exists(dxf_path):
        pytest.skip(f"Test file {dxf_path} not found")
    return DXFProcessor(dxf_path)

def test_get_layers(processor):
    """Test layer retrieval."""
    layers = processor.get_layers()
    assert isinstance(layers, list)
    assert len(layers) > 0
    assert all(isinstance(layer, str) for layer in layers)
    
def test_nonexistent_layer(processor):
    """Test handling of non-existent layer."""
    with pytest.raises(KeyError):  
        processor.extract_points_from_layer("nonexistent_layer")

def test_extract_points_from_layer(processor):
    """Test point extraction from z value TN layer."""
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
        assert not np.isnan(point.x)
        assert not np.isnan(point.y)
        assert not np.isnan(point.z)

def test_break_lines(processor):
    """Test break line functionality."""
    points = processor.extract_points_from_layer("z value TN")
    
    # Get first three point IDs
    point_ids = list(points.keys())[:3]
    
    # Add break line
    assert processor.add_break_line(point_ids)
    
    # Try to add invalid break line
    assert not processor.add_break_line([999999])  
    assert not processor.add_break_line([])  
    assert not processor.add_break_line([1])  
    
    # Get break lines
    break_lines = processor.get_break_lines()
    assert len(break_lines) == 1
    assert break_lines[0] == point_ids
    
    # Try to add duplicate break line
    assert not processor.add_break_line(point_ids)

def test_point_cloud_conversion(processor):
    """Test conversion to numpy point cloud."""
    points = processor.extract_points_from_layer("z value TN")
    cloud = processor.get_point_cloud()  
    
    assert isinstance(cloud, np.ndarray)
    assert cloud.shape[1] == 3  
    assert len(cloud) == len(points)
    
    # Test that all points are included
    for point in points.values():
        point_array = np.array([point.x, point.y, point.z])
        assert any(np.allclose(row, point_array) for row in cloud)

def test_invalid_dxf_file():
    """Test handling of invalid DXF file."""
    with pytest.raises(Exception):
        DXFProcessor("nonexistent.dxf")
        
def test_empty_layer(processor):
    """Test handling of empty layer."""
    # Get first available layer
    layers = processor.get_layers()
    test_layer = layers[0]  # Use first layer instead of a hardcoded name
    points = processor.extract_points_from_layer(test_layer)
    assert isinstance(points, dict)
