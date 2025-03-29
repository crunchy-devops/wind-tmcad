"""Test module for DXF file handling."""
import os
import pytest
import ezdxf
from dxf_processor import DXFProcessor

@pytest.fixture
def dxf_path():
    """Get path to test DXF file."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'project.dxf')
    if not os.path.exists(path):
        pytest.skip("DXF test file not found")
    return path

@pytest.fixture
def processor(dxf_path):
    """Create DXFProcessor instance."""
    return DXFProcessor(dxf_path)

def test_file_readable(dxf_path):
    """Test that DXF file is readable."""
    doc = ezdxf.readfile(dxf_path)
    assert doc is not None

def test_layers_exist(dxf_path):
    """Test that DXF file contains layers."""
    doc = ezdxf.readfile(dxf_path)
    layers = [layer.dxf.name for layer in doc.layers]
    assert len(layers) > 0

def test_processor_integration(processor):
    """Test DXFProcessor integration."""
    layers = processor.get_layers()
    assert isinstance(layers, list)
    assert len(layers) > 0

def test_point_extraction(processor):
    """Test point extraction from layers."""
    layers = processor.get_layers()
    
    for layer in layers:
        try:
            points = processor.extract_points_from_layer(layer)
            if points:  # Some layers might not have points
                assert isinstance(points, dict)
                point = next(iter(points.values()))
                assert isinstance(point.x, float)
                assert isinstance(point.y, float)
                assert isinstance(point.z, float)
        except Exception as e:
            pytest.fail(f"Error processing layer '{layer}': {e}")
