"""Test module for DXF file handling."""
import os
import unittest
import ezdxf
from dxf_processor import DXFProcessor

class TestDXF(unittest.TestCase):
    """Test cases for DXF file handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dxf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'project.dxf')
        self.assertTrue(os.path.exists(self.dxf_path), "DXF test file not found")
        
    def test_file_readable(self):
        """Test that DXF file is readable."""
        doc = ezdxf.readfile(self.dxf_path)
        self.assertIsNotNone(doc)
        
    def test_layers_exist(self):
        """Test that DXF file contains layers."""
        doc = ezdxf.readfile(self.dxf_path)
        layers = [layer.dxf.name for layer in doc.layers]
        self.assertGreater(len(layers), 0)
        
    def test_processor_integration(self):
        """Test DXFProcessor integration."""
        processor = DXFProcessor(self.dxf_path)
        layers = processor.get_layers()
        self.assertIsInstance(layers, list)
        self.assertGreater(len(layers), 0)
        
    def test_point_extraction(self):
        """Test point extraction from layers."""
        processor = DXFProcessor(self.dxf_path)
        layers = processor.get_layers()
        
        for layer in layers:
            with self.subTest(layer=layer):
                try:
                    points = processor.extract_points_from_layer(layer)
                    if points:  # Some layers might not have points
                        self.assertIsInstance(points, dict)
                        point = next(iter(points.values()))
                        self.assertIsInstance(point.x, float)
                        self.assertIsInstance(point.y, float)
                        self.assertIsInstance(point.z, float)
                except Exception as e:
                    self.fail(f"Error processing layer '{layer}': {e}")

if __name__ == '__main__':
    unittest.main()
