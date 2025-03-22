"""Test script to check DXF file contents."""
import os
import ezdxf
from dxf_processor import DXFProcessor

def main():
    """Main function to test DXF file."""
    dxf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'plan-masse.dxf')
    print(f"DXF file path: {dxf_path}")
    print(f"File exists: {os.path.exists(dxf_path)}")
    
    try:
        # Try to read the DXF file directly with ezdxf
        doc = ezdxf.readfile(dxf_path)
        print("\nLayers in DXF file:")
        for layer in doc.layers:
            print(f"- {layer.dxf.name}")
            
        # Try to read points using our processor
        print("\nTrying DXFProcessor:")
        processor = DXFProcessor(dxf_path)
        layers = processor.get_layers()
        print(f"Available layers: {layers}")
        
        # Try to extract points from each layer
        for layer in layers:
            try:
                points = processor.extract_points_from_layer(layer)
                print(f"\nLayer '{layer}': {len(points)} points found")
                if points:
                    point = next(iter(points.values()))
                    print(f"Sample point: {point}")
            except Exception as e:
                print(f"Error processing layer '{layer}': {e}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
