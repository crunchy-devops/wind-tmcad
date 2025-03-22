"""Demonstration of DXF processor usage."""
from dxf_processor import DXFProcessor
from point_cloud import PointCloud
import numpy as np

def main():
    # Initialize processor with DXF file
    processor = DXFProcessor("data/plan-masse.dxf")
    
    # Print available layers
    print("Available layers:")
    for layer in processor.get_layers():
        print(f"- {layer}")
    
    # Extract points from z value TN layer
    points = processor.extract_points_from_layer("z value TN")
    print(f"\nExtracted {len(points)} points from 'z value TN' layer")
    
    # Create point cloud and add points
    cloud = PointCloud()
    for point in points.values():
        cloud.add_point(point)
    
    # Print first 5 points as example
    print("\nFirst 5 points:")
    for point_id in list(points.keys())[:5]:
        point = points[point_id]
        print(f"Point {point.id}: ({point.x}, {point.y}, {point.z})")
    
    # Example of adding break lines
    point_ids = list(points.keys())[:3]
    processor.add_break_line(point_ids)
    print(f"\nAdded break line with points: {point_ids}")
    
    # Convert to numpy point cloud
    point_cloud = processor.get_point_cloud()
    print(f"\nPoint cloud shape: {point_cloud.shape}")
    
    # Compute Delaunay triangulation
    cloud.compute_delaunay()
    
    # Example interpolation points
    test_points = [
        (1035.0, 3000.0),  # Inside the point cloud
        (1040.0, 3002.0),  # Another point inside
    ]
    
    print("\nInterpolation examples:")
    for x, y in test_points:
        try:
            z_delaunay = cloud.interpolate_z_delaunay(x, y)
            z_idw = cloud.interpolate_z_idw(x, y, k=3)
            print(f"\nPoint ({x}, {y}):")
            print(f"  Delaunay interpolation Z: {z_delaunay:.2f}")
            print(f"  IDW interpolation Z: {z_idw:.2f}")
        except ValueError as e:
            print(f"\nCannot interpolate at ({x}, {y}): {e}")

if __name__ == "__main__":
    main()
