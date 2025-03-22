"""Demonstration of DXF processor usage."""
from dxf_processor import DXFProcessor, Point3d

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
    
    # Example of packing/unpacking points
    print("\nDemonstrating point packing/unpacking:")
    first_point = points[1]
    packed = first_point.pack()
    unpacked = Point3d.unpack(packed)
    print(f"Original point: ({first_point.x}, {first_point.y}, {first_point.z})")
    print(f"Unpacked point: ({unpacked.x}, {unpacked.y}, {unpacked.z})")

if __name__ == "__main__":
    main()
