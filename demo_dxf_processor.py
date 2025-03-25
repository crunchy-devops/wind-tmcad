"""Demonstration of DXF processor usage."""
from dxf_processor import DXFProcessor
from point_cloud import PointCloud
from models import init_db, Session, Project, PointCloud as DBPointCloud, Breakline, DelaunayTriangle
import numpy as np

def save_point_cloud_to_db(cloud: PointCloud, project_name: str):
    """Save point cloud data to SQLite database.
    
    Args:
        cloud: PointCloud instance containing the point cloud data
        project_name: Name of the project to save
    """
    init_db()  # Ensure database tables exist
    session = Session()
    
    try:
        # Create new project
        project = Project(name=project_name)
        session.add(project)
        session.flush()  # Get the project ID
        
        # Save points and keep track of original IDs to database IDs
        point_map = {}  # Maps original point IDs to database IDs
        for point_id, point in cloud.points.items():
            db_point = DBPointCloud(
                project_id=project.id,
                x=point.x,
                y=point.y,
                z=point.z
            )
            session.add(db_point)
            session.flush()
            point_map[point_id] = db_point.id
        
        # Save breaklines if they exist
        if hasattr(cloud, 'break_lines'):
            for breakline in cloud.break_lines:
                start_id = point_map[breakline[0]]
                end_id = point_map[breakline[1]]
                db_breakline = Breakline(
                    project_id=project.id,
                    start_point3d_id=start_id,
                    end_point3d_id=end_id
                )
                session.add(db_breakline)
        
        # Save Delaunay triangles if they exist
        if hasattr(cloud, '_triangulation') and cloud._triangulation is not None:
            for triangle in cloud._triangulation.simplices:
                db_triangle = DelaunayTriangle(
                    project_id=project.id,
                    point3d_id1=point_map[list(cloud.points.keys())[triangle[0]]],
                    point3d_id2=point_map[list(cloud.points.keys())[triangle[1]]],
                    point3d_id3=point_map[list(cloud.points.keys())[triangle[2]]]
                )
                session.add(db_triangle)
        
        session.commit()
        print(f"Successfully saved project '{project_name}' to database")
        
    except Exception as e:
        session.rollback()
        print(f"Error saving to database: {str(e)}")
        raise
    
    finally:
        session.close()

def main():
    # Initialize processor with DXF file
    processor = DXFProcessor("data/project.dxf")
    
    # Print available layers
    print("Available layers:")
    for layer in processor.get_layers():
        print(f"- {layer}")
    
    # Extract points from z value projet layer
    points = processor.extract_points_from_layer("z value TN")
    print(f"\nExtracted {len(points)} points from 'z value projet' layer")
    
    # Create point cloud and add points
    cloud = PointCloud()
    cloud.break_lines = []  # Initialize break_lines list
    for point in points.values():
        cloud.add_point(point)
    
    # Print first 5 points as example
    print("\nFirst 5 points:")
    for point_id in list(points.keys())[:5]:
        point = points[point_id]
        print(f"Point {point.id}: ({point.x}, {point.y}, {point.z})")
    
    # Example of adding break lines
    point_ids = list(points.keys())[:3]
    cloud.break_lines.append((point_ids[0], point_ids[1]))  # Add first two points as a breakline
    print(f"\nAdded break line with points: {point_ids[:2]}")
    
    # Convert to numpy point cloud
    point_cloud = processor.get_point_cloud()
    print(f"\nPoint cloud shape: {point_cloud.shape}")
    
    # Example interpolation points
    test_points = [
        (1035.0, 3000.0),  # Inside the point cloud
        (1040.0, 3002.0),  # Another point inside
    ]
    
    print("\nInterpolation examples:")
    for x, y in test_points:
        try:
            # Use IDW interpolation instead since it's available
            z = cloud.interpolate_z_idw(x, y)
            print(f"Point ({x}, {y}): z = {z:.2f}")
        except ValueError as e:
            print(f"Error interpolating point ({x}, {y}): {str(e)}")
    
    # Save point cloud to database
    try:
        save_point_cloud_to_db(cloud, "MNT")
    except Exception as e:
        print(f"Failed to save to database: {str(e)}")

if __name__ == "__main__":
    main()
