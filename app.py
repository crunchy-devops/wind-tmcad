"""Flask web server for point cloud visualization."""
from flask import Flask, render_template, jsonify, request
from dxf_processor import DXFProcessor
from point_cloud import PointCloud
from point3d import Point3d
import plotly.graph_objects as go
import numpy as np
import logging
import os
import json
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def validate_contours(points, contours):
    """Validate contour lines for proper closure and intersection."""
    try:
        # Check if we have enough points for meaningful contours
        if len(points) < 3:
            return False, "Not enough points for contour generation"
            
        # Get model boundary dimensions for relative gap tolerance
        x_coords = np.array([p.x for p in points.values()])
        y_coords = np.array([p.y for p in points.values()])
        model_width = x_coords.max() - x_coords.min()
        model_height = y_coords.max() - y_coords.min()
        model_diagonal = np.sqrt(model_width**2 + model_height**2)
        
        # Maximum allowed gap is 2% of model diagonal for boundary contours
        # and 1% for internal contours
        max_boundary_gap = model_diagonal * 0.02
        max_internal_gap = model_diagonal * 0.01
            
        # Check for closed contours with adaptive tolerance
        for level, paths in contours.items():
            for path in paths:
                if len(path) < 3:
                    return False, f"Invalid contour at level {level}: too few points"
                
                # Get contour bounding box
                path_array = np.array(path)
                min_x, min_y = path_array.min(axis=0)
                max_x, max_y = path_array.max(axis=0)
                
                # Check if contour touches the model boundary
                touches_boundary = (
                    np.isclose(min_x, x_coords.min(), rtol=1e-3) or
                    np.isclose(max_x, x_coords.max(), rtol=1e-3) or
                    np.isclose(min_y, y_coords.min(), rtol=1e-3) or
                    np.isclose(max_y, y_coords.max(), rtol=1e-3)
                )
                
                # Check closure with appropriate tolerance
                start = np.array(path[0])
                end = np.array(path[-1])
                gap = np.linalg.norm(end - start)
                max_allowed_gap = max_boundary_gap if touches_boundary else max_internal_gap
                
                if gap > max_allowed_gap:
                    gap_percent = (gap / model_diagonal) * 100
                    return False, (
                        f"Unclosed contour detected at level {level} "
                        f"(gap: {gap:.2f}m, {gap_percent:.1f}% of model size)"
                    )
        
        # Check for invalid intersections
        z_values = np.array([point.z for point in points.values()])
        z_min, z_max = z_values.min(), z_values.max()
        
        for level, paths in contours.items():
            if float(level) < z_min or float(level) > z_max:
                return False, f"Contour level {level} outside of terrain bounds [{z_min:.2f}, {z_max:.2f}]"
                
        return True, None
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def create_point_cloud_plot(file_path, layer, intervals):
    """Create a 2D plot with points and contour lines."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"DXF file not found at: {file_path}")
        
        # Load points from DXF
        processor = DXFProcessor(file_path)
        points = processor.extract_points_from_layer(layer)
        
        if not points:
            raise ValueError(f"No points found in layer: {layer}")
            
        # Create point cloud
        cloud = PointCloud()
        for point in points.values():
            # Points from DXF processor are already Point3d objects
            cloud.add_point(point)
            
        # Generate contours for selected intervals
        contour_data = []
        for interval, enabled in intervals.items():
            if enabled:
                level_contours = cloud.get_contour_lines([float(interval)])
                
                # Validate contours
                is_valid, error_msg = validate_contours(points, level_contours)
                if not is_valid:
                    raise ValueError(f"Invalid contours at {interval}m interval: {error_msg}")
                
                # Add contour lines to plot
                for level, paths in level_contours.items():
                    width = 1 if float(interval) == 0.25 else (2 if float(interval) == 0.5 else 3)
                    color = 'orange' if float(interval) == 0.25 else ('green' if float(interval) == 0.5 else 'blue')
                    opacity = 0.6 if float(interval) == 0.25 else (0.8 if float(interval) == 0.5 else 1.0)
                    
                    for path in paths:
                        x, y = zip(*[(p[0], p[1]) for p in path])
                        contour_data.append({
                            'type': 'scatter',
                            'x': list(x),
                            'y': list(y),
                            'mode': 'lines',
                            'line': {'width': width, 'color': color},
                            'opacity': opacity,
                            'name': f'Contour {float(interval)}m',
                            'showlegend': True
                        })
        
        # Create scatter plot of points
        x = [point.x for point in points.values()]
        y = [point.y for point in points.values()]
        z = [point.z for point in points.values()]
        
        point_data = {
            'type': 'scatter',
            'x': x,
            'y': y,
            'mode': 'markers',
            'marker': {
                'size': 5,
                'color': z,
                'colorscale': 'Viridis',
                'showscale': True,
                'colorbar': {'title': 'Elevation (m)'}
            },
            'name': 'Points',
            'text': [f'Elevation: {z_val:.2f}m' for z_val in z],
            'hoverinfo': 'text'
        }
        
        # Combine point and contour data
        data = [point_data] + contour_data
        
        # Create layout
        layout = {
            'title': 'Point Cloud with Contour Lines',
            'showlegend': True,
            'hovermode': 'closest',
            'yaxis': {
                'scaleanchor': 'x',
                'scaleratio': 1
            }
        }
        
        return {'data': data, 'layout': layout}
        
    except Exception as e:
        logger.error(f"Error creating plot: {str(e)}")
        raise

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/get_layers', methods=['POST'])
def get_layers():
    """Get available layers from uploaded DXF file."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
            
        if not file.filename.endswith('.dxf'):
            return jsonify({'error': 'Invalid file type. Please upload a DXF file.'})
            
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Get available layers
        processor = DXFProcessor(file_path)
        layers = processor.get_layers()
        
        return jsonify({'layers': layers})
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/process_layer', methods=['POST'])
def process_layer():
    """Process selected layer and generate visualization."""
    try:
        data = request.get_json()
        layer = data.get('layer')
        intervals = data.get('intervals', {})
        
        if not layer:
            return jsonify({'error': 'No layer selected'})
            
        # Find the most recently uploaded DXF file
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        dxf_files = [f for f in files if f.endswith('.dxf')]
        if not dxf_files:
            return jsonify({'error': 'No DXF file found'})
            
        latest_file = max(dxf_files, key=lambda f: os.path.getctime(os.path.join(app.config['UPLOAD_FOLDER'], f)))
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], latest_file)
        
        # Create plot
        plot_data = create_point_cloud_plot(file_path, layer, intervals)
        return jsonify(plot_data)
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
