"""Flask web server for point cloud visualization."""
from flask import Flask, render_template, jsonify
from dxf_processor import DXFProcessor
from point_cloud import PointCloud
import plotly.graph_objects as go
import numpy as np
import logging
import os
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get absolute path to DXF file
DXF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'plan-masse.dxf')

def create_point_cloud_plot():
    """Create a 2D plot with points and contour lines."""
    try:
        logger.info(f"Loading DXF file from: {DXF_FILE}")
        
        if not os.path.exists(DXF_FILE):
            logger.error(f"DXF file not found at: {DXF_FILE}")
            raise FileNotFoundError(f"DXF file not found at: {DXF_FILE}")
        
        # Load points from DXF
        processor = DXFProcessor(DXF_FILE)
        
        # Get available layers
        layers = processor.get_layers()
        logger.info(f"Available layers: {layers}")
        
        # Try to find the correct layer
        target_layer = None
        for layer in layers:
            if 'z value' in layer.lower():
                target_layer = layer
                break
        
        if not target_layer:
            logger.error("No layer containing 'z value' found")
            raise ValueError("No layer containing 'z value' found. Available layers: " + ", ".join(layers))
        
        logger.info(f"Using layer: {target_layer}")
        points = processor.extract_points_from_layer(target_layer)
        
        if not points:
            logger.error("No points found in the DXF file")
            raise ValueError("No points found in the DXF file")
        
        # Create point cloud and generate contours
        cloud = PointCloud()
        for point in points.values():
            cloud.add_point(point)
        
        # Generate contour lines at different intervals
        intervals = [0.25, 0.5, 1.0]  # 25cm, 50cm, and 1m intervals
        contours = cloud.get_contour_lines(intervals)
        
        # Create arrays for point plotting
        x_coords = []
        y_coords = []
        z_coords = []
        text_labels = []
        
        for point in points.values():
            x_coords.append(point.x)
            y_coords.append(point.y)
            z_coords.append(point.z)
            text_labels.append(f"ID: {point.id}<br>Z: {point.z:.2f}")
        
        logger.info(f"Loaded {len(points)} points for plotting")
        
        # Create data list for plotting
        data = []
        
        # Add contour lines with different styles for each interval
        colors = {0.25: 'rgba(255,127,14,0.6)', 0.5: 'rgba(44,160,44,0.8)', 1.0: 'rgba(31,119,180,1)'}
        widths = {0.25: 1, 0.5: 2, 1.0: 3}
        
        for interval in intervals:
            for level, paths in contours.items():
                # Find which interval this level belongs to
                level_interval = min(intervals, key=lambda x: level % x)
                if level_interval == interval:
                    for path in paths:
                        # Convert path to lists explicitly
                        x_path = [float(x) for x, _ in path]
                        y_path = [float(y) for _, y in path]
                        
                        data.append({
                            'type': 'scatter',
                            'mode': 'lines',
                            'x': x_path,
                            'y': y_path,
                            'line': {
                                'color': colors[interval],
                                'width': widths[interval]
                            },
                            'name': f'Contour {level:.2f}m',
                            'showlegend': True
                        })
        
        # Add points
        data.append({
            'type': 'scatter',
            'mode': 'markers',
            'x': x_coords,
            'y': y_coords,
            'marker': {
                'color': z_coords,
                'colorscale': 'Viridis',
                'showscale': True,
                'colorbar': {'title': 'Elevation (Z)'}
            },
            'text': text_labels,
            'hoverinfo': 'text',
            'name': 'Points'
        })
        
        # Calculate plot bounds with some padding
        x_range = [min(x_coords), max(x_coords)]
        y_range = [min(y_coords), max(y_coords)]
        x_padding = (x_range[1] - x_range[0]) * 0.05
        y_padding = (y_range[1] - y_range[0]) * 0.05
        
        # Create layout
        layout = {
            'title': 'Point Cloud with Elevation Contours',
            'xaxis': {'title': 'X', 'range': [x_range[0] - x_padding, x_range[1] + x_padding]},
            'yaxis': {'title': 'Y', 'range': [y_range[0] - y_padding, y_range[1] + y_padding], 'scaleanchor': 'x', 'scaleratio': 1},
            'showlegend': True,
            'legend': {'title': 'Contour Lines', 'yanchor': 'top', 'y': 0.99, 'xanchor': 'left', 'x': 1.05},
            'margin': {'l': 50, 'r': 150, 'b': 50, 't': 50},  # Increased right margin for legend
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'hovermode': 'closest'
        }
        
        # Create figure and return as JSON
        return jsonify({
            'data': data,
            'layout': layout
        })
        
    except Exception as e:
        logger.exception("Error creating plot")
        raise

@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')

@app.route('/get_plot_data')
def get_plot_data():
    """Return plot data as JSON."""
    try:
        plot_data = create_point_cloud_plot()
        return plot_data
    except Exception as e:
        logger.exception("Error in get_plot_data")
        return jsonify({
            'error': str(e),
            'data': [],
            'layout': {'title': 'Error loading plot: ' + str(e)}
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
