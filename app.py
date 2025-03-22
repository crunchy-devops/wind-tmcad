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
    """Create a 2D scatter plot of the point cloud with Z values as colors."""
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
        
        # Create arrays for plotting
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
        
        # Create scatter plot
        scatter = go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=8,
                color=z_coords,
                colorscale='Viridis',
                colorbar=dict(
                    title='Elevation (Z)',
                    titleside='right'
                ),
                showscale=True,
                opacity=0.8
            ),
            text=text_labels,
            hoverinfo='text'
        )
        
        # Calculate plot bounds with some padding
        x_range = [min(x_coords), max(x_coords)]
        y_range = [min(y_coords), max(y_coords)]
        x_padding = (x_range[1] - x_range[0]) * 0.05
        y_padding = (y_range[1] - y_range[0]) * 0.05
        
        # Create layout
        layout = go.Layout(
            title='Point Cloud Elevation Map',
            xaxis=dict(
                title='X',
                range=[x_range[0] - x_padding, x_range[1] + x_padding]
            ),
            yaxis=dict(
                title='Y',
                range=[y_range[0] - y_padding, y_range[1] + y_padding],
                scaleanchor='x',  # Make the scale of x and y axes equal
                scaleratio=1
            ),
            showlegend=False,
            margin=dict(l=50, r=50, b=50, t=50),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='closest'
        )
        
        # Create figure and return as JSON
        fig = go.Figure(data=[scatter], layout=layout)
        return jsonify(fig.to_dict())
        
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
    app.run(debug=True, port=5000)
