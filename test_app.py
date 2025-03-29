"""Tests for Flask web application."""
import os
import json
import pytest
from app import app, validate_contours, create_point_cloud_plot
from point3d import Point3d
from werkzeug.datastructures import FileStorage
import numpy as np
import tempfile
import shutil
from unittest.mock import patch, MagicMock

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    app.config['UPLOAD_FOLDER'] = 'test_uploads'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    with app.test_client() as client:
        yield client
        
    # Cleanup test uploads
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
    os.rmdir(app.config['UPLOAD_FOLDER'])

@pytest.fixture
def sample_points():
    """Create sample points for testing."""
    points = {
        1: Point3d(id=1, x=0.0, y=0.0, z=10.0),
        2: Point3d(id=2, x=10.0, y=0.0, z=20.0),
        3: Point3d(id=3, x=10.0, y=10.0, z=30.0),
        4: Point3d(id=4, x=0.0, y=10.0, z=40.0)
    }
    return points

@pytest.fixture
def sample_contours():
    """Create sample contour data for testing."""
    return {
        "20.0": [
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 10.0],
                [0.0, 10.0],
                [0.0, 0.0]
            ]
        ]
    }

@pytest.fixture
def mock_dxf_file():
    """Create a mock DXF file for testing."""
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'test.dxf')
    with open(temp_file, 'w') as f:
        f.write('Mock DXF content')
    yield temp_file
    shutil.rmtree(temp_dir)

def test_index_route(client):
    """Test the index route."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'<!DOCTYPE html>' in response.data

def test_get_layers_no_file(client):
    """Test get_layers route without file."""
    response = client.post('/get_layers')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'No file uploaded'

def test_get_layers_empty_filename(client):
    """Test get_layers route with empty filename."""
    response = client.post('/get_layers', data={
        'file': (FileStorage(stream=None, filename=''), '')
    })
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'No file selected'

def test_get_layers_invalid_extension(client):
    """Test get_layers route with invalid file extension."""
    response = client.post('/get_layers', data={
        'file': (FileStorage(stream=None, filename='test.txt'), 'test.txt')
    })
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'Invalid file type. Please upload a DXF file.'

def test_process_layer_no_layer(client):
    """Test process_layer route without layer selection."""
    response = client.post('/process_layer', 
                         json={'layer': None, 'intervals': {}})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'No layer selected'

def test_process_layer_no_dxf(client):
    """Test process_layer route without DXF file."""
    response = client.post('/process_layer', 
                         json={'layer': 'test_layer', 'intervals': {}})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'No DXF file found'

@patch('app.DXFProcessor')
def test_process_layer_success(mock_dxf, client, mock_dxf_file, sample_points):
    """Test successful layer processing."""
    # Setup mock
    mock_instance = MagicMock()
    mock_instance.extract_points_from_layer.return_value = sample_points
    mock_dxf.return_value = mock_instance
    
    # Create test DXF file
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    shutil.copy(mock_dxf_file, os.path.join(app.config['UPLOAD_FOLDER'], 'test.dxf'))
    
    response = client.post('/process_layer', 
                         json={'layer': 'test_layer', 'intervals': {'10': True}})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'data' in data
    assert 'layout' in data

def test_validate_contours_not_enough_points(sample_points):
    """Test contour validation with insufficient points."""
    points = {1: sample_points[1]}  # Only one point
    contours = {}
    valid, msg = validate_contours(points, contours)
    assert not valid
    assert "Not enough points" in msg

def test_validate_contours_invalid_level(sample_points, sample_contours):
    """Test contour validation with invalid contour level."""
    # Add a contour level outside the terrain bounds
    invalid_contours = sample_contours.copy()
    invalid_contours["100.0"] = sample_contours["20.0"]
    
    valid, msg = validate_contours(sample_points, invalid_contours)
    assert not valid
    assert "outside of terrain bounds" in msg

def test_validate_contours_unclosed(sample_points):
    """Test contour validation with unclosed contour."""
    # Create an unclosed contour path
    unclosed_contours = {
        "20.0": [
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 10.0],
                [0.0, 10.0]
                # Missing closing point
            ]
        ]
    }
    
    valid, msg = validate_contours(sample_points, unclosed_contours)
    assert not valid
    assert "Unclosed contour" in msg

def test_validate_contours_valid(sample_points, sample_contours):
    """Test contour validation with valid data."""
    valid, msg = validate_contours(sample_points, sample_contours)
    assert valid
    assert msg is None

def test_save_project_missing_data(client):
    """Test save_project route with missing data."""
    response = client.post('/save_project', json={})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'error' in data

@patch('app.Session')
def test_save_project_success(mock_session, client):
    """Test successful project saving."""
    # Setup mock session
    session_instance = MagicMock()
    mock_session.return_value = session_instance
    
    project_data = {
        'projectName': 'Test Project',
        'points': [{'id': 1, 'x': 0.0, 'y': 0.0, 'z': 0.0}],
        'breaklines': [{'start': 1, 'end': 1}],
        'triangles': [{'p1': 1, 'p2': 1, 'p3': 1}],
        'contours': [{'x': 0.0, 'y': 0.0, 'z': 0.0}]
    }
    
    response = client.post('/save_project', json=project_data)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'success' in data
    assert data['success'] is True

@patch('app.DXFProcessor')
@patch('os.path.exists')
def test_create_point_cloud_plot(mock_exists, mock_dxf, sample_points):
    """Test point cloud plot creation."""
    # Setup mocks
    mock_exists.return_value = True
    mock_instance = MagicMock()
    mock_instance.extract_points_from_layer.return_value = sample_points
    mock_dxf.return_value = mock_instance
    
    plot_data = create_point_cloud_plot('test.dxf', 'test_layer', {'10': True})
    assert 'data' in plot_data
    assert 'layout' in plot_data
    assert len(plot_data['data']) > 0

def test_create_point_cloud_plot_no_file():
    """Test plot creation with missing file."""
    with pytest.raises(FileNotFoundError):
        create_point_cloud_plot('nonexistent.dxf', 'test_layer', {})
