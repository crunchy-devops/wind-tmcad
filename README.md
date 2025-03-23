# Wind TMCAD

A Python application for processing and visualizing terrain point cloud data from DXF files, with a focus on wind turbine micro-siting analysis.

## Features

### Point Cloud Processing
- Import 3D point data from DXF files
- Memory-efficient point cloud storage using HDF5
- Spatial indexing with KD-trees for fast neighbor queries
- Point interpolation using various methods:
  - Inverse Distance Weighting (IDW)
  - Barycentric interpolation
  - Natural Neighbor interpolation

### Elevation Analysis
- Generate isometric contour lines at multiple intervals:
  - 25 cm (fine detail)
  - 50 cm (medium detail)
  - 1 meter (major contours)
- Interactive 2D visualization with:
  - Points colored by elevation
  - Contour lines with different styles per interval
  - Hover information showing exact elevations
  - Equal aspect ratio for accurate representation

### Web Interface
- Flask-based web server
- Interactive visualization using Plotly
- File upload for DXF processing
- Real-time point cloud and contour display
- Responsive layout with legend and controls

## Installation

1. Clone the repository:
```bash
git clone https://github.com/crunchy-devops/wind-tmcad.git
cd wind-tmcad
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the web server:
```bash
python app.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

3. Upload a DXF file containing point cloud data

4. View and interact with the visualization:
   - Toggle contour levels using the legend
   - Hover over points to see exact elevations
   - Use the modebar for zoom, pan, and other controls
   - Download the plot as PNG if needed

## Technical Details

### Point Cloud Class
- Efficient point storage using dictionary mapping
- Spatial indexing with scipy's cKDTree
- Support for various interpolation methods
- HDF5-based file storage for large datasets

### Contour Generation
- Uses matplotlib's contour generator
- Cubic interpolation for smooth contours
- Grid-based approach with customizable resolution
- Multiple elevation intervals with distinct styling

### Visualization
- Points:
  - Colored using Viridis colorscale
  - Size and opacity optimized for clarity
  - Hover information showing coordinates and elevation
- Contours:
  - 25cm: Thin orange lines (semi-transparent)
  - 50cm: Medium green lines (more opaque)
  - 1m: Thick blue lines (fully opaque)
  - Interactive legend for toggling visibility

## Dependencies
- Flask: Web framework
- Plotly: Interactive visualization
- NumPy: Numerical operations
- SciPy: Spatial operations and interpolation
- Matplotlib: Contour generation
- ezdxf: DXF file processing
- h5py: HDF5 file handling

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.
