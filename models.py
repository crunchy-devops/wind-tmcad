"""Database models for the point cloud application."""
from sqlalchemy import create_engine, Column, Integer, Float, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()

class Project(Base):
    """Project model to store project information."""
    __tablename__ = 'projects'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    
    # Relationships
    points = relationship('PointCloud', back_populates='project', cascade='all, delete-orphan')
    breaklines = relationship('Breakline', back_populates='project', cascade='all, delete-orphan')
    triangles = relationship('DelaunayTriangle', back_populates='project', cascade='all, delete-orphan')
    contour_lines = relationship('ContourLine', back_populates='project', cascade='all, delete-orphan')

class PointCloud(Base):
    """PointCloud model to store point cloud data."""
    __tablename__ = 'pointcloud'
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    z = Column(Float, nullable=False)
    
    # Relationships
    project = relationship('Project', back_populates='points')
    breakline_starts = relationship('Breakline', foreign_keys='Breakline.start_point3d_id', back_populates='start_point')
    breakline_ends = relationship('Breakline', foreign_keys='Breakline.end_point3d_id', back_populates='end_point')

class Breakline(Base):
    """Breakline model to store breakline data."""
    __tablename__ = 'breaklines'
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    start_point3d_id = Column(Integer, ForeignKey('pointcloud.id'), nullable=False)
    end_point3d_id = Column(Integer, ForeignKey('pointcloud.id'), nullable=False)
    
    # Relationships
    project = relationship('Project', back_populates='breaklines')
    start_point = relationship('PointCloud', foreign_keys=[start_point3d_id], back_populates='breakline_starts')
    end_point = relationship('PointCloud', foreign_keys=[end_point3d_id], back_populates='breakline_ends')

class DelaunayTriangle(Base):
    """DelaunayTriangle model to store triangulation data."""
    __tablename__ = 'delaunay_triangles'
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    point3d_id1 = Column(Integer, ForeignKey('pointcloud.id'), nullable=False)
    point3d_id2 = Column(Integer, ForeignKey('pointcloud.id'), nullable=False)
    point3d_id3 = Column(Integer, ForeignKey('pointcloud.id'), nullable=False)
    
    # Relationships
    project = relationship('Project', back_populates='triangles')
    point1 = relationship('PointCloud', foreign_keys=[point3d_id1])
    point2 = relationship('PointCloud', foreign_keys=[point3d_id2])
    point3 = relationship('PointCloud', foreign_keys=[point3d_id3])

class ContourLine(Base):
    """ContourLine model to store contour line data."""
    __tablename__ = 'contour_lines'
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    z = Column(Float, nullable=False)
    
    # Relationships
    project = relationship('Project', back_populates='contour_lines')

# Create database engine and session
engine = create_engine('sqlite:///point_cloud.db')
Session = sessionmaker(bind=engine)

def init_db():
    """Initialize the database by creating all tables."""
    Base.metadata.create_all(engine)
