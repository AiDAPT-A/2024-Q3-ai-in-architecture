"""
A library of functions used in the third tutorial of the course:

T3 From Geometries to Graphs (ed. 24 - 25)

Contains the following functions:
- Make GeoPandas Dataframe contain only valid geometries
- Rotate (Multi-)Polygon - used for Extract access graph
- Polygon to list conversion - used for Extract access graph
- Extract access graph
- Draw (Multi-)Polygon on an axis
- Draw Graph on an axis

"""

# ----
# Imports
# ----

# General
import os
import random
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Dataframes
import pandas as pd
import geopandas as gpd

# Shapes
import shapely
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import rotate, scale

# Graphs
import networkx as nx

# ----
# Random color generator
# ----

def generate_color():
  return random.choice(list(mcolors.XKCD_COLORS.keys()))
  

# ----
# Dataframes
# ----

def make_gdf_valid(gdf):
  """Makes the geometries in the given GeoPandas dataframe valid by buffering them."""
  
  gdf['geometry'] = gdf['geometry'].apply(lambda poly: poly if poly.is_valid else poly.buffer(0))
  return gdf

# ----
# Shapes
# ----

def draw_shape(ax, poly, label=None, **kwargs):
    """Draws a Shapely Polygon or MultiPolygon with customizable fill and edge colors. """
  
    # Check if input is a MultiPolygon or Polygon
    if isinstance(poly, MultiPolygon):
        for subpoly in poly.geoms:
            x, y = subpoly.exterior.xy
            ax.fill(x, y, label=label, **kwargs)
    elif isinstance(poly, Polygon):
        x, y = poly.exterior.xy
        ax.fill(x, y, label=label, **kwargs)
    else:
        raise TypeError("Input must be a Polygon or MultiPolygon")


def rotate_polygon(rect: Polygon, scale_factor=0.5, angle=90):
    """Rotates and scales a polygonal shape. 
    By default the rotation is 90 degrees.
    By default the scale is 1 (no change)."""
  
    # Compute centroid (center point)
    centroid = rect.centroid

    # Scale the rectangle (relative to the centroid)
    scaled_rect = scale(rect, xfact=scale_factor, yfact=scale_factor, origin=centroid)

    # Rotate the scaled rectangle around its center
    rotated_rect = rotate(scaled_rect, angle, origin=centroid)

    return rotated_rect


def polygon_to_list(polygon: Polygon) -> list:
    """Converts a polygon into a list of coordinates."""
    return list(zip(*polygon.exterior.coords.xy))


# ----
# Graphs
# ----

def extract_access_graph(polygons_room,  # Polygons of the rooms
                                          polygons_door,   # Polygons of the doors
                                          names_room,  # Names of the rooms
                                          name_unit,  # Name of the unit
                                          t_passage=0.01,  # Threshold for distance [in m] to determine passage between rooms
                                          t_adjacent=0.5):  # Threshold for distance [in m] to determine room adjacency
    """Extract the access graph of a floor plan from the room shapes."""
                                            
    # Accumulation of NODES (i.e., the rooms)
    nodes = {}
    for key, (room, name) in enumerate(zip(polygons_room, names_room)):
        # For the center we use a representative point instead of the real center
        center = room.representative_point()
        nodes[key] = {
            "polygon": room,
            "room name": name,
            "centroid": np.array((center.xy[0][0], center.xy[1][0]))
        }

    # Accumulation of EDGES (i.e., room to room connectivity)
    edges = []
    for (i, v1), (j, v2) in combinations(enumerate(polygons_room), 2):

        edge = False

        # (Option 1) Passage (i.e., direct access := no wall in between)
        distance = v1.distance(v2)
        if distance < t_passage:
            edges.append([i, j, {'connectivity': 1}])
            edge = True

        # (Option 2) Door (i.e., door in between two rooms)
        else:
            for door in polygons_door:
                door_rotated = rotate_polygon(door, scale_factor=1)
                if door_rotated.intersection(v1) and door_rotated.intersection(v2):
                    # Adds the geometry of the door as well (slightly different from paper)
                    edges.append([i, j, {'connectivity': 1}])
                    edge = True
                else: continue

        #  (Option 3) Adjacent-only (i.e., wall in-between, but no door)
        if (not edge) &  (distance < t_adjacent):
            edges.append([i, j, {'connectivity': 0}])

    # Creation of the GRAPH
    G = nx.Graph()
    G.graph["apartment name"] = name_unit
    # Node attributes / features
    G.add_nodes_from([(u, v) for u, v in nodes.items()])
    # Edge attributes / features
    G.add_edges_from(edges)

    return G


def draw_graph(G, ax,
               node_color='black', edge_color='black', 
               node_size=20, edge_width=2):
    """Draw a graph ('G') on a given axis ('ax') with customizable coloring and dimensions."""
    
    # Create positions for the nodes
    pos = {n: d for n, d in G.nodes(data="centroid")}

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, 
                           ax=ax, 
                           node_size=node_size, node_color=node_color)

    # Draw the door edges (thick line)
    edges = [(u, v) for (u, v, d) in G.edges(data="connectivity") if d == 1]
    nx.draw_networkx_edges(G, pos, 
                           ax=ax, 
                           edgelist=edges, edge_color=edge_color, width=edge_width)
    
    # Draw the adjacent edges (dotted line)
    edges = [(u, v) for (u, v, d) in G.edges(data="connectivity") if d == 0]
    nx.draw_networkx_edges(G, pos, 
                           ax=ax, 
                           edgelist=edges, edge_color=edge_color, width=edge_width, 
                           style='--')
