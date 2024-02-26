# Quadtree Index

Set of classes that define a quadtree index.

## Flexible Grid

The nestedgrid class specifies a coordinate reference system (CRS) and the hierarchical set of nested grids built on top of it
- e.g. the origin of the grid and resolution of the highest-resolution level

## Morton Curve (z-order index)

Defines the order used to traverse the Quadtree

## Quadtree implementation

Provides functions to navigate the quadtree
- e.g. parent, children, breadth-first search, depth-first search, indexing polygons, gridding

## Partitioning based on Quadtree Index

Functionality to partition data indexed with the quadtree