---
title: What's near me? Space Filling Curves and Radix Tree in geospatial applications
layout: single
---

In geospatial applications, indexing and searching for things around a given point is the most common operation. Since there is latitude and longitude information, a regular look up table kind of approach is difficult and slow. The alternate approach is to use space filling curves. 

![Space Filling Curve (Source: Wikipedia)](/assets/images/space_filling_curve.png)

We can divide the area by smaller squares and draw a non-overlapping curve passing through the squares. Each turn in the curve can be represented by a continuous value (one dimensional coordinate).  Smaller the square, the more accurate the point location in the curve will be.  For more details on Space Filling Curves, please watch the video below. 

{% include video id="DgL49CFlM" provider="youtube" %}

## S2 Geometry
![S2Geometry (Source: s2geometry)](/assets/images/s2geometry.png)

S2Geometry uses a radix tree implementation along with geometrical operations (near, point in line, point in polygon, overlap etc).  
##  Openstreetmap data and spatial indexing
Openstreetmap is the wikipedia of maps and geospatial information where everything is community contributed. As a developer if I have to get geospatial data (how many malls near a given location, which is the nearest highway etc), the first place to go to is openstreetmap. Being free (as in free beer), it is doing an amazing job on serving such queries.  But the best way to leverage the data is to take a dump and transform to a R-Tree. 

I tried to look at the python ecosystem with openstreetmap and database creation and indexing (without postgres or other spatial support DB) is almost non-existant. Geopandas supports only smaller datasets and performance is too low. 

### The OSM_Roads Package
I wrote a small script which can take openstreetmap dumps, convert to a R-Tree using geohash (one of the implementations of space filling curve for latitude, longitude). The package currently can do map matching of point to a road. 

![S2Geometry (Source: s2geometry)](/assets/images/sample_output_osm_roads.png)

#### Github Link 
[OSM Roads](https://github.com/sahyagiri/osm_roads)
