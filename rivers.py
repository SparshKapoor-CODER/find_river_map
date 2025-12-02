import geopandas as gpd
import matplotlib.pyplot as plt
import os
import pandas as pd
import requests
import zipfile
import contextily as ctx
import numpy as np
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

data_folder = 'find_river_map/data'
output_folder = 'find_river_map/output'

if not os.path.exists(data_folder):
    os.mkdir(data_folder)
if not os.path.exists(output_folder):
    os.mkdir(output_folder)


def download_and_extract(url, extract_to):
    filename = os.path.join(data_folder, os.path.basename(url))
    if not os.path.exists(filename):
        with requests.get(url, stream=True, allow_redirects=True) as r:
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    print('Downloaded', filename)


data_url = "https://github.com/spatialthoughts/python-dataviz-web/releases/download/"

# This is a subset of the main HydroRivers dataset of all
# rivers having `UPLAND_SKM` value  greater than 100 sq. km.

hydrorivers_file = "hydrorivers_100.gpkg"
hydrorivers_url = data_url + 'hydrosheds/'
countries_file = 'ne_10m_admin_0_countries_ind.zip'
countries_url = data_url + 'naturalearth/'


download_and_extract(hydrorivers_url + hydrorivers_file, data_folder)
download_and_extract(countries_url + countries_file, data_folder)


countries_filepath = os.path.join(data_folder, countries_file)


country_gdf = gpd.read_file(countries_filepath)
print(sorted(country_gdf.SOVEREIGNT.unique()))

country = "italy"  # change country name as needed, e.g., "india", "brazil", "australia"

# Select country (case-insensitive). Try exact match first, then contains.
country_input = country.strip()
selected_country = country_gdf[
    (country_gdf["SOVEREIGNT"].str.lower() == country_input.lower()) &
    (country_gdf["TYPE"] != "Dependency")
]
if selected_country.empty:
    # try a contains match
    mask = country_gdf["SOVEREIGNT"].str.lower().str.contains(country_input.lower())
    selected_country = country_gdf[mask & (country_gdf["TYPE"] != "Dependency")]

if selected_country.empty:
    print(f"ERROR: country '{country}' not found in countries dataset. Available samples:\n", 
          list(country_gdf.SOVEREIGNT.unique())[:10])
    import sys
    sys.exit(1)

# Determine mainland: pick the largest polygon (in projected units) when country has multiple parts
print('Computing mainland (largest polygon) if country has multiple parts...')
selected_3857 = selected_country.to_crs(epsg=3857)
parts = []
for geom in selected_3857.geometry:
    if geom is None:
        continue
    if geom.geom_type == 'MultiPolygon' or geom.geom_type == 'GeometryCollection':
        for part in geom.geoms:
            parts.append(part)
    else:
        parts.append(geom)

if len(parts) == 0:
    # fallback: use the original selected_country as mask
    mainland_mask = selected_country.geometry.unary_union
    mainland_gdf = selected_country
    print('No multipart geometry found; using full country geometry as mainland.')
else:
    areas = [g.area for g in parts]
    largest_idx = int(np.argmax(areas))
    largest = parts[largest_idx]
    # convert largest back to original CRS and force mainland usage
    mainland_3857 = gpd.GeoSeries([largest], crs=3857)
    mainland_mask = mainland_3857.to_crs(selected_country.crs).geometry.iloc[0]
    mainland_gdf = gpd.GeoDataFrame(geometry=[mainland_mask], crs=selected_country.crs)
    print('Using largest polygon as mainland mask (forced mainland-only view).')


# make the map like a terrain and produce prettier static + interactive maps
hydrorivers_filepath = os.path.join(data_folder, hydrorivers_file)
# Read rivers clipped to mainland_mask (focus on mainland for small/multi-part countries)
river_gdf = gpd.read_file(hydrorivers_filepath, mask=mainland_mask)


def _add_scale_bar(ax, length_m=100000, location='lower right'):
    fontprops = fm.FontProperties(size=10)
    scalebar = AnchoredSizeBar(ax.transData,
                               length_m,
                               f"{int(length_m/1000)} km",
                               loc=location,
                               pad=0.4,
                               color='black',
                               frameon=False,
                               size_vertical=2,
                               fontproperties=fontprops)
    ax.add_artist(scalebar)


def _add_north_arrow(ax, x=0.95, y=0.15, size=0.06):
    ax.annotate('N', xy=(x, y + size), xycoords='axes fraction',
                ha='center', va='center', fontsize=12, fontweight='bold')
    ax.annotate('', xy=(x, y + size), xytext=(x, y - size), xycoords='axes fraction',
                arrowprops=dict(facecolor='black', width=2, headwidth=8))


def plot_stylish_map(country_gdf, rivers_gdf, country_name, out_folder=output_folder):
    print('Reprojecting to Web Mercator...')
    country_3857 = country_gdf.to_crs(epsg=3857)
    rivers_3857 = rivers_gdf.to_crs(epsg=3857)

    print('Preparing figure...')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=150)
    # soft land color
    country_3857.plot(ax=ax, color='#f3efe0', edgecolor='#444444', linewidth=0.8)

    # Compute stroke widths based on UPLAND_SKM if available, else geometry length
    print('Computing river size metric...')
    if 'UPLAND_SKM' in rivers_3857.columns:
        metric = rivers_3857['UPLAND_SKM'].fillna(0).astype(float)
    else:
        metric = rivers_3857.geometry.length

    # Normalize and categorize into three classes to speed up plotting
    if metric.max() > 0:
        rel = metric / metric.max()
    else:
        rel = metric

    # create categories: small (<0.33), medium (0.33-0.66), large (>0.66)
    cats = pd.cut(rel, bins=[-1, 0.33, 0.66, 1.0], labels=['small', 'medium', 'large'])
    rivers_3857 = rivers_3857.assign(_size_cat=cats)

    print('Plotting rivers by size classes (fast grouped plotting)...')
    # Plot each class with a representative linewidth (fast vectorized plot)
    class_styles = {
        'small': {'linewidth': 0.8, 'alpha': 0.8},
        'medium': {'linewidth': 1.8, 'alpha': 0.9},
        'large': {'linewidth': 4.0, 'alpha': 0.95},
    }

    for cat, style in class_styles.items():
        subset = rivers_3857[rivers_3857['_size_cat'] == cat]
        if len(subset):
            subset.plot(ax=ax, color='#1f78b4', linewidth=style['linewidth'], alpha=style['alpha'])

    # Set axis limits explicitly to country bounds with a small buffer to avoid huge extents
    minx, miny, maxx, maxy = country_3857.total_bounds
    xbuffer = (maxx - minx) * 0.03 if (maxx - minx) > 0 else 10000
    ybuffer = (maxy - miny) * 0.03 if (maxy - miny) > 0 else 10000
    ax.set_xlim(minx - xbuffer, maxx + xbuffer)
    ax.set_ylim(miny - ybuffer, maxy + ybuffer)

    # Add hillshade-ish basemap (terrain)
    print('Adding basemap (may take a few seconds)...')
    # Choose a reasonable zoom level for basemap based on country width (meters)
    width_m = maxx - minx
    if width_m <= 0:
        zoom = 6
    elif width_m > 5_000_000:
        zoom = 4
    elif width_m > 2_000_000:
        zoom = 5
    elif width_m > 1_000_000:
        zoom = 6
    elif width_m > 500_000:
        zoom = 7
    elif width_m > 100_000:
        zoom = 8
    else:
        zoom = 10

    try:
        ctx.add_basemap(ax, source=ctx.providers.Stamen.TerrainBackground, zoom=zoom)
    except Exception as e:
        print('Warning: basemap failed:', e)
        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=zoom)
        except Exception as e2:
            print('Warning: fallback basemap failed:', e2)

    # Title and styling
    ax.set_title(f'Rivers of {country_name}', fontsize=18, fontweight='bold')
    ax.axis('off')

    # Scale bar and north arrow
    _add_scale_bar(ax, length_m=100000)
    _add_north_arrow(ax)

    # Create a small legend for river stroke widths
    handles = [Line2D([0], [0], color='#1f78b4', lw=1, label='Small river'),
               Line2D([0], [0], color='#1f78b4', lw=2.5, label='Medium river'),
               Line2D([0], [0], color='#1f78b4', lw=5, label='Large river')]
    ax.legend(handles=handles, loc='upper right')

    plt.show()

    # Save PNG
    out_png = os.path.join(out_folder, f'rivers_of_{country_name.lower().replace(" ","_")}.png')
    print('Saving PNG to', out_png)

    # Safe save: try high quality first, but handle extremely large image errors by retrying with lower DPI
    try:
        fig.savefig(out_png, bbox_inches='tight', dpi=300)
    except ValueError as e:
        print('Warning: high-res save failed:', e)
        try:
            print('Retrying without bbox_inches and lower DPI (150)...')
            fig.savefig(out_png, dpi=150)
        except Exception as e2:
            print('Retry failed:', e2)
            try:
                print('Final retry: reduce figure size and DPI (10x6 inches, 100 DPI)...')
                fig.set_size_inches(10, 6)
                fig.savefig(out_png, dpi=100)
            except Exception as e3:
                print('All save attempts failed:', e3)
                raise
    finally:
        plt.close(fig)
    print('Saved', out_png)


# Interactive (HTML) export removed — static PNG only now.


# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

plot_stylish_map(mainland_gdf, river_gdf, country)