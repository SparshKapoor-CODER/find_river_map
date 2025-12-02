import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt
import contextily as ctx

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
)

from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

# ─────────────────────────────
# Base paths (IMPORTANT FIX)
# ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

DATA_URL = "https://github.com/spatialthoughts/python-dataviz-web/releases/download/"
HYDRORIVERS_FILE = "hydrorivers_100.gpkg"
COUNTRIES_FILE = "ne_10m_admin_0_countries_ind.zip"

# ─────────────────────────────
# List of available countries (for dropdown)
# ─────────────────────────────
COUNTRIES = [
    'Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antarctica',
    'Antigua and Barbuda', 'Argentina', 'Armenia', 'Australia', 'Austria',
    'Azerbaijan', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium',
    'Belize', 'Benin', 'Bhutan', 'Bir Tawil', 'Bolivia',
    'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brazilian Island',
    'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cambodia',
    'Cameroon', 'Canada', 'Central African Republic', 'Chad', 'Chile', 'China',
    'Colombia', 'Comoros', 'Costa Rica', 'Croatia', 'Cuba', 'Cyprus',
    'Czechia', 'Democratic Republic of the Congo', 'Denmark', 'Djibouti',
    'Dominica', 'Dominican Republic', 'East Timor', 'Ecuador', 'Egypt',
    'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia',
    'Federated States of Micronesia', 'Fiji', 'Finland', 'France', 'Gabon',
    'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala',
    'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary',
    'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel',
    'Italy', 'Ivory Coast', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan',
    'Kenya', 'Kiribati', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon',
    'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg',
    'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta',
    'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico', 'Moldova',
    'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar',
    'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua',
    'Niger', 'Nigeria', 'North Korea', 'North Macedonia', 'Norway', 'Oman',
    'Pakistan', 'Palau', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru',
    'Philippines', 'Poland', 'Portugal', 'Qatar', 'Republic of Serbia',
    'Republic of the Congo', 'Romania', 'Russia', 'Rwanda',
    'Saint Kitts and Nevis', 'Saint Lucia',
    'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Saudi Arabia',
    'Scarborough Reef', 'Senegal', 'Seychelles', 'Sierra Leone', 'Singapore',
    'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa',
    'South Korea', 'South Sudan', 'Spain', 'Spratly Islands', 'Sri Lanka',
    'Sudan', 'Suriname', 'Sweden', 'Switzerland', 'Syria',
    'SÃ£o TomÃ© and Principe', 'Taiwan', 'Tajikistan', 'Thailand',
    'The Bahamas', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia',
    'Turkey', 'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine',
    'United Arab Emirates', 'United Kingdom',
    'United Republic of Tanzania', 'United States of America', 'Uruguay',
    'Uzbekistan', 'Vanuatu', 'Vatican', 'Venezuela', 'Vietnam', 'Yemen',
    'Zambia', 'Zimbabwe', 'eSwatini'
]

# ─────────────────────────────
# Helper: download if missing
# ─────────────────────────────
def download_and_save(url, filename):
    filepath = os.path.join(DATA_FOLDER, filename)
    if not os.path.exists(filepath):
        import requests
        with requests.get(url, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return filepath

# ─────────────────────────────
# Helpers: scale bar + north arrow
# ─────────────────────────────
def add_scale_bar(ax, length_m=100000, location='lower right'):
    fontprops = fm.FontProperties(size=10)
    scalebar = AnchoredSizeBar(
        ax.transData,
        length_m,
        f"{int(length_m/1000)} km",
        loc=location,
        pad=0.4,
        color='black',
        frameon=False,
        size_vertical=2,
        fontproperties=fontprops
    )
    ax.add_artist(scalebar)

def add_north_arrow(ax, x=0.95, y=0.15, size=0.06):
    ax.annotate(
        'N',
        xy=(x, y + size),
        xycoords='axes fraction',
        ha='center',
        va='center',
        fontsize=12,
        fontweight='bold'
    )
    ax.annotate(
        '',
        xy=(x, y + size),
        xytext=(x, y - size),
        xycoords='axes fraction',
        arrowprops=dict(facecolor='black', width=2, headwidth=8)
    )

# ─────────────────────────────
# Core map drawing
# ─────────────────────────────
def plot_stylish_map(country_gdf, rivers_gdf, country_name, stem_name):
    print("Reprojecting to Web Mercator...")
    country_3857 = country_gdf.to_crs(epsg=3857)
    rivers_3857 = rivers_gdf.to_crs(epsg=3857)

    print("Preparing figure...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=150)
    country_3857.plot(ax=ax, color="#f3efe0", edgecolor="#444444", linewidth=0.8)

    print("Computing river size metric...")
    if "UPLAND_SKM" in rivers_3857.columns:
        metric = rivers_3857["UPLAND_SKM"].fillna(0).astype(float)
    else:
        metric = rivers_3857.geometry.length

    if metric.max() > 0:
        rel = metric / metric.max()
    else:
        rel = metric

    cats = pd.cut(rel, bins=[-1, 0.33, 0.66, 1.0],
                  labels=["small", "medium", "large"])
    rivers_3857 = rivers_3857.assign(_size_cat=cats)

    print("Plotting rivers by size classes...")
    class_styles = {
        "small": {"linewidth": 0.8, "alpha": 0.8},
        "medium": {"linewidth": 1.8, "alpha": 0.9},
        "large": {"linewidth": 4.0, "alpha": 0.95},
    }
    for cat, style in class_styles.items():
        subset = rivers_3857[rivers_3857["_size_cat"] == cat]
        if len(subset):
            subset.plot(
                ax=ax,
                color="#1f78b4",
                linewidth=style["linewidth"],
                alpha=style["alpha"],
            )

    minx, miny, maxx, maxy = country_3857.total_bounds
    xbuffer = (maxx - minx) * 0.03 if (maxx - minx) > 0 else 10000
    ybuffer = (maxy - miny) * 0.03 if (maxy - miny) > 0 else 10000
    ax.set_xlim(minx - xbuffer, maxx + xbuffer)
    ax.set_ylim(miny - ybuffer, maxy + ybuffer)

    print("Adding basemap...")
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
        print("Warning: basemap failed:", e)
        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=zoom)
        except Exception as e2:
            print("Warning: fallback basemap failed:", e2)

    ax.set_title(f"Rivers of {country_name}", fontsize=18, fontweight="bold")
    ax.axis("off")

    add_scale_bar(ax, length_m=100000)
    add_north_arrow(ax)

    handles = [
        Line2D([0], [0], color="#1f78b4", lw=1, label="Small river"),
        Line2D([0], [0], color="#1f78b4", lw=2.5, label="Medium river"),
        Line2D([0], [0], color="#1f78b4", lw=5, label="Large river"),
    ]
    ax.legend(handles=handles, loc="upper right")

    png_name = f"{stem_name}.png"
    pdf_name = f"{stem_name}.pdf"

    png_path = os.path.join(OUTPUT_FOLDER, png_name)
    pdf_path = os.path.join(OUTPUT_FOLDER, pdf_name)

    print("Saving PNG:", png_path)
    fig.savefig(png_path, bbox_inches='tight', dpi=200)

    print("Saving PDF:", pdf_path)
    fig.savefig(pdf_path, bbox_inches='tight')

    plt.close(fig)
    return png_name, pdf_name

def generate_country_map(country_name: str):
    hydro_url = DATA_URL + "hydrosheds/" + HYDRORIVERS_FILE
    countries_url = DATA_URL + "naturalearth/" + COUNTRIES_FILE

    countries_fp = download_and_save(countries_url, COUNTRIES_FILE)
    hydrorivers_fp = download_and_save(hydro_url, HYDRORIVERS_FILE)

    country_gdf = gpd.read_file(countries_fp)
    country_input = country_name.strip()

    selected_country = country_gdf[
        (country_gdf["SOVEREIGNT"].str.lower() == country_input.lower())
        & (country_gdf["TYPE"] != "Dependency")
    ]

    if selected_country.empty:
        mask = country_gdf["SOVEREIGNT"].str.lower().str.contains(country_input.lower())
        selected_country = country_gdf[mask & (country_gdf["TYPE"] != "Dependency")]

    if selected_country.empty:
        raise ValueError(f"Country '{country_name}' not found. Try a different name.")

    selected_3857 = selected_country.to_crs(epsg=3857)
    parts = []
    for geom in selected_3857.geometry:
        if geom is None:
            continue
        if geom.geom_type in ["MultiPolygon", "GeometryCollection"]:
            for part in geom.geoms:
                parts.append(part)
        else:
            parts.append(geom)

    if len(parts) == 0:
        mainland_mask = selected_country.geometry.unary_union
        mainland_gdf = selected_country
    else:
        areas = [g.area for g in parts]
        largest_idx = int(np.argmax(areas))
        largest = parts[largest_idx]
        mainland_3857 = gpd.GeoSeries([largest], crs=3857)
        mainland_mask = mainland_3857.to_crs(selected_country.crs).geometry.iloc[0]
        mainland_gdf = gpd.GeoDataFrame(geometry=[mainland_mask], crs=selected_country.crs)

    river_gdf = gpd.read_file(hydrorivers_fp, mask=mainland_mask)
    stem = f"rivers_of_{country_name.lower().replace(' ', '_')}"
    return plot_stylish_map(mainland_gdf, river_gdf, country_name, stem)

# ─────────────────────────────
# Flask app
# ─────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-secret"
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

@app.route("/", methods=["GET"])
def index():
    # default: show India pre-selected
    return render_template("index.html", countries=COUNTRIES, country="India")

@app.route("/generate", methods=["POST"])
def generate():
    country = request.form.get("country", "").strip()
    if not country:
        flash("Please select a country.")
        return redirect(url_for("index"))

    try:
        png_name, pdf_name = generate_country_map(country)
        return render_template(
            "index.html",
            countries=COUNTRIES,
            country=country,
            map_png=png_name,
            map_pdf=pdf_name,
        )
    except Exception as e:
        flash(str(e))
        return redirect(url_for("index"))

@app.route("/output/<path:filename>")
def output_file(filename):
    # for <img> display
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

@app.route("/download/png/<path:filename>")
def download_png(filename):
    return send_from_directory(
        app.config["OUTPUT_FOLDER"],
        filename,
        as_attachment=True,
        mimetype="image/png",
    )

@app.route("/download/pdf/<path:filename>")
def download_pdf(filename):
    return send_from_directory(
        app.config["OUTPUT_FOLDER"],
        filename,
        as_attachment=True,
        mimetype="application/pdf",
    )

if __name__ == "__main__":
    app.run(debug=True)
