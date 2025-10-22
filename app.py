# app_streamlit_cobb.py
import streamlit as st
import tempfile
import os
import joblib
import numpy as np
import pandas as pd
import open3d as o3d
import plotly.graph_objects as go

# Optional extras - if not installed ML volume may be None
try:
    import trimesh
except Exception:
    trimesh = None

# --------------------
# Config - adjust paths if needed
# --------------------
MODEL_PATH = r"D:\Professional\Projects\Aiims\Gait\Data\project work\Surface topography data\All_cropped\svm_cobb_model.pkl"

# Default XY/Z used by your crop function
DEFAULT_X_RANGE = (-0.6, 0.5)
DEFAULT_Y_RANGE = (-0.6, 0.73)
DEFAULT_Z_FRAC = (0.15, 1.0)  # top 85% by default (you used variations earlier)

# --------------------
# Minimal crop + feature extractor (uses your logic)
# If you already have implementations, import them instead and remove below.
# --------------------
def crop_torso_freeze_xy_dynamic_z(
    stl_path,
    x_range=DEFAULT_X_RANGE,
    y_range=DEFAULT_Y_RANGE,
    z_frac=DEFAULT_Z_FRAC,
    visualize_bbox=False
):
    mesh = o3d.io.read_triangle_mesh(stl_path)
    if mesh.is_empty():
        raise ValueError("Mesh is empty or failed to load.")

    min_bound = np.array(mesh.get_min_bound())
    max_bound = np.array(mesh.get_max_bound())
    dims = max_bound - min_bound

    z_min = min_bound[2] + z_frac[0] * dims[2]
    z_max = min_bound[2] + z_frac[1] * dims[2]

    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(x_range[0], y_range[0], z_min),
        max_bound=(x_range[1], y_range[1], z_max)
    )

    torso_mesh = mesh.crop(bbox)
    if torso_mesh.is_empty():
        raise ValueError("Cropped mesh is empty. Try relaxing Z fraction or XY ranges.")

    torso_mesh.remove_duplicated_vertices()
    torso_mesh.remove_duplicated_triangles()
    torso_mesh.remove_degenerate_triangles()
    try:
        torso_mesh.remove_non_manifold_edges()
    except Exception:
        pass

    torso_mesh.compute_vertex_normals()
    torso_mesh.compute_triangle_normals()

    return torso_mesh, (x_range, y_range, (z_min, z_max))

# feature extractor (must match training features)
def extract_features_from_mesh(mesh, slice_percentiles=[0.95, 0.9, 0.85, 0.8, 0.75]):
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)

    if verts.size == 0:
        raise ValueError("Mesh contains no vertices.")

    aabb = mesh.get_axis_aligned_bounding_box()
    bbox_min = aabb.min_bound
    bbox_max = aabb.max_bound
    bbox_dims = bbox_max - bbox_min

    num_vertices = int(verts.shape[0])
    num_triangles = int(tris.shape[0])
    surface_area = float(mesh.get_surface_area())
    centroid = verts.mean(axis=0)
    height = float(bbox_dims[2])

    # volume via trimesh if available
    volume = None
    is_watertight = None
    if trimesh is not None:
        try:
            t = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
            volume = float(getattr(t, "volume", 0.0))
            is_watertight = bool(getattr(t, "is_watertight", False))
        except Exception:
            volume = None
            is_watertight = None

    # slice circumferences
    slice_info = {}
    zmin, zmax = bbox_min[2], bbox_max[2]
    slab_thickness = max(1e-6, 0.005 * max(1e-6, height))
    for p in slice_percentiles:
        zq = zmin + p * (zmax - zmin)
        mask = (verts[:,2] >= (zq - slab_thickness)) & (verts[:,2] <= (zq + slab_thickness))
        pts2d = verts[mask][:, :2]
        circ = 0.0
        if pts2d.shape[0] >= 3:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pts2d)
                hull_pts = pts2d[hull.vertices]
                diffs = np.diff(np.vstack([hull_pts, hull_pts[0]]), axis=0)
                circ = float(np.sqrt((diffs**2).sum(axis=1)).sum())
            except Exception:
                circ = 0.0
        slice_info[f"slice_{int(p*100)}"] = circ

    features = {
        "num_vertices": num_vertices,
        "num_triangles": num_triangles,
        "surface_area": surface_area,
        "volume": volume,
        "is_watertight": is_watertight,
        "bbox_dx": float(bbox_dims[0]),
        "bbox_dy": float(bbox_dims[1]),
        "bbox_dz": float(bbox_dims[2]),
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        "height": height,
    }
    # add slices
    features.update(slice_info)
    return features

#### Visualise mesh

def mesh_to_plotly(mesh, color='lightblue', opacity=1.0):
    """
    Convert an open3d TriangleMesh to a plotly go.Figure (Mesh3d).
    """
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    if verts.size == 0 or tris.size == 0:
        raise ValueError("Mesh has no vertices or triangles")

    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    i, j, k = tris[:, 0], tris[:, 1], tris[:, 2]

    mesh3d = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color=color,
        opacity=opacity,
        flatshading=True,
        hoverinfo="skip"
    )

    # compute axis ranges to keep aspect ratio sensible
    x_range = [x.min(), x.max()]
    y_range = [y.min(), y.max()]
    z_range = [z.min(), z.max()]
    # center
    xc = 0.5 * (x_range[0] + x_range[1])
    yc = 0.5 * (y_range[0] + y_range[1])
    zc = 0.5 * (z_range[0] + z_range[1])
    max_span = max(x_range[1]-x_range[0], y_range[1]-y_range[0], z_range[1]-z_range[0], 1e-6)

    layout = go.Layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z= (z_range[1]-z_range[0]) / (max_span + 1e-12))
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig = go.Figure(data=[mesh3d], layout=layout)
    return fig


# --------------------
# Helper to load model
# --------------------
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        return None
    data = joblib.load(model_path)
    # Expect dict with keys model, scaler, features
    model = data.get("model") if isinstance(data, dict) else data
    scaler = data.get("scaler") if isinstance(data, dict) else None
    feat_names = data.get("features") if isinstance(data, dict) else None
    return {"model": model, "scaler": scaler, "features": feat_names}

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="Cobb Angle from 3D STL", layout="centered")
st.title("Predict Cobb angle from 3D Scan")

st.markdown(
    """
Upload a **cropped** or **uncropped** STL. The app will:
1. Crop the torso.
2. Extract geometric features.
3. Load the trained SVM model and predict Cobb angle.

 """)

uploaded = st.file_uploader("Upload STL file", type=["stl", "ply", "obj"])
st.markdown("**Cropping parameters (you can tweak)**")
col1, col2, col3 = st.columns(3)
with col1:
    x_min = st.number_input("x_min", value=DEFAULT_X_RANGE[0], format="%.3f")
    x_max = st.number_input("x_max", value=DEFAULT_X_RANGE[1], format="%.3f")
with col2:
    y_min = st.number_input("y_min", value=DEFAULT_Y_RANGE[0], format="%.3f")
    y_max = st.number_input("y_max", value=DEFAULT_Y_RANGE[1], format="%.3f")
with col3:
    z_low_frac = st.number_input("z_low_frac", value=DEFAULT_Z_FRAC[0], format="%.3f")
    z_high_frac = st.number_input("z_high_frac", value=DEFAULT_Z_FRAC[1], format="%.3f")

use_cropped_toggle = st.checkbox("I already uploaded a cropped mesh (skip cropping step)", value=False)

model_box = st.empty()
model_info = load_model(MODEL_PATH)
if model_info is None or model_info.get("model") is None:
    model_box.error(f"Trained model not found at: {MODEL_PATH}. Prediction will be disabled. Place your joblib model there.")
else:
    model_box.success("Model loaded successfully.")

if uploaded is not None:
    # save to a temp file
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, uploaded.name)
    with open(tmp_path, "wb") as fh:
        fh.write(uploaded.getbuffer())

    st.write("Saved upload to:", tmp_path)
    # Show basic file info
    st.write("Filename:", uploaded.name)
    try:
        if use_cropped_toggle:
            mesh = o3d.io.read_triangle_mesh(tmp_path)
            torso_mesh = mesh
            used_bounds = None
        else:
            # crop
            torso_mesh, used_bounds = crop_torso_freeze_xy_dynamic_z(
                tmp_path,
                x_range=(x_min, x_max),
                y_range=(y_min, y_max),
                z_frac=(z_low_frac, z_high_frac),
                visualize_bbox=False
            )
        st.success("Cropping/extraction OK")
    except Exception as e:
        st.error(f"Error while loading/cropping mesh: {e}")
        st.stop()


    # Extract features
    try:
        feats = extract_features_from_mesh(torso_mesh)
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        st.stop()

    #     # --- visualize cropped mesh ---
    # try:
    #     st.subheader("Cropped mesh preview (interactive)")
    #     fig = mesh_to_plotly(torso_mesh, color='lightblue', opacity=1.0)
    #     st.plotly_chart(fig, use_container_width=True)
    # except Exception as e:
    #     st.warning(f"Could not render 3D preview: {e}")


    st.subheader("Extracted features (sample)")
    # show a few key features
    short = {k: feats[k] for k in ["bbox_dx","bbox_dy","bbox_dz","height","surface_area","volume"] if k in feats}
    st.json(short)

    # Prepare model input
    if model_info is None or model_info.get("model") is None:
        st.info("Model missing — cannot predict. Displaying extracted features only.")
    else:
        model = model_info["model"]
        scaler = model_info.get("scaler")
        feat_names = model_info.get("features")

        # If feature names not embedded, attempt to use a reasonable order:
        if not feat_names:
            # Infer names from extracted dict ordering (sorted stable)
            feat_names = sorted([k for k in feats.keys() if isinstance(feats[k], (int, float))])
            st.warning("Model does not include feature list. Using inferred numeric feature list: " + ", ".join(feat_names))

        # Build feature vector in model order, fill missing with 0
        x = np.array([feats.get(fn, 0.0) if feats.get(fn, None) is not None else 0.0 for fn in feat_names], dtype=float).reshape(1, -1)

        # Scale if scaler present
        if scaler is not None:
            x = scaler.transform(x)

        # Predict
        try:
            pred = model.predict(x)[0]
            st.success(f"Predicted Cobb angle: **{pred:.2f}°**")
            st.write("Feature vector used (first 10):")
            display_df = pd.DataFrame([x.flatten()], columns=feat_names)
            st.dataframe(display_df.iloc[:, :10])
            if used_bounds:
                st.info(f"Used crop bounds: X{used_bounds[0]} Y{used_bounds[1]} Z{used_bounds[2]}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Offer download of cropped mesh (optional)
    st.markdown("---")
    if st.button("Save cropped mesh to disk"):
        outfn = os.path.join(tmp_dir, f"cropped_{uploaded.name}")
        o3d.io.write_triangle_mesh(outfn, torso_mesh)
        st.success(f"Saved cropped mesh to: {outfn}")
        with open(outfn, "rb") as fh:
            st.download_button("Download cropped mesh", fh.read(), file_name=os.path.basename(outfn), mime="application/octet-stream")
