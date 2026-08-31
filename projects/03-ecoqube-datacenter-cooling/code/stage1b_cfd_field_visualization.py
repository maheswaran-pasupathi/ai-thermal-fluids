# Stage 1b - render the actual 3D CFD temperature field, not just the 1D
# exhaust-height profile from stage1_data_exploration.py.
#
# Needs the ECO-Qube "postProcess" archive (118.7 MB) - not committed, see
# ../data/cfd_field/README.md for the download command. This is real solved
# OpenFOAM output (previous/old design case), not raw case setup - it includes
# an actual solved timestep (500) with T, U, p, k, omega fields.
#
# Credit: ECO-Qube EU project (CORDIS 956059) and contributing partners - see
# README.md for full citation.

# %%
import pyvista as pv
pv.OFF_SCREEN = True

CASE_DIR = "../data/cfd_field/postProcess"
FOAM_FILE = f"{CASE_DIR}/case21newBoundaryOldDesign.foam"

reader = pv.OpenFOAMReader(FOAM_FILE)
print("Available time values:", reader.time_values)
reader.set_active_time_value(reader.time_values[-1])  # the solved timestep, not t=0
mesh = reader.read().combine()
print("Domain bounds:", mesh.bounds)
print("T range (K):", mesh["T"].min(), mesh["T"].max())

# %%
# Vertical slice through the middle of the aisle - side view through the racks.
y_mid = (mesh.bounds[2] + mesh.bounds[3]) / 2
slice_xz = mesh.slice(normal="y", origin=(0, y_mid, 0))

plotter = pv.Plotter(off_screen=True, window_size=[1400, 700])
plotter.add_mesh(slice_xz, scalars="T", cmap="coolwarm", clim=[294, 328],
                  scalar_bar_args={"title": "Temperature (K)"})
plotter.view_xz()
plotter.camera.zoom(1.3)
plotter.add_text("Previous design - CFD temperature field, side view through aisle", font_size=10)
plotter.screenshot("../results/stage1b_cfd_field_side.png")

# %%
# Horizontal slice at mid-height - plan view, showing the hotspot near the
# rack exhaust boundary.
z_mid = (mesh.bounds[4] + mesh.bounds[5]) / 2
slice_xy = mesh.slice(normal="z", origin=(0, 0, z_mid))

plotter2 = pv.Plotter(off_screen=True, window_size=[1400, 700])
plotter2.add_mesh(slice_xy, scalars="T", cmap="coolwarm", clim=[294, 328],
                   scalar_bar_args={"title": "Temperature (K)"})
plotter2.view_xy()
plotter2.camera.zoom(1.3)
plotter2.add_text("Previous design - CFD temperature field, plan view at mid-height", font_size=10)
plotter2.screenshot("../results/stage1b_cfd_field_plan.png")

# %%
# Next: fill in stage1_learning_notes.md - does the hot-air-rises pattern in
# the side view and the exhaust-boundary hotspot in the plan view match the
# height profile you already found from the sensor logs and exhaust CSV?
