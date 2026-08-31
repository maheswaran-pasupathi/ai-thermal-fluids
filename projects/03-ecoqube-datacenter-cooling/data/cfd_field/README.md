# CFD field data (not committed)

The solved OpenFOAM case behind `stage1b_cfd_field_visualization.py` isn't committed here - it's 611 MB extracted (118.7 MB compressed), well over anything worth putting in a portfolio repo.

Download and extract it yourself:

```
curl -L -o postProcess.tar.xz "https://zenodo.org/records/7035829/files/postProcess.tar.xz?download=1"
tar -xf postProcess.tar.xz -C .
```

This gives you `postProcess/case21newBoundaryOldDesign.foam` plus the full solved case (mesh, boundary conditions, and a solved timestep at t=500 with T/U/p/k/omega fields) - real ECO-Qube CFD output for the previous ("old") data center cooling design, not raw unsolved case setup.

Requires `pyvista` (in `requirements.txt`) to read - it wraps VTK's native OpenFOAM reader.
