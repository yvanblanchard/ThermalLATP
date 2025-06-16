# Thermal LATP - Thermal Analysis for Laser-Assisted Tape Placement

CalculiX-based thermal analysis tool for automated fiber placement processes.

## Features

- Mesh generation matching FreeFem++ geometry
- Anisotropic thermal conductivity modeling
- Boundary condition setup for various interfaces
- Healing degree calculation
- VTK output for visualization

## Usage

```python
from thermal_model_ccx import modelEF, generate_inp_only

# Generate CalculiX input file only
inp_file = generate_inp_only(velocity=0.5, Troller=50.0, k_longi=2.0)

# Run full simulation
Tmax, Dhfinal = modelEF(velocity=0.5, Troller=50.0, k_longi=2.0, verbosity=True)
```

## Requirements

- Python 3.7+
- NumPy
- CalculiX (ccx) installed and in PATH
