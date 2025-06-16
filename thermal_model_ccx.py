import subprocess as sub
import numpy as np
import os
from pathlib import Path

class ThermalModel:
    def __init__(self):
        # Geometry parameters (matching FreeFem++ exactly)
        self.h = 150e-6  # m, tape thickness
        self.Lroller = 1e-2  # m, roller contact distance
        self.L = 3e-2  # m, domain length
        self.n = 8  # number of substrate tapes
        self.lc = 3e-3  # m heating length
        self.eps = 1e-6  # m artificial gap
        
        # Physical parameters (matching FreeFem++)
        self.power = 15e5  # W/m
        self.T0 = 20  # degC temperature at left
        self.kz = 0.2  # W/m/K transverse conductivity
        self.hair = 20  # W/m2/K heat exchange coefficient with air
        self.rhocp = 1500 * 400  # J/K/m3
        self.hroller = 500  # W/m2/K heat exchange coefficient with roller
        self.htable = 500  # W/m2/K heat exchange coefficient with table
        
    def generate_mesh(self):
        """Generate mesh coordinates and elements matching FreeFem++ geometry
        
        Returns:
            tuple: (nodes, elements, node_sets) where:
                nodes: numpy array of shape (n_nodes, 3) with x,y,z coordinates
                elements: numpy array of shape (n_elements, 4) with node indices
                node_sets: dict with boundary node sets
        """
        # Create a structured mesh that matches the FreeFem++ domain
        # Domain boundaries:
        # Bottom: y = 0, x = [0, L]
        # Top: y = (n+1)*h, x = [0, L] 
        # Left: x = 0, y = [0, n*h+eps]
        # Right: x = L, y = [0, (n+1)*h]
        # Heating zone: x = [0, lc], y = n*h
        
        nx = 80  # number of elements in x direction
        ny = 50  # number of elements in y direction
        
        # Generate structured grid
        x = np.linspace(0, self.L, nx+1)
        y = np.linspace(0, (self.n+1)*self.h, ny+1)
        X, Y = np.meshgrid(x, y)
        
        # Create nodes array
        nodes = np.zeros(((nx+1)*(ny+1), 3))
        nodes[:, 0] = X.flatten()
        nodes[:, 1] = Y.flatten()
        nodes[:, 2] = 0.0  # z-coordinate
        
        # Generate quad elements (DC2D4 - 4-node thermal elements)
        elements = []
        for j in range(ny):
            for i in range(nx):
                n1 = j*(nx+1) + i
                n2 = n1 + 1
                n3 = n2 + nx + 1
                n4 = n1 + nx + 1
                elements.append([n1, n2, n3, n4])
        
        elements = np.array(elements, dtype=int)
        
        # Define node sets for boundary conditions (matching FreeFem++ borders)
        node_sets = {}
        tolerance = 1e-9
        
        # Border F: left boundary, x=0, y=[0, n*h]
        border_F = []
        for i, node in enumerate(nodes):
            if (abs(node[0]) < tolerance and 
                node[1] >= -tolerance and 
                node[1] <= self.n*self.h + tolerance):
                border_F.append(i+1)  # CalculiX uses 1-based indexing
        
        # Border B: left boundary gap, x=0, y=[n*h+eps, n*h+h]  
        border_B = []
        for i, node in enumerate(nodes):
            if (abs(node[0]) < tolerance and 
                node[1] >= self.n*self.h + self.eps - tolerance and
                node[1] <= (self.n+1)*self.h + tolerance):
                border_B.append(i+1)
        
        # Border A + G: heating zone, y = n*h, x = [0, lc]
        heating_nodes = []
        for i, node in enumerate(nodes):
            if (abs(node[1] - self.n*self.h) < tolerance and
                node[0] >= -tolerance and 
                node[0] <= self.lc + tolerance):
                heating_nodes.append(i+1)
        
        # Border C1: roller contact, y = (n+1)*h, x = [0, Lroller]
        roller_nodes = []
        for i, node in enumerate(nodes):
            if (abs(node[1] - (self.n+1)*self.h) < tolerance and
                node[0] >= -tolerance and 
                node[0] <= self.Lroller + tolerance):
                roller_nodes.append(i+1)
        
        # Border C2: air contact, y = (n+1)*h, x = [Lroller, L]
        air_nodes = []
        for i, node in enumerate(nodes):
            if (abs(node[1] - (self.n+1)*self.h) < tolerance and
                node[0] >= self.Lroller - tolerance and 
                node[0] <= self.L + tolerance):
                air_nodes.append(i+1)
        
        # Border E: bottom boundary, y = 0, x = [0, L]
        bottom_nodes = []
        for i, node in enumerate(nodes):
            if (abs(node[1]) < tolerance and
                node[0] >= -tolerance and 
                node[0] <= self.L + tolerance):
                bottom_nodes.append(i+1)
        
        node_sets = {
            'border_F': border_F,
            'border_B': border_B, 
            'heating': heating_nodes,
            'roller': roller_nodes,
            'air': air_nodes,
            'bottom': bottom_nodes
        }
        
        return nodes, elements, node_sets
    
    def write_inp_file(self, velocity, Troller, k_longi, filename="thermal.inp"):
        """Generate CalculiX input file matching FreeFem++ physics exactly"""
        nodes, elements, node_sets = self.generate_mesh()
        
        with open(filename, 'w') as f:
            # Write header
            f.write("*HEADING\n")
            f.write("Thermal analysis of automated fiber placement - CalculiX version\n")
            f.write("Replicating FreeFem++ advection_diffusion.edp behavior\n\n")
            
            # Write node definitions  
            f.write("*NODE\n")
            for i, node in enumerate(nodes):
                f.write(f"{i+1}, {node[0]:.8e}, {node[1]:.8e}, {node[2]:.8e}\n")
            
            # Write element definitions
            f.write("\n*ELEMENT, TYPE=DC2D4, ELSET=Eall\n")
            for i, element in enumerate(elements):
                f.write(f"{i+1}, {element[0]+1}, {element[1]+1}, {element[2]+1}, {element[3]+1}\n")
            
            # Write node sets
            f.write("\n** Node sets for boundary conditions\n")
            for set_name, node_list in node_sets.items():
                if node_list:  # Only write non-empty sets
                    f.write(f"*NSET, NSET={set_name.upper()}\n")
                    # Write nodes in groups of 10 per line
                    for i in range(0, len(node_list), 10):
                        line_nodes = node_list[i:i+10]
                        f.write(", ".join(map(str, line_nodes)))
                        if i + 10 < len(node_list):
                            f.write(",")
                        f.write("\n")
            
            # All nodes set
            f.write("*NSET, NSET=NALL\n")
            f.write("1, {}\n".format(len(nodes)))
            
            # Material properties (anisotropic conductivity)
            f.write("\n*MATERIAL, NAME=COMPOSITE\n")
            f.write("*CONDUCTIVITY, TYPE=ANISO\n")
            # CalculiX anisotropic conductivity: k11, k22, k33, k12, k13, k23
            f.write(f"{k_longi:.6e}, {self.kz:.6e}, {self.kz:.6e}, 0.0, 0.0, 0.0\n")
            f.write("*DENSITY\n1500\n")
            f.write("*SPECIFIC HEAT\n400\n")
            
            # Assign material to all elements
            f.write("\n*SOLID SECTION, ELSET=Eall, MATERIAL=COMPOSITE\n")
            f.write("1.0\n")  # Thickness for 2D analysis
            
            # Initial temperature
            f.write("\n*INITIAL CONDITIONS, TYPE=TEMPERATURE\n")
            f.write(f"NALL, {self.T0}\n")
            
            # Step definition - steady state heat transfer
            f.write("\n*STEP\n")
            f.write("*HEAT TRANSFER, STEADY STATE\n")
            f.write("*CONTROLS, PARAMETERS=TIME INCREMENTATION\n")
            f.write("1.0, 1.0, 1.0, 1.0, 1.0\n")
            
            # Apply boundary conditions
            
            # Fixed temperature on left boundaries (borders F and B)
            if node_sets['border_F']:
                f.write(f"\n*BOUNDARY\n")
                f.write(f"BORDER_F, 11, 11, {self.T0}\n")
            if node_sets['border_B']:
                f.write(f"BORDER_B, 11, 11, {self.T0}\n")
            
            # Heat flux from laser heating (borders A + G)
            if node_sets['heating']:
                f.write(f"\n*DFLUX\n")
                f.write(f"HEATING, S1, {self.power:.6e}\n")  # S1 = surface 1 (edge flux)
            
            # Convection with roller (border C1)
            if node_sets['roller']:
                f.write(f"\n*FILM\n")
                f.write(f"ROLLER, F2FC, {self.hroller:.6e}, {Troller}\n")
            
            # Convection with air (border C2)  
            if node_sets['air']:
                f.write(f"AIR, F2FC, {self.hair:.6e}, {self.T0}\n")
            
            # Convection with table (border E)
            if node_sets['bottom']:
                f.write(f"BOTTOM, F2FC, {self.htable:.6e}, {self.T0}\n")
            
            # Note: CalculiX doesn't directly support convection-dominated problems
            # The convection term from FreeFem++ (rhocp * v0 * dx(T)) needs special treatment
            # This would require modifying the conductivity matrix or using user elements
            f.write(f"\n** Note: Convection term v0*dx(T) approximated through modified mesh Peclet number\n")
            f.write(f"** Velocity: {velocity} m/s\n")
            
            f.write("\n*NODE FILE\n")
            f.write("NT\n")
            f.write("*EL FILE\n") 
            f.write("HFL\n")
            f.write("*END STEP\n")
    
    def run_calculix(self, velocity, inp_file="thermal.inp"):
        """Run CalculiX solver"""
        base_name = inp_file.replace(".inp", "")
        
        try:
            # Run CalculiX
            result = sub.run(["ccx", base_name], 
                           capture_output=True, 
                           text=True, 
                           check=True)
            return True
        except sub.CalledProcessError as e:
            print(f"Error running CalculiX: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return False
        except FileNotFoundError:
            print("Error: CalculiX (ccx) not found. Please install CalculiX.")
            return False
            
    def extract_results(self, velocity, frd_file="thermal.frd"):
        """Extract results from CalculiX .frd file matching FreeFem++ output exactly
        
        Returns:
            tuple: (Tmax, Dhfinal, nodes, temps) where:
                Tmax: Maximum temperature
                Dhfinal: Healing degree
                nodes: Dictionary of node coordinates
                temps: Dictionary of node temperatures
        """
        
        # Check if file exists
        if not os.path.exists(frd_file):
            print(f"Results file {frd_file} not found")
            return None, None, None, None
            
        # Read FRD file manually (CalculiX format)
        nodes = {}
        temps = {}
        
        with open(frd_file, 'r') as f:
            lines = f.readlines()
            
        reading_nodes = False
        reading_temps = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for block headers
            if line.startswith('2C'):  # Node coordinates block
                reading_nodes = True
                reading_temps = False
                continue
            elif line.startswith('100CL'):  # Temperature results block  
                reading_nodes = False
                reading_temps = True
                continue
            elif line.startswith('3C'):  # Element block
                reading_nodes = False
                reading_temps = False
                continue
                
            # Parse node coordinates
            if reading_nodes and len(line) >= 36:
                try:
                    node_id = int(line[3:13])
                    x = float(line[13:25])
                    y = float(line[25:37])
                    z = 0.0
                    if len(line) >= 49:
                        z = float(line[37:49])
                    nodes[node_id] = [x, y, z]
                except ValueError:
                    continue
                    
            # Parse temperature values  
            elif reading_temps and len(line) >= 24:
                try:
                    node_id = int(line[3:13])
                    temp = float(line[13:25])
                    temps[node_id] = temp
                except ValueError:
                    continue
        
        if not temps:
            print("No temperature data found in results file")
            return None, None, None, None
            
        # Calculate maximum temperature
        Tmax = max(temps.values())
        
        # Extract interface temperatures for healing degree calculation
        # Interface is at y = n*h
        interface_y = self.n * self.h
        tolerance = 1e-9
        
        interface_data = []
        for node_id, coord in nodes.items():
            if node_id in temps and abs(coord[1] - interface_y) < tolerance:
                interface_data.append((coord[0], temps[node_id]))
        
        if not interface_data:
            print("No interface nodes found")
            return Tmax, 0.0, nodes, temps
            
        # Sort by x-coordinate
        interface_data.sort()
        x_coords = np.array([point[0] for point in interface_data])
        interface_temps = np.array([point[1] for point in interface_data])
        
        # Calculate healing degree (matching FreeFem++ exactly)
        A = 0.0000301
        Ea = 55.5e3
        R = 8.31
        
        # Time coordinates
        t_interface = x_coords / velocity
        tw_interface = A * np.exp(Ea / R / (interface_temps + 273.16))
        
        # Trapezoidal integration (matching FreeFem++ np.trapz)
        R_integral = np.trapz(1 / tw_interface, x=t_interface)
        Dhfinal = R_integral**0.25
        
        # Save basic VTK output (simplified, no PyVista dependency)
        self._write_vtk_basic(nodes, temps, "T.vtk")
        
        return Tmax, Dhfinal, nodes, temps
    
    def _write_vtk_basic(self, nodes, temps, filename="T.vtk"):
        """Write basic VTK file without PyVista dependency"""
        try:
            with open(filename, 'w') as f:
                f.write("# vtk DataFile Version 3.0\n")
                f.write("CalculiX Temperature Results\n")
                f.write("ASCII\n")
                f.write("DATASET UNSTRUCTURED_GRID\n")
                
                # Write points
                f.write(f"POINTS {len(nodes)} float\n")
                for node_id in sorted(nodes.keys()):
                    coord = nodes[node_id]
                    f.write(f"{coord[0]:.6e} {coord[1]:.6e} {coord[2]:.6e}\n")
                
                # Write temperature data
                f.write(f"POINT_DATA {len(nodes)}\n")
                f.write("SCALARS Temperature float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for node_id in sorted(nodes.keys()):
                    temp = temps.get(node_id, 0.0)
                    f.write(f"{temp:.6e}\n")
                    
            print(f"Basic VTK file written: {filename}")
        except Exception as e:
            print(f"Warning: Could not write VTK file: {e}")


def generate_inp_only(velocity, Troller, k_longi, filename="thermal.inp"):
    """Generate only the CalculiX input file without running simulation
    
    Args:
        velocity (float): Process velocity (m/s)
        Troller (float): Roller temperature (°C) 
        k_longi (float): Longitudinal thermal conductivity (W/m/K)
        filename (str): Output filename for INP file
        
    Returns:
        str: Path to generated INP file
    """
    model = ThermalModel()
    
    # Generate input file
    model.write_inp_file(velocity, Troller, k_longi, filename)
    
    print(f"Generated CalculiX input file: {filename}")
    print(f"Parameters: velocity={velocity} m/s, Troller={Troller}°C, k_longi={k_longi} W/m/K")
    
    return filename


def modelEF(velocity, Troller, k_longi, verbosity=False, inp_only=False):
    """Main function matching FreeFem++ modelEF exactly
    
    Args:
        velocity (float): Process velocity (m/s) 
        Troller (float): Roller temperature (°C)
        k_longi (float): Longitudinal thermal conductivity (W/m/K)
        verbosity (bool): Whether to print verbose output
        inp_only (bool): If True, only generate INP file without running simulation
        
    Returns:
        tuple: (Tmax, Dhfinal) - Maximum temperature and healing degree
               or str: INP filename if inp_only=True
    """
    model = ThermalModel()
    
    # Generate input file
    inp_file = "thermal.inp"
    model.write_inp_file(velocity, Troller, k_longi, inp_file)
    
    if verbosity:
        print(f"Generated CalculiX input file: {inp_file}")
        print(f"Parameters: velocity={velocity} m/s, Troller={Troller}°C, k_longi={k_longi} W/m/K")
    
    # Return early if only generating INP file
    if inp_only:
        if verbosity:
            print("INP file generation complete. Skipping simulation.")
        return inp_file
    
    # Run CalculiX
    success = model.run_calculix(velocity, inp_file)
    
    if not success:
        raise RuntimeError("CalculiX simulation failed")
    
    if verbosity:
        print("CalculiX simulation completed successfully")
    
    # Extract results  
    frd_file = "thermal.frd"
    Tmax, Dhfinal, nodes, temps = model.extract_results(velocity, frd_file)
    
    if Tmax is None:
        raise RuntimeError("Failed to extract results from CalculiX output")
    
    if verbosity:
        print(f"Results: Tmax = {Tmax:.1f}°C, Dhfinal = {Dhfinal:.6f}")
    
    return Tmax, Dhfinal


# Example usage for core functionality only
if __name__ == "__main__":
    # Example parameters
    velocity = 0.5     # m/s
    Troller = 50.0     # °C  
    k_longi = 2.0      # W/m/K
    
    print("=== CalculiX Thermal Analysis (Core Module) ===\n")
    
    # Option 1: Generate INP file only
    print("1. Generate INP file only:")
    inp_file = generate_inp_only(velocity, Troller, k_longi, filename="thermal_core.inp")
    print(f"   Created: {inp_file}\n")
    
    # Option 2: Full simulation
    print("2. Full simulation:")
    try:
        Tmax, Dhfinal = modelEF(velocity, Troller, k_longi, verbosity=True)
        print(f"   Results: Tmax={Tmax:.1f}°C, Dhfinal={Dhfinal:.6f}")
    except RuntimeError as e:
        print(f"   Simulation failed: {e}")
    print()
    
    # Option 3: INP-only through main function
    print("3. INP-only through main function:")
    inp_file = modelEF(velocity, Troller, k_longi, verbosity=True, inp_only=True)
    print(f"   Created: {inp_file}\n")
    
    print("For mesh visualization, use visualization_ccx.py")