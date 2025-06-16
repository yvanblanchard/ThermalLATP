import numpy as np
import os
from thermal_model_ccx import ThermalModel, modelEF, generate_inp_only

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("PyVista not available. Install with: pip install pyvista")


class ThermalVisualization:
    """Visualization wrapper for ThermalModel using PyVista"""
    
    def __init__(self):
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required for visualization. Install with: pip install pyvista")
        self.model = ThermalModel()
    
    def visualize_mesh(self, show_boundaries=True, show_labels=True):
        """Visualize the mesh geometry and boundaries using PyVista
        
        Args:
            show_boundaries (bool): Whether to highlight boundary regions
            show_labels (bool): Whether to show dimension labels
            
        Returns:
            pv.UnstructuredGrid: PyVista grid object
        """
        # Generate mesh from thermal model
        nodes, elements, node_sets = self.model.generate_mesh()
        
        # Create PyVista unstructured grid
        # Convert elements to PyVista format (VTK_QUAD = 9)
        cells = []
        for element in elements:
            cells.extend([4] + element.tolist())  # 4 nodes per quad element
        
        cell_types = np.full(len(elements), 9)  # VTK_QUAD
        
        # Create grid
        grid = pv.UnstructuredGrid(cells, cell_types, nodes)
        
        # Create plotter
        plotter = pv.Plotter(window_size=(1200, 800))
        
        # Add main mesh
        plotter.add_mesh(grid, 
                        color='lightblue',
                        show_edges=True,
                        edge_color='black',
                        line_width=0.5,
                        opacity=0.8)
        
        if show_boundaries:
            # Highlight different boundary regions with colors
            boundary_colors = {
                'heating': 'red',      # Laser heating zone
                'roller': 'orange',    # Roller contact
                'air': 'cyan',         # Air convection  
                'bottom': 'brown',     # Table contact
                'border_F': 'green',   # Left boundary (main)
                'border_B': 'lime'     # Left boundary (gap)
            }
            
            for boundary_name, node_ids in node_sets.items():
                if node_ids and boundary_name in boundary_colors:
                    # Create points for boundary nodes
                    boundary_nodes = nodes[np.array(node_ids) - 1]  # Convert to 0-based
                    boundary_cloud = pv.PolyData(boundary_nodes)
                    
                    plotter.add_mesh(boundary_cloud,
                                   color=boundary_colors[boundary_name],
                                   point_size=8,
                                   render_points_as_spheres=True,
                                   label=boundary_name.replace('_', ' ').title())
        
        # Add coordinate system
        plotter.add_axes(interactive=True)
        
        if show_labels:
            # Add dimension annotations
            annotations = [
                f"Domain Length: {self.model.L*1000:.1f} mm",
                f"Domain Height: {(self.model.n+1)*self.model.h*1000:.2f} mm", 
                f"Tape Thickness: {self.model.h*1000:.3f} mm",
                f"Number of Tapes: {self.model.n}",
                f"Heating Length: {self.model.lc*1000:.1f} mm",
                f"Roller Contact: {self.model.Lroller*1000:.1f} mm"
            ]
            
            plotter.add_text("\n".join(annotations), 
                           position='upper_left', 
                           font_size=12,
                           color='black')
            
            # Add boundary legend if boundaries are shown
            if show_boundaries:
                legend_text = "Boundary Conditions:\n"
                legend_text += "🔴 Laser Heating\n🟠 Roller Contact\n🔵 Air Convection\n"
                legend_text += "🟤 Table Contact\n🟢 Fixed Temperature"
                
                plotter.add_text(legend_text,
                               position='upper_right',
                               font_size=10,
                               color='black')
        
        # Set view and show
        plotter.view_xy()
        plotter.camera.zoom(1.2)
        plotter.show()
        
        return grid
    
    def visualize_results(self, velocity, Troller, k_longi, frd_file="thermal.frd"):
        """Visualize thermal analysis results
        
        Args:
            velocity (float): Process velocity (m/s)
            Troller (float): Roller temperature (°C)
            k_longi (float): Longitudinal thermal conductivity (W/m/K)
            frd_file (str): Path to CalculiX results file
            
        Returns:
            tuple: (Tmax, Dhfinal, grid) - Results and PyVista grid
        """
        # Extract results using thermal model
        Tmax, Dhfinal, nodes, temps = self.model.extract_results(velocity, frd_file)
        
        if Tmax is None:
            print("No results to visualize")
            return None, None, None
        
        # Create PyVista grid from results
        grid = self._create_results_grid(nodes, temps)
        
        # Create multi-window plotter
        pl = pv.Plotter(shape=(2, 1))
        
        # Temperature field plot
        pl.subplot(0, 0)
        pl.add_mesh(grid, 
                    scalars="Temperature",
                    cmap="coolwarm",
                    show_edges=True,
                    edge_color='black',
                    scalar_bar_args={'title': 'Temperature (°C)'})
        pl.add_text("Temperature Field", position='upper_left')
        pl.view_xy()
        
        # Interface temperature plot
        pl.subplot(1, 0)
        
        # Extract interface nodes
        y_interface = self.model.n * self.model.h
        tolerance = 1e-10
        
        points = grid.points
        temp_data = grid.point_data["Temperature"]
        
        # Find interface nodes
        interface_mask = np.abs(points[:, 1] - y_interface) < tolerance
        interface_points = points[interface_mask]
        interface_temps = temp_data[interface_mask]
        
        if len(interface_points) > 0:
            # Sort by x-coordinate
            sort_idx = np.argsort(interface_points[:, 0])
            x_coords = interface_points[sort_idx, 0]
            sorted_temps = interface_temps[sort_idx]
            
            # Create line for interface temperature
            line = pv.Line(pointa=(x_coords[0], 0, 0), pointb=(x_coords[-1], 0, 0))
            line["Temperature"] = sorted_temps
            
            pl.add_mesh(line, 
                        scalars="Temperature",
                        render_lines_as_tubes=True,
                        line_width=5,
                        cmap="coolwarm",
                        scalar_bar_args={'title': 'Interface Temperature (°C)'})
        
        pl.add_text("Interface Temperature Distribution", position='upper_left')
        
        # Add analysis results text
        results_text = f"Process Parameters:\n"
        results_text += f"Velocity: {velocity:.2f} m/s\n"
        results_text += f"Roller Temperature: {Troller:.1f}°C\n"
        results_text += f"Conductivity: {k_longi:.1f} W/m/K\n\n"
        results_text += f"Results:\n"
        results_text += f"Maximum Temperature: {Tmax:.1f}°C\n"
        results_text += f"Healing Degree: {Dhfinal:.3f}"
        
        pl.add_text(results_text, position='upper_right')
        
        # Show the combined plots
        pl.show()
        
        return Tmax, Dhfinal, grid
    
    def _create_results_grid(self, nodes, temps):
        """Create PyVista structured grid from thermal results"""
        # Determine grid dimensions from node coordinates
        x_coords = sorted(set(node[0] for node in nodes.values()))
        y_coords = sorted(set(node[1] for node in nodes.values()))
        
        nx = len(x_coords)
        ny = len(y_coords)
        
        # Create coordinate arrays
        X, Y = np.meshgrid(x_coords, y_coords)
        Z = np.zeros_like(X)
        
        # Create temperature array
        temp_array = np.zeros((ny, nx))
        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                # Find corresponding node
                for node_id, coord in nodes.items():
                    if (abs(coord[0] - x) < 1e-9 and 
                        abs(coord[1] - y) < 1e-9 and 
                        node_id in temps):
                        temp_array[i, j] = temps[node_id]
                        break
        
        # Create StructuredGrid
        grid = pv.StructuredGrid(X, Y, Z)
        grid.point_data["Temperature"] = temp_array.flatten()
        
        # Save enhanced VTK file
        grid.save("T_pyvista.vtk")
        
        return grid


def visualize_mesh_only(show_boundaries=True, show_labels=True):
    """Standalone function to visualize mesh without running simulation
    
    Args:
        show_boundaries (bool): Whether to highlight boundary regions
        show_labels (bool): Whether to show dimension labels
        
    Returns:
        pv.UnstructuredGrid: PyVista grid object or None if PyVista unavailable
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista not available for visualization")
        return None
        
    viz = ThermalVisualization()
    return viz.visualize_mesh(show_boundaries, show_labels)


def generate_and_visualize_inp(velocity, Troller, k_longi, filename="thermal.inp"):
    """Generate INP file and show mesh visualization
    
    Args:
        velocity (float): Process velocity (m/s)
        Troller (float): Roller temperature (°C)
        k_longi (float): Longitudinal thermal conductivity (W/m/K)
        filename (str): Output INP filename
        
    Returns:
        tuple: (inp_filename, grid) - INP file path and PyVista grid
    """
    # Generate INP file using core module
    inp_file = generate_inp_only(velocity, Troller, k_longi, filename)
    
    # Show mesh visualization
    print("Displaying mesh visualization...")
    grid = visualize_mesh_only(show_boundaries=True, show_labels=True)
    
    return inp_file, grid


def run_and_visualize(velocity, Troller, k_longi, verbosity=False):
    """Run complete thermal analysis with visualization
    
    Args:
        velocity (float): Process velocity (m/s)
        Troller (float): Roller temperature (°C)
        k_longi (float): Longitudinal thermal conductivity (W/m/K)
        verbosity (bool): Whether to print verbose output
        
    Returns:
        tuple: (Tmax, Dhfinal, grid) - Results and PyVista grid
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista not available. Running analysis without visualization...")
        return modelEF(velocity, Troller, k_longi, verbosity)
    
    # Run thermal analysis
    Tmax, Dhfinal = modelEF(velocity, Troller, k_longi, verbosity)
    
    # Visualize results
    viz = ThermalVisualization()
    Tmax_viz, Dhfinal_viz, grid = viz.visualize_results(velocity, Troller, k_longi)
    
    return Tmax, Dhfinal, grid


def compare_with_freefem(velocity, Troller, k_longi, freefem_vtk="T.vtk"):
    """Compare CalculiX results with FreeFem++ output
    
    Args:
        velocity (float): Process velocity (m/s)
        Troller (float): Roller temperature (°C)
        k_longi (float): Longitudinal thermal conductivity (W/m/K)
        freefem_vtk (str): Path to FreeFem++ VTK output file
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista not available for comparison visualization")
        return
    
    # Run CalculiX analysis
    print("Running CalculiX analysis...")
    Tmax_ccx, Dhfinal_ccx, grid_ccx = run_and_visualize(velocity, Troller, k_longi, verbosity=True)
    
    # Try to load FreeFem++ results if available
    if os.path.exists(freefem_vtk):
        print(f"Loading FreeFem++ results from {freefem_vtk}...")
        grid_ff = pv.read(freefem_vtk)
        
        # Create comparison plot
        pl = pv.Plotter(shape=(1, 2))
        
        # FreeFem++ results
        pl.subplot(0, 0)
        pl.add_mesh(grid_ff, scalars="Temperature", cmap="coolwarm")
        pl.add_text("FreeFem++ Results", position='upper_left')
        
        # CalculiX results  
        pl.subplot(0, 1)
        pl.add_mesh(grid_ccx, scalars="Temperature", cmap="coolwarm")
        pl.add_text("CalculiX Results", position='upper_left')
        
        pl.show()
        
        # Print comparison
        if hasattr(grid_ff.point_data, 'Temperature'):
            Tmax_ff = np.max(grid_ff.point_data['Temperature'])
            print(f"\nComparison:")
            print(f"FreeFem++ Max Temperature: {Tmax_ff:.1f}°C")
            print(f"CalculiX Max Temperature: {Tmax_ccx:.1f}°C")
            print(f"Difference: {abs(Tmax_ff - Tmax_ccx):.1f}°C")
    else:
        print(f"FreeFem++ results file {freefem_vtk} not found")


# Example usage and demonstrations
if __name__ == "__main__":
    # Example parameters
    velocity = 0.5     # m/s
    Troller = 50.0     # °C  
    k_longi = 2.0      # W/m/K
    
    print("=== CalculiX Thermal Analysis with Visualization ===\n")
    
    if not PYVISTA_AVAILABLE:
        print("PyVista not available. Please install with: pip install pyvista")
        print("Running core functionality only...\n")
        
        # Run without visualization
        Tmax, Dhfinal = modelEF(velocity, Troller, k_longi, verbosity=True)
        print(f"Results: Tmax={Tmax:.1f}°C, Dhfinal={Dhfinal:.6f}")
    else:
        # Option 1: Visualize mesh only
        print("1. Visualizing mesh geometry...")
        visualize_mesh_only(show_boundaries=True, show_labels=True)
        
        # Option 2: Generate INP and visualize
        print("\n2. Generate INP file with visualization...")
        inp_file, grid = generate_and_visualize_inp(velocity, Troller, k_longi, 
                                                   filename="thermal_viz.inp")
        
        # Option 3: Full analysis with visualization
        print("\n3. Running full analysis with visualization...")
        try:
            Tmax, Dhfinal, grid = run_and_visualize(velocity, Troller, k_longi, verbosity=True)
            print(f"Results: Tmax={Tmax:.1f}°C, Dhfinal={Dhfinal:.6f}")
        except RuntimeError as e:
            print(f"Analysis failed: {e}")
        
        # Option 4: Compare with FreeFem++ if available
        print("\n4. Comparing with FreeFem++ results (if available)...")
        compare_with_freefem(velocity, Troller, k_longi)
    
    print("\nVisualization module complete.")