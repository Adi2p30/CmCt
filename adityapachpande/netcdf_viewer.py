import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json

class NetCDFExplorer:
    def __init__(self, root):
        self.root = root
        self.root.title("NetCDF File Explorer")
        self.root.geometry("1200x800")
        
        self.dataset = None
        self.current_file = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main menu
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open NetCDF File", command=self.open_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # File info section
        file_frame = ttk.LabelFrame(main_frame, text="File Information", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Label(file_frame, text="File:").grid(row=0, column=0, sticky=tk.W)
        self.file_label = ttk.Label(file_frame, text="No file loaded", foreground="gray")
        self.file_label.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        
        ttk.Button(file_frame, text="Browse", command=self.open_file).grid(row=0, column=2, padx=(10, 0))
        
        # Left panel - Structure
        left_frame = ttk.LabelFrame(main_frame, text="Dataset Structure", padding="5")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)
        
        # Tree view for dataset structure
        self.tree = ttk.Treeview(left_frame, height=15)
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        tree_scroll = ttk.Scrollbar(left_frame, orient="vertical", command=self.tree.yview)
        tree_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.tree.configure(yscrollcommand=tree_scroll.set)
        
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        
        # Right panel - Details and visualization
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), rowspan=2)
        
        # Details tab
        details_frame = ttk.Frame(self.notebook)
        self.notebook.add(details_frame, text="Details")
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)
        
        self.details_text = scrolledtext.ScrolledText(details_frame, height=20, width=60)
        self.details_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Data preview tab
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="Data Preview")
        data_frame.columnconfigure(0, weight=1)
        data_frame.rowconfigure(0, weight=1)
        
        self.data_text = scrolledtext.ScrolledText(data_frame, height=20, width=60)
        self.data_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Visualization tab
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="Visualization")
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(1, weight=1)
        
        # Visualization controls
        viz_controls = ttk.Frame(viz_frame)
        viz_controls.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        ttk.Label(viz_controls, text="Plot Type:").grid(row=0, column=0, padx=(0, 5))
        self.plot_type = ttk.Combobox(viz_controls, values=["Line Plot", "Histogram", "2D Heatmap"], state="readonly")
        self.plot_type.grid(row=0, column=1, padx=(0, 10))
        self.plot_type.set("Line Plot")
        
        ttk.Button(viz_controls, text="Plot", command=self.create_plot).grid(row=0, column=2)
        
        # Matplotlib canvas
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
    def open_file(self):
        file_path = filedialog.askopenfilename(
            title="Select NetCDF File",
            filetypes=[("NetCDF files", "*.nc *.nc4"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                if self.dataset:
                    self.dataset.close()
                
                self.dataset = nc.Dataset(file_path, 'r')
                self.current_file = file_path
                self.file_label.config(text=file_path, foreground="black")
                self.populate_tree()
                self.show_global_attributes()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open file: {str(e)}")
    
    def populate_tree(self):
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        if not self.dataset:
            return
        
        # Add dimensions
        dim_node = self.tree.insert("", "end", text="Dimensions", tags=("dimension",))
        for dim_name, dim in self.dataset.dimensions.items():
            size_text = f" (size: {len(dim)})"
            unlimited_text = " [UNLIMITED]" if dim.isunlimited() else ""
            self.tree.insert(dim_node, "end", text=f"{dim_name}{size_text}{unlimited_text}", 
                           values=(dim_name, "dimension"))
        
        # Add variables
        var_node = self.tree.insert("", "end", text="Variables", tags=("variable",))
        for var_name, var in self.dataset.variables.items():
            shape_text = f" {var.shape}" if var.shape else ""
            dtype_text = f" ({var.dtype})"
            self.tree.insert(var_node, "end", text=f"{var_name}{shape_text}{dtype_text}", 
                           values=(var_name, "variable"))
        
        # Add global attributes
        attr_node = self.tree.insert("", "end", text="Global Attributes", tags=("attributes",))
        for attr_name in self.dataset.ncattrs():
            self.tree.insert(attr_node, "end", text=attr_name, 
                           values=(attr_name, "global_attribute"))
        
        # Expand all nodes
        for item in self.tree.get_children():
            self.tree.item(item, open=True)
    
    def on_tree_select(self, event):
        selection = self.tree.selection()
        if not selection or not self.dataset:
            return
        
        item = selection[0]
        values = self.tree.item(item, "values")
        
        if len(values) >= 2:
            name, item_type = values[0], values[1]
            
            if item_type == "variable":
                self.show_variable_details(name)
            elif item_type == "dimension":
                self.show_dimension_details(name)
            elif item_type == "global_attribute":
                self.show_global_attribute_details(name)
    
    def show_variable_details(self, var_name):
        var = self.dataset.variables[var_name]
        
        details = f"Variable: {var_name}\n"
        details += f"Shape: {var.shape}\n"
        details += f"Dimensions: {var.dimensions}\n"
        details += f"Data Type: {var.dtype}\n"
        
        if var.ncattrs():
            details += f"\nAttributes:\n"
            for attr in var.ncattrs():
                attr_value = getattr(var, attr)
                details += f"  {attr}: {attr_value}\n"
        
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, details)
        
        # Show data preview
        try:
            if var.size > 0:
                if var.size <= 1000:  # Show all data if small
                    data_preview = str(var[:])
                else:  # Show subset if large
                    if len(var.shape) == 1:
                        preview_data = var[:10]
                        data_preview = f"First 10 values:\n{preview_data}\n\n... (showing 10 of {var.size} total values)"
                    elif len(var.shape) == 2:
                        preview_data = var[:5, :5]
                        data_preview = f"First 5x5 subset:\n{preview_data}\n\n... (showing subset of {var.shape} array)"
                    else:
                        data_preview = f"Array shape: {var.shape}\nData type: {var.dtype}\n(Use visualization tab for multi-dimensional data)"
                
                # Add statistics if numeric
                if np.issubdtype(var.dtype, np.number):
                    try:
                        data_sample = var[:].flatten()[:10000]  # Sample for stats
                        stats = f"\n\nStatistics:\n"
                        stats += f"Min: {np.min(data_sample)}\n"
                        stats += f"Max: {np.max(data_sample)}\n"
                        stats += f"Mean: {np.mean(data_sample):.6f}\n"
                        stats += f"Std: {np.std(data_sample):.6f}\n"
                        data_preview += stats
                    except:
                        pass
            else:
                data_preview = "No data available"
                
        except Exception as e:
            data_preview = f"Error reading data: {str(e)}"
        
        self.data_text.delete(1.0, tk.END)
        self.data_text.insert(1.0, data_preview)
    
    def show_dimension_details(self, dim_name):
        dim = self.dataset.dimensions[dim_name]
        
        details = f"Dimension: {dim_name}\n"
        details += f"Size: {len(dim)}\n"
        details += f"Unlimited: {'Yes' if dim.isunlimited() else 'No'}\n"
        
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, details)
        
        self.data_text.delete(1.0, tk.END)
        self.data_text.insert(1.0, "Select a variable to view data")
    
    def show_global_attribute_details(self, attr_name):
        attr_value = getattr(self.dataset, attr_name)
        
        details = f"Global Attribute: {attr_name}\n"
        details += f"Value: {attr_value}\n"
        details += f"Type: {type(attr_value).__name__}\n"
        
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, details)
        
        self.data_text.delete(1.0, tk.END)
        self.data_text.insert(1.0, "Select a variable to view data")
    
    def show_global_attributes(self):
        if not self.dataset:
            return
        
        details = "Dataset Global Information:\n\n"
        
        for attr in self.dataset.ncattrs():
            attr_value = getattr(self.dataset, attr)
            details += f"{attr}: {attr_value}\n"
        
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, details)
    
    def create_plot(self):
        selection = self.tree.selection()
        if not selection or not self.dataset:
            messagebox.showwarning("Warning", "Please select a variable to plot")
            return
        
        item = selection[0]
        values = self.tree.item(item, "values")
        
        if len(values) < 2 or values[1] != "variable":
            messagebox.showwarning("Warning", "Please select a variable to plot")
            return
        
        var_name = values[0]
        var = self.dataset.variables[var_name]
        
        if not np.issubdtype(var.dtype, np.number):
            messagebox.showwarning("Warning", "Selected variable is not numeric")
            return
        
        try:
            self.ax.clear()
            plot_type = self.plot_type.get()
            
            if plot_type == "Line Plot":
                if len(var.shape) == 1:
                    self.ax.plot(var[:])
                    self.ax.set_title(f"Line Plot: {var_name}")
                    self.ax.set_xlabel("Index")
                    self.ax.set_ylabel(var_name)
                elif len(var.shape) == 2:
                    # Plot first row/column
                    self.ax.plot(var[0, :])
                    self.ax.set_title(f"Line Plot: {var_name} (first row)")
                    self.ax.set_xlabel("Index")
                    self.ax.set_ylabel(var_name)
                else:
                    messagebox.showwarning("Warning", "Line plot not suitable for this data shape")
                    return
            
            elif plot_type == "Histogram":
                data = var[:].flatten()
                self.ax.hist(data, bins=50, alpha=0.7)
                self.ax.set_title(f"Histogram: {var_name}")
                self.ax.set_xlabel(var_name)
                self.ax.set_ylabel("Frequency")
            
            elif plot_type == "2D Heatmap":
                if len(var.shape) >= 2:
                    if len(var.shape) == 2:
                        data = var[:, :]
                    else:
                        # Take first slice of higher dimensional data
                        data = var[0, :, :] if len(var.shape) == 3 else var[(0,) * (len(var.shape) - 2)]
                    
                    im = self.ax.imshow(data, aspect='auto', cmap='viridis')
                    self.ax.set_title(f"Heatmap: {var_name}")
                    plt.colorbar(im, ax=self.ax)
                else:
                    messagebox.showwarning("Warning", "2D heatmap requires at least 2D data")
                    return
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to create plot: {str(e)}")

def main():
    root = tk.Tk()
    app = NetCDFExplorer(root)
    root.mainloop()

if __name__ == "__main__":
    main()