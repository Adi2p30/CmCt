import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageEnhance, ImageTk


class TIFFViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("TIFF File Viewer")
        self.root.geometry("1400x900")

        self.current_image = None
        self.original_image = None
        self.current_file = None
        self.zoom_factor = 1.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.canvas_image_id = None

        # Image enhancement parameters
        self.brightness = 1.0
        self.contrast = 1.0
        self.current_band = 0

        self.setup_ui()

    def setup_ui(self):
        # Main menu
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open TIFF File", command=self.open_file)
        file_menu.add_separator()
        file_menu.add_command(label="Save Current View", command=self.save_current_view)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Zoom In", command=self.zoom_in)
        view_menu.add_command(label="Zoom Out", command=self.zoom_out)
        view_menu.add_command(label="Fit to Window", command=self.fit_to_window)
        view_menu.add_command(label="Reset View", command=self.reset_view)

        # Main container
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel for image display
        left_frame = ttk.Frame(main_container)
        main_container.add(left_frame, weight=3)

        # Right panel for controls and info
        right_frame = ttk.Frame(main_container)
        main_container.add(right_frame, weight=1)

        self.setup_image_panel(left_frame)
        self.setup_control_panel(right_frame)

    def setup_image_panel(self, parent):
        # File info bar
        file_frame = ttk.Frame(parent)
        file_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(file_frame, text="File:").pack(side=tk.LEFT)
        self.file_label = ttk.Label(
            file_frame, text="No file loaded", foreground="gray"
        )
        self.file_label.pack(side=tk.LEFT, padx=(10, 0))

        ttk.Button(file_frame, text="Browse", command=self.open_file).pack(
            side=tk.RIGHT
        )

        # Image display frame
        image_frame = ttk.LabelFrame(parent, text="Image Display")
        image_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas with scrollbars
        canvas_frame = ttk.Frame(image_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(canvas_frame, bg="gray90")

        # Scrollbars
        h_scrollbar = ttk.Scrollbar(
            canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview
        )
        v_scrollbar = ttk.Scrollbar(
            canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview
        )

        self.canvas.configure(
            xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set
        )

        # Grid layout
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.do_pan)
        self.canvas.bind("<MouseWheel>", self.zoom_wheel)
        self.canvas.bind("<Double-Button-1>", self.fit_to_window)

        # Status bar
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(5, 0))

        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT)

        self.zoom_label = ttk.Label(status_frame, text="Zoom: 100%")
        self.zoom_label.pack(side=tk.RIGHT)

    def setup_control_panel(self, parent):
        # Notebook for different control tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Image Info Tab
        info_frame = ttk.Frame(notebook)
        notebook.add(info_frame, text="Image Info")

        self.info_text = scrolledtext.ScrolledText(info_frame, height=15, width=40)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Controls Tab
        controls_frame = ttk.Frame(notebook)
        notebook.add(controls_frame, text="Controls")

        # Zoom controls
        zoom_frame = ttk.LabelFrame(controls_frame, text="Zoom Controls")
        zoom_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(zoom_frame, text="Zoom In", command=self.zoom_in).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(zoom_frame, text="Zoom Out", command=self.zoom_out).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(zoom_frame, text="Fit to Window", command=self.fit_to_window).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(zoom_frame, text="Actual Size", command=self.actual_size).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(zoom_frame, text="Reset View", command=self.reset_view).pack(
            fill=tk.X, pady=2
        )

        # Enhancement controls
        enhance_frame = ttk.LabelFrame(controls_frame, text="Image Enhancement")
        enhance_frame.pack(fill=tk.X, padx=5, pady=5)

        # Brightness
        ttk.Label(enhance_frame, text="Brightness:").pack(anchor=tk.W)
        self.brightness_var = tk.DoubleVar(value=1.0)
        brightness_scale = ttk.Scale(
            enhance_frame,
            from_=0.1,
            to=3.0,
            variable=self.brightness_var,
            command=self.update_enhancement,
        )
        brightness_scale.pack(fill=tk.X, pady=2)

        # Contrast
        ttk.Label(enhance_frame, text="Contrast:").pack(anchor=tk.W)
        self.contrast_var = tk.DoubleVar(value=1.0)
        contrast_scale = ttk.Scale(
            enhance_frame,
            from_=0.1,
            to=3.0,
            variable=self.contrast_var,
            command=self.update_enhancement,
        )
        contrast_scale.pack(fill=tk.X, pady=2)

        ttk.Button(
            enhance_frame, text="Reset Enhancement", command=self.reset_enhancement
        ).pack(fill=tk.X, pady=5)

        # Band selection for multi-band images
        band_frame = ttk.LabelFrame(controls_frame, text="Band Selection")
        band_frame.pack(fill=tk.X, padx=5, pady=5)

        self.band_var = tk.IntVar(value=0)
        self.band_spinbox = ttk.Spinbox(
            band_frame,
            from_=0,
            to=0,
            textvariable=self.band_var,
            command=self.change_band,
            width=10,
        )
        self.band_spinbox.pack(fill=tk.X, pady=2)

        self.band_info_label = ttk.Label(band_frame, text="Single band image")
        self.band_info_label.pack(anchor=tk.W, pady=2)

        # Histogram Tab
        hist_frame = ttk.Frame(notebook)
        notebook.add(hist_frame, text="Histogram")

        # Matplotlib figure for histogram
        self.hist_fig = Figure(figsize=(5, 4), dpi=80)
        self.hist_ax = self.hist_fig.add_subplot(111)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, hist_frame)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ttk.Button(
            hist_frame, text="Update Histogram", command=self.update_histogram
        ).pack(pady=5)

    def open_file(self):
        file_path = filedialog.askopenfilename(
            title="Select TIFF File",
            filetypes=[
                ("TIFF files", "*.tif *.tiff *.TIF *.TIFF"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            try:
                self.original_image = Image.open(file_path)
                self.current_image = self.original_image.copy()
                self.current_file = file_path

                # Update file label
                filename = os.path.basename(file_path)
                self.file_label.config(text=filename, foreground="black")

                # Reset view parameters
                self.zoom_factor = 1.0
                self.brightness = 1.0
                self.contrast = 1.0
                self.current_band = 0

                # Update controls
                self.brightness_var.set(1.0)
                self.contrast_var.set(1.0)
                self.band_var.set(0)

                # Update band selection for multi-band images
                if hasattr(self.original_image, "n_frames"):
                    max_frames = self.original_image.n_frames - 1
                    self.band_spinbox.config(to=max_frames)
                    self.band_info_label.config(text=f"Frames: 0-{max_frames}")
                elif len(self.original_image.getbands()) > 1:
                    max_bands = len(self.original_image.getbands()) - 1
                    self.band_spinbox.config(to=max_bands)
                    self.band_info_label.config(
                        text=f"Bands: {self.original_image.getbands()}"
                    )
                else:
                    self.band_spinbox.config(to=0)
                    self.band_info_label.config(text="Single band image")

                self.display_image()
                self.update_image_info()
                self.update_histogram()
                self.status_label.config(text=f"Loaded: {filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to open file: {str(e)}")

    def display_image(self):
        if not self.current_image:
            return

        try:
            # Apply enhancements
            enhanced_image = self.current_image.copy()

            # Brightness and contrast
            if self.brightness != 1.0:
                enhancer = ImageEnhance.Brightness(enhanced_image)
                enhanced_image = enhancer.enhance(self.brightness)

            if self.contrast != 1.0:
                enhancer = ImageEnhance.Contrast(enhanced_image)
                enhanced_image = enhancer.enhance(self.contrast)

            # Calculate display size
            display_width = int(enhanced_image.width * self.zoom_factor)
            display_height = int(enhanced_image.height * self.zoom_factor)

            # Resize for display if needed
            if self.zoom_factor != 1.0:
                display_image = enhanced_image.resize(
                    (display_width, display_height), Image.LANCZOS
                )
            else:
                display_image = enhanced_image

            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(display_image)

            # Update canvas
            self.canvas.delete("all")
            self.canvas_image_id = self.canvas.create_image(
                0, 0, anchor=tk.NW, image=self.photo
            )

            # Update scroll region
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

            # Update zoom label
            zoom_percent = int(self.zoom_factor * 100)
            self.zoom_label.config(text=f"Zoom: {zoom_percent}%")

        except Exception as e:
            messagebox.showerror("Display Error", f"Failed to display image: {str(e)}")

    def update_image_info(self):
        if not self.original_image:
            return

        info = f"File: {os.path.basename(self.current_file)}\n"
        info += f"Format: {self.original_image.format}\n"
        info += f"Mode: {self.original_image.mode}\n"
        info += f"Size: {self.original_image.width} x {self.original_image.height}\n"

        if hasattr(self.original_image, "n_frames"):
            info += f"Frames: {self.original_image.n_frames}\n"

        # Get image info/metadata
        if hasattr(self.original_image, "info") and self.original_image.info:
            info += "\nMetadata:\n"
            for key, value in self.original_image.info.items():
                info += f"  {key}: {value}\n"

        # File size
        if self.current_file:
            file_size = os.path.getsize(self.current_file)
            if file_size > 1024 * 1024:
                size_str = f"{file_size / (1024 * 1024):.1f} MB"
            elif file_size > 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size} bytes"
            info += f"\nFile Size: {size_str}\n"

        # Image statistics if possible
        try:
            if self.original_image.mode in ["L", "RGB", "RGBA"]:
                img_array = np.array(self.original_image)
                if len(img_array.shape) == 2:  # Grayscale
                    info += f"\nStatistics:\n"
                    info += f"  Min: {np.min(img_array)}\n"
                    info += f"  Max: {np.max(img_array)}\n"
                    info += f"  Mean: {np.mean(img_array):.2f}\n"
                    info += f"  Std: {np.std(img_array):.2f}\n"
                elif len(img_array.shape) == 3:  # Color
                    info += f"\nColor Statistics:\n"
                    for i, channel in enumerate(
                        ["Red", "Green", "Blue"][: img_array.shape[2]]
                    ):
                        channel_data = img_array[:, :, i]
                        info += f"  {channel} - Min: {np.min(channel_data)}, Max: {np.max(channel_data)}, Mean: {np.mean(channel_data):.2f}\n"
        except:
            pass

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info)

    def update_histogram(self):
        if not self.current_image:
            return

        try:
            self.hist_ax.clear()

            img_array = np.array(self.current_image)

            if len(img_array.shape) == 2:  # Grayscale
                self.hist_ax.hist(
                    img_array.flatten(), bins=256, alpha=0.7, color="gray"
                )
                self.hist_ax.set_title("Histogram")
            elif len(img_array.shape) == 3:  # Color
                colors = ["red", "green", "blue"]
                for i in range(min(3, img_array.shape[2])):
                    self.hist_ax.hist(
                        img_array[:, :, i].flatten(),
                        bins=256,
                        alpha=0.5,
                        color=colors[i],
                        label=colors[i].capitalize(),
                    )
                self.hist_ax.legend()
                self.hist_ax.set_title("Color Histogram")

            self.hist_ax.set_xlabel("Pixel Value")
            self.hist_ax.set_ylabel("Frequency")
            self.hist_canvas.draw()

        except Exception as e:
            self.hist_ax.clear()
            self.hist_ax.text(
                0.5,
                0.5,
                f"Error creating histogram:\n{str(e)}",
                ha="center",
                va="center",
                transform=self.hist_ax.transAxes,
            )
            self.hist_canvas.draw()

    def zoom_in(self):
        self.zoom_factor *= 1.25
        self.display_image()

    def zoom_out(self):
        self.zoom_factor *= 0.8
        self.display_image()

    def zoom_wheel(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def fit_to_window(self, event=None):
        if not self.current_image:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            zoom_x = canvas_width / self.current_image.width
            zoom_y = canvas_height / self.current_image.height
            self.zoom_factor = min(zoom_x, zoom_y) * 0.95  # 95% to leave some margin
            self.display_image()

    def actual_size(self):
        self.zoom_factor = 1.0
        self.display_image()

    def reset_view(self):
        self.zoom_factor = 1.0
        self.brightness = 1.0
        self.contrast = 1.0
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        self.display_image()

    def start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def do_pan(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def update_enhancement(self, event=None):
        self.brightness = self.brightness_var.get()
        self.contrast = self.contrast_var.get()
        self.display_image()

    def reset_enhancement(self):
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        self.update_enhancement()

    def change_band(self):
        if not self.original_image:
            return

        try:
            band_num = self.band_var.get()

            if (
                hasattr(self.original_image, "n_frames")
                and self.original_image.n_frames > 1
            ):
                # Multi-frame TIFF
                self.original_image.seek(band_num)
                self.current_image = self.original_image.copy()
            elif len(self.original_image.getbands()) > 1:
                # Multi-band image
                bands = self.original_image.split()
                if band_num < len(bands):
                    self.current_image = bands[band_num]

            self.display_image()
            self.update_histogram()

        except Exception as e:
            messagebox.showerror("Band Error", f"Failed to change band: {str(e)}")

    def save_current_view(self):
        if not self.current_image:
            messagebox.showwarning("Warning", "No image loaded")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Current View",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("TIFF files", "*.tif"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            try:
                # Apply current enhancements
                save_image = self.current_image.copy()

                if self.brightness != 1.0:
                    enhancer = ImageEnhance.Brightness(save_image)
                    save_image = enhancer.enhance(self.brightness)

                if self.contrast != 1.0:
                    enhancer = ImageEnhance.Contrast(save_image)
                    save_image = enhancer.enhance(self.contrast)

                save_image.save(file_path)
                self.status_label.config(text=f"Saved: {os.path.basename(file_path)}")

            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save image: {str(e)}")


def main():
    root = tk.Tk()
    app = TIFFViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
