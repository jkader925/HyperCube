#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 15:01:22 2025

@author: justin
"""

import sys
import numpy as np
import pandas as pd
import ast  # For safely evaluating strings as Python literals
import re
from functools import partial
import csv
import gc
import psutil
import os 
from pathlib import Path
import warnings

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.patches as patches

from astropy.io import fits
from astropy.wcs import WCS

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QVBoxLayout, QPushButton, QLineEdit, QScrollArea, QGridLayout,
    QWidget, QLabel, QFrame, QMenu, QTextEdit, QMainWindow,
    QHBoxLayout, QMenuBar, QProgressBar, QDialog, QSplashScreen,
    QAction, QSplitter, QFileDialog, QApplication)
from PyQt5.QtGui import QFontMetrics, QPixmap
from PyQt5.QtGui import QKeySequence

from lmfit import Model, Parameters

import HyperCube_ModelFunctions







# Global variables for application-wide FITS access
global FITS_HEADER, FITS_DATA
FITS_HEADER = None
FITS_DATA = None

global viewing_fit
viewing_fit = False

global spectral_count
spectral_count = None

global df_obs, df, df_cont, df_fit

global fit_results
fit_results = []
constraints = []

global spectrum

global snr_map
global snr_value
snr_value = 0

data_observation_init = {'sourcename': [],
                    'redshift': [],
                    'resolvingpower': []}

df_obs = pd.DataFrame(data_observation_init)


# Sample data for continuum and lines
data_cont_init = {'Continuum Name': [],
             'x1': [],
             'x2': [],
             'Slope_0': [],
             'Intercept_0': [],
             'Slope_fit': [],
             'Intercept_fit': [],
             'region_ID': [],
             'lineactor:': []}

df_cont = pd.DataFrame(data_cont_init)

data_lines_init = {'Line_ID': [],
        'Line_Name': [],
        'SNR': [],
        'Rest Wavelength':[],
        'Amp_0': [],
        'Amp_0_lowlim': [],
        'Amp_0_highlim': [],
        'Centroid_0': [],
        'Centroid_0_lowlim': [],
        'Centroid_0_highlim': [],
        'Sigma_0': [],
        'Sigma_0_lowlim': [],
        'Sigma_0_highlim': [],
        'Constraints': np.array(5),
        'Amp_fit': [],
        'Centroid_fit': [],
        'Sigma_fit': [],
        'region_ID': [],
        'curveactor': []}

df = pd.DataFrame(data_lines_init)


df_fit = pd.DataFrame({})


def convert_velocity_to_centroid(velocity_constraint, line_name, df, galaxy_redshift):
    """
    Convert velocity-based constraints to centroid-based constraints.
    
    Parameters:
    - velocity_constraint (str): The velocity constraint (e.g., "vel == vel_[sii_1]")
    - line_name (str): The current line for which we're modifying the constraints
    - df (DataFrame): The DataFrame containing line information
    - galaxy_redshift (float): The galaxy's redshift (z)
    
    Returns:
    - str: A new centroid-based constraint.
    """
    # Extract the reference line from the constraint (e.g., "sii_1" from "vel == vel_[sii_1]")
    match = re.search(r"vel == vel_\[([\w\d_]+)\]", velocity_constraint)
    if not match:
        return velocity_constraint  # Return unchanged if no match
    
    ref_line_name = match.group(1)

    # Get rest wavelengths of the current and reference lines
    current_rest_wl = np.float64(df.loc[df['Line_Name'] == line_name, 'Rest Wavelength'].values[0])
    ref_rest_wl = np.float64(df.loc[df['Line_Name'] == ref_line_name, 'Rest Wavelength'].values[0])

    # Ensure the reference line exists
    if not ref_rest_wl:
        raise ValueError(f"Reference line {ref_line_name} not found in DataFrame.")

    # Convert velocity to centroid relation
    ref_cen_param = f"cen_[{ref_line_name}]"
    current_cen_param = "cen"

    # Centroid relation: λ_obs = λ_rest * (1 + z) * (1 + v/c)
    # Since v_ref == v_current, we derive:
    centroid_relation = f"{current_cen_param} == {ref_cen_param} * {current_rest_wl / ref_rest_wl:.6f}"

    return centroid_relation

def update_constraints_with_velocity(df, galaxy_redshift):
    """
    Update DataFrame constraints by converting velocity constraints to centroid constraints.
    
    Parameters:
    - df (DataFrame): Input DataFrame with Gaussian fitting information.
    - galaxy_redshift (float): The galaxy redshift.
    
    Returns:
    - DataFrame: Updated DataFrame with corrected centroid constraints.
    """
    updated_constraints = []

    for idx, row in df.iterrows():
        line_name = row['Line_Name']
        rest_wavelength = np.float64(row['Rest Wavelength'])
        constraints = row['constraints']
        
        wl_emitted_redshift = rest_wavelength * (1 + galaxy_redshift)
        
        
        # If 'vel' appears in constraints, process it
        if any('vel' in c for c in constraints):
            new_constraints = [
                convert_velocity_to_centroid(c, line_name, df, galaxy_redshift) if 'vel' in c else c
                for c in constraints
            ]
        else:
            new_constraints = constraints

        print(f'centroid constraint converted from velocity: {new_constraints}')
        updated_constraints.append(new_constraints)

    # Update the DataFrame
    df['constraints'] = updated_constraints
    return df


def add_dataframe_constraints_to_params(df, params):
    """
    Adds constraints to lmfit Parameters object, using:
    - Additive offsets for centroid inequalities (cen1 >= cen2 → cen1 = cen2 + offset)
    - Multiplicative ratios for all other inequalities (amp1 >= amp2 → amp1 = amp2 * ratio)
    - Direct expressions for equality constraints
    """
    if 'constraints' not in df.columns:
        print("Warning: 'constraints' column not found in DataFrame.")
        return

    for index, row in df.iterrows():
        line_id = int(row['Line_ID'])
        line_name = row['Line_Name']
        constraints_list_str = row['constraints']

        try:
            # Parse constraints string into list
            if isinstance(constraints_list_str, str):
                constraints_list = ast.literal_eval(constraints_list_str)
            elif isinstance(constraints_list_str, list):
                constraints_list = constraints_list_str
            else:
                constraints_list = ['', '', '', '', '']

            if not isinstance(constraints_list, list) or len(constraints_list) != 5:
                print(f"Warning: Invalid constraints format for Line_ID {line_id} ({line_name}): {constraints_list_str}")
                continue

            for constraint in constraints_list:
                constraint = constraint.strip()
                if not constraint:
                    continue

                # Split into left and right parts of the main operator
                operators = ['<=', '<', '>=', '>', '==']
                found_op = None
                for op in operators:
                    if op in constraint:
                        parts = constraint.split(op, 1)
                        found_op = op
                        break

                if not found_op:
                    print(f"Warning: Unsupported constraint format (no valid operator): {constraint}")
                    continue

                left_side = parts[0].strip()
                right_side = parts[1].strip()

                # Parse left side parameter
                param1_base = ''.join(filter(str.isalpha, left_side)).lower()
                param1_num = line_id + 1
                param1_name = f"{param1_base}{param1_num}"

                if param1_name not in params:
                    print(f"Warning: Parameter not found in params: {param1_name}")
                    continue

                # SPECIAL CASE: Centroid inequalities use additive offsets
                if param1_base == 'cen' and found_op in ['>=', '>', '<=', '<']:
                    if '_[' in right_side and ']' in right_side:
                        # Reference to another line's centroid
                        param2_base = ''.join(filter(str.isalpha, right_side.split('_[')[0])).lower()
                        other_line_name = right_side.split('_[')[1].split(']')[0]
                        
                        try:
                            other_row = df[df['Line_Name'] == other_line_name].iloc[0]
                            other_line_id = int(other_row['Line_ID'])
                            param2_num = other_line_id + 1
                            param2_name = f"{param2_base}{param2_num}"
                            
                            if param2_name not in params:
                                print(f"Warning: Parameter not found in params: {param2_name}")
                                continue

                            # Get current centroid values
                            cen1_init = params[param1_name].value
                            cen2_init = params[param2_name].value
                            
                            # Calculate initial offset value (ensure it's positive)
                            initial_offset = max(0.1, abs(cen1_init - cen2_init))
                            
                            # Determine max offset based on spectral range constraints
                            if found_op in ['>=', '>']:
                                # cen1 >= cen2 → cen1 = cen2 + offset
                                # Max offset should keep cen1 within x1_end
                                if 'x1_end' in params:
                                    max_offset = max(initial_offset, params['x1_end'].value - cen2_init)
                                else:
                                    max_offset = 100.0  # fallback
                                
                                offset_name = f"offset_{param2_name}_{param1_name}"
                                if offset_name not in params:
                                    params.add(
                                        offset_name,
                                        value=initial_offset,
                                        min=0.0,
                                        max=max_offset,
                                        vary=True
                                    )
                                params[param1_name].expr = f"{param2_name} + {offset_name}"
                                print(f"Centroid constraint: {param1_name} >= {param2_name} "
                                      f"(offset={initial_offset:.1f}, max={max_offset:.1f})")
                            
                            elif found_op in ['<=', '<']:
                                # cen1 <= cen2 → cen1 = cen2 - offset
                                # Max offset should keep cen1 within x1_start
                                if 'x1_start' in params:
                                    max_offset = max(initial_offset, cen2_init - params['x1_start'].value)
                                else:
                                    max_offset = 100.0  # fallback
                                
                                offset_name = f"offset_{param1_name}_{param2_name}"
                                if offset_name not in params:
                                    params.add(
                                        offset_name,
                                        value=initial_offset,
                                        min=0.0,
                                        max=max_offset,
                                        vary=True
                                    )
                                params[param1_name].expr = f"{param2_name} - {offset_name}"
                                print(f"Centroid constraint: {param1_name} <= {param2_name} "
                                      f"(offset={initial_offset:.1f}, max={max_offset:.1f})")
                        
                        except (IndexError, KeyError) as e:
                            print(f"Warning: Could not find line '{other_line_name}' for constraint: {constraint} - {e}")
                    else:
                        print(f"Warning: Centroid constraint must reference another line: {constraint}")
                    continue

                # DEFAULT CASE: Non-centroid parameters use ratio method
                # Parse right side (could be simple parameter or complex expression)
                if '_[' in right_side and ']' in right_side:
                    # Case 1: Reference to another line parameter (e.g., amp_[Halpha_c1])
                    param2_base = ''.join(filter(str.isalpha, right_side.split('_[')[0])).lower()
                    other_line_name = right_side.split('_[')[1].split(']')[0]
                    
                    try:
                        other_row = df[df['Line_Name'] == other_line_name].iloc[0]
                        other_line_id = int(other_row['Line_ID'])
                        param2_num = other_line_id + 1
                        param2_name = f"{param2_base}{param2_num}"
                        
                        if param2_name not in params:
                            print(f"Warning: Parameter not found in params: {param2_name}")
                            continue
                            
                        # Handle multiplicative factors
                        if '*' in right_side:
                            left_part, right_part = [x.strip() for x in right_side.split('*')]
                        
                            # Check which part is numeric and which is the parameter
                            if left_part.replace('.', '', 1).isdigit():
                                factor = float(left_part)
                                expr = f"{factor} * {param2_name}"
                            elif right_part.replace('.', '', 1).isdigit():
                                factor = float(right_part)
                                expr = f"{factor} * {param2_name}"
                            else:
                                print(f"Warning: Could not parse factor in constraint: {constraint}")
                                continue
                        else:
                            expr = param2_name
                            
                        # Apply the constraint
                        if found_op in ['<=', '<']:
                            # For A <= B, we express A = factor * B where factor <= 1
                            factor_name = f"ratio_{param1_name}_{param2_name}"
                            if factor_name not in params:
                                params.add(factor_name, value=0.9, min=0.0, max=1.0, vary=True)
                            params[param1_name].expr = f"{factor_name} * {expr}"
                            print(f"Constraint added: {param1_name} <= {expr} (using factor {factor_name})")
                        elif found_op in ['>=', '>']:
                            # For A >= B, we express A = B / factor where factor <= 1
                            factor_name = f"ratio_{param2_name}_{param1_name}"
                            if factor_name not in params:
                                params.add(factor_name, value=0.9, min=0.0, max=1.0, vary=True)
                            params[param1_name].expr = f"{expr} / {factor_name}"
                            print(f"Constraint added: {param1_name} >= {expr} (using factor {factor_name})")
                        elif found_op == '==':
                            params[param1_name].expr = expr
                            print(f"Constraint added: {param1_name} == {expr}")
                            
                    except (IndexError, KeyError):
                        print(f"Warning: Could not find line '{other_line_name}' for constraint: {constraint}")
                
                elif any(op in right_side for op in ['+', '-', '*', '/']):
                    # Case 2: Mathematical expression (e.g., "0.1 * amp_[Halpha_c1]")
                    try:
                        # Find all parameter references in the expression
                        expr_parts = right_side.split()
                        parsed_expr = []
                        
                        for part in expr_parts:
                            if '_[' in part and ']' in part:
                                # It's a parameter reference
                                param_base = ''.join(filter(str.isalpha, part.split('_[')[0])).lower()
                                other_line_name = part.split('_[')[1].split(']')[0]
                                
                                other_row = df[df['Line_Name'] == other_line_name].iloc[0]
                                other_line_id = int(other_row['Line_ID'])
                                param_num = other_line_id + 1
                                param_name = f"{param_base}{param_num}"
                                
                                if param_name not in params:
                                    print(f"Warning: Parameter not found in params: {param_name}")
                                    raise ValueError
                                    
                                parsed_expr.append(param_name)
                            else:
                                parsed_expr.append(part)
                                
                        expr = ' '.join(parsed_expr)
                        
                        # Apply the constraint
                        if found_op in ['<=', '<']:
                            factor_name = f"ratio_{param1_name}_expr_{hash(expr)}"[:30]
                            if factor_name not in params:
                                params.add(factor_name, value=0.9, min=0.0, max=1.0, vary=True)
                            params[param1_name].expr = f"{factor_name} * ({expr})"
                            print(f"Constraint added: {param1_name} <= {expr} (using factor {factor_name})")
                        elif found_op in ['>=', '>']:
                            factor_name = f"ratio_expr_{hash(expr)}_{param1_name}"[:30]
                            if factor_name not in params:
                                params.add(factor_name, value=1.1, min=1.0, max=10.0, vary=True)
                            params[param1_name].expr = f"({expr}) / {factor_name}"
                            print(f"Constraint added: {param1_name} >= {expr} (using factor {factor_name})")
                        elif found_op == '==':
                            params[param1_name].expr = f"({expr})"
                            print(f"Constraint added: {param1_name} == {expr}")
                            
                    except (ValueError, IndexError, KeyError):
                        print(f"Warning: Could not parse complex expression in constraint: {constraint}")
                
                else:
                    print(f"Warning: Unsupported right side format in constraint: {constraint}")

        except (ValueError, SyntaxError) as e:
            print(f"Warning: Could not parse constraints string for Line_ID {line_id} ({line_name}): {constraints_list_str} - {e}")



def generalized_model(x, slope, intercept, *gaussian_params):
    """
    Generalized Gaussian model for N Gaussians.
    :param x: Wavelength data.
    :param slope: Continuum slope.
    :param intercept: Continuum intercept.
    :param gaussian_params: Model parameters for N Gaussians [Amp_0, Centroid_0, Sigma_0, ...]
    """
    model = slope * x + intercept  # Linear model for continuum
    num_gaussians = len(gaussian_params) // 3
    for i in range(num_gaussians):
        amp = gaussian_params[3 * i]
        centroid = gaussian_params[3 * i + 1]
        sigma = gaussian_params[3 * i + 2]
        model += amp * np.exp(-(x - centroid)**2 / (2 * sigma**2))

    return model

# def load_stylesheet(app, filename):
#     with open(filename, 'r') as f:
#         style = f.read()
#         app.setStyleSheet(style)

# Apply PyQt5-like styling to Matplotlib
def apply_mpl_qss_style(fig, ax, line):
    """Apply a PyQt5-style QSS theme to the given Matplotlib figure, axes, and line."""
    
    # Set the figure (entire canvas) background color
    fig.patch.set_facecolor('#323232')  # Matches QWidget background
    
    # Axis background color
    ax.set_facecolor('#2E3440')  
    
    # Axis labels and title (matches QWidget text color)
    ax.xaxis.label.set_color('#b1b1b1')
    ax.yaxis.label.set_color('#b1b1b1')
    ax.title.set_color('#b1b1b1')
    
    # Ticks (color and size)
    ax.tick_params(axis='x', colors='#b1b1b1', labelsize=12)
    ax.tick_params(axis='y', colors='#b1b1b1', labelsize=12)
    
    # Axis borders (spines)
    for spine in ax.spines.values():
        spine.set_edgecolor('#b1b1b1')
        spine.set_linewidth(0.75)
    
    # Grid (optional – enable for a clearer view)
    ax.grid(color='#4C566A', linestyle='--', linewidth=0.5)
    
    if line != None:
        # Adjust the spectrum line styling
        line.set_color('whitesmoke')  # Hover color from QPushButton
        line.set_linewidth(0.75)  # Slightly thicker for better visibility
    
    # Legend styling (if needed)
    # legend = ax.legend()
    # if legend:
    #     legend.get_frame().set_facecolor('#2E3440')
    #     legend.get_frame().set_edgecolor('#D8DEE9')
    #     plt.setp(legend.get_texts(), color='#D8DEE9')

    # Tighten layout to ensure good spacing
    fig.tight_layout()

class SpaxelButton(QPushButton):
    def __init__(self, text, col, *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        self.col = col

class FrameButton(QPushButton):
    def __init__(self, text, row, col, frame_id, feature, *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        self.row = row
        self.col = col
        self.frame_id = frame_id  # Store frame information in the button
        self.feature = feature


class ViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.fit_params_window = None  # Placeholder, initially None
        self.fits_header = None
        self.fits_data = None
        self.current_spaxel = None  # Stores the currently highlighted spaxel (x, y)
        self.locked = False  # Initially unlocked
        self.spectrum_cursor_pos = None  # Add tracking for spectrum cursor position
        self.spectrum_ax = None
        self.drawing_line = None
        self.slope = 0  # Default slope
        self.intercept = 0  # Default intercept
        self.gaussian_active = False  # Toggle for Gaussian fitting
        self.current_gaussian = None  # Store the Gaussian plot
        self.initial_x = None  # Stores initial cursor position for Gaussian
        self.current_sigma = 10  # Default sigma
        self.current_amplitude = 1  # Default amplitude
        self.line_region = None
        
        self.zoom_rect = None  # Store the rectangle
        self.zoom_start_x = None  # Start position for zoom
        self.zoom_active = False
        self.zoom_limits = None  # Store zoom limits globally
        self.zoom_pad = 0.2
        self.modelfit = None
        
        self.initUI()

    def initUI(self):
        self.setWindowTitle("HyperCube")
        self.setGeometry(100, 100, 1000, 600)

        # Main layout
        main_layout = QVBoxLayout()

        # Splitter for two large panels
        self.splitter = QSplitter(Qt.Horizontal)

        # Left panel for white-light image
        self.left_panel = QWidget(self)
        self.left_layout = QVBoxLayout(self.left_panel)
        self.canvas = FigureCanvas(plt.Figure())  # Matplotlib figure canvas
        self.left_layout.addWidget(self.canvas)
        self.left_panel.setLayout(self.left_layout)  # Ensure layout is set
        self.splitter.addWidget(self.left_panel)

        # Right panel for spectrum
        self.right_panel = QWidget(self)
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_panel.setLayout(self.right_layout)  # Ensure layout is set
        self.splitter.addWidget(self.right_panel)

        # Set equal size stretch factors (ensures both panels start equally sized)
        self.splitter.setSizes([300, 300])  # Start with equal sizes
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)

        main_layout.addWidget(self.splitter)

        # Bottom buttons layout
        bottom_layout = QHBoxLayout()

        # Small "Open Fit Params" button (lower-left corner)
        self.open_fit_params_button = QPushButton("Fit Params", self)
        self.open_fit_params_button.setFixedSize(80, 30)
        self.open_fit_params_button.clicked.connect(self.open_fit_params_window)
        bottom_layout.addWidget(self.open_fit_params_button, alignment=Qt.AlignLeft)

        # Spacer to push buttons apart
        bottom_layout.addStretch()

        # "Open FITS File" button (lower-right corner)
        self.open_fits_button = QPushButton("Open FITS", self)
        self.open_fits_button.setFixedSize(120, 40)
        self.open_fits_button.clicked.connect(self.open_fits_file)
        bottom_layout.addWidget(self.open_fits_button, alignment=Qt.AlignRight)

        # self.open_fits_file()

        main_layout.addLayout(bottom_layout)

        # Set up central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Set menu bar
        self.setMenuBar(self.create_menu_bar_VisualizerWindow())

        # Ensure the window comes to the front
        # self.show()
        self.raise_()
        self.activateWindow()

        # Enable mouse tracking for the white-light image
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        self.cursor_pos = None

    def set_fit_params_window(self, fit_params_window):
        """Store the reference to FitParamsWindow after it's created."""
        self.fit_params_window = fit_params_window


    def on_mouse_move(self, event):
        global spectrum
        """Update rectangle position and spectrum in right panel."""
        if self.locked:
            return  # Do nothing if locked
        
        if event.inaxes:
            x, y = int(event.xdata), int(event.ydata)
            self.cursor_pos = (x, y)
            
            if self.fits_data is not None:
                self.current_spaxel = (x, y)
                spectrum = self.get_spectrum_at_spaxel(x, y)
                self.update_spectrum_panel(spectrum)
                
                # Update red rectangle position
                self.red_rect.set_xy((x - 0.5, y - 0.5))
                self.red_rect.set_visible(True)
                self.canvas.draw_idle()
                
                # Now update the button texts with the new cursor position
                if self.fit_params_window is not None:
                    self.update_buttons(x, y)




    def on_spectrum_mouse_move(self, event):
        """Handle mouse movement in the spectrum plot."""
        if event.inaxes == self.spectrum_ax:
            # Update the cursor position in the spectrum plot
            self.spectrum_cursor_pos = (event.xdata, event.ydata)
    
            # If we are in the middle of drawing a line, update the line's end position
            if self.drawing_line and self.current_line:
                # Get current position and update the line end coordinates
                end_x, end_y = self.spectrum_cursor_pos
                self.current_line.set_xdata([self.line_start[0], end_x])
                self.current_line.set_ydata([self.line_start[1], end_y])
    
                # Redraw the canvas to reflect the updated line
                self.spectrum_canvas.draw()


    def on_mouse_press(self, event):
        """Start zoom selection."""
        if event.button == 1 and event.inaxes == self.spectrum_ax:
            self.zoom_start_x = event.xdata
            if self.zoom_rect:
                self.zoom_rect.remove()
            self.zoom_rect = self.spectrum_ax.axvspan(self.zoom_start_x, self.zoom_start_x, color='grey', alpha=0.3)
            self.spectrum_canvas.draw_idle()

    def on_mouse_drag(self, event):
        """Update zoom selection rectangle dynamically."""
        if event.button == 1 and event.inaxes == self.spectrum_ax and self.zoom_start_x is not None:
            x_end = event.xdata
            if x_end is not None:
                if self.zoom_rect:
                    self.zoom_rect.remove()
                self.zoom_rect = self.spectrum_ax.axvspan(self.zoom_start_x, x_end, color='grey', alpha=0.3)
                self.spectrum_canvas.draw_idle()

    def on_mouse_release(self, event):
        """Apply zoom when mouse is released."""
        if event.button == 1 and event.inaxes == self.spectrum_ax and self.zoom_start_x is not None:
            zoom_end_x = event.xdata
            if zoom_end_x is not None and self.zoom_start_x != zoom_end_x:
                x_min, x_max = min(self.zoom_start_x, zoom_end_x), max(self.zoom_start_x, zoom_end_x)
                self.spectrum_ax.set_xlim(x_min, x_max)
                self.zoom_limits = (x_min, x_max)  # Store zoom limits globally
                self.zoom_active = True
    
                # Dynamically adjust the y-axis based on the zoomed x-axis range
                mask = (wavelengths >= x_min) & (wavelengths <= x_max)
                if np.any(mask):  # Ensure there are valid points in the selected range
                    y_min, y_max = np.nanmin(self.spectrum_line.get_ydata()[mask]), np.nanmax(self.spectrum_line.get_ydata()[mask])
                else:  # If no valid points, use full y-range
                    y_min, y_max = np.nanmin(self.spectrum_line.get_ydata()), np.nanmax(self.spectrum_line.get_ydata())
    
                padding = self.zoom_pad * (y_max - y_min)
                self.spectrum_ax.set_ylim(y_min - padding/2, y_max + padding)
    
                self.spectrum_ax.figure.canvas.draw_idle()  # Ensure immediate update
    
            if self.zoom_rect:
                self.zoom_rect.remove()
                self.zoom_rect = None
            self.zoom_start_x = None

    def show_context_menu(self, event):
        if event.button == 3:  # Right-click
            menu = QMenu(self)
            reset_action = menu.addAction("Reset Zoom")
            action = menu.exec_(self.spectrum_canvas.mapToGlobal(event.guiEvent.pos()))
            if action == reset_action:
                self.reset_zoom()

    def reset_zoom(self):
        """Reset zoom to the full spectrum range for both x and y axes."""
        self.spectrum_ax.set_xlim(wavelengths.min(), wavelengths.max())
    
        # Compute full y-range and apply padding
        y_min, y_max = np.nanmin(self.spectrum_line.get_ydata()), np.nanmax(self.spectrum_line.get_ydata())
        padding = self.zoom_pad * (y_max - y_min)
        self.spectrum_ax.set_ylim(y_min - padding/2, y_max + padding)
    
        self.zoom_active = False
        self.spectrum_canvas.draw_idle()  # Ensure the reset happens immediately


    def open_fits_file(self):
        """Opens a FITS file and stores header & data"""
        global FITS_HEADER, FITS_DATA, wavelengths, snr_map
        print('Opening file...')
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "FITS Files (*.fits);;All Files (*)", options=options)
        
        # file_path = '../../Images/IIIZw035/IIIZw035_coadd_BH1L.fits'
        # file_path = '../../Images/IIIZw035/IIIZw035_coadd_BH1L_crop.fits'
        # file_path = '../../Images/IIIZw035/IIIZw035_coadd_RH2L_crop.fits'
        # file_path = '../../Images/IIIZw035/IIIZw035_coadd_RH2L_northernComp.fits'
        # file_path = '../../Images/VV114/MUSE_Deep_20161013_WCSFix.fits'
        # file_path = '../../Images/VV114/VV114_MUSE_Crop5.fits'
        # file_path = '../../Images/Mrk273/Mrk273_coadd_BH3L.fits'
        # file_path = '../../Images/IRAS_F09320+6134/UGC05101_coadd_BH1L.fits'
        
        if file_path:
            with fits.open(file_path) as hdul:
                self.fits_header = None
                self.fits_data = None
        
                # Try to find a valid WCS header
                for ext in range(len(hdul)):
                    header = hdul[ext].header
                    data = hdul[ext].data
        
                    # Skip extensions with no usable data
                    if data is None or np.size(data) <= 1:
                        continue
                    
                    # Check for common WCS keywords
                    if all(key in header for key in ['CRVAL1', 'CRVAL2', 'CTYPE1', 'CTYPE2']):
                        print(f"Using WCS from extension {ext}")
                        self.fits_header = header
                        self.fits_data = data
                        break  # Stop searching once a valid WCS is found
                
                # If no valid WCS was found, default to the primary extension
                if self.fits_header is None:
                    print("No valid WCS found, defaulting to primary extension.")
                    self.fits_header = hdul[0].header
                    self.fits_data = hdul[0].data if np.size(hdul[0].data) > 1 else hdul[1].data
                    
                    
                    
                    
                    
                FITS_HEADER = self.fits_header
                FITS_DATA = self.fits_data# * 1E19
                print("FITS file loaded successfully!")
                self.draw_image(FITS_DATA,cmap='Grays',scale='linear',from_fits=True)
                
                # Initialize WCS from the FITS header
                self.wcs = WCS(self.fits_header)
                
                # Initiate SNR map
                num_wavelengths, nx, ny = np.shape(FITS_DATA)  # Cube dimensions
                snr_map = np.zeros((nx, ny)) + 1E8
                
                # Extract spectral information
                spectral_unit_keys = ["CUNIT3", "BUNIT", "SPECSYS"]
                specunit = None
                for key in spectral_unit_keys:
                    if key in FITS_HEADER:
                        specunit = FITS_HEADER[key]
                        break
    
                if specunit:
                    specunit = specunit.lower()
                
                if specunit == 'angstrom': specunit = r'$\rm{\AA}$'
                if specunit == 'micron': specunit = r'$\mu$m'
                
                # Determine spectral axis and sampling
                spectral_sampling = FITS_HEADER.get('CDELT3', FITS_HEADER.get('CD3_3', None))
                if spectral_sampling is None:
                    raise ValueError("Unable to determine spectral sampling method")
                
                wavelengths = FITS_HEADER['CRVAL3'] + spectral_sampling * (np.arange(FITS_DATA.shape[0]) - FITS_HEADER['CRPIX3'] + 1)
                
                # Extract additional information for the observation
                source_name = FITS_HEADER.get('OBJECT', '')  # Assuming 'OBJECT' holds the source name
                source_redshift = FITS_HEADER.get('REDSHIFT', '')  # Assuming 'REDSHIFT' holds the redshift
                resolving_power = FITS_HEADER.get('RESOLVINGP', '')  # Assuming 'RESOLVINGP' holds resolving power
                
                # Append extracted data to df_obs
                df_obs.loc[0] = [source_name, source_redshift, resolving_power]



    def pixel_to_ra_dec(self, x, y):
        """Convert pixel coordinates (x, y) to RA, Dec using the WCS information from FITS header."""
        # Ensure x, y are numpy arrays (needed for all_pix2world)
        x = np.array(x)
        y = np.array(y)
    
        # For 3D WCS, you need to provide the correct number of dimensions, so we will include a spectral coordinate (e.g., 0)
        # Spectral axis (assuming 0, you can adjust this if needed)
        z = np.zeros_like(x)  # If you have a 3D WCS with a spectral axis, use the corresponding z values
    
        # Use all_pix2world for the 3D WCS (x, y, z)
        ra, dec, _ = self.wcs.all_pix2world(x, y, z, 0)  # 0 is for the spectral axis (CDELT3)
    
        return ra, dec


    # Helper function to convert decimal to sexagesimal
    def decimal_to_sexagesimal(self, ra_dec, is_ra=True):
        """
        Convert decimal RA or Dec to sexagesimal format (RA: h:m:s, Dec: d:m:s).
        :param ra_dec: decimal RA or Dec.
        :param is_ra: True if it's RA (hours), False if it's Dec (degrees).
        :return: sexagesimal string.
        """
        if is_ra:
            # Convert RA from degrees to hours, minutes, seconds
            total_seconds = ra_dec * 3600.0 / 15.0  # RA in hours, 15 degrees per hour
        else:
            # Convert Dec from degrees to degrees, arcminutes, arcseconds
            total_seconds = ra_dec * 3600.0
    
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
    
        if is_ra:
            return f"{hours:02}:{minutes:02}:{seconds:05.2f}"  # RA: h:m:s
        else:
            sign = '+' if ra_dec >= 0 else '-'
            degrees = abs(int(ra_dec))
            minutes = int((abs(ra_dec) * 60) % 60)
            seconds = (abs(ra_dec) * 3600) % 60
            return f"{sign}{degrees:02}:{minutes:02}:{seconds:05.2f}"  # Dec: ±d:m:s


    
    def update_buttons(self, x, y):
        """Update the RA, Dec, X, Y, N values displayed on the buttons based on mouse position."""
        
        # Call pixel_to_ra_dec to get RA, Dec from pixel coordinates
        ra, dec = self.pixel_to_ra_dec(x, y)
        
        # Convert RA and Dec to sexagesimal format
        ra_sexagesimal = self.decimal_to_sexagesimal(ra, is_ra=True)  # Explicitly pass `is_ra=True`
        dec_sexagesimal = self.decimal_to_sexagesimal(dec, is_ra=False)  # Explicitly pass `is_ra=False`
        
        # Update RA and Dec buttons with the calculated values
        self.fit_params_window.ra_button.setText(f"RA: {ra_sexagesimal}")
        self.fit_params_window.dec_button.setText(f"Dec: {dec_sexagesimal}")
        
        # Optionally, update other buttons like X, Y, N with appropriate values if needed
        self.fit_params_window.x_button.setText(f"X: {x}")
        self.fit_params_window.y_button.setText(f"Y: {y}")
        
        # If you have additional functionality for the N button, you can update it as needed.
        n = self.get_spaxel_number(x, y)
        self.fit_params_window.n_button.setText(f"N: {n}")
        
        
        for line in self.spectrum_ax.get_lines():
            if line.get_label() != "_child0":  # Keep Line2D(_child0)
                line.remove()
        self.spectrum_canvas.draw()
        
        # self.rebuild_plot(0,from_file=False,show_init=False,show_fit=True,x=x,y=y)
        for region_ID in np.unique(np.int64(df_cont['region_ID'])):
            # df_cont.loc[(df_cont["region_ID"] == region_ID)]['lineactor'].item().remove()
            self.rebuild_plot(region_ID,from_file=False,show_init=False,show_fit=True,x=x,y=y)


        # Mapping from button keys to the column names in df_fit
        key_to_column_mapping = {
            'Slope_fit': 'slope_fit',
            'Intercept_fit': 'intercept_fit',
            'Line_Name': 'LineName',
            'Amp_fit': 'amp_fit',
            'Centroid_fit': 'cen_fit',
            'Sigma_fit': 'sigma_fit',
            'v_fit': 'vel_fit'
        }
        # print(self.fit_params_window.buttons_dict.items())
        for key, button in self.fit_params_window.buttons_dict.items():
            
            # Only process keys related to 'fit'
            if ('fit' in key[1]) or ('Line_Name' in key[1]):
                
                # Extract the region_ID and gaussian_number from the key
                region_ID = key[0]
                
                if 'continuum' in key[1]:
                    if key[1].split('~')[1] == 'Slope_fit':
                        column_name = 'cont_region'+str(region_ID+1)+'_slope_fit'
                        try:# Query the corresponding value from df_fit for the selected region_ID and gaussian_number
                            dfslice = df_fit.loc[(np.int64(df_fit['spaxel_x']) == x) &
                                               (np.int64(df_fit['spaxel_y']) == y) &
                                               (np.int64(df_fit['region_ID']) == np.int64(region_ID))]
                            value = dfslice.iloc[0][column_name]
                        except:
                            value = ''
                    if key[1].split('~')[1] == 'Intercept_fit':
                        column_name = 'cont_region'+str(region_ID+1)+'_intercept_fit'
                        try:# Query the corresponding value from df_fit for the selected region_ID and gaussian_number
                            dfslice = df_fit.loc[(np.int64(df_fit['spaxel_x']) == x) &
                                               (np.int64(df_fit['spaxel_y']) == y) &
                                               (np.int64(df_fit['region_ID']) == np.int64(region_ID))]
                            value = dfslice.iloc[0][column_name]
                        except:
                            value = ''
                
                    # print(column_name,value)
                    # if 'Name' not in column_name:
                    #     button.setStyleSheet("""
                    #         font-family: "Segoe UI";
                    #         font-size: 10pt;
                    #         border: 2px solid;
                    #         border-color: green;
                    #         border-radius: 5px;
                    #         padding-right: 10px;
                    #         padding-left: 10px;
                    #         padding-top: 5px;
                    #         padding-bottom: 5px;
                    #         background-color: limegreen;
                    #         highlight-color; limegreen;
                    #         color: k;
                    #         font: bold;
                    #         width: 64px;
                    #     """)
                    # Set the button's text
                    if type(value) == str:
                        button.setText(value)
                    else:
                        button.setText(str(round(value, 3)))
                
                
                # Now update the gaussian parameter buttons:
                    
                gaussian_number_str = key[1].split('~')[0]
                
                try:
                    gaussian_number = float(gaussian_number_str)
                except ValueError:
                    continue  # Skip this button if it's not a valid number
        
                # Check if the button's key has a matching mapping in key_to_column_mapping
                if key[1].split('~')[1] in key_to_column_mapping:
                    # Get the corresponding column name
                    column_name = key_to_column_mapping[key[1].split('~')[1]]
                    # print(column_name)
        
                    # # Check if the column exists in df_fit
                    if column_name in df_fit.columns:
                        try:# Query the corresponding value from df_fit for the selected region_ID and gaussian_number
                            dfslice = df_fit.loc[(np.int64(df_fit['spaxel_x']) == x) &
                                               (np.int64(df_fit['spaxel_y']) == y) &
                                               (np.int64(df_fit['region_ID']) == np.int64(region_ID)) & 
                                                (np.int64(df_fit['LineID']) == np.int64(gaussian_number))]
                            value = dfslice[column_name].item()
                            plotcolor = dfslice['color'].item()
                            if 'Name' in column_name:
                                button.setStyleSheet(f"""
                                        background-color: {plotcolor};
                                        color: k;
                                    """)
                        except:
                            value = ''
                # Set the button's style
                # print(column_name,value)

                # if 'Name' not in column_name:
                #     button.setStyleSheet("""
                #         font-family: "Segoe UI";
                #         font-size: 10pt;
                #         border: 2px solid;
                #         border-color: green;
                #         border-radius: 5px;
                #         padding-right: 10px;
                #         padding-left: 10px;
                #         padding-top: 5px;
                #         padding-bottom: 5px;
                #         background-color: limegreen;
                #         highlight-color; limegreen;
                #         color: k;
                #         font: bold;
                #         width: 64px;
                #     """)
                # else:
                #     button.setStyleSheet(f"""
                #         font-family: "Segoe UI";
                #         font-size: 10pt;
                #         border: 2px solid;
                #         border-color: green;
                #         border-radius: 5px;
                #         padding-right: 10px;
                #         padding-left: 10px;
                #         padding-top: 5px;
                #         padding-bottom: 5px;
                #         background-color: {plotcolor};
                #         highlight-color; limegreen;
                #         color: k;
                #         font: bold;
                #         width: 64px;
                #     """)
                # Set the button's text
                if type(value) == str:
                    button.setText(value)
                else:
                    button.setText(str(round(value, 3)))
            else:
                pass
                # print(f"Column {column_name} not found in df_fit!")


    def get_spaxel_number(self, x, y):
        """Get the spaxel number for the given pixel coordinates."""
        # Example: convert (x, y) to a spaxel index (could be linear index or 2D)
        return y #* self.image_width + x  # Example of converting to a 1D spaxel number




    def update_spectrum_panel(self, spectrum):
        """Efficiently update the right panel to show the 1D spectrum."""
        if spectrum is not None:
            if not hasattr(self, 'spectrum_fig'):
                # Initialize the spectrum figure and canvas
                self.spectrum_fig = plt.Figure(figsize=(5, 4))
                self.spectrum_canvas = FigureCanvas(self.spectrum_fig)
                self.spectrum_ax = self.spectrum_fig.add_subplot(111)
                
                # Create the spectrum line (step plot)
                self.spectrum_line, = self.spectrum_ax.step([], [], lw=0.5, color='w')
                
                # Apply the updated QSS styling
                apply_mpl_qss_style(self.spectrum_fig, self.spectrum_ax, self.spectrum_line)
                
                # Set axis labels and title
                self.spectrum_ax.set_title("1D Spectrum at Cursor Position")
                self.spectrum_ax.set_xlabel("Wavelength")
                self.spectrum_ax.set_ylabel("Flux")
                
                # Mouse event connections
                self.spectrum_canvas.mpl_connect('button_press_event', self.on_mouse_press)
                self.spectrum_canvas.mpl_connect('motion_notify_event', self.on_mouse_drag)
                self.spectrum_canvas.mpl_connect('button_release_event', self.on_mouse_release)
                self.spectrum_canvas.mpl_connect("button_press_event", self.show_context_menu)
                self.spectrum_canvas.mpl_connect('motion_notify_event', self.on_spectrum_mouse_move)
                
                # Ensure canvas reflects the QSS style (PyQt5-level styling)
                self.spectrum_canvas.setStyleSheet("background-color: #2E3440; border: none;")
                
                # Add the Matplotlib canvas to the layout
                layout = self.right_panel.layout()
                layout.addWidget(self.spectrum_canvas)

            # Update the data instead of recreating the plot
            self.spectrum_line.set_xdata(wavelengths)
            self.spectrum_line.set_ydata(spectrum)

            if not self.zoom_active:
                # Default full range
                self.spectrum_ax.set_xlim(wavelengths.min(), wavelengths.max())
                y_min, y_max = np.nanmin(spectrum), np.nanmax(spectrum)
            else:
                # Get the zoomed x-axis range
                xlim_min, xlim_max = self.spectrum_ax.get_xlim()
    
                # Select only the part of the spectrum within the zoomed range
                mask = (wavelengths >= xlim_min) & (wavelengths <= xlim_max)
                if np.any(mask):  # Ensure there are valid values in range
                    y_min, y_max = np.nanmin(spectrum[mask]), np.nanmax(spectrum[mask])
                else:  # If no valid data, use default
                    y_min, y_max = np.nanmin(spectrum), np.nanmax(spectrum)
    
            # Apply 10% padding to the y-limits
            padding = self.zoom_pad * (y_max - y_min)
            self.spectrum_ax.set_ylim(y_min - padding/2, y_max + padding)

            # Refresh the figure
            self.spectrum_canvas.draw()

    def keyPressEvent(self, event):
        key = event.text().lower()

        global df_cont, df
        """Handle key press events for locking and drawing a line."""
        if event.key() == Qt.Key_L:
            # Toggle the locking state
            self.locked = not self.locked
            print(f"Lock {'enabled' if self.locked else 'disabled'}")
        
        if event.key() == Qt.Key_F:
            print(df)
            # for line in self.spectrum_ax.get_lines():
            #     print(line)
                
            # self.active_line.remove()
            # self.spectrum_canvas.draw_idle()  # Efficient redrawing
            
            
        if event.key() == Qt.Key_D:
            # Start or finish drawing a line in the spectrum plot
            if not self.drawing_line:
                # Start drawing a line: store the first point (cursor position)
                self.line_start = self.spectrum_cursor_pos
                self.drawing_line = True
                
                # Create a new line object in the spectrum plot (only x-coordinates for the start)
                start_x, start_y = self.line_start
                self.current_line, = self.spectrum_ax.plot([start_x, start_x], [start_y, start_y], color='lime', lw=1)
                self.spectrum_canvas.draw()  # Refresh the canvas with the new line
            else:
                # Finish the line at the current cursor position
                self.finalize_line()
                self.drawing_line = False
                # print(f"Line finalized from {self.line_start} to {self.spectrum_cursor_pos}")
                self.slope = (self.spectrum_cursor_pos[1]-self.line_start[1])/(self.spectrum_cursor_pos[0]-self.line_start[0])
                self.intercept = self.line_start[1] - self.slope * self.line_start[0]
                data_cont = {'Continuum Name': ['Continuum'],
                             'x1': [round(self.line_start[0],2)],
                             'x2': [round(self.spectrum_cursor_pos[0],2)],
                             'Slope_0': [self.slope],
                             'Intercept_0': [self.intercept],
                             'Slope_fit': [np.nan],
                             'Intercept_fit': [np.nan],
                             'region_ID': [len(df_cont)],
                             'lineactor': [self.current_line]}
                df_cont_new = pd.DataFrame(data_cont)
                if len(df_cont) == 0:
                    df_cont = df_cont_new
                else:
                    df_cont = pd.concat([df_cont, df_cont_new], ignore_index=True)
                
                # When opening the FitParamsWindow, pass the current_line
                if len(df_cont) == 1:
                    self.fit_params_window = FitParamsWindow(viewer_window=self, df=df, df_cont=df_cont, current_line=self.current_line)
                    self.fit_params_window.show()
                if len(df_cont) > 1:
                    self.fit_params_window.add_spectral_frame(f"Spectral Region {len(df_cont)}",df_cont,df,ID=len(df_cont)-1,addframe=True)

        if event.key() == Qt.Key_G:
            if not self.gaussian_active:
                # First press of 'g': Start Gaussian fitting
                x_cursor, y_cursor = self.spectrum_cursor_pos  # Get cursor position
                self.line_region = self.get_line_region(x_cursor)  # Check if cursor is in x1, x2 range
                if self.line_region is not None:
                    # self.gaussian_active = True
                    self.initial_x = x_cursor  # Lock centroid position
                    self.current_sigma = 10
                    self.current_amplitude = y_cursor
                    
                    # Plot initial Gaussian
                    self.start_gaussian_adjustment(self.line_region['region_ID'])
                    # self.draw_gaussian(line_region, self.initial_x, self.current_sigma, self.current_amplitude)
            else:
                # Second press of 'g': Lock the Gaussian
                self.gaussian_active = False
                self.save_gaussian(self.line_region['region_ID'])


    def start_gaussian_adjustment(self, region_ID):
        """Initialize Gaussian drawing when 'G' is pressed, linked to mouse motion."""
        if self.gaussian_active:
            # If Gaussian adjustment is already active, finalize it
            self.gaussian_active = False
            return  
        
        self.gaussian_active = True
        
        # Retrieve the line associated with frame_id
        line_obj = df_cont.loc[df_cont["region_ID"] == region_ID, 'lineactor'].item()
    
        if not isinstance(line_obj, plt.Line2D):
            print(f"Error: No valid line object found.")
            return
    
        # Store the line and initialize Gaussian parameters
        self.active_line = line_obj
        region = df_cont.loc[df_cont["region_ID"] == region_ID]
        self.x1, self.x2 = region["x1"].item(), region["x2"].item()
        self.m, self.b = region["Slope_0"].item(), region["Intercept_0"].item()
    
        # Capture cursor position for Gaussian center
        if self.spectrum_cursor_pos:
            self.gaussian_x0, _ = self.spectrum_cursor_pos  # x0 is fixed at cursor position
    
        # self.gaussian_sigma = 5  # Initial sigma
        # self.gaussian_amplitude = 1  # Initial amplitude
        self.gaussian_sigma = 20 * abs(wavelengths[1]-wavelengths[0])
        self.gaussian_amplitude = 0.1 * abs(np.diff(self.spectrum_ax.get_ylim())[0])
    
        # Store Gaussian component lines for individual display
        self.gaussian_component_lines = []
    
        # Set flag to indicate active adjustment
        self.gaussian_active = True
    
        # Connect the mouse move event
        self.spectrum_canvas.mpl_connect("motion_notify_event", self.update_gaussian_dynamic)
    
    
    def update_gaussian_dynamic(self, event):
        """Update the Gaussian shape dynamically based on cursor movement."""
        if not self.gaussian_active or event.inaxes != self.spectrum_ax:
            return
    
        # Adjust sigma (horizontal motion) and amplitude (vertical motion)
        dx = event.xdata - self.gaussian_x0
        dy = event.ydata
    
        self.gaussian_sigma = max(0.01, abs(dx))  # Sigma changes with horizontal movement
    
        # Compute the continuum level at the Gaussian's center position
        y_continuum_x0 = self.m * self.gaussian_x0 + self.b
    
        # Adjust amplitude so that peak of the Gaussian reaches dy
        self.gaussian_amplitude = max(0.001, dy - y_continuum_x0)
    
        # Generate updated Gaussian + Line function
        x_vals = np.linspace(self.x1, self.x2, 1000)
        y_line = self.m * x_vals + self.b  # The baseline (without Gaussians)
    
        # Recompute all Gaussians from scratch (prevents runaway summing)
        y_gaussian_total = np.zeros_like(x_vals)
    
        # Clear existing individual component lines
        while self.gaussian_component_lines:
            line = self.gaussian_component_lines.pop()
            line.remove()
    
        # Sum over all existing Gaussians in df and plot them
        for _, row in df.iterrows():
            y_gaussian = row["Amp_0"] * np.exp(-((x_vals - row["Centroid_0"]) ** 2) / (2 * row["Sigma_0"] ** 2))
            y_gaussian_total += y_gaussian
            
            # Draw individual Gaussian component (dashed green line)
            component_line, = self.spectrum_ax.plot(x_vals, y_gaussian + y_line, color='#d7801a', linestyle='--', linewidth=1)
            self.gaussian_component_lines.append(component_line)
    
        # If we're actively adjusting a new Gaussian, include and plot it
        y_gaussian_new = self.gaussian_amplitude * np.exp(-((x_vals - self.gaussian_x0) ** 2) / (2 * self.gaussian_sigma ** 2))
        y_gaussian_total += y_gaussian_new
    
        new_component_line, = self.spectrum_ax.plot(x_vals, y_gaussian_new + y_line, color='#d7801a', linestyle='--', linewidth=1)
        self.gaussian_component_lines.append(new_component_line)
    
        # Final y-values for the sum (solid green line)
        y_new = y_line + y_gaussian_total
    
        # Update the existing line object for the summed model
        self.active_line.set_data(x_vals, y_new)
    
        # Redraw dynamically
        self.spectrum_ax.figure.canvas.draw_idle()





    def save_gaussian(self, region_ID):
        global df
        """Save final Gaussian parameters to df_cont"""
        
        if len(df) == 0: 
            line_id = 0
        else:
            line_id = np.max(np.int64(df['Line_ID']))+1
        
        df_new = pd.DataFrame({'Line_ID': [line_id],
                'Line_Name': [f'Line '+str(len(df.loc[df["region_ID"]==region_ID]))],
                'SNR': [0],
                'Rest Wavelength': [np.nan],
                'Amp_0': [self.gaussian_amplitude],
                'Amp_0_lowlim': [0],
                'Amp_0_highlim': [9999],
                'Centroid_0': [self.gaussian_x0],
                'Centroid_0_lowlim': [-9999],
                'Centroid_0_highlim': [9999],
                'Sigma_0': [self.gaussian_sigma],
                'Sigma_0_lowlim': [0],
                'Sigma_0_highlim': [9999],
                'Amp_fit': [np.nan],
                'Centroid_fit': [np.nan],
                'Sigma_fit': [np.nan],
                'region_ID': [region_ID],
                'curveactor': [self.active_line]})
        
        
        
        
        df = pd.concat([df,df_new])

        self.fit_params_window.on_addline_button_click(df,frame_id=region_ID)
        
        
        
    def rebuild_plot(self,region_ID, from_file, show_init, show_fit, x, y):
        """used after deleting emission line, adding new line, or editing line"""
        global df, df_cont
        
        region = df_cont.loc[np.int64(df_cont["region_ID"]) == np.int64(region_ID)]
        # '''
        
        if show_init == True:
            self.x1, self.x2 = region["x1"].item(), region["x2"].item()
            self.m, self.b = region["Slope_0"].item(), region["Intercept_0"].item()
            
            self.x1 = np.float64(self.x1)
            self.x2 = np.float64(self.x2)
            self.m = np.float64(self.m)
            self.b = np.float64(self.b)
            
            # Generate updated Gaussian + Line function
            x_vals = np.linspace(self.x1, self.x2, 1000)
            y_line = self.m * x_vals + self.b  # The baseline (without Gaussians)
            
            # '''
            gaussians = df.loc[np.int64(df["region_ID"]) == region_ID]
    
            # Recompute all Gaussians from scratch (prevents runaway summing)
            y_gaussian_total = np.zeros_like(x_vals)
        
            # Sum over all existing Gaussians in df
            i = 0
            for _, row in gaussians.iterrows():
                row['Amp_0'] = np.float64(row['Amp_0'])
                row['Centroid_0'] = np.float64(row['Centroid_0'])
                row['Sigma_0'] = np.float64(row['Sigma_0'])
                gaussian = row["Amp_0"] * np.exp(-((x_vals - row["Centroid_0"]) ** 2) / (2 * row["Sigma_0"] ** 2))
                new_curve, = self.spectrum_ax.plot(x_vals, gaussian+y_line, color='#d7801a', lw=1,linestyle='--')
                df.loc[(np.int64(df['region_ID']==region_ID)) & (df['Line_ID'] == i), 'curveactor'] = new_curve
                y_gaussian_total += gaussian
                i = i+1
                
        
            # Final y-values
            y_new = y_line + y_gaussian_total
        
            # Update the existing line object
            if from_file == False:
                line_obj = df_cont.loc[np.int64(df_cont["region_ID"]) == region_ID, 'lineactor'].item()
                line_obj.set_data(x_vals, y_new)
                
            
            new_line, = self.spectrum_ax.plot(x_vals, y_new, color='lime', lw=1)
            df_cont.loc[np.int64(df_cont["region_ID"]) == region_ID, 'lineactor'] = new_line  # Update reference


        if (show_fit == True) and (len(df_fit) > 0):
            region = df_fit.loc[(np.int64(df_fit['spaxel_x'])==x) &
                                 (np.int64(df_fit['spaxel_y'])==y) &
                                 (np.int64(df_fit['region_ID'])==np.int64(region_ID))]
            
            
            self.x1, self.x2 = region.iloc[0]["cont_region"+str(region_ID+1)+"_x_start"].item(), region.iloc[0]["cont_region"+str(region_ID+1)+"_x_end"].item()
            self.m, self.b = region.iloc[0]["cont_region"+str(region_ID+1)+"_slope_fit"].item(), region.iloc[0]["cont_region"+str(region_ID+1)+"_intercept_fit"].item()
            
            self.x1 = np.float64(self.x1)
            self.x2 = np.float64(self.x2)
            self.m = np.float64(self.m)
            self.b = np.float64(self.b)
            
            # Generate updated Gaussian + Line function
            x_vals = np.linspace(self.x1, self.x2, 1000)
            y_line = self.m * x_vals + self.b  # The baseline (without Gaussians)
            
            # '''
            # gaussians = df.loc[np.int64(df["region_ID"]) == region_ID]
    
            # Recompute all Gaussians from scratch (prevents runaway summing)
            y_gaussian_total = np.zeros_like(x_vals)
        
            # Sum over all existing Gaussians in df
            for _, row in region.iterrows():
                row['amp_fit'] = np.float64(row['amp_fit'])
                row['cen_fit'] = np.float64(row['cen_fit'])
                row['sigma_fit'] = np.float64(row['sigma_fit'])
                plotcolor = row['color']
                y_gaussian = row["amp_fit"] * np.exp(-((x_vals - row["cen_fit"]) ** 2) / (2 * row["sigma_fit"] ** 2))
                y_gaussian_total += y_gaussian
                fit_component, = self.spectrum_ax.plot(x_vals,y_line+y_gaussian,lw=1,linestyle='--',color=plotcolor)
                
            
            # Final y-values
            y_new = y_line + y_gaussian_total
        
            # Update the existing line object
            if from_file == False:
                line_obj = df_cont.loc[np.int64(df_cont["region_ID"]) == np.int64(region_ID), 'lineactor'].item()
                line_obj.set_data(x_vals, y_new)
                
            
            
            new_line, = self.spectrum_ax.plot(x_vals, y_new, color='red', lw=0.5,label=f'rChi2 = {row["rchisq"]}')
            df_cont.loc[np.int64(df_cont["region_ID"]) == region_ID, 'lineactor'] = new_line  # Update reference
            
            self.spectrum_ax.legend()

        # Redraw dynamically
        self.spectrum_ax.figure.canvas.draw_idle()
        
        
        
    def get_line_region(self, x_cursor):
        """Check if cursor is within x1, x2 of any existing lines"""
        for index, row in df_cont.iterrows():
            if row["x1"] <= x_cursor <= row["x2"]:
                return row  # Return the matching line's data
        return None

                
    def draw_image(self, data, cmap, scale, from_fits=False):
        """Generates and displays the white-light image in the left panel."""
        if from_fits:
            image = np.nansum(data, axis=0)
        else:
            image = data
            if scale == 'log':
                image = np.log10(image)
        
        
        min_value = np.nanmin(image)
        if min_value < 0:
            image += abs(min_value)
        
        # Filter out extreme outliers using percentiles
        lower_bound, upper_bound = np.nanpercentile(image, [5, 95])
        
        # Clip the image data to reduce the influence of extreme outliers
        image = np.clip(image, lower_bound, upper_bound)
    
        vmin = np.nanmedian(image) - 3 * np.nanstd(image)
        vmax = np.nanmedian(image) + 3 * np.nanstd(image)
    
        self.canvas.figure.clear()
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        apply_mpl_qss_style(self.canvas.figure, self.ax, None)

        # Initialize red rectangle but keep it hidden initially
        self.red_rect = patches.Rectangle((0, 0), 1, 1, linewidth=1.5, edgecolor='#d7801a', facecolor='none', visible=False)
        self.ax.add_patch(self.red_rect)
        self.canvas.draw()





    def get_spectrum_at_spaxel(self, x, y):
        """Fetch the 1D spectrum corresponding to the given spaxel (x, y)."""
        spectrum = FITS_DATA[:, y, x]
        return spectrum


    def finalize_line(self):
        """Finalize the drawn line when the 'd' key is pressed again."""
        if self.drawing_line and self.line_start is not None:
            # Get the final position of the cursor in the spectrum plot
            end_x, end_y = self.spectrum_cursor_pos
            self.current_line.set_xdata([self.line_start[0], end_x])
            self.current_line.set_ydata([self.line_start[1], end_y])
            
            # Redraw the canvas to reflect the final line
            self.spectrum_canvas.draw()
    
    def update_line(self, x1, x2):
        """Update the line in the plot while keeping y1 and y2 fixed."""
        if hasattr(self, 'current_line') and hasattr(self, 'line_start') and hasattr(self, 'spectrum_cursor_pos'):
            y1 = self.line_start[1]  # Keep the original y1
            y2 = self.spectrum_cursor_pos[1]  # Keep the original y2
            
            # Recalculate the slope and intercept to maintain (y1, y2)
            if x1 != x2:  # Avoid division by zero
                self.slope = (y2 - y1) / (x2 - x1)
                self.intercept = y1 - self.slope * x1
            else:
                print("Warning: x1 and x2 are the same. Vertical line assumed.")
                self.slope = float('inf')  # Indicates a vertical line
                self.intercept = None  # Not defined for vertical lines
    
            self.current_line.set_data([x1, x2], [y1, y2])  # Update with new x1, x2
            self.spectrum_canvas.draw()  # Refresh the plot
            # print(f"Updated line: x1={x1}, y1={y1}, x2={x2}, y2={y2}, slope={self.slope}, intercept={self.intercept}")

    
    def update_line_with_slope_intercept(self, x1, x2, slope, intercept):
        """Update the line in the plot based on the updated slope and intercept."""
        if hasattr(self, 'current_line'):
            y1 = slope * x1 + intercept
            y2 = slope * x2 + intercept
            print(y1,y2)
            self.current_line.set_data([x1, x2], [y1, y2])
            self.spectrum_canvas.draw()  # Refresh the plot
            print(f"Redrawing line with updated slope={slope} and intercept={intercept}")
        
        # Store the slope and intercept for future updates
        self.slope = slope
        self.intercept = intercept






    def create_menu_bar_VisualizerWindow(self):
        """Creates and returns the menu bar for ViewerWindow"""
        menu_bar = QMenuBar(self)

        # File Menu
        file_menu = menu_bar.addMenu("File")
        open_action = QAction("Import (Cmd+I)", self)
        open_action.setShortcut(QKeySequence("Cmd+O" if sys.platform == "darwin" else "Ctrl+O"))
        save_action = QAction("Save (Cmd+S)", self)
        save_action.setShortcut(QKeySequence("Cmd+S" if sys.platform == "darwin" else "Ctrl+S"))
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)

        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        # View Menu
        view_menu = menu_bar.addMenu("View")
        view_menu.addAction(QAction("Zoom In", self))
        view_menu.addAction(QAction("Zoom Out", self))

        # Help Menu
        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction(QAction("About", self))

        return menu_bar

    def showEvent(self, event):
        """Ensure ViewerWindow always has its own menu bar when focused"""
        self.setMenuBar(self.create_menu_bar_VisualizerWindow())
        super().showEvent(event)


    def open_fit_params_window(self):
        """Opens the FitParamsWindow when the button is clicked"""
        if self.fit_params_window is None or not self.fit_params_window.isVisible():
            self.fit_params_window = FitParamsWindow(viewer_window=self, df=df, df_cont=df_cont, current_line='')
            self.fit_params_window.show()
        
    def closeEvent(self, event):
        """Override closeEvent to ensure the console is still active after closing the window"""
        app = QApplication.instance()
        if app:
            app.quit()  # Quit the QApplication when the window is closed
        event.accept()  # Accept the event and close the window
        

class FitParamsWindow(QtWidgets.QMainWindow):
    buttons_dict = {}

    def __init__(self, viewer_window, df_cont, df, current_line):
        super().__init__()
        self.viewer_window = viewer_window
        df_cont = df_cont
        df = df
        self.fitloaded = False
        self.current_line = current_line
        self.setWindowTitle("Fit Parameters")
        
        # self.create_menu_bar_FitParamsWindow()  # Add menu bar
        # Assign the menu bar explicitly to this window
        self.setMenuBar(self.create_menu_bar_FitParamsWindow())

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout(self.main_widget)

        # Scroll Area
        self.scroll_area = QScrollArea(self.main_widget)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)

        # Container for frames inside the scroll area
        self.scroll_container = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_container)
        self.scroll_layout.setSpacing(20)
        self.scroll_container.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_container)

        self.add_spaxel_info_frame('Observation Data', df_cont, df, df_obs)
        
        if len(df_cont) == 0:
            self.spectral_count = 0
        else:
            self.spectral_count = 1
            self.add_spectral_frame("Spectral Region 1", df_cont, df, ID=0, addframe=False)

        self.setGeometry(100, 100, 1200, 400)
        self.show()

    def create_menu_bar_FitParamsWindow(self):
        """Creates the menu bar for this window"""
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)  # Assign it to the window

        # File Menu
        file_menu = menu_bar.addMenu("File")
        open_action = QAction("Open (Cmd+O)", self)
        open_action.setShortcut(QKeySequence("Cmd+O" if sys.platform == "darwin" else "Ctrl+O"))
        save_action = QAction("Save (Cmd+S)", self)
        save_action.setShortcut(QKeySequence("Cmd+S" if sys.platform == "darwin" else "Ctrl+S"))
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)

        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        # Edit Menu
        edit_menu = menu_bar.addMenu("Edit")
        edit_menu.addAction(QAction("Undo", self))
        edit_menu.addAction(QAction("Redo", self))

        # View Menu
        view_menu = menu_bar.addMenu("View")
        view_menu.addAction(QAction("Zoom In", self))
        view_menu.addAction(QAction("Zoom Out", self))

        # Help Menu
        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction(QAction("About", self))

        return menu_bar  # Return the menu bar

    def add_continuum_buttons(self, regionID, grid_layout):
        # Add Labels (continuum button labels)
        for col, col_name in enumerate(df_cont.columns):
            if (col_name != 'region_ID') & (col_name != 'lineactor'):
                label = QLabel(col_name)
                # label.setStyleSheet("""
                #     font-family: "Segoe UI";
                #     font-size: 12pt;
                #     border: none;
                #     padding-right: 10px;
                #     padding-left: 10px;
                #     padding-top: 5px;
                #     padding-bottom: 5px;
                #     background-color: snow;
                #     color: black;
                #     font: bold;
                #     width: 64px;
                # """)
                grid_layout.addWidget(label, 0, col)
            
        # Add Data (continuum buttons)
        for row in range(df_cont.shape[0]):
           for col, col_name in enumerate(df_cont.columns):
               if (col_name != 'region_ID') & (col_name != 'lineactor'):
                   if np.int64(df_cont.iloc[row,7]) == np.int64(regionID):
                       button_name = 'continuum~'+col_name
                       # button_text = str(df_cont.iloc[row, col]) if 'fit' not in col_name.lower() else ''
                       if (col_name == 'Continuum Name') or (type(df_cont.iloc[row, col]) == str):
                             button_text = str(df_cont.iloc[row, col]) if 'fit' not in col_name.lower() else ''
                       else:
                             button_text = str(round(df_cont.iloc[row, col],4)) if 'fit' not in col_name.lower() else ''
                       button = FrameButton(button_text, row, col, regionID, button_name)
                       # button.setStyleSheet("""
                       #     font-family: "Segoe UI";
                       #     font-size: 10pt;
                       #     border: 2px solid;
                       #     border-color: steelblue;
                       #     border-radius: 5px;
                       #     padding-right: 10px;
                       #     padding-left: 10px;
                       #     padding-top: 5px;
                       #     padding-bottom: 5px;
                       #     background-color: lightskyblue;
                       #     highlight-color; cyan;
                       #     color: k;
                       #     font: bold;
                       #     width: 64px;
                       # """)
                       
                       if col >= 5:  # Check the 'fit' column in df_cont
                           # button.setStyleSheet("""
                           #     font-family: "Segoe UI";
                           #     font-size: 12pt;
                           #     border: 2px solid;
                           #     border-color: black;
                           #     border-radius: 5px;
                           #     padding-right: 10px;
                           #     padding-left: 10px;
                           #     padding-top: 5px;
                           #     padding-bottom: 5px;
                           #     background-color: darkgrey;
                           #     color: white;
                           #     font: bold;
                           #     width: 64px;
                           # """)
                           button.setDisabled(True)  # Disable the button
                       else:
                           button.clicked.connect(partial(self.on_button_click, data_frame=df_cont, frame_id=regionID, button_name=button_name))
                       
                       grid_layout.addWidget(button, 1, col)
       
                       # Store the button reference in buttons_dict by row/col
                       self.buttons_dict[(regionID, button_name)] = button
                       
                       if col == 6:
                            button_name = 'continuum_delete'
                            delete_region_button = FrameButton('-', row, col, regionID, button_name)
                            delete_region_button.setStyleSheet("""
                                background-color: lightcoral;
                                color: k;
                            """)
                            
                            delete_region_button.clicked.connect(partial(self.on_deleteregion_button_click, frame_id=regionID))

                            grid_layout.addWidget(delete_region_button, 1, col+1)
                            # Store the button reference in buttons_dict by row/col
                            self.buttons_dict[(regionID, button_name)] = delete_region_button
                       
                       
    def add_spectral_lines_button_header(self, grid_layout):
        button_columns = [
            'Line Name', 
            'SNR',
            '<i>&lambda;</i><sub>rest</sub>', 
            '<i>f</i><sub>&lambda;,0</sub>', 
            '<i>f</i><sub>&lambda;,0,min</sub>', 
            '<i>f</i><sub>&lambda;,0,max</sub>', 
            '<i>&lambda;</i><sub>obs,0</sub>', 
            '<i>&lambda;</i><sub>obs,0,min</sub>',
            '<i>&lambda;</i><sub>obs,0,max</sub>',
            '&sigma;<sub>0</sub>', 
            '&sigma;<sub>0,min</sub>', 
            '&sigma;<sub>0,max</sub>', 
            '<i>f</i><sub>&lambda;,fit</sub>', 
            '<i>&lambda;</i><sub>obs,fit</sub>', 
            'v<sub>fit</sub>', 
            '&sigma;<sub>fit</sub>'
        ]
    
        for col, col_name in enumerate(button_columns):
            label = QLabel(col_name)
            label.setTextFormat(Qt.RichText)  # Enable HTML formatting
            if col == 0:
                label.setStyleSheet("""
                    font-size: 12pt;
                """)
            else:
                label.setStyleSheet("""
                    font-size: 12pt;
                    padding-left: 10px;
                """)
            # label.setStyleSheet("""
            #     font-family: "Segoe UI";
            #     font-size: 12pt;
            #     border: none;
            #     padding-right: 10px;
            #     padding-left: 10px;
            #     padding-top: 5px;
            #     padding-bottom: 5px;
            #     background-color: snow;
            #     color: black;
            #     font: bold;
            #     width: 64px;
            # """)
            grid_layout.addWidget(label, 2, col)



    def add_spectral_lines(self, regionID, grid_layout):
        global df_cont, df
        # Add Data (spectral line buttons)
        df_region = df.loc[np.int64(df['region_ID']) == np.int64(regionID)]
        
        for row in range(df_region.shape[0]):
            button_columns = ['Line_Name', 'SNR','Rest Wavelength', 
                              'Amp_0', 'Amp_0_lowlim', 'Amp_0_highlim',
                              'Centroid_0', 'Centroid_0_lowlim', 'Centroid_0_highlim',
                              'Sigma_0','Sigma_0_lowlim','Sigma_0_highlim',
                              'Amp_fit', 'Centroid_fit', 'v_fit', 'Sigma_fit']
            
            main_row_index = 3 + row * 2  # Reserve extra row space
            limits_row_index = main_row_index + 1  # Place limit buttons in a separate row
            
            button_height = 24  # Standardized button height
            
            for col, col_name in enumerate(button_columns):
                # print(type(df_region.iloc[row][col_name]),df_region.iloc[row][col_name])
                button_name = str(np.int64(df_region.iloc[row]['Line_ID']))+'~'+str(col_name)
                if col_name in ['Rest Wavelength', 'Amp_0', 'Centroid_0', 'Sigma_0']:
                    df_region.iloc[row][col_name] = np.float64(df_region.iloc[row][col_name])
                    button_text = str(round(df_region.iloc[row][col_name],3))
                else:
                    
                    if ('fit' in col_name.lower()) | ('SNR' in col_name) | ('Rest' in col_name):
                        button_text = ''
                    else: 
                        button_text = str(df_region.iloc[row][col_name])
                        
                # button_text = str(df_region.iloc[row][col_name]) if 'fit' not in col_name.lower() else ''
                button = FrameButton(button_text, row, col, regionID, button_name)
                # button.setStyleSheet("""
                #     font-family: "Segoe UI";
                #     font-size: 8pt;
                #     border: 2px solid;
                #     border-color: steelblue;
                #     border-radius: 5px;
                #     padding: 5px 10px;
                #     background-color: lightskyblue;
                #     color: black;
                #     font: bold;
                # """)
                button.setFixedWidth(76)
                
                if 'fit' in col_name:  # Check the 'fit' column
                    # button.setStyleSheet("""
                    #     font-family: "Segoe UI";
                    #     font-size: 12pt;
                    #     border: 2px solid;
                    #     border-color: black;
                    #     border-radius: 5px;
                    #     padding: 5px 10px;
                    #     background-color: darkgrey;
                    #     color: white;
                    #     font: bold;
                    # """)
                    # button.setDisabled(True)  # Disable the button
                    button.clicked.connect(partial(self.on_fit_button_click, frame_id=regionID, button_name=button_name))
                else:
                    button.clicked.connect(partial(self.on_button_click, data_frame=df, frame_id=regionID, button_name=button_name))
    
                grid_layout.addWidget(button, main_row_index, col)
                
                # Store button reference
                self.buttons_dict[(regionID, button_name)] = button
                
                line_id = np.float64(df_region.iloc[row]['Line_ID'])
                line_id = np.int64(line_id)
                
                '''
                # Add lower and upper limit buttons for specific columns
                if col_name in ['Amp_0', 'Centroid_0', 'Sigma_0']:
                    limits_container = QWidget()
                    limits_layout = QHBoxLayout()
                    limits_container.setLayout(limits_layout)
                    
                    # **REMOVE BORDERS** from limits_container
                    limits_container.setStyleSheet("border: none;")
                    
                    
                    # Lower limit button
                    button_name_lowerlim = str(line_id)+'~'+col_name + '_lowlim'
                    button_text_lowerlim = str(df_region.iloc[row][col_name+'_lowlim'])
                    button_lowlim = FrameButton(button_text_lowerlim, row+1, col, regionID, button_name_lowerlim)
                    # button_lowlim.setStyleSheet("""
                    #     font-family: "Segoe UI";
                    #     font-size: 8pt;
                    #     border: 2px solid;
                    #     border-color: steelblue;
                    #     border-radius: 5px;
                    #     padding: 5px 10px;
                    #     background-color: lightskyblue;
                    #     color: black;
                    #     font: bold;
                    #     padding: 0px;  /* Remove padding */
                    #     margin: 0px;   /* Remove margins */
                    #     text-align: center;
                    # """)
                    button_lowlim.setFixedWidth(38)
                    button_lowlim.setFixedHeight(button_height)  # Ensure consistent height
    
                    # Upper limit button
                    
                    button_name_upperlim = str(line_id)+'~'+col_name + '_highlim'
                    button_text_upperlim = str(df_region.iloc[row][col_name+'_highlim'])
                    button_highlim = FrameButton(button_text_upperlim, row+1, col, regionID, button_name_upperlim)
                    # button_highlim.setStyleSheet("""
                    #     font-family: "Segoe UI";
                    #     font-size: 8pt;
                    #     border: 2px solid;
                    #     border-color: steelblue;
                    #     border-radius: 5px;
                    #     padding: 5px 10px;
                    #     background-color: lightskyblue;
                    #     color: black;
                    #     font: bold;
                    #     padding: 0px;  /* Remove padding */
                    #     margin: 0px;   /* Remove margins */
                    #     text-align: center;
                    # """)

                    button_highlim.setFixedWidth(38)
                    button_highlim.setFixedHeight(button_height)  # Ensure consistent height
    
                    button_lowlim.setMinimumSize(38, 24)
                    button_highlim.setMinimumSize(38, 24)
    
    
                    # Add both buttons inside the container with spacing
                    limits_layout.addWidget(button_lowlim)
                    limits_layout.addWidget(button_highlim)
    
                    # Adjust margins of the limits_layout to shift the buttons around
                    limits_layout.setContentsMargins(0, 0, 24, 0)  
                    
        
                    # **Adjust padding between buttons**
                    # limits_layout.setSpacing(16)  # Change this value to adjust spacing
                    
                    # limits_container.setStyleSheet("background-color: transparent; border: none;")
                    # limits_container.setStyleSheet("background-color: yellow;")

                    # Add the limits_container to the grid layout
                    grid_layout.addWidget(limits_container, limits_row_index, col)
    
                    # Store button references
                    self.buttons_dict[(regionID, button_name_lowerlim)] = button_lowlim
                    self.buttons_dict[(regionID, button_name_upperlim)] = button_highlim
                    
                    button_lowlim.clicked.connect(partial(self.on_button_click, data_frame=df, frame_id=regionID, button_name=button_name_lowerlim))
                    button_highlim.clicked.connect(partial(self.on_button_click, data_frame=df, frame_id=regionID, button_name=button_name_upperlim))
                    
                    
                    '''

                # # Add a delete button at the last column
                if col_name == 'Sigma_fit':
                    button_name = 'line_' + str(row+1)
                    delete_line_button = FrameButton('-', row, col, regionID, button_name)
                    delete_line_button.setStyleSheet("""
                        background-color: lightcoral;
                        color: black;
                    """)
                    
                    delete_line_button.clicked.connect(partial(self.on_deleteline_button_click, 
                                                                frame_id=regionID, 
                                                                line_id=line_id))
                    
                    # delete_line_button.clicked.connect(partial(self.on_addline_button_click,data_frame=df, frame_id=regionID))
                    grid_layout.addWidget(delete_line_button, main_row_index, col+1)



    def on_fit_button_click(self, frame_id, button_name):
        print(f'frame ID: {frame_id}, button name: {button_name}')

        if len(df_fit) > 0:
            print(df_fit.iloc[0])
            gaussian_number = np.float64(button_name.split('~')[0])
            print(f'gaussian number = {gaussian_number}')
            # Filter df_fit for the specified line_name and region_ID
            filtered_df = df_fit[(df_fit['LineID'] == np.float64(gaussian_number)) & (df_fit['region_ID'] == frame_id)]

            # Convert spaxel_x and spaxel_y to integers for indexing
            filtered_df['spaxel_x'] = filtered_df['spaxel_x'].astype(int)
            filtered_df['spaxel_y'] = filtered_df['spaxel_y'].astype(int)
            
            # Determine image dimensions
            x_max = filtered_df['spaxel_x'].max() + 1
            y_max = filtered_df['spaxel_y'].max() + 1
            
            # Create an empty image array
            image_array = np.full((y_max, x_max), np.nan)  # Using NaN for unfilled values
            
            
            
            
            # Mapping of button key parts to the column names in df_fit
            key_to_column_mapping = {
                'continuum~Slope_fit': 'slope_fit',
                'continuum~Intercept_fit': 'intercept_fit',
                'Line_Name': 'LineName',
                'Amp_fit': 'amp_fit',
                'Centroid_fit': 'cen_fit',
                'Sigma_fit': 'sigma_fit',
                'v_fit': 'vel_fit'
            }
            
            df_param = key_to_column_mapping[button_name.split('~')[1]]
            
            
            print(f'df_param: {df_param}')
            # Populate the image array with amp_fit values
            for _, row in filtered_df.iterrows():
                # print(row[df_param])
                image_array[row['spaxel_y'], row['spaxel_x']] = row[df_param]
                

            # print('shape of img array'+str(np.shape(image_array)))

            if df_param == 'amp_fit': 
                cmap='viridis'
                scale='log'
                # image_array = np.nan_to_num(image_array,nan=abs(np.nanmedian(image_array)))
            if df_param == 'cen_fit': 
                cmap='bwr'
                scale='linear'
            if df_param == 'vel_fit': 
                cmap='bwr'
                scale='linear'
            if df_param == 'sigma_fit': 
                cmap='plasma'
                scale='linear'
            
            # fig,ax=plt.subplots()
            # ax.imshow(image_array,origin='lower',aspect='auto',cmap=cmap)
            self.viewer_window.draw_image(image_array, cmap=cmap, scale=scale,from_fits=False)


    def save_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "CSV Files (*.csv);;All Files (*)", options=options)
    
        if file_path:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
    
                # Convert region_ID to int64
                df_cont['region_ID'] = np.int64(df_cont['region_ID'])
                df['region_ID'] = np.int64(df['region_ID'])
    
                # Write df_obs (Observation parameters)
                writer.writerow(df_obs.columns)  # Write header
                for row in df_obs.itertuples(index=False, name=None):
                    writer.writerow(row)  # Flatten and write each row
    
                # Insert an empty row to separate the sections
                writer.writerow([])
    
                # Write df_cont (Continuum parameters)
                writer.writerow(df_cont.columns)  # Header for df_cont
                writer.writerows(df_cont.values)  # Data for df_cont
    
                # Insert an empty row to separate the sections
                writer.writerow([])
    
                # Write df header (Emission line parameters)
                writer.writerow(df.columns)  # Header for df
    
                # Sort df by 'region_ID' before writing
                df_sorted = df.sort_values(by='region_ID', ascending=True)
                writer.writerows(df_sorted.values)  # Data for df
    
            print(f"File saved: {file_path}")

    
    def open_file(self):
        global df, df_cont, df_obs
        """Open a CSV file and load it into the dataframes"""
        print('Opening file...')
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;All Files (*)", options=options)
    
        # Clear existing plot actors
        for frame_id in np.unique(df_cont['region_ID']):
            df_cont.loc[(df_cont["region_ID"] == frame_id)]['lineactor'].item().remove()
    
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    reader = csv.reader(file)
                    rows = list(reader)
    
                # Identify section indices (empty rows)
                empty_indices = [i for i, row in enumerate(rows) if not any(row)]
    
                if len(empty_indices) < 2:
                    print("Error: Invalid file format.")
                    return
    
                # Split the CSV data into three sections
                obs_data = rows[:empty_indices[0]]
                cont_data = rows[empty_indices[0]+1:empty_indices[1]]
                line_data = rows[empty_indices[1]+1:]
    
                # Load Observation Data (df_obs)
                df_obs_columns = obs_data[0]
                df_obs_values = obs_data[1:]
                df_obs = pd.DataFrame(df_obs_values, columns=df_obs_columns)
    
                # Ensure numeric columns are properly cast
                for col in ['redshift', 'resolvingpower']:
                    if col in df_obs.columns:
                        df_obs[col] = pd.to_numeric(df_obs[col], errors='coerce')
    
                # Load Continuum Data (df_cont)
                df_cont_columns = cont_data[0]
                df_cont_values = cont_data[1:]
                df_cont = pd.DataFrame(df_cont_values, columns=df_cont_columns)
    
                # Convert data types for df_cont
                df_cont['x1'] = np.float64(df_cont['x1'])
                df_cont['x2'] = np.float64(df_cont['x2'])
                df_cont['Slope_0'] = np.float64(df_cont['Slope_0'])
                df_cont['Intercept_0'] = np.float64(df_cont['Intercept_0'])
                df_cont['Slope_fit'] = pd.to_numeric(df_cont['Slope_fit'], errors='coerce')
                df_cont['Intercept_fit'] = pd.to_numeric(df_cont['Intercept_fit'], errors='coerce')
                df_cont['region_ID'] = np.int64(df_cont['region_ID'])
    
                # Load Emission Line Data (df)
                df_columns = line_data[0]
                df_values = line_data[1:]
                df = pd.DataFrame(df_values, columns=df_columns)
    
                # Convert numeric columns in df
                float_columns = [
                    'spaxel_x', 'spaxel_y', 'Line_ID', 'Rest Wavelength',
                    'Amp_0', 'Amp_0_lowlim', 'Amp_0_highlim',
                    'Centroid_0', 'Centroid_0_lowlim', 'Centroid_0_highlim',
                    'Sigma_0', 'Sigma_0_lowlim', 'Sigma_0_highlim', 'region_ID'
                ]
                for col in float_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if col in ['Line_ID', 'region_ID']:
                            df[col] = np.int64(df[col])
    
                # Clear the existing layout
                for i in reversed(range(self.scroll_layout.count())):
                    widget = self.scroll_layout.itemAt(i).widget()
                    if isinstance(widget, QFrame):
                        self.scroll_layout.removeWidget(widget)
                        widget.deleteLater()
    
                # Populate the UI with the loaded data
                self.add_spaxel_info_frame('Observation Data', df_cont, df, df_obs)
                for ID in np.unique(df_cont['region_ID']):
                    ID = np.int64(ID)
                    self.add_spectral_frame(f'Spectral Region {ID+1}', df_cont, df, ID, addframe=False)
                    self.viewer_window.rebuild_plot(region_ID=ID, from_file=True, show_init=True, show_fit=False, x=0, y=0)
    
                print(f"File loaded: {file_path}")
    
            except Exception as e:
                print(f"Error opening file: {e}")


            
    def toggle_fullscreen(self):
        """Toggles full screen mode"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key_S and (event.modifiers() & Qt.ControlModifier):  # Check for Cmd/Ctrl + S
            self.save_file()  # Trigger save file dialog
        if event.key() == Qt.Key_O and (event.modifiers() & Qt.ControlModifier):  # Check for Cmd/Ctrl + S
            self.open_file()  # Trigger save file dialog
        # if event.key() == Qt.Key_D:  # If 'D' key is pressed
        #     self.spectral_count += 1
        #     ID = self.spectral_count - 1
        #     IDName = ID +1
        #     self.add_spectral_frame(f"Spectral Region {IDName}", df_cont, df, ID, addframe=True)
        #debugging
        if event.key() == Qt.Key_F:
            print(df)
            # print(self.buttons_dict)
            # print(df.iloc[0])
            # print(df_fit.iloc[0])
            
            # for col, val in df_cont.iloc[0].items():  # Assuming a single row
            #     print(f"{col}: {val}")

    def add_spaxel_info_frame(self, title_text, df_cont, df, df_obs):
        
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Raised)
        frame.setObjectName(f"observation_data")
        frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # frame.setStyleSheet("""
        # border: 2px solid black;
        # background-color: snow;
        # border-radius: 15px;  /* Adjust the value to change the roundness */
        # padding: 10px;        /* Optional: Adds spacing inside the frame */
        # """)
        
        # Main layout for the frame
        main_layout = QHBoxLayout(frame)
    
        # Title
        title = QLabel(title_text)
        title.setAlignment(Qt.AlignCenter)
        # title.setStyleSheet("""
        #     font-family: "Segoe UI";
        #     font-size: 16pt;
        #     border: none;
        #     padding-right: 10px;
        #     padding-left: 10px;
        #     padding-top: 5px;
        #     padding-bottom: 5px;
        #     background-color: snow;
        #     color: black;
        #     font: bold;
        #     width: 64px;
        # """)
    
        # Create the left frame to hold buttons and labels
        button_frame = QFrame()
        button_frame.setFrameShape(QFrame.StyledPanel)
        button_frame.setFrameShadow(QFrame.Raised)
        button_frame.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        button_layout = QVBoxLayout(button_frame)
        
        # Define the rows of information with labels and buttons
        spaxel_info_rows = [['RA', 'Dec'], ['Pixel X', 'Pixel Y'], ['Pixel N']]  # Group the labels and buttons into pairs
        
        # Create buttons here for later use
        self.ra_button = SpaxelButton('RA: ', 'RA')
        self.dec_button = SpaxelButton('Dec: ', 'Dec')
        self.x_button = SpaxelButton('X: ', 'X')
        self.y_button = SpaxelButton('Y: ', 'Y')
        self.n_button = SpaxelButton('N: ', 'N')
        
        # Loop through and create rows for labels and buttons
        for row in spaxel_info_rows:
            # Create a horizontal layout for the labels
            label_layout = QHBoxLayout()
        
            for label_name in row:
                # Create label
                label = QLabel(label_name)
                # label.setStyleSheet("""
                #     font-family: "Segoe UI";
                #     font-size: 12pt;
                #     border: none;
                #     padding-right: 10px;
                #     padding-left: 10px;
                #     padding-top: 5px;
                #     padding-bottom: 5px;
                #     background-color: snow;
                #     color: black;
                #     font: bold;
                #     width: 64px;
                # """)
                label_layout.addWidget(label)  # Add label to horizontal layout
            
            # Add label layout (side-by-side) to the main button layout
            button_layout.addLayout(label_layout)
        
            # Create a horizontal layout for the buttons (below the labels)
            button_layout_2 = QHBoxLayout()
            
            for label_name in row:
                # Dynamically add the correct button for each label
                if label_name == 'RA':
                    button = self.ra_button
                elif label_name == 'Dec':
                    button = self.dec_button
                elif label_name == 'Pixel X':
                    button = self.x_button
                elif label_name == 'Pixel Y':
                    button = self.y_button
                elif label_name == 'Pixel N':
                    button = self.n_button
                else:
                    button = SpaxelButton('test', label_name)  # fallback for any other label name
        
                # button.setStyleSheet("""
                #     font-family: "Segoe UI";
                #     font-size: 10pt;
                #     border: 2px solid;
                #     border-color: black;
                #     border-radius: 5px;
                #     padding-right: 10px;
                #     padding-left: 10px;
                #     padding-top: 5px;
                #     padding-bottom: 5px;
                #     color: k;
                #     font: bold;
                #     width: 12px;
                # """)
                
                # Adjust the width of the button
                button.setFixedWidth(120)  # Set a fixed width for the button (adjust this value as needed)
                button_layout_2.addWidget(button)  # Add button to horizontal layout (side-by-side)
        
            # Add button layout (side-by-side) to the main button layout
            button_layout.addLayout(button_layout_2)


    
        # Add the button frame to the left side of the main layout
        main_layout.addWidget(button_frame)
    
        # Main content area (right side)
        content_frame = QFrame()
        content_frame.setFrameShape(QFrame.StyledPanel)
        content_frame.setFrameShadow(QFrame.Raised)
        content_layout = QVBoxLayout(content_frame)
    
        # Add the title to the content frame
        content_layout.addWidget(title)
    
        # Add other content or widgets to the main content frame here
        # ...
        self.source_name_button = SpaxelButton('Source Name: ', 'Source Name')  # New button
        self.source_redshift_button = SpaxelButton('Source Redshift: ', 'Source Redshift')  # New button
        self.resolving_power_button = SpaxelButton('Resolving Power: ', 'Resolving Power')  # New button

        # Add the source information to the content layout
        self.source_name_button.setText(f"Source Name: {df_obs.loc[0, 'sourcename']}")
        self.source_redshift_button.setText(f"Source Redshift: {df_obs.loc[0, 'redshift']}")
        self.resolving_power_button.setText(f"Resolving Power: {df_obs.loc[0, 'resolvingpower']}")
        
        # for button in [self.source_name_button,self.source_redshift_button,self.resolving_power_button]:
        #     button.setStyleSheet("""
        #         font-family: "Segoe UI";
        #         font-size: 10pt;
        #         border: 2px solid;
        #         border-color: steelblue;
        #         border-radius: 5px;
        #         padding-right: 10px;
        #         padding-left: 10px;
        #         padding-top: 5px;
        #         padding-bottom: 5px;
        #         background-color: lightskyblue;
        #         color: k;
        #         font: bold;
        #         width: 12px;
        #     """)
        
        
        
        self.source_name_button.clicked.connect(partial(self.on_button_click, data_frame=df_obs, frame_id=0, button_name='sourcename'))
        self.source_redshift_button.clicked.connect(partial(self.on_button_click, data_frame=df_obs, frame_id=0, button_name='redshift'))
        self.resolving_power_button.clicked.connect(partial(self.on_button_click, data_frame=df_obs, frame_id=0, button_name='resolvingpower'))


        # Add the observation buttons to the content layout
        content_layout.addWidget(self.source_name_button)
        content_layout.addWidget(self.source_redshift_button)
        content_layout.addWidget(self.resolving_power_button)
    
        # Add the content frame to the main layout
        main_layout.addWidget(content_frame)
        
        
        # Fitting Panel
        
        # Title
        title = QLabel('Spectral Fitting')
        title.setAlignment(Qt.AlignCenter)
        # title.setStyleSheet("""
        #     font-family: "Segoe UI";
        #     font-size: 16pt;
        #     border: none;
        #     padding-right: 10px;
        #     padding-left: 10px;
        #     padding-top: 5px;
        #     padding-bottom: 5px;
        #     background-color: snow;
        #     color: black;
        #     font: bold;
        #     width: 64px;
        # """)
        
        # Main content area (right side)
        fit_frame = QFrame()
        fit_frame.setFrameShape(QFrame.StyledPanel)
        fit_frame.setFrameShadow(QFrame.Raised)
        fit_layout = QVBoxLayout(fit_frame)
    
        # Add the title to the content frame
        fit_layout.addWidget(title)
        
        
        # Add other content or widgets to the main content frame here
        # ...
        self.fit_spaxel_button = QPushButton('Fit Current Spaxel')  # New button
        # self.fit_aperture_button = QPushButton('Fit Aperture')  # New button
        self.fit_cube_button = QPushButton('Fit Cube')  # New button
        self.fix_fit_button = QPushButton('Rectify Bad Fits')
        self.save_cube_fit_button = QPushButton('Save Cube Fit')
        self.save_cube_fit_fitsfile_button = QPushButton('Save Fit to Fits File')
        self.load_cube_fit_button = QPushButton('Load Cube Fit')

        # Add the source information to the content layout
        self.fit_spaxel_button.setText(f"Fit Current Spaxel")
        # self.fit_aperture_button.setText(f"Fit Aperture")
        self.fit_cube_button.setText(f"Fit Cube")
        self.fix_fit_button.setText(f"Rectify Bad Fits")
        self.save_cube_fit_button.setText(f"Save Cube Fit")
        self.save_cube_fit_fitsfile_button.setText(f'Save Fit to Fits File')
        self.load_cube_fit_button.setText(f"Load Cube Fit")
        
        
        # for button in [self.fit_spaxel_button,self.fit_aperture_button,self.fit_cube_button,
        #                self.save_cube_fit_button, self.load_cube_fit_button]:
        #     button.setStyleSheet("""
        #         font-family: "Segoe UI";
        #         font-size: 10pt;
        #         border: 2px solid;
        #         border-color: steelblue;
        #         border-radius: 5px;
        #         padding-right: 10px;
        #         padding-left: 10px;
        #         padding-top: 5px;
        #         padding-bottom: 5px;
        #         background-color: lightskyblue;
        #         color: k;
        #         font: bold;
        #         width: 12px;
        #     """)
        
        # Button press functions
        self.fit_spaxel_button.clicked.connect(partial(self.fit_spaxel))
        self.fit_cube_button.clicked.connect(partial(self.fit_cube))
        self.save_cube_fit_button.clicked.connect(partial(self.save_cube_fit))
        self.fix_fit_button.clicked.connect(partial(self.fix_fits))
        self.save_cube_fit_fitsfile_button.clicked.connect(partial(self.save_fit_result_fitsfile))
        self.load_cube_fit_button.clicked.connect(partial(self.load_cube_fit))
        
        
        # Add the observation buttons to the content layout
        fit_layout.addWidget(self.fit_spaxel_button)
        # fit_layout.addWidget(self.fit_aperture_button)
        fit_layout.addWidget(self.fit_cube_button)
        fit_layout.addWidget(self.fix_fit_button)
        fit_layout.addWidget(self.save_cube_fit_button)
        fit_layout.addWidget(self.save_cube_fit_fitsfile_button)
        fit_layout.addWidget(self.load_cube_fit_button)
        
        # Add the content frame to the main layout
        main_layout.addWidget(fit_frame)
        
        
        
        self.scroll_layout.addWidget(frame)



    def fix_fits(self):
        global df, df_cont, df_fit
        
        # identify spaxels with bad fit
        if len(df_fit) > 0:
            rchisq_max = 0.01
            self.fit_cube(refit=True, rchisq_thresh=rchisq_max)
        
                


    def save_fit_result_fitsfile(self):
        """Save df_fit results as a multi-extension FITS file with WCS-based RA/Dec mapping."""
        wcs = self.viewer_window.wcs
        print('Saving cube fit to FITS file')
    
        # Prompt user for file save location
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "FITS Files (*.fits);;All Files (*)", options=options)
        
        if not file_path:
            print("File save canceled.")
            return
    
        # Replace "nan" and "NaN" strings with actual NaNs
        df_fit.replace(["nan", "NaN"], np.nan, inplace=True)
    
        # Convert coordinate columns to int
        df_fit[["spaxel_x", "spaxel_y"]] = df_fit[["spaxel_x", "spaxel_y"]].astype(int)
    
        #   Convert spaxel coordinates to RA, Dec using the 3D WCS
        spaxel_x = df_fit["spaxel_x"].values
        spaxel_y = df_fit["spaxel_y"].values
        spaxel_z = np.zeros_like(spaxel_x)  # Placeholder for spectral axis
        
        ra_vals, dec_vals, _ = wcs.all_pix2world(spaxel_x, spaxel_y, spaxel_z, 0)

        
        # Identify unique spectral lines
        unique_lines = np.unique(df_fit['LineName'])
        extensions = []
    
        # Create a Primary HDU
        primary_hdu = fits.PrimaryHDU()
        extensions.append(primary_hdu)
    
        # Generate image maps for each line and each fit parameter
        for line in unique_lines:
            line_df = df_fit[df_fit['LineName'] == line]
            for param in ["amp_fit", "cen_fit", "vel_fit", "sigma_fit"]:
                # Create an empty map with NaNs
                image_map = np.full((max(df_fit["spaxel_y"]) + 1, max(df_fit["spaxel_x"]) + 1), np.nan)
    
                # Populate map using RA/Dec indices
                for (_, row), ra, dec in zip(line_df.iterrows(), ra_vals, dec_vals):
                    x, y = int(row["spaxel_x"]), int(row["spaxel_y"])
                    image_map[y, x] = row[param]
    
                # Create FITS HDU with WCS header
                hdu = fits.ImageHDU(data=image_map, name=f"{param[:-4]}_{line}")
                hdu.header.update(wcs.to_header())  # Add WCS info
                extensions.append(hdu)
    
        # Write all extensions to FITS file
        hdulist = fits.HDUList(extensions)
        hdulist.writeto(file_path, overwrite=True)
    
        print(f"FITS file saved to {file_path}")
        

    def save_cube_fit(self):
        global df, df_cont, df_obs, df_fit
        """Save csv storing galaxy, observation, and cube fit data"""
        print('Saving cube fit to csv file')
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        
        if file_path:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
    
                # Write df_obs (Observation information)
                writer.writerow(df_obs.columns)  # Header for df_obs
                writer.writerows(df_obs.values)  # Data for df_obs
    
                # Insert an empty row for separation
                writer.writerow([])
                
                # Write df_cont (continuum information)
                writer.writerow(df_cont.columns)  # Header for df_cont
                writer.writerows(df_cont.values)  # Data for df_cont
    
                # Insert an empty row for separation
                writer.writerow([])
    
                # Write df_cont (line information)
                writer.writerow(df.columns)  # Header for df
                writer.writerows(df.values)  # Data for df
    
                # Insert an empty row for separation
                writer.writerow([])
                
                # Write df_fit (Fitting results)
                writer.writerow(df_fit.columns)  # Header for df_fit
                writer.writerows(df_fit.values)  # Data for df_fit
    
            print(f"CSV File saved: {file_path}")
            
            
        
    def load_cube_fit(self):
        global df, df_cont, df_fit, df_obs, viewing_fit
        """Open a CSV file and load it into the dataframes"""
        print('opening file')
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;All Files (*)", options=options)
    
        if file_path:
            with open(file_path, 'r', newline='') as file:
                reader = csv.reader(file)
                rows = [row for row in reader]  # Don't remove empty rows yet
    
            # Find all the empty rows that separate the dataframes
            split_indices = [i for i, row in enumerate(rows) if not any(row)]
            
            if len(split_indices) < 3:
                raise ValueError("CSV file doesn't have enough sections separated by blank rows")
            
            # Read df_obs (first section)
            df_obs = pd.DataFrame(rows[:split_indices[0]][1:], columns=rows[0])  # Exclude header row from data
            df_obs = df_obs.apply(pd.to_numeric, errors='ignore')  # Convert numeric columns if possible
    
            self.source_name_button.setText('Source Name: '+str(df_obs['sourcename'].item()))
            self.source_redshift_button.setText('Source Redshift: '+str(df_obs['redshift'].item()))
            self.resolving_power_button.setText('Resolving Power: '+str(df_obs['resolvingpower'].item()))
            
            # Read df_cont (second section, after first empty row)
            df_cont_header_index = split_indices[0] + 1
            while df_cont_header_index < len(rows) and not any(rows[df_cont_header_index]):
                df_cont_header_index += 1
                
            df_cont_data_start = df_cont_header_index + 1
            df_cont_data_end = split_indices[1]
            
            df_cont = pd.DataFrame(rows[df_cont_data_start:df_cont_data_end], 
                                  columns=rows[df_cont_header_index])
            df_cont = df_cont.apply(pd.to_numeric, errors='ignore')
            
            # Read df (third section, after second empty row)
            df_header_index = split_indices[1] + 1
            while df_header_index < len(rows) and not any(rows[df_header_index]):
                df_header_index += 1
                
            df_data_start = df_header_index + 1
            df_data_end = split_indices[2]
            
            df = pd.DataFrame(rows[df_data_start:df_data_end], 
                             columns=rows[df_header_index])
            df = df.apply(pd.to_numeric, errors='ignore')
            
            # Read df_fit (fourth section, after third empty row)
            df_fit_header_index = split_indices[2] + 1
            while df_fit_header_index < len(rows) and not any(rows[df_fit_header_index]):
                df_fit_header_index += 1
                
            df_fit_data_start = df_fit_header_index + 1
            
            df_fit = pd.DataFrame(rows[df_fit_data_start:], 
                                 columns=rows[df_fit_header_index])
            df_fit = df_fit.apply(pd.to_numeric, errors='ignore')
            
            # Define column groups for conversion
            float_columns = [
                col for col in df_fit.columns 
                if col not in ["color", "success", "LineName", "LineID", "spaxel_x", "spaxel_y", "region_ID"]
            ]
            
            int_columns = ["LineID", "spaxel_x", "spaxel_y", "region_ID"]
            
            # Replace "nan" and "NaN" strings with actual NaNs
            df_fit.replace(["nan", "NaN"], np.nan, inplace=True)
            
            # Convert columns to float
            df_fit[float_columns] = df_fit[float_columns].astype(float)
            
            # Convert integer columns (use Int64 dtype to allow NaNs)
            df_fit[int_columns] = df_fit[int_columns].astype("Int64")
            
            # Remove any current actors from the spectrum plot
            # for frame_id in np.unique(df_cont['region_ID']):
            #     df_cont.loc[(df_cont["region_ID"] == frame_id)]['lineactor'].item().remove()
            # self.viewer_window.spectrum_canvas.draw()
            
            # Iterate over all items in the layout and remove them
            for i in reversed(range(self.scroll_layout.count())):  # Reverse to avoid index shifting
                widget = self.scroll_layout.itemAt(i).widget()
                if isinstance(widget, QFrame):
                    self.scroll_layout.removeWidget(widget)
                    widget.deleteLater()
                    
            # Regenerate the UI using updated df and df_cont
            self.add_spaxel_info_frame('Observation Data', df_cont, df, df_obs)
            for ID in np.unique(df_cont['region_ID']):
                ID = np.int64(ID)
                self.add_spectral_frame('Spectral Region '+str(ID+1), df_cont, df, ID, addframe=False)
                self.viewer_window.rebuild_plot(region_ID=ID,from_file=True,show_init=False,show_fit=True,x=df_fit.iloc[0]['spaxel_x'],y=df_fit.iloc[0]['spaxel_y'])
                    
            self.fitloaded = True
            viewing_fit = True
            print(f"File loaded: {file_path}")



    def line(self, x, m, b):
        return m*x+b
    
    # Define a Gaussian function
    def gaussian_function(self, x, amp, mu, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def generalized_model(self, x, m, b, *gaussian_params):
        """
        Model combining a linear continuum and multiple Gaussian components.
    
        Parameters:
        - x: ndarray, input x values
        - m: float, slope of the linear continuum
        - b: float, intercept of the linear continuum
        - *gaussian_params: list, flattened parameters for each Gaussian in the form (A1, mu1, sigma1, A2, mu2, sigma2, ...)
    
        Returns:
        - y: ndarray, model values
        """
        # Linear component
        y = self.line(x, m, b)
        
        # Add each Gaussian
        num_gaussians = len(gaussian_params) // 3
        for i in range(num_gaussians):
            A = gaussian_params[3 * i]
            mu = gaussian_params[3 * i + 1]
            sigma = gaussian_params[3 * i + 2]
            y += self.gaussian_function(x, A, mu, sigma)
        
        return y
    


    def fit_spaxel(self,z,max_nfev=256,params_to_use=None):
        global df_fit, df, df_cont
        
        spectrum = self.viewer_window.get_spectrum_at_spaxel(
            self.viewer_window.current_spaxel[0], self.viewer_window.current_spaxel[1]
        )
        
        spectrum = np.nan_to_num(spectrum)
        
        # Constants for velocity calculation
        c = 299792.458  # km/s
        
        # Perform the fit using lmfit
        try:
            result = piecewise_model.fit(spectrum, params_to_use, x=wavelengths, max_nfev=max_nfev)
            
            # Map parameter prefixes to continuum regions
            cont_map = {
                f"x{i + 1}_start": i + 1 for i in range(len(df_cont))
            }
            
            # Loop through each emission line in df
            for line_idx, line in enumerate(df.itertuples(), start=1):
                line_id = line.Line_ID
                line_name = line.Line_Name
                rest_wavelength = np.float64(df.loc[df['Line_ID']==line_id]['Rest Wavelength'])
            
                # Identify corresponding region for this line
                region_id = line.region_ID
                region_index = df_cont[df_cont['region_ID'] == region_id].index[0] + 1
            
                # Collect continuum model parameters
                cont_params = {
                    f'cont_region{region_index}_x_start': params_to_use[f'x{region_index}_start'].value,
                    f'cont_region{region_index}_x_end': params_to_use[f'x{region_index}_end'].value,
                    f'cont_region{region_index}_slope_init': params_to_use[f'slope{region_index}'].init_value,
                    f'cont_region{region_index}_slope_fit': result.params[f'slope{region_index}'].value,
                    f'cont_region{region_index}_intercept_init': params_to_use[f'intercept{region_index}'].init_value,
                    f'cont_region{region_index}_intercept_fit': result.params[f'intercept{region_index}'].value
                }
            
                # If there is an intermediate region, capture its parameters
                if region_index < len(df_cont):
                    cont_params.update({
                        f'cont_region{region_index}_x_int_start': params_to_use[f'x_int_{region_index}_start'].value,
                        f'cont_region{region_index}_x_int_end': params_to_use[f'x_int_{region_index}_end'].value,
                        f'cont_region{region_index}_slope_int_init': params_to_use[f'slope_int_{region_index}'].init_value,
                        f'cont_region{region_index}_slope_int_fit': result.params[f'slope_int_{region_index}'].value,
                    })
            
                # Get parameter keys for this line
                amp_key = f'amp{line_idx}'
                cen_key = f'cen{line_idx}'
                sigma_key = f'sigma{line_idx}'
                
                # Calculate velocities
                if np.isfinite(rest_wavelength):
                    # Initial velocity from initial parameters
                    vel_init = c * ((params_to_use[cen_key].init_value / (rest_wavelength*(z+1))) - 1)
                    # Fitted velocity from fitted parameters
                    vel_fit = c * ((result.params[cen_key].value / (rest_wavelength*(z+1))) - 1)
                    # Velocity from parameter if it exists
                    # vel_param = result.params[vel_key].value if vel_key in result.params else vel_fit
                    # vel_std = result.params[vel_key].stderr if vel_key in result.params else np.nan
                else:
                    vel_init = vel_fit = vel_param = vel_std = np.nan
            
                fit_entry = {
                    'spaxel_x': self.viewer_window.current_spaxel[0],
                    'spaxel_y': self.viewer_window.current_spaxel[1],
                    'region_ID': region_id,
                    'LineName': line_name,
                    'LineID': line_id,
            
                    # Amplitude information
                    'amp_init': params_to_use[amp_key].init_value,
                    'amp_fit': result.params[amp_key].value,
                    'amp_std': result.params[amp_key].stderr,
            
                    # Centroid information
                    'cen_init': params_to_use[cen_key].init_value,
                    'cen_fit': result.params[cen_key].value,
                    'cen_std': result.params[cen_key].stderr,
                    
                    # Velocity information
                    'vel_init': vel_init,
                    'vel_fit': vel_fit,
                    # 'vel_param': vel_param,
                    # 'vel_std': vel_std,
                    'rest_wavelength': rest_wavelength,
                    
                    # Sigma (width) information
                    'sigma_init': params_to_use[sigma_key].init_value,
                    'sigma_fit': result.params[sigma_key].value,
                    'sigma_std': result.params[sigma_key].stderr,
            
                    # Fit statistics
                    'BIC': result.bic,
                    'rchisq': result.redchi,
                    'success': result.success
                }
            
                # Add continuum parameters
                fit_entry.update(cont_params)
            
                fit_results.append(fit_entry)
    
        except RuntimeError as e:
            # print(f"Fit failed for spaxel {self.viewer_window.current_spaxel}: {e}")
            # Add a failed entry with basic information
            fit_results.append({
                'spaxel_x': self.viewer_window.current_spaxel[0],
                'spaxel_y': self.viewer_window.current_spaxel[1],
                'fit_success': False,
                'error': str(e)
            })
                
            

    def calculate_snr_map(self, linewl, Nsigma, search_window_factor=5, continuum_offset_factor=20, continuum_width_factor=30):
        """
        Calculate an SNR map and draw a red contour for the given Nsigma level, dynamically adjusting
        the search window and continuum offset based on the pixel scale from the 'wavelengths' array.
    
        Parameters:
            linewl (float): Central wavelength of the emission line.
            Nsigma (float): SNR threshold for contour drawing.
            search_window_factor (int): Multiplier to scale the search window based on pixel scale.
            continuum_offset_factor (int): Multiplier to scale the continuum offset based on pixel scale.
            continuum_width_factor (int): Multiplier to scale the continuum width based on pixel scale.
    
        Returns:
            snr_map (numpy.ndarray): 2D array of max SNR values across emission lines for each spaxel.
        """
        global snr_map
        num_wavelengths, nx, ny = np.shape(FITS_DATA)  # Cube dimensions
        snr_map = np.zeros((nx, ny))  # Store max SNR for each spaxel
    
        # Calculate the pixel scale (delta_wavelength) from the 'wavelengths' array
        delta_wavelength = wavelengths[1] - wavelengths[0]  # Assumes evenly spaced wavelengths
        
        print(f'Calculating S/N map for line_wl = {linewl}, masking at {Nsigma}-sigma!')
    
        for i in range(nx):
            for j in range(ny):
                spectrum = FITS_DATA[:, i, j]
                snr_values = []
                for index, row in df.iterrows():
                    centroid_idx = int(np.round(row['Centroid_0']))  # Ensure it is a single integer
    
                    # Dynamically determine the search window size based on the pixel scale (delta_wavelength)
                    search_window = int(search_window_factor * delta_wavelength)
                    continuum_offset = int(continuum_offset_factor * delta_wavelength)
                    continuum_width = int(continuum_width_factor * delta_wavelength)
                    
                    # Define search window around centroid for peak detection
                    mask = (wavelengths >= centroid_idx - search_window) & (wavelengths <= centroid_idx + search_window)
                    indices_within_range = np.where(mask)[0]
                    line_region = spectrum[indices_within_range]
                    peak_flux = np.percentile(line_region, 95)  # 95th percentile peak flux
    
                    # Define continuum region (avoiding emission line)
                    mask_left = (wavelengths >= centroid_idx - continuum_offset - continuum_width) & (wavelengths <= centroid_idx - continuum_offset)
                    mask_right = (wavelengths <= centroid_idx + continuum_offset + continuum_width) & (wavelengths >= centroid_idx + continuum_offset)
                    indices_left = np.where(mask_left)[0]
                    indices_right = np.where(mask_right)[0]
                    continuum_flux = np.concatenate([spectrum[indices_left], spectrum[indices_right]])
    
                    # Compute noise as the standard deviation of the continuum
                    noise_level = np.std(continuum_flux)
    
                    # Compute SNR for this emission line
                    snr = peak_flux / noise_level
                    snr_values.append(snr)
    
                # Store the maximum SNR across all lines
                snr_map[i, j] = np.max(snr_values)
    
        # Draw the contour on the viewer window at the specified Nsigma level
        contour = self.viewer_window.ax.contour(snr_map, levels=[Nsigma], colors='red', linewidths=1.5)
    
        # Update the canvas
        self.viewer_window.canvas.draw()
        self.viewer_window.spectrum_canvas.draw_idle()
    
        return snr_map




    def fit_cube(self, refit=False, rchisq_thresh=None):
        global df, df_fit, fit_results, snr_mask, piecewise_model, line, new_results#,params
        
        nx, ny = np.shape(FITS_DATA)[2], np.shape(FITS_DATA)[1]
        total_spaxels = nx * ny
    
        app = QApplication.instance() or QApplication([])
    
        # Progress UI
        progress_window = QWidget()
        progress_window.setWindowTitle("Fitting Progress")
    
        progress_bar = QProgressBar()
        progress_bar.setRange(0, total_spaxels)
    
        status_label = QLabel(f"Fitting spaxel 0 / {total_spaxels}")
        status_label.setAlignment(Qt.AlignCenter)
        status_label.setStyleSheet("font-size: 14px; font-weight: bold;")
    
        layout = QVBoxLayout()
        layout.addWidget(progress_bar)
        layout.addWidget(status_label)
        progress_window.setLayout(layout)
        progress_window.show()
    
        # Constants for velocity calculation
        c = 299792.458  # km/s
        
        # Get galaxy redshift
        if df_obs['redshift'].item() == '':
            z = 0
        else:
            z = df_obs['redshift'].item()
    
        # Model parameters
        params = Parameters()
        
        # Sort continuum regions by x1 (start wavelength)
        df_cont_sort = df_cont.sort_values(by='x1').reset_index(drop=True)
        
        # Ensure lines are sorted by Centroid_0 within each region
        df_sort = df.sort_values(by=['region_ID', 'Centroid_0']).reset_index(drop=True)
    
        # Loop through each continuum region and add parameters
        for i, row in df_cont.iterrows():
            region_id = row['region_ID']
    
            # Set x-limits for each region (fixed)
            params.add(f'x{i + 1}_start', value=row['x1'], vary=False)
            params.add(f'x{i + 1}_end', value=row['x2'], vary=False)
    
            # Linear continuum parameters for each region
            slope = row['Slope_0'] if np.isfinite(row['Slope_0']) else 0
            intercept = row['Intercept_0'] if np.isfinite(row['Intercept_0']) else 0
    
            params.add(f'slope{i + 1}', value=slope, vary=True)
            params.add(f'intercept{i + 1}', value=intercept, vary=True)
    
            # Add parameters for the linear region between this and the next region
            if i < len(df_cont) - 1:
                next_row = df_cont.iloc[i + 1]
                params.add(f'x_int_{i + 1}_start', value=row['x2'], vary=False)
                params.add(f'x_int_{i + 1}_end', value=next_row['x1'], vary=False)
    
                params.add(f'slope_int_{i + 1}', value=slope, vary=True)
                params.add(f'intercept_int_{i + 1}', value=intercept, vary=True)
    
            # Extract Gaussians associated with this region
            region_lines = df[df['region_ID'] == region_id]
    
            # Add Gaussian parameters for this region
            for j, line in enumerate(df.itertuples(), start=1):
                params.add(f'amp{j}', value=line.Amp_0, vary=True, min=line.Amp_0_lowlim, max=line.Amp_0_highlim)
                params.add(f'cen{j}', value=line.Centroid_0, vary=True, min=line.Centroid_0_lowlim, max=line.Centroid_0_highlim)
                params.add(f'sigma{j}', value=line.Sigma_0, vary=True, min=line.Sigma_0_lowlim, max=line.Sigma_0_highlim)
                
                
                # Add velocity parameter for this line
                if df.iloc[j-1]['Rest Wavelength']:
                    if type(df.iloc[j-1]['Rest Wavelength']) == str:
                        rest_wavelength = ast.literal_eval(df.iloc[j-1]['Rest Wavelength'])
                    else:
                        rest_wavelength = df.iloc[j-1]['Rest Wavelength']
                    
                else:
                    rest_wavelength = np.nan
                
                if (np.isfinite(rest_wavelength)) & ('constraints' in df.columns):
                    # Update constraints in df
                    df = update_constraints_with_velocity(df, z)
    
            # Add a "fake" parameter to store the number of Gaussians in this region
            params.add(f'NR{i + 1}', value=len(region_lines), vary=False)
    
        # choose the model based on parameters
        Nregions = len(df_cont)
        Nlines = len(np.unique(df['Line_ID']))
        
        # parameter constraints 
        add_dataframe_constraints_to_params(df, params)
    
        # print("\nParameters with Constraints:")
        # params.pretty_print()
        
        piecewise_model = Model(HyperCube_ModelFunctions.model_chooser(Nregions, Nlines))
        
        if len(fit_results) > 0 and not refit:
            fit_results = []
            
        # Handle refit case with rchisq threshold
        if refit and rchisq_thresh is not None and 'df_fit' in globals() and 'rchisq' in df_fit.columns:
            fit_results = []
            
            # Get indices of spaxels that need refitting
            bad_fits = df_fit[df_fit['rchisq'] > rchisq_thresh]
            bad_spaxels = bad_fits[['spaxel_x', 'spaxel_y']].drop_duplicates()
            
            # Create a copy of original parameters to perturb
            refit_params = params.copy()
            
            substrings = ['amp', 'cen', 'sigma']
            
            # Perturb all varying parameters by ±10%
            for name, param in refit_params.items():
                if param.vary:
                    if any(sub in name for sub in substrings):
                        current_val = param.value
                        perturbation = np.random.uniform(0.9, 1.1)  # 10% random variation
                        refit_params[name].value = current_val * perturbation
                        
                    # print('original parameters:')
                    # params.pretty_print()
                    # print('refit parameters:')
                    # refit_params.pretty_print()
                        
            # Refit only the bad spaxels with perturbed parameters
            for i, j in bad_spaxels.itertuples(index=False):
                if snr_map[j, i] >= snr_value:
                    # Get original amplitudes for this spaxel - properly extract as dict
                    original_data = df_fit[
                        (df_fit['spaxel_x'] == i) & 
                        (df_fit['spaxel_y'] == j)
                    ]
                    original_amplitudes = dict(zip(original_data['LineID'], original_data['amp_fit']))
                    
                    # Perform the fit
                    self.viewer_window.current_spaxel = (i, j)
                    self.fit_spaxel(z, max_nfev=512, params_to_use=refit_params)
                    
                    # Get new results (last n entries where n=number of lines in spaxel)
                    new_results = [r for r in fit_results[-len(original_amplitudes):] 
                                  if r['spaxel_x'] == i and r['spaxel_y'] == j]
                    
                    # Print comparison
                    print(f"\nSpaxel ({i}, {j}) - Amplitude Comparison:")
                    print(f"{'LineID':<10} | {'Old Amp':<10} | {'New Amp':<10} | {'Change (%)':<10}")
                    print("-" * 45)
                    
                    for new_r in new_results:
                        line_id = new_r['LineID']
                        try:
                            old_amp = original_amplitudes[line_id]
                            new_amp = new_r['amp_fit']
                            change_pct = (new_amp - old_amp)/old_amp * 100
                            print(
                                f"{line_id:<10} | {old_amp:<10.3f} | {new_amp:<10.3f} | "
                                f"{change_pct:>+10.1f}%"
                            )
                        except KeyError:
                            print(f"{line_id:<10} | {'N/A':<10} | {new_amp:<10.3f} | {'New':>10}")

        
                    current_spaxel = i * ny + j + 1
                    progress_bar.setValue(current_spaxel)
                    status_label.setText(f"Refitting spaxel {current_spaxel} / {total_spaxels}")
                    QApplication.processEvents()
        
                    if current_spaxel % 500 == 0 and psutil.virtual_memory().percent > 80:
                        gc.collect()
            
            # Merge results (keeping LineID matching)
            new_results = pd.DataFrame(fit_results)
            
            # First, drop the rows from df_fit that match the rows in df_new (on the three columns)
            df_fit_filtered = df_fit.merge(
                new_results[['spaxel_x', 'spaxel_y', 'LineID']],
                on=['spaxel_x', 'spaxel_y', 'LineID'],
                how='left',
                indicator=True
            ).query('_merge == "left_only"').drop(columns='_merge')
            
            # Then, append the new rows
            df_fit = pd.concat([df_fit_filtered, new_results], ignore_index=True)
            # new_results = pd.DataFrame(fit_results[-len(bad_spaxels):])
            # composite_keys = (df_fit['spaxel_x'].astype(str) + '_' + 
            #                  df_fit['spaxel_y'].astype(str) + '_' + 
            #                  df_fit['LineID'].astype(str))
            # new_keys = (new_results['spaxel_x'].astype(str) + '_' + 
            #             new_results['spaxel_y'].astype(str) + '_' + 
            #             new_results['LineID'].astype(str))
            
            # df_fit = df_fit[~composite_keys.isin(new_keys)]
            # df_fit = pd.concat([df_fit, new_results], ignore_index=True)
            
        else:
            # Normal fitting procedure for all spaxels
            for i in range(nx):
                for j in range(ny):
                    if snr_map[j,i] >= snr_value:
                        if progress_window.isVisible() and psutil.virtual_memory().percent > 80:
                            QApplication.processEvents()
        
                        self.viewer_window.current_spaxel = (i, j)
                        self.fit_spaxel(z, max_nfev=256, params_to_use=params)
        
                        current_spaxel = i * ny + j + 1
                        progress_bar.setValue(current_spaxel)
                        status_label.setText(f"Fitting spaxel {current_spaxel} / {total_spaxels}")
                        QApplication.processEvents()
        
                        if current_spaxel % 500 == 0 and psutil.virtual_memory().percent > 80:
                            gc.collect()
            
            # After fitting, create the results dataframe
            df_fit = pd.DataFrame(fit_results)
        
        # Clean up the dataframe
        if 'fit_success' in df_fit.columns:
            # Handle failed fits
            df_fit['success'] = df_fit.get('success', True) & ~df_fit['fit_success'].isna()
            df_fit.drop(columns=['fit_success'], inplace=True, errors='ignore')
        
        # Assign colors to the lines for plotting:
        df_fit['LineID'] = df_fit['LineID'].astype(float)
        unique_lines = df_fit['LineID'].unique()
        
        svg_colors = [
            'deeppink', 'dodgerblue', 'plum', 'orange', 'springgreen',
            'brown', 'cyan', 'magenta', 'burlywood', 'gold'
        ]
        
        lineid_to_color = {line: svg_colors[i % len(svg_colors)] for i, line in enumerate(unique_lines)}
        df_fit['color'] = df_fit['LineID'].map(lineid_to_color)
        
        progress_window.close()
        print("Fitting complete for entire cube.")

        

    def add_spectral_frame(self, title_text, df_cont, df, ID, addframe):
        
        
        """Creates a frame for a spectral region and adds it to the scroll area"""


        """ Initialize new region dataframes"""
        data_cont_initreg = {'Continuum Name': ['Continuum'],
                      'x1': [5000],
                      'x2': [5010],
                      'Slope_0': [0],
                      'Intercept_0': [0.002],
                      'Slope_fit': [np.nan],
                      'Intercept_fit': [np.nan],
                      'region_ID': [ID],
                      'lineactor': [self.current_line]}

        data_lines_initreg = {'Line_ID': [0, 1],
                'Line_Name': ['line 1', 'line 2'],
                'SNR': [np.nan,np.nan],
                'Amp_0': [0.2348, 0.343],
                'Centroid_0': [5504, 6533],
                'Sigma_0': [0.345, 0.45],
                'Amp_fit': [np.nan,np.nan],
                'Centroid_fit': [np.nan,np.nan],
                'Sigma_fit': [np.nan,np.nan],
                'region_ID': [ID,ID]}

        df_cont_new = pd.DataFrame(data_cont_initreg)
        df_new = pd.DataFrame(data_lines_initreg)
        
        
        # if ID > 0:
        if addframe == True:
            df_cont = pd.concat([df_cont, df_cont_new], ignore_index=True)
            df = pd.concat([df, df_new], ignore_index=True)
        
        # Create frame
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Raised)
        frame.setObjectName(f"frame_{ID}")
        
        # Set max width to prevent exceeding main window width
        max_frame_width = self.width()  # Allow some padding
        frame.setMaximumWidth(max_frame_width)
        frame.setMinimumWidth(1150)  # Adjust based on content needs
        frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
    
        # frame.setStyleSheet("""
        #     border: 2px solid black;
        #     background-color: snow;
        #     border-radius: 15px;  /* Adjust roundness */
        #     padding: 10px;
        # """)
    
        # Frame layout
        frame_layout = QVBoxLayout(frame)


        # Title
        title = QLabel(title_text)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 24pt;
            font: bold;
            width: 64px;
        """)
        # title.setStyleSheet("""
        #     font-family: "Segoe UI";
        #     font-size: 24pt;
        #     border: none;
        #     padding-right: 10px;
        #     padding-left: 10px;
        #     padding-top: 5px;
        #     padding-bottom: 5px;
        #     background-color: snow;
        #     color: black;
        #     font: bold;
        #     width: 64px;
        # """)
        frame_layout.addWidget(title)

        
        # Grid Layout
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(10)  # Adjust horizontal spacing
        grid_layout.setVerticalSpacing(5)    # Adjust vertical spacing

        # Ensure grid does not stretch frame
        grid_layout.setContentsMargins(10, 10, 10, 10)
    
        frame_layout.addLayout(grid_layout)
        
        
        # Add headers and buttons
        self.add_continuum_buttons(ID, grid_layout)
        self.add_spectral_lines_button_header(grid_layout)
        self.add_spectral_lines(ID, grid_layout)
           
                        
        # Add emission line row of buttons
        add_line_button = QPushButton('+')
        # add_line_button.setStyleSheet("""
        #     font-family: "Segoe UI";
        #     font-size: 12pt;
        #     border: 2px solid;
        #     border-color: green;
        #     border-radius: 5px;
        #     padding-right: 10px;
        #     padding-left: 10px;
        #     padding-top: 5px;
        #     padding-bottom: 5px;
        #     background-color: limegreen;
        #     color: k;
        #     font: bold;
        #     width: 24px;
        # """)

        # add_line_button.clicked.connect(partial(self.on_addline_button_click, data_frame=df, frame_id=ID,send_new_df=False))
        add_line_button.clicked.connect(partial(self.on_addline_widget, data_frame=df, frame_id=ID))
        grid_layout.addWidget(add_line_button,4+len(df)*2,0)
        
        # Append this frame to the scroll layout
        self.scroll_layout.addWidget(frame)
        
        

    def update_button_value(self, frame_id, button_name, new_value):
        """Update the button text based on row and column"""
        if any(x in button_name for x in ['sourcename','redshift','resolvingpower']):
            self.source_name_button.setText(f"Source Name: {df_obs.loc[0, 'sourcename']}")
            self.source_redshift_button.setText(f"Source Redshift: {df_obs.loc[0, 'redshift']}")
            self.resolving_power_button.setText(f"Resolving Power: {df_obs.loc[0, 'resolvingpower']}")
        else:
        
            if (frame_id, button_name) in self.buttons_dict:
                button = self.buttons_dict[(frame_id, button_name)]
                button.setText(str(new_value))
    
    def on_deleteregion_button_click(self, frame_id):
        global df, df_cont
        """Deletes a spectral line from df, removes the UI frame if empty, and removes the line from the spectrum plot."""
        print(f"Deleting frame {frame_id}")
    
        # Fetch the line object from df_cont
        line_actor = df_cont.loc[np.int64(df_cont["region_ID"]) == frame_id, 'lineactor'].item()
    
        if line_actor is not None:
            print(f"Removing line: {line_actor}")
            line_actor.remove()  # Remove the line from the plot
            
            # Redraw the figure via ViewerWindow
            if hasattr(self, 'viewer_window') and hasattr(self.viewer_window, 'spectrum_canvas'):
                self.viewer_window.spectrum_canvas.draw_idle()  # Efficient redrawing
            else:
                print("Warning: No reference to spectrum_canvas found!")
        else:
            print(f"No valid line found for frame_id={frame_id}")
    
        # Remove frame from dataframe
        df = df.reset_index(drop=True)  # Ensure unique indices
        df = df.drop(df.index[np.int64(df["region_ID"]) == frame_id]).reset_index(drop=True)
        df_cont = df_cont.drop(df_cont.index[np.int64(df_cont["region_ID"]) == frame_id]).reset_index(drop=True)
    
        # Iterate over all items in the layout and remove them
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if isinstance(widget, QFrame):
                self.scroll_layout.removeWidget(widget)
                widget.deleteLater()
    
        # Regenerate UI
        self.add_spaxel_info_frame('Observation Data', df_cont, df, df_obs)
        for ID in np.unique(df_cont['region_ID']):
            ID = np.int64(ID)
            self.add_spectral_frame('Spectral Region ' + str(ID + 1), df_cont, df, ID, addframe=False)


    def on_deleteline_button_click(self, frame_id, line_id):
        global df, df_cont
        """Deletes a spectral line from df and removes the frame if empty"""
        print(f"Deleting line {line_id} from frame {frame_id}")
    
        # Delete all curveactors for the given frame_id
        for actor in df.loc[df["region_ID"] == frame_id, 'curveactor']:
            if actor and actor in self.viewer_window.spectrum_ax.lines:
                actor.remove()
        # self.viewer_window.spectrum_ax.cla()
        # xmin,xmax=self.viewer_window.spectrum_ax.get_xlim()
        # ymin,ymax=self.viewer_window.spectrum_ax.get_ylim()
        # self.viewer_window.spectrum_ax.step(wavelengths,spectrum,lw=0.5,color='w')
        # self.viewer_window.spectrum_ax.set_xlim(xmin,xmax)
        # self.viewer_window.spectrum_ax.set_ylim(ymin,ymax)
        
        
        
        # Reset DataFrame indices and remove the line from df
        df = df.reset_index(drop=True)  
        df = df.drop(df[(df["region_ID"].astype(int) == frame_id) & (df["Line_ID"].astype(int) == line_id)].index)
        
        # Clean up the UI frames
        for i in reversed(range(self.scroll_layout.count())):  
            widget = self.scroll_layout.itemAt(i).widget()
            if isinstance(widget, QFrame):
                self.scroll_layout.removeWidget(widget)
                widget.deleteLater()
        
        # Refresh the UI and redraw the plot
        self.add_spaxel_info_frame('Observation Data', df_cont, df, df_obs)
        for ID in np.unique(df_cont['region_ID']):
            ID = np.int64(ID)
            self.add_spectral_frame(f'Spectral Region {ID + 1}', df_cont, df, ID, addframe=False)
        
        # Safely remove lineactor objects
        for actor in df_cont.loc[df_cont["region_ID"] == frame_id, 'lineactor']:
            if actor and actor in self.viewer_window.spectrum_ax.lines:
                actor.remove()
        
        self.viewer_window.spectrum_canvas.draw_idle()  
        self.viewer_window.rebuild_plot(region_ID=frame_id, from_file=False, show_init=True, show_fit=False, x=0, y=0)
            
    def on_addline_widget(self, data_frame, frame_id):
        global df, df_cont
        # Retrieve the line associated with frame_id
        line_obj = df_cont.loc[df_cont["region_ID"] == frame_id, 'lineactor'].item()
    
        if not isinstance(line_obj, plt.Line2D):
            print(f"Error: No valid line object found.")
            return
    
        # Store the line and initialize Gaussian parameters
        self.active_line = line_obj
        region = df_cont.loc[df_cont["region_ID"] == frame_id]
        self.x1, self.x2 = region["x1"].item(), region["x2"].item()
        self.m, self.b = region["Slope_0"].item(), region["Intercept_0"].item()
        # Generate updated Gaussian + Line function
        x_vals = np.linspace(self.x1, self.x2, 1000)
        y_line = self.m * x_vals + self.b  # The baseline (without Gaussians)
        
        
        # Recompute all Gaussians from scratch (prevents runaway summing)
        y_gaussian_total = np.zeros_like(x_vals)
    
        # Sum over all existing Gaussians in df
        for _, row in df.iterrows():
            y_gaussian_total += row["Amp_0"] * np.exp(-((x_vals - row["Centroid_0"]) ** 2) / (2 * row["Sigma_0"] ** 2))
    
        self.gaussian_amplitude = 0.1
        self.gaussian_x0 = np.mean(x_vals)
        self.gaussian_sigma = 1
        # If we're actively adjusting a new Gaussian, include it
        y_gaussian_new = self.gaussian_amplitude * np.exp(-((x_vals - self.gaussian_x0) ** 2) / (2 * self.gaussian_sigma ** 2))
        y_gaussian_total += y_gaussian_new
    
        # Final y-values
        y_new = y_line + y_gaussian_total
    
        # Update the existing line object
        self.active_line.set_data(x_vals, y_new)
    
        # Redraw dynamically
        self.viewer_window.spectrum_ax.figure.canvas.draw_idle()
        
        if len(df) == 0: 
            line_id = 0
        else:
            line_id = np.max(np.int64(df['Line_ID']))+1
        
        df_new = pd.DataFrame({'Line_ID': [line_id],
                'Line_Name': [f'Line '+str(len(df.loc[df["region_ID"]==frame_id]))],
                'SNR':[np.nan],
                'Rest Wavelength': [np.nan],
                'Amp_0': [self.gaussian_amplitude],
                'Amp_0_lowlim': [0],
                'Amp_0_highlim': [9999],
                'Centroid_0': [self.gaussian_x0],
                'Centroid_0_lowlim': [-9999],
                'Centroid_0_highlim': [9999],
                'Sigma_0': [self.gaussian_sigma],
                'Sigma_0_lowlim': [0],
                'Sigma_0_highlim': [9999],
                'Amp_fit': [np.nan],
                'Centroid_fit': [np.nan],
                'Sigma_fit': [np.nan],
                'region_ID': [frame_id],
                'curveactor': [self.active_line]})
        df = pd.concat([df,df_new])
            
        self.on_addline_button_click(data_frame=df, frame_id=frame_id)
        
        
        
    def print_active_frames(self):
        print("Active frames in the scroll window:")
        for i in range(self.scroll_layout.count()):
            widget = self.scroll_layout.itemAt(i).widget()
            if isinstance(widget, QFrame):
                print(f"Frame ID: {widget.objectName()}")
            else:
                print("Non-frame widget found in scroll layout.")
        
    
    def on_addline_button_click(self, data_frame, frame_id):
        global df

        # Find the corresponding frame in the UI
        frame = None
        for i in range(self.scroll_layout.count()):
            widget = self.scroll_layout.itemAt(i).widget()
            if isinstance(widget, QFrame) and str(frame_id) in widget.objectName():
                frame = widget
                break
    
        # self.print_active_frames()
    
        if not frame:
            print("Frame not found!")
            return
    
    
        # Get the grid layout inside the frame
        grid_layout = frame.layout().itemAt(1).layout()  # Assuming it's the second layout added
    
        # Find and remove the existing button
        add_line_button = None
        for i in range(grid_layout.count()):
            item = grid_layout.itemAt(i)
            if isinstance(item.widget(), QPushButton) and item.widget().text() == '+':
                add_line_button = item.widget()
                grid_layout.removeWidget(add_line_button)
                break
    
        # Determine the new row index
        row = df.shape[0]*2 - 1  # New row index
        button_name = f'line_{row}'
    
        # Add new row of buttons
        self.add_spectral_lines(frame_id, grid_layout)
                
        # Re-add the add_line_button below the last row
        if add_line_button:
            grid_layout.addWidget(add_line_button, 5 + row, 0)

        # """
    
    def clear_snr_visualizer(self, frame_id, button_name):
        global df
        self.viewer_window.draw_image(FITS_DATA,cmap='Grays',scale='linear',from_fits=True)
        self.update_button_value(frame_id, button_name, np.nan)
        df.loc[(np.int64(df['region_ID'])==frame_id) & 
                        (np.float64(df['Line_ID'])==np.float64(button_name.split('~')[0])),'SNR'] = np.nan


    def on_button_click(self, data_frame, frame_id, button_name):
        """Handle button click, show a QLineEdit for editing the value"""
        global df_cont, df
        print(str(frame_id)+', '+button_name)
        if 'SNR' in button_name:
            # Create a dialog window with two input fields
            snr_window = QDialog(self)
            snr_window.setWindowTitle(f'Line Name and Parameter Constraints')
            
            # Create layout
            layout = QVBoxLayout()
            
            # Input box with placeholder text
            text_box = QLineEdit()
            # print(df)
            # print(f'frame_id = {frame_id}')
            # print(f'button_name = {button_name}')
            snr_value = df.loc[
                (df['region_ID'].astype(np.int64) == frame_id) & 
                (df['Line_ID'].astype(np.float64) == np.float64(button_name.split('~')[0])),
                'SNR'].item()
            # Convert to float safely
            try:
                snr_value = np.float64(snr_value)  # Converts valid numbers and "nan" strings to float
            except ValueError:
                snr_value = np.nan  # If conversion fails, assume NaN
                
            # Check for NaN properly
            if not np.isnan(snr_value):
                text_box.setText(str(snr_value))
            else:
                text_box.setPlaceholderText("Enter SNR threshold")
            
            layout.addWidget(text_box)    
            
            # Button layout (side-by-side buttons)
            button_layout = QHBoxLayout()
            
            # "Visualize" button
            visualize_button = QPushButton("Visualize")
            visualize_button.clicked.connect(lambda: self.on_submit(text_box, data_frame, frame_id, button_name, button_type='SNR'))
            button_layout.addWidget(visualize_button)
            
            # "Clear" button to close the window
            clear_button = QPushButton("Clear")
            # Reset to the white light image
            clear_button.clicked.connect(lambda: self.clear_snr_visualizer(frame_id, button_name))
            clear_button.clicked.connect(snr_window.close)
            button_layout.addWidget(clear_button)
            
            # Add button layout to main layout
            layout.addLayout(button_layout)
            
            # Set layout and show the window
            snr_window.setLayout(layout)
            snr_window.show()
            
        if ('_highlim' in button_name) or ('_lowlim' in button_name):
            string = str(df.loc[(np.int64(df['region_ID'])==frame_id) & 
                            (np.float64(df['Line_ID'])==np.int64(button_name.split('~')[0]))][button_name.split('~')[1]].item())
            text_box = QLineEdit()
            text_box.setText(string)  # Set the initial value from the dataframe
            text_box.setGeometry(200, 200, 100, 30)
            text_box.show()
            # # # # Connect the returnPressed signal to handle text submission
            text_box.returnPressed.connect(lambda: self.on_submit(text_box, data_frame, frame_id, button_name, button_type='param_limit'))
        
        else:
        
            if any(x in button_name for x in ['sourcename','redshift','resolvingpower']):
                string = df_obs[button_name].item()
                text_box = QLineEdit()
                text_box.setText(string)  # Set the initial value from the dataframe
                text_box.setGeometry(200, 200, 100, 30)
                text_box.show()
                # # # # Connect the returnPressed signal to handle text submission
                text_box.returnPressed.connect(lambda: self.on_submit(text_box, data_frame, frame_id, button_name, button_type='obs_button'))
            else:
            
            
                if button_name.split('~')[0] == 'continuum':
                    string = str(data_frame.loc[np.int64(data_frame['region_ID']) == frame_id][button_name.split('~')[1]].item())
                    button_type = 'continuum'
                # elif button_name.split('~')[1] == 'Line_Name':
                #     string = str(data_frame.loc[(data_frame['region_ID'] == np.float64(frame_id)) & (np.float64(data_frame['Line_ID'])==np.float64(button_name.split(',')[0]))][button_name.split('~')[1]].item())
                else:
                    string = str(data_frame.loc[(data_frame['region_ID'] == np.float64(frame_id)) & (np.float64(data_frame['Line_ID'])==np.float64(button_name.split('~')[0]))][button_name.split('~')[1]].item())
                    button_type = 'line'
                
            if 'Rest Wavelength' in button_name:
                text_box = QLineEdit()
                # text_box.setText('')  # Set the initial value from the dataframe
                text_box.show()
                text_box.returnPressed.connect(lambda: self.on_submit(text_box, data_frame, frame_id, button_name, button_type='RestWavelength'))
    
            if 'Line_Name' in button_name:
                
                lineid = np.int64(button_name.split('~')[0])
                linename = df.loc[(df['region_ID'] == frame_id) & (np.float64(df['Line_ID'])==np.int64(button_name.split('~')[0]))]['Line_Name'].item()
                parname = button_name.split("~")[1].split("_")[0]            
    
                # Create a dialog window with two input fields
                dialog = QDialog(self)
                dialog.setWindowTitle(f'Line Name and Parameter Constraints')
    
                # Layout for the dialog
                layout = QVBoxLayout(dialog)
    
                # Create a horizontal layout for the label and checkmark
                label_layout = QHBoxLayout()
    
                # Label for Initial Guess
                label_line_name = QLabel('Line name:', dialog)
                label_layout.addWidget(label_line_name)
    
                # Create the confirmation label (green checkmark) and hide it initially
                confirm_label = QLabel('✔️', dialog)
                confirm_label.setStyleSheet("color: limegreen; font-size: 16px;")
                confirm_label.hide()
    
                # Add the checkmark and ensure it's right next to the label
                label_layout.addWidget(confirm_label)
    
                # Ensure no additional stretching pushes them apart - add a stretch to push to the right
                label_layout.addStretch(1)
    
                # Add the horizontal layout to the main vertical layout
                layout.addLayout(label_layout)
    
                # Function to show and hide the checkmark after 2 seconds
                def on_input_confirmed():
                    confirm_label.show()
                    QTimer.singleShot(2000, confirm_label.hide)
    
                # Text box for line name
                text_box = QLineEdit(dialog)
                text_box.setText(linename)  # Set the initial value from the dataframe
                text_box.setPlaceholderText('Enter new line name')
                text_box.returnPressed.connect(lambda: [self.on_submit(text_box, data_frame, frame_id, button_name, button_type='line_name'), on_input_confirmed()]) # Connect return pressed signal
    
                # Add the text box to the main vertical layout
                layout.addWidget(text_box)
    
                # Label for constraints box
                label_constraints = QLabel('Parameter constraints:', dialog)
                layout.addWidget(label_constraints)
    
                # Helper function to create a numbered QTextEdit with dynamic height
                def create_numbered_textbox(dialog, number, placeholder_text, initial_text=''):
                    box_layout = QHBoxLayout()
    
                    # Add the enumerator (e.g., 1., 2., etc.)
                    label_number = QLabel(f'{number}.', dialog)
                    box_layout.addWidget(label_number)
    
                    # Create the QTextEdit box
                    text_box_constraints = QTextEdit(dialog)
                    text_box_constraints.setPlainText(initial_text)  # Set initial text
                    text_box_constraints.setPlaceholderText(placeholder_text)
    
                    # Set dynamic height based on text metrics with padding
                    font_metrics = QFontMetrics(text_box_constraints.font())
                    text_height = font_metrics.height()
                    padding = 12  # Small padding for better appearance
                    text_box_constraints.setFixedHeight(text_height + padding)
    
                    box_layout.addWidget(text_box_constraints)
                    layout.addLayout(box_layout)
                    return text_box_constraints
    
                # Get existing constraints if available
                constraints_for_line = []
                if 'constraints' in df.columns:
                    try:
                        constraints_value = df.loc[(df['region_ID'] == frame_id) & (df['Line_Name'] == linename), 'constraints'].item()
                        if isinstance(constraints_value, str):
                            # Try to safely evaluate the string as a Python literal (list)
                            try:
                                constraints_list = ast.literal_eval(constraints_value)
                                if isinstance(constraints_list, list) and len(constraints_list) == 5:
                                    constraints_for_line = constraints_list
                                    print(f'constraints (from string): {constraints_for_line}')
                            except (ValueError, SyntaxError):
                                print(f"Warning: Could not parse constraints string: {constraints_value}")
                                constraints_for_line = ['', '', '', '', '']
                        elif isinstance(constraints_value, list) and len(constraints_value) == 5:
                            constraints_for_line = constraints_value
                            print(f'constraints (from list): {constraints_for_line}')
                        else:
                            constraints_for_line = ['', '', '', '', '']
                    except (KeyError, ValueError):
                        constraints_for_line = ['', '', '', '', '']
                else:
                    constraints_for_line = ['', '', '', '', '']
    
                # Create the constraints text boxes with enumerators and initial values
                text_box_constraints_01 = create_numbered_textbox(dialog, 1, f'e.g., amp <= amp_[line name]', constraints_for_line[0] if len(constraints_for_line) > 0 else '')
                text_box_constraints_02 = create_numbered_textbox(dialog, 2, '', constraints_for_line[1] if len(constraints_for_line) > 1 else '')
                text_box_constraints_03 = create_numbered_textbox(dialog, 3, '', constraints_for_line[2] if len(constraints_for_line) > 2 else '')
                text_box_constraints_04 = create_numbered_textbox(dialog, 4, '', constraints_for_line[3] if len(constraints_for_line) > 3 else '')
                text_box_constraints_05 = create_numbered_textbox(dialog, 5, '', constraints_for_line[4] if len(constraints_for_line) > 4 else '')
    
                # Submit button to send constraints
                submit_button = QPushButton('Submit Constraints', dialog, default=False, autoDefault=False)
                submit_button.clicked.connect(lambda: self.submit_constraints([
                    text_box_constraints_01,
                    text_box_constraints_02,
                    text_box_constraints_03,
                    text_box_constraints_04,
                    text_box_constraints_05
                ], frame_id, lineid))
                layout.addWidget(submit_button)
    
    
                # Increase the width by 10%
                current_width = dialog.width()
                current_height = dialog.height()
                new_width = int(current_width * 2)  # 10% wider
                dialog.resize(350, 100)
    
                # Connect returnPressed for text_box (normal behavior)
                text_box.returnPressed.connect(lambda: [self.on_submit(text_box, data_frame, frame_id, button_name, button_type='line_name'), on_input_confirmed()])
    
                # Show the dialog
                dialog.exec_()

    def submit_constraints(self, text_box_constraints_list, frame_id, lineid):
        global df  # Assuming 'df' is a global DataFrame
        """Handles the submission of the constraints from all text boxes and adds them to the DataFrame."""
        all_constraints_text = [text_box.toPlainText() for text_box in text_box_constraints_list]
        # Ensure all_constraints_text has exactly 5 elements
        while len(all_constraints_text) < 5:
            all_constraints_text.append('')
        all_constraints_text = all_constraints_text[:5]
    
        # Find the row to update
        df = df.reset_index(drop=True)
        linename = df.loc[ (df['region_ID'] == frame_id) & (df['Line_ID'] == lineid)]['Line_Name']
        mask = (df['region_ID'] == frame_id) & (df['Line_ID'] == lineid)
    
        if 'constraints' not in df.columns:
            df['constraints'] = [[] for _ in range(len(df))]  # Initialize with empty lists
        for index, row in df.iterrows():
            if mask[index]:
                df.at[index, 'constraints'] = all_constraints_text
            elif not df.at[index, 'constraints'] or not isinstance(df.at[index, 'constraints'], list) or len(df.at[index, 'constraints']) != 5:
                # Only initialize with blank strings if the 'constraints' column
                # doesn't exist or is not a valid 5-element list
                df.at[index, 'constraints'] = ['', '', '', '', '']
    
        print(f"Constraints submitted for region_ID: {frame_id}, Line_Name: {linename}")
        print("Constraints:", all_constraints_text)



    def on_submit(self, text_box, data_frame, frame_id, button_name, button_type):
        global df_cont, df, snrmap, snr_value
        """Submit the value from the text box, update the dataframe, and adjust related parameters."""
        
        if button_type == 'SNR':
            snr_value = np.float64(text_box.text())
            linewl = df.loc[(np.int64(df['region_ID'])==frame_id) & 
                            (np.float64(df['Line_ID'])==np.int64(button_name.split('~')[0]))]['Centroid_0']
            self.viewer_window.ax.cla()
            self.viewer_window.draw_image(FITS_DATA,cmap='Grays',scale='linear',from_fits=True)
            # snrmap = self.calculate_snr_map(linewl, snr_value, search_window=5, continuum_offset=20, continuum_width=30)
            snrmap = self.calculate_snr_map(linewl, snr_value, search_window_factor=5, continuum_offset_factor=20, continuum_width_factor=30)
            self.update_button_value(frame_id, button_name, snr_value)
        if button_type == 'obs_button':
            new_value = np.float64(text_box.text()) if 'name' not in button_name else str(text_box.text())
            data_frame[button_name] = new_value
            self.update_button_value(frame_id, button_name, new_value)
            
        elif button_type == 'param_limit':
            if type(text_box.text()) == str:
                new_value = np.float64(text_box.text())
                # new_value = text_box.text()
            # else:
            #     new_value = np.float64(text_box.text())

            df.loc[(df['region_ID']==frame_id) & 
                            (np.int64(df['Line_ID']==np.int64(button_name.split('~')[0]))),button_name.split('~')[1]] = new_value
            self.update_button_value(frame_id, button_name, new_value)
            
        elif button_type == 'RestWavelength':
            new_value = ast.literal_eval(text_box.text())
            df.loc[(df['region_ID']==frame_id) & 
                            (np.int64(df['Line_ID']==np.int64(button_name.split('~')[0]))),'Rest Wavelength'] = new_value
            self.update_button_value(frame_id, button_name, text_box.text())
        
        else:
            
            try:
                
                # Extract new value from the text box
                new_value = np.float64(text_box.text()) if 'Name' not in button_name else str(text_box.text())
        
                # Identify which parameter is being updated
                param = button_name.split('~')[1]
                
            except ValueError:
                print(f"Invalid input: {text_box.text()} is not valid for this column.")
        
                if button_type == 'continuum':
                    
                    # Directly update the dataframe with the new value
                    data_frame.loc[data_frame['region_ID'] == frame_id, param] = new_value
                    
                    # **Important**: Fetch updated values *after* setting the new value
                    x1 = data_frame.loc[data_frame['region_ID'] == frame_id, 'x1'].values[0]
                    x2 = data_frame.loc[data_frame['region_ID'] == frame_id, 'x2'].values[0]
                    slope = data_frame.loc[data_frame['region_ID'] == frame_id, 'Slope_0'].values[0]
                    intercept = data_frame.loc[data_frame['region_ID'] == frame_id, 'Intercept_0'].values[0]
    
        
                    # Adjust other values based on what changed
                    if param == 'x1' or param == 'x2':
                        # Ensure y1 and y2 remain fixed, update slope and intercept accordingly
                        y1 = slope * x1 + intercept
                        y2 = slope * x2 + intercept
                        slope = (y2 - y1) / (x2 - x1) if x1 != x2 else 0
                        intercept = y1 - slope * x1
            
                        # **Re-store slope & intercept in dataframe** after updating them
                        data_frame.loc[data_frame['region_ID'] == frame_id, 'Slope_0'] = slope
                        data_frame.loc[data_frame['region_ID'] == frame_id, 'Intercept_0'] = intercept
            
                    elif param == 'Slope_0':
                        # Recalculate y-values based on the new slope
                        y1 = slope * x1 + intercept
                        y2 = slope * x2 + intercept
            
                    elif param == 'Intercept_0':
                        # Recalculate y-values based on the new intercept
                        y1 = slope * x1 + intercept
                        y2 = slope * x2 + intercept
            
                    # **Ensure UI updates instantly**: Force refresh of button values
                    self.update_button_value(frame_id, 'continuum~x1', x1)
                    self.update_button_value(frame_id, 'continuum~x2', x2)
                    self.update_button_value(frame_id, 'continuum~Slope_0', slope)
                    self.update_button_value(frame_id, 'continuum~Intercept_0', intercept)
            
                    # **Redraw the updated line with correct values**
                    self.viewer_window.update_line_with_slope_intercept(x1, x2, slope, intercept)
        
            if button_type == 'line':
                df.loc[(df['region_ID'] == np.float64(frame_id)) & (df['Line_ID'] == np.float64(button_name.split('~')[0])), param] = new_value
                self.update_button_value(frame_id, button_name, new_value)
                df_cont.loc[(df_cont["region_ID"] == frame_id)]['lineactor'].item().remove()
                self.viewer_window.spectrum_canvas.draw_idle()  # Efficient redrawing
                self.viewer_window.rebuild_plot(frame_id, from_file=False,show_init=True,show_fit=False,x=0,y=0)
            
            if button_type == 'line_name':
                df.loc[(df['region_ID'] == np.float64(frame_id)) & (df['Line_ID'] == np.float64(button_name.split('~')[0])), param] = new_value
                print(f'frame id: {frame_id}, button name: {button_name}, new name: {new_value}')
                self.update_button_value(frame_id, button_name, new_value)
                df_cont.loc[(df_cont["region_ID"] == frame_id)]['lineactor'].item().remove()
                self.viewer_window.spectrum_canvas.draw_idle()  # Efficient redrawing
                self.viewer_window.rebuild_plot(frame_id, from_file=False,show_init=True,show_fit=False,x=0,y=0)
    
        # Hide the text box after submission
        if (button_type != 'line_name') & (button_type != 'SNR'):
            text_box.hide()

def load_pixmap(image_path):
    """Load QPixmap with proper error handling"""
    pixmap = QPixmap()
    if not pixmap.load(image_path):
        # Try alternative loading method
        with open(image_path, 'rb') as f:
            pixmap.loadFromData(f.read())
    if pixmap.isNull():
        raise ValueError(f"Failed to load image: {image_path}\n"
                       f"Supported formats: {', '.join(QPixmap.supportedImageFormats())}")
    return pixmap

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = Path(sys._MEIPASS)
    except Exception:
        base_path = Path(__file__).parent

    # Try multiple possible locations
    search_paths = [
        base_path / relative_path,
        base_path / 'Resources' / relative_path,
        Path.cwd() / relative_path
    ]
    
    for path in search_paths:
        if path.exists():
            print(f"Found resource at: {path}")
            return str(path)
    
    raise FileNotFoundError(f"Resource not found in any of: {search_paths}")
    
    
def load_stylesheet(app,filename):
    try:
        stylesheet_path = resource_path(filename)
        print(f"Loading stylesheet from: {stylesheet_path}")
        
        with open(stylesheet_path, "r") as f:
            style = f.read()
            app.setStyleSheet(style)
    except Exception as e:
        print(f"Error loading stylesheet: {e}")
        # Fallback to basic dark theme
        app.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
        """)    
    
def main():
    """Entry point for Hypercube spectral analysis tool."""
    try:
        app = QApplication(sys.argv)
        load_stylesheet(app, 'QDarkOrange_style.qss')
        
        
        print("Application starting...")
        
        # Load splash image
        try:
            image_path = resource_path("hypercube_logo.png")
            print(f"Loading image from: {image_path}")
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at: {image_path}")
            
            pixmap = load_pixmap(image_path)
            pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            print("Image loaded successfully")
            
        except Exception as e:
            print(f"Image loading failed: {str(e)}")
            # Fallback to blank pixmap
            pixmap = QPixmap(400, 300)
            pixmap.fill(Qt.white)
        
        
        # Show splash screen
        splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
        splash.show()
        print("Splash screen shown")
        
        splash.raise_()
        splash.repaint()
        app.processEvents()
        print("Events processed")
        
        # Create main window
        print("Creating main window...")
        viewer = ViewerWindow()
        
        # Set up transition
        def show_main_window():
            print("Closing splash and showing main window")
            splash.finish(viewer)
            viewer.show()
        
        QTimer.singleShot(2000, show_main_window)
        print("Timer set")
        
        print("Entering main loop")
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()