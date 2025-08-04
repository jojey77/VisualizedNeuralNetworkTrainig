"""
Seven-Segment Neural Network Trainer and Visualizer

A GUI application for training and visualizing a neural network that recognizes
seven-segment display patterns for digits 0-9.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration constants."""
    
    # Model parameters
    INPUT_SIZE: int = 7
    HIDDEN_SIZE: int = 16
    OUTPUT_SIZE: int = 10
    LEARNING_RATE: float = 0.1
    EPOCHS_PER_BATCH: int = 200
    
    # UI parameters
    WINDOW_TITLE: str = "Seven-Segment Neural Network Trainer"
    CANVAS_WIDTH: int = 120
    CANVAS_HEIGHT: int = 200
    SEGMENT_LENGTH: int = 20
    SEGMENT_THICKNESS: int = 5
    
    # Colors
    SEGMENT_ON_COLOR: str = '#00FF00'
    SEGMENT_OFF_COLOR: str = '#330000'
    CANVAS_BG_COLOR: str = '#000000'
    POSITIVE_WEIGHT_COLOR: str = '#00AA00'
    NEGATIVE_WEIGHT_COLOR: str = '#AA0000'
    NEURON_ACTIVE_COLOR: str = '#FFA500'
    NEURON_INACTIVE_COLOR: str = '#CCCCCC'


class TrainingData:
    """Manages the training data for seven-segment displays."""
    
    SEGMENT_PATTERNS = {
        0: [1, 1, 1, 1, 1, 1, 0],
        1: [0, 1, 1, 0, 0, 0, 0],
        2: [1, 1, 0, 1, 1, 0, 1],
        3: [1, 1, 1, 1, 0, 0, 1],
        4: [0, 1, 1, 0, 0, 1, 1],
        5: [1, 0, 1, 1, 0, 1, 1],
        6: [1, 0, 1, 1, 1, 1, 1],
        7: [1, 1, 1, 0, 0, 0, 0],
        8: [1, 1, 1, 1, 1, 1, 1],
        9: [1, 1, 1, 1, 0, 1, 1],
    }
    
    @classmethod
    def get_training_tensors(cls) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get training data as PyTorch tensors."""
        X = torch.tensor(list(cls.SEGMENT_PATTERNS.values()), dtype=torch.float32)
        y = torch.tensor(list(cls.SEGMENT_PATTERNS.keys()), dtype=torch.long)
        return X, y


@dataclass
class PredictionResult:
    """Result of a neural network prediction."""
    predicted_digit: int
    probabilities: List[float]
    confidence: float
    
    @property
    def confidence_percentage(self) -> str:
        """Get confidence as percentage string."""
        return f"{self.confidence * 100:.1f}%"


class SevenSegmentNN(nn.Module):
    """Neural network for seven-segment digit recognition."""
    
    def __init__(self, input_size: int = Config.INPUT_SIZE, 
                 hidden_size: int = Config.HIDDEN_SIZE, 
                 output_size: int = Config.OUTPUT_SIZE):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of input neurons (7 segments)
            hidden_size: Number of hidden neurons
            output_size: Number of output neurons (10 digits)
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class NetworkVisualizer:
    """Handles network visualization in separate windows."""
    
    @staticmethod
    def create_network_visualization(parent: tk.Widget, model: SevenSegmentNN, 
                                   input_tensor: torch.Tensor, h1: torch.Tensor, 
                                   output: torch.Tensor, epoch: int, 
                                   segments: List[int]) -> None:
        """Create a visualization window showing network activations."""
        win = tk.Toplevel(parent)
        
        # Window positioning
        width, height = 800, 600
        screen_width = win.winfo_screenwidth()
        x = screen_width - width - 50
        y = 50
        win.geometry(f"{width}x{height}+{x}+{y}")
        
        seg_str = ",".join(str(s) for s in segments)
        win.title(f"Neural Network Activation - Epoch {epoch} - Segments({seg_str})")
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6))
        layers = [Config.INPUT_SIZE, Config.HIDDEN_SIZE, Config.OUTPUT_SIZE]
        positions = []
        
        # Calculate neuron positions
        for i, n in enumerate(layers):
            x_pos = i * 3
            y_positions = [(j + 1) * (6 / (n + 1)) for j in range(n)]
            positions.append([(x_pos, y) for y in y_positions])
        
        # Draw connections with weights
        NetworkVisualizer._draw_connections(ax, positions, model)
        
        # Draw neurons with activations
        probs = F.softmax(output, dim=0)
        NetworkVisualizer._draw_neurons(ax, positions, input_tensor, h1, probs)
        
        ax.set_xlim(-1, 7)
        ax.set_ylim(0, 6.5)
        ax.axis('off')
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack()
        
        # Add MSE information
        NetworkVisualizer._add_mse_info(win, probs)
        
        canvas.draw()
    
    @staticmethod
    def _draw_connections(ax, positions: List[List[Tuple[float, float]]], 
                         model: SevenSegmentNN) -> None:
        """Draw network connections with weight visualization."""
        weights1 = model.fc1.weight.detach().numpy()
        weights2 = model.fc2.weight.detach().numpy()
        
        # Input to hidden connections
        for i, (x0, y0) in enumerate(positions[0]):
            for j, (x1, y1) in enumerate(positions[1]):
                weight = weights1[j][i]
                color = Config.POSITIVE_WEIGHT_COLOR if weight > 0 else Config.NEGATIVE_WEIGHT_COLOR
                width = max(0.1, abs(weight) * 1.5)
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=width, alpha=0.7)
        
        # Hidden to output connections
        for i, (x0, y0) in enumerate(positions[1]):
            for j, (x1, y1) in enumerate(positions[2]):
                weight = weights2[j][i]
                color = Config.POSITIVE_WEIGHT_COLOR if weight > 0 else Config.NEGATIVE_WEIGHT_COLOR
                width = max(0.1, abs(weight) * 1.5)
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=width, alpha=0.7)
    
    @staticmethod
    def _draw_neurons(ax, positions: List[List[Tuple[float, float]]], 
                     input_tensor: torch.Tensor, h1: torch.Tensor, 
                     probs: torch.Tensor) -> None:
        """Draw neurons with activation values."""
        activations = [input_tensor, h1, probs]
        
        for layer_idx, layer_pos in enumerate(positions):
            for neuron_idx, (x, y) in enumerate(layer_pos):
                activation = activations[layer_idx][neuron_idx].item()
                
                color = Config.NEURON_ACTIVE_COLOR if activation > 0.5 else Config.NEURON_INACTIVE_COLOR
                circle = patches.Circle((x, y), 0.2, color=color, ec='black', linewidth=1)
                ax.add_patch(circle)
                
                ax.text(x, y, f"{activation:.2f}", ha='center', va='center', 
                       fontsize=8, fontweight='bold')
    
    @staticmethod
    def _add_mse_info(window: tk.Toplevel, probs: torch.Tensor) -> None:
        """Add MSE information to the visualization window."""
        predicted_class = torch.argmax(probs).item()
        target_one_hot = torch.zeros_like(probs)
        target_one_hot[predicted_class] = 1.0
        mse = torch.mean((probs - target_one_hot) ** 2).item()
        
        info_frame = ttk.Frame(window)
        info_frame.pack(pady=10)
        
        mse_label = ttk.Label(info_frame, text=f"MSE: {mse:.6f}", 
                             font=("Arial", 12), foreground="blue")
        mse_label.pack()


class WeightAnalyzer:
    """Handles weight change analysis and visualization."""
    
    @staticmethod
    def show_weight_changes(parent: tk.Widget, model: SevenSegmentNN, 
                           criterion, optimizer, X: torch.Tensor, y: torch.Tensor,
                           neuron_filter: Optional[int] = None) -> None:
        """Show weight changes after one training step."""
        try:
            # Save weights before training
            old_weights = model.fc1.weight.clone().detach()
            
            # Perform one training step
            output = model(X)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate weight changes
            new_weights = model.fc1.weight.clone().detach()
            delta = new_weights - old_weights
            
            # Create visualization window
            WeightAnalyzer._create_weight_change_window(parent, delta, neuron_filter)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze weight changes: {str(e)}")
    
    @staticmethod
    def _create_weight_change_window(parent: tk.Widget, delta: torch.Tensor, 
                                   neuron_filter: Optional[int]) -> None:
        """Create window showing weight changes."""
        win = tk.Toplevel(parent)
        
        if neuron_filter is not None:
            title = f"Weight Changes for Input Neuron {neuron_filter + 1}"
            changes = delta[:, neuron_filter].numpy()
            labels = [f"I{neuron_filter + 1}→H{j + 1}" for j in range(len(changes))]
            fig_size = (10, 5)
        else:
            title = "Weight Changes (All Connections)"
            changes = delta.T.flatten().numpy()
            labels = [f"I{i + 1}→H{j + 1}" for i in range(delta.size(1)) 
                     for j in range(delta.size(0))]
            fig_size = (max(14, delta.numel() / 8), 6)
        
        win.title(title)
        
        # Create plot
        fig, ax = plt.subplots(figsize=fig_size)
        colors = [Config.POSITIVE_WEIGHT_COLOR if v >= 0 else Config.NEGATIVE_WEIGHT_COLOR 
                 for v in changes]
        
        bars = ax.bar(range(len(changes)), changes, color=colors, alpha=0.7)
        ax.set_title("Weight Changes After One Training Epoch")
        ax.set_xlabel("Connection")
        ax.set_ylabel("Δ Weight")
        ax.set_xticks(range(len(changes)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, changes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}', ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=6)
        
        fig.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack()
        canvas.draw()


class SevenSegmentApp:
    """
    Main application class for the Seven-Segment Neural Network Trainer.
    
    This application provides a GUI for training and testing a neural network
    that recognizes seven-segment display patterns for digits 0-9.
    """
    
    def __init__(self, root: tk.Tk) -> None:
        """
        Initialize the application.
        
        Args:
            root: The main tkinter window
        """
        self.root = root
        self._setup_model()
        self._setup_variables()
        self._apply_styling()
        self._create_widgets()
        self._initialize_display()
    
    def _setup_model(self) -> None:
        """Initialize the neural network and training components."""
        self.model = SevenSegmentNN()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.X, self.y = TrainingData.get_training_tensors()
    
    def _setup_variables(self) -> None:
        """Initialize application variables."""
        self.current_epoch = 0
        self.segment_buttons: List[ttk.Checkbutton] = []
        self.segment_state = [0] * Config.INPUT_SIZE
        self.segment_shapes: List[int] = []
        self.loss_values: List[float] = []
        self.probability_values = [0.0] * Config.OUTPUT_SIZE
    
    def _apply_styling(self) -> None:
        """Apply modern styling to the application."""
        self.root.title(Config.WINDOW_TITLE)
        self.root.configure(bg='#f0f0f0')
        
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 10, 'bold'))
        style.configure('Action.TButton', font=('Arial', 9, 'bold'))
    
    def _create_widgets(self) -> None:
        """Create and arrange all GUI widgets."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        self._create_control_panel(main_frame)
        self._create_segment_display(main_frame)
        self._create_prediction_panel(main_frame)
        self._create_training_panel(main_frame)
        self._create_analysis_panel(main_frame)
        self._create_visualization_panel(main_frame)
    
    def _create_control_panel(self, parent: ttk.Widget) -> None:
        """Create the segment control buttons."""
        control_frame = ttk.LabelFrame(parent, text="Segment Controls", padding="10")
        control_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(control_frame, text="Click to toggle segments:", 
                 style='Subtitle.TLabel').pack(anchor="w")
        
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill="x", pady=(5, 0))
        
        for i in range(Config.INPUT_SIZE):
            var = tk.IntVar()
            btn = ttk.Checkbutton(
                button_frame,
                text=f"Segment {i + 1}",
                variable=var,
                command=self._on_segment_changed
            )
            btn.var = var
            btn.grid(row=i // 4, column=i % 4, padx=5, pady=2, sticky="w")
            self.segment_buttons.append(btn)
    
    def _create_segment_display(self, parent: ttk.Widget) -> None:
        """Create the seven-segment display visualization."""
        display_frame = ttk.LabelFrame(parent, text="Seven-Segment Display", padding="10")
        display_frame.pack(fill="x", pady=(0, 10))
        
        self.segment_canvas = tk.Canvas(
            display_frame,
            width=Config.CANVAS_WIDTH,
            height=Config.CANVAS_HEIGHT,
            bg=Config.CANVAS_BG_COLOR,
            relief="sunken",
            borderwidth=2
        )
        self.segment_canvas.pack()
    
    def _create_prediction_panel(self, parent: ttk.Widget) -> None:
        """Create the prediction display panel."""
        pred_frame = ttk.LabelFrame(parent, text="Prediction Results", padding="10")
        pred_frame.pack(fill="x", pady=(0, 10))
        
        self.prediction_label = ttk.Label(pred_frame, text="Recognized Digit: -", 
                                        style='Subtitle.TLabel')
        self.prediction_label.pack(anchor="w")
        
        self.probabilities_label = ttk.Label(pred_frame, text="Probabilities: -")
        self.probabilities_label.pack(anchor="w")
    
    def _create_training_panel(self, parent: ttk.Widget) -> None:
        """Create the training control panel."""
        training_frame = ttk.LabelFrame(parent, text="Training Controls", padding="10")
        training_frame.pack(fill="x", pady=(0, 10))
        
        button_frame = ttk.Frame(training_frame)
        button_frame.pack(fill="x")
        
        self.train_button = ttk.Button(
            button_frame,
            text="Train Network",
            command=self._train_model_with_progress,
            style='Action.TButton'
        )
        self.train_button.pack(side="left", padx=(0, 10))
        
        self.predict_button = ttk.Button(
            button_frame,
            text="Predict Digit",
            command=self._predict_digit,
            style='Action.TButton'
        )
        self.predict_button.pack(side="left")
    
    def _create_analysis_panel(self, parent: ttk.Widget) -> None:
        """Create the weight analysis panel."""
        analysis_frame = ttk.LabelFrame(parent, text="Weight Analysis", padding="10")
        analysis_frame.pack(fill="x", pady=(0, 10))
        
        input_frame = ttk.Frame(analysis_frame)
        input_frame.pack(fill="x")
        
        ttk.Label(input_frame, text="Input Neuron (1-7, optional):").pack(side="left")
        
        self.neuron_entry = ttk.Entry(input_frame, width=10)
        self.neuron_entry.pack(side="left", padx=(5, 10))
        
        self.weight_button = ttk.Button(
            input_frame,
            text="Show Weight Changes",
            command=self._show_weight_changes,
            style='Action.TButton'
        )
        self.weight_button.pack(side="left")
    
    def _create_visualization_panel(self, parent: ttk.Widget) -> None:
        """Create the plotting panels."""
        viz_frame = ttk.LabelFrame(parent, text="Training Visualization", padding="10")
        viz_frame.pack(fill="both", expand=True)
        
        # Main plots
        self.fig, (self.ax_loss, self.ax_probs) = plt.subplots(2, 1, figsize=(6, 6))
        self.fig.subplots_adjust(hspace=0.4)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
        
        # Recent loss plot
        self.fig2, self.ax2 = plt.subplots(figsize=(6, 3))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=viz_frame)
        self.canvas2.get_tk_widget().pack(side="right", fill="both", expand=True)
    
    def _initialize_display(self) -> None:
        """Initialize the seven-segment display."""
        self.segment_shapes = self._draw_seven_segment()
        self._update_segment_display()
        self._update_plots()
    
    def _on_segment_changed(self) -> None:
        """Handle segment button state changes."""
        self._update_segment_state()
        self._update_segment_display()
    
    def _update_segment_state(self) -> None:
        """Update internal segment state from UI."""
        for i, btn in enumerate(self.segment_buttons):
            self.segment_state[i] = btn.var.get()
    
    def _predict_digit(self) -> None:
        """Perform digit prediction and update UI."""
        try:
            self._update_segment_state()
            input_tensor = torch.tensor(self.segment_state, dtype=torch.float32)
            
            with torch.no_grad():
                h1 = torch.relu(self.model.fc1(input_tensor))
                output = self.model.fc2(h1)
                probabilities = torch.softmax(output, dim=0)
                prediction = torch.argmax(probabilities).item()
            
            # Update UI
            self.prediction_label.config(text=f"Recognized Digit: {prediction}")
            prob_str = ", ".join(f"{i}:{p:.3f}" for i, p in enumerate(probabilities))
            self.probabilities_label.config(text=f"Probabilities: {prob_str}")
            
            # Update visualization
            self.probability_values = probabilities.tolist()
            self._update_plots()
            
            # Show network visualization
            NetworkVisualizer.create_network_visualization(
                self.root, self.model, input_tensor, h1, output,
                self.current_epoch, self.segment_state
            )
            
        except Exception as e:
            messagebox.showerror("Prediction Error", 
                               f"An error occurred during prediction: {str(e)}")
    
    def _train_model_with_progress(self) -> None:
        """Train the model with progress indication."""
        try:
            # Disable train button during training
            self.train_button.config(state="disabled")
            
            # Create progress window
            progress_window_data = self._create_progress_window()
            progress_window, progress_var, status_label = progress_window_data
            
            def train_batch(epoch_start: int) -> None:
                if epoch_start >= Config.EPOCHS_PER_BATCH:
                    progress_window.destroy()  # Close progress window
                    self.train_button.config(state="normal")  # Re-enable button
                    messagebox.showinfo("Training Complete", 
                                    "Neural network training finished successfully!")
                    return
                
                # Train for a small batch
                batch_size = min(10, Config.EPOCHS_PER_BATCH - epoch_start)
                for _ in range(batch_size):
                    output = self.model(self.X)
                    loss = self.criterion(output, self.y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    self.loss_values.append(loss.item())
                    self.current_epoch += 1
                
                # Update progress
                progress_var.set(epoch_start + batch_size)
                status_label.config(text=f"Epoch {epoch_start + batch_size}/{Config.EPOCHS_PER_BATCH}")
                
                # Update plots periodically
                if (epoch_start + batch_size) % 20 == 0:
                    self._update_plots()
                
                # Schedule next batch
                self.root.after(50, lambda: train_batch(epoch_start + batch_size))
            
            train_batch(0)
            
        except Exception as e:
            self.train_button.config(state="normal")
            messagebox.showerror("Training Error", f"Training failed: {str(e)}")

    def _create_progress_window(self) -> Tuple[tk.Toplevel, tk.DoubleVar, ttk.Label]:
        """Create and return progress window components."""
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Training Progress")
        progress_window.geometry("400x120")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Center the window
        progress_window.update_idletasks()
        x = (progress_window.winfo_screenwidth() // 2) - (400 // 2)
        y = (progress_window.winfo_screenheight() // 2) - (120 // 2)
        progress_window.geometry(f"400x120+{x}+{y}")
        
        frame = ttk.Frame(progress_window, padding="20")
        frame.pack(fill="both", expand=True)
        
        ttk.Label(frame, text="Training Neural Network...", 
                style='Subtitle.TLabel').pack()
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            frame,
            variable=progress_var,
            maximum=Config.EPOCHS_PER_BATCH,
            length=300,
            mode='determinate'
        )
        progress_bar.pack(pady=10)
        
        status_label = ttk.Label(frame, text="Initializing training...")
        status_label.pack()
        
        return progress_window, progress_var, status_label
    
    def _show_weight_changes(self) -> None:
        """Show weight change analysis."""
        neuron_filter = self._validate_neuron_input()
        if neuron_filter is False:  # Validation failed
            return
        
        WeightAnalyzer.show_weight_changes(
            self.root, self.model, self.criterion, self.optimizer,
            self.X, self.y, neuron_filter
        )
    
    def _validate_neuron_input(self) -> Optional[int]:
        """Validate neuron index input."""
        try:
            value = self.neuron_entry.get().strip()
            if not value:
                return None
            
            index = int(value) - 1  # Convert to 0-based indexing
            if not 0 <= index < Config.INPUT_SIZE:
                raise ValueError("Index out of range")
            return index
            
        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                f"Please enter a number between 1 and {Config.INPUT_SIZE}, "
                "or leave empty for all neurons."
            )
            return False
    
    def _update_plots(self) -> None:
        """Update all visualization plots."""
        # Loss plot
        self.ax_loss.clear()
        if self.loss_values:
            self.ax_loss.plot(self.loss_values, label="Training Loss", color='blue')
            self.ax_loss.set_title("Training Progress")
            self.ax_loss.set_xlabel("Epoch")
            self.ax_loss.set_ylabel("Loss")
            self.ax_loss.legend()
            self.ax_loss.grid(True, alpha=0.3)
        
        # Probability plot
        self.ax_probs.clear()
        bars = self.ax_probs.bar(range(Config.OUTPUT_SIZE), self.probability_values, 
                                color='skyblue', alpha=0.7)
        self.ax_probs.set_title("Class Probabilities")
        self.ax_probs.set_xlabel("Digit")
        self.ax_probs.set_ylabel("Probability")
        self.ax_probs.set_ylim(0, 1)
        self.ax_probs.set_xticks(range(Config.OUTPUT_SIZE))
        self.ax_probs.grid(True, alpha=0.3)
        
        # Add probability values on bars
        for bar, prob in zip(bars, self.probability_values):
            height = bar.get_height()
            if height > 0.01:  # Only show labels for visible bars
                self.ax_probs.text(bar.get_x() + bar.get_width()/2., height,
                                 f'{prob:.2f}', ha='center', va='bottom', fontsize=8)
        
        self.canvas.draw()
        
        # Recent loss plot
        self.ax2.clear()
        if len(self.loss_values) > 0:
            recent_losses = self.loss_values[-200:] if len(self.loss_values) > 200 else self.loss_values
            start_epoch = self.current_epoch - len(recent_losses)
            epochs = range(start_epoch, self.current_epoch)
            
            self.ax2.plot(epochs, recent_losses, label="Recent Loss", color='orange')
            self.ax2.set_title("Training Progress (Last 200 Epochs)")
            self.ax2.set_xlabel("Epoch")
            self.ax2.set_ylabel("Loss")
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3)
        
        self.canvas2.draw()
    
    def _draw_seven_segment(self) -> List[int]:
        """Draw the seven-segment display and return shape IDs."""
        segments = []
        cx = Config.CANVAS_WIDTH // 2
        cy = Config.CANVAS_HEIGHT // 2
        s = Config.SEGMENT_LENGTH
        t = Config.SEGMENT_THICKNESS
        
        # Define segment positions (A-G)
        segment_coords = [
            # Segment A - top horizontal
            (cx - s, cy - 3*s, cx + s, cy - 3*s + t),
            # Segment B - top right vertical
            (cx + s, cy - 3*s + t, cx + s + t, cy - s),
            # Segment C - bottom right vertical
            (cx + s, cy - s + t, cx + s + t, cy + s),
            # Segment D - bottom horizontal
            (cx - s, cy + s, cx + s, cy + s + t),
            # Segment E - bottom left vertical
            (cx - s - t, cy - s + t, cx - s, cy + s),
            # Segment F - top left vertical
            (cx - s - t, cy - 3*s + t, cx - s, cy - s),
            # Segment G - middle horizontal
            (cx - s, cy - s, cx + s, cy - s + t),
        ]
        
        for coords in segment_coords:
            segment_id = self.segment_canvas.create_rectangle(
                *coords, fill=Config.SEGMENT_OFF_COLOR, outline='gray'
            )
            segments.append(segment_id)
        
        return segments
    
    def _update_segment_display(self) -> None:
        """Update the visual representation of the seven-segment display."""
        for i, segment_id in enumerate(self.segment_shapes):
            color = Config.SEGMENT_ON_COLOR if self.segment_state[i] else Config.SEGMENT_OFF_COLOR
            self.segment_canvas.itemconfig(segment_id, fill=color)


def main() -> None:
    """Main application entry point."""
    try:
        root = tk.Tk()
        app = SevenSegmentApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Application Error", f"Failed to start application: {str(e)}")


if __name__ == "__main__":
    main()