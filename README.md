# Seven-Segment Neural Network Trainer

A Python GUI application for training and visualizing a neural network that recognizes seven-segment display patterns for digits 0-9. This project demonstrates backpropagation and neural network training in an interactive, educational environment.

## Features

- **Interactive Seven-Segment Display**: Click to toggle individual segments and see real-time digit recognition
- **Neural Network Training**: Train a custom neural network with visual progress tracking
- **Weight Visualization**: Analyze how network weights change during training
- **Network Architecture Visualization**: See network activations and connections in real-time
- **Training Progress Monitoring**: Multiple plots showing loss curves and probability distributions
- **Simple Efficient GUI**: Clean, responsive interface built with tkinter and ttk

## Screenshots

The application includes:
- Seven-segment display simulator
- Real-time prediction results with probability scores
- Training progress visualization with loss curves
- Interactive weight change analysis
- Network architecture visualization with neuron activations

## Neural Network Architecture

- **Input Layer**: 7 neurons (one for each segment)
- **Hidden Layer**: 16 neurons with ReLU activation
- **Output Layer**: 10 neurons (digits 0-9) with CrossEntropy loss
- **Optimizer**: Stochastic Gradient Descent (SGD)

## Installation

### Prerequisites

Make sure you have Python 3.7+ installed on your system.

### Required Dependencies

Install the required packages using pip:

```bash
pip install torch torchvision matplotlib tkinter
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

### For Windows Users

If you're using Windows and encounter issues with tkinter, it should be included with most Python installations. If not, reinstall Python with tkinter support.

## Usage

### Running the Application

1. **Basic Version (trainer_v1.py)**:
   ```bash
   python trainer_v1.py
   ```

2. **Advanced Version (trainer_v2.py)**:
   ```bash
   python trainer_v2.py
   ```

### How to Use

1. **Toggle Segments**: Click the segment buttons to create different digit patterns
2. **Train the Network**: Click "Train Network" to train the neural network for 200 epochs
3. **Predict Digits**: Click "Predict Digit" to see what digit the network recognizes
4. **Analyze Weights**: Enter a neuron number (1-7) and click "Show Weight Changes" to see how training affects specific connections
5. **Monitor Progress**: Watch the loss curves and probability distributions update in real-time

### Training Data

The network is trained on the standard seven-segment patterns for digits 0-9:

| Digit | Segments (A,B,C,D,E,F,G) |
|-------|--------------------------|
| 0     | 1,1,1,1,1,1,0           |
| 1     | 0,1,1,0,0,0,0           |
| 2     | 1,1,0,1,1,0,1           |
| 3     | 1,1,1,1,0,0,1           |
| 4     | 0,1,1,0,0,1,1           |
| 5     | 1,0,1,1,0,1,1           |
| 6     | 1,0,1,1,1,1,1           |
| 7     | 1,1,1,0,0,0,0           |
| 8     | 1,1,1,1,1,1,1           |
| 9     | 1,1,1,1,0,1,1           |

## File Structure

```
├── trainer_v1.py          # Basic version with core functionality
├── trainer_v2.py          # Advanced version with improved UI and features
├── README.md              # This file
├── LICENSE                # License information
└── requirements.txt       # Python dependencies 
```

## Key Components

### SevenSegmentNN Class
- Defines the neural network architecture
- 2-layer feedforward network with ReLU activation

### TrainingData Class
- Manages the seven-segment display patterns
- Provides training tensors for the network

### NetworkVisualizer Class
- Creates interactive visualizations of network activations
- Shows connections between neurons with weight thickness
- Displays neuron activation levels

### WeightAnalyzer Class
- Analyzes weight changes during training
- Provides detailed insights into learning dynamics

## Educational Value

This project is excellent for:
- Understanding neural network fundamentals
- Visualizing backpropagation in action
- Learning about gradient descent optimization
- Exploring the relationship between network architecture and performance
- Demonstrating overfitting and generalization concepts

## Technical Details

### Dependencies
- **PyTorch**: Neural network framework
- **Matplotlib**: Plotting and visualization
- **Tkinter/TTK**: GUI framework
- **NumPy**: Numerical computations (via PyTorch)

### Configuration
Key parameters can be modified in the `Config` class (trainer_v2.py):
- Learning rate: 0.1
- Hidden layer size: 16 neurons
- Training epochs per batch: 200
- Canvas dimensions and colors

## Troubleshooting

### Common Issues

1. **ImportError for tkinter**: 
   - On Ubuntu/Debian: `sudo apt-get install python3-tk`
   - On CentOS/RHEL: `sudo yum install tkinter`

2. **PyTorch Installation Issues**:
   - Visit [PyTorch.org](https://pytorch.org/) for platform-specific installation instructions

3. **Matplotlib Backend Issues**:
   - Try setting: `matplotlib.use('TkAgg')` at the start of the script

### Performance Tips

- The network trains quickly due to the small dataset
- Training progress updates every 10-20 epochs for smooth visualization
- Weight change analysis works best with single training steps

## Contributing

Feel free to submit issues, feature requests, or pull requests. Some areas for improvement:
- Additional network architectures
- More visualization options
- Export/import of trained models
- Batch training with different datasets

## License

This project is available under the MIT License. See LICENSE file for details.

## Acknowledgments

This project was created for educational purposes to demonstrate neural network concepts in an interactive and visual way. It's particularly useful for students learning about:
- Backpropagation algorithms
- Neural network training dynamics
- Weight optimization
- Pattern recognition

## Version History

- **v1**: Basic functionality with core training and prediction
- **v2**: Enhanced UI, progress tracking, and advanced visualization features

---

*Built with ❤️ for neural network education*

## Project Origin

This project was born out of an educational challenge: explaining the complex concept of backpropagation to my fellow students at university. With just one night to prepare for a presentation, I needed to create something that would make this fundamental neural network algorithm both understandable and engaging.

The seven-segment display was chosen as the perfect example because:
- **Visual Simplicity**: Everyone understands how seven-segment displays work
- **Clear Input-Output Mapping**: 7 inputs (segments) to 10 outputs (digits) creates an intuitive learning problem
- **Interactive Learning**: Students could toggle segments and immediately see how the network responds
- **Real-time Visualization**: Watching weights change during training makes backpropagation tangible

What started as a late-night coding session to create visual aids for a presentation evolved into a comprehensive educational tool that demonstrates not just backpropagation, but the entire neural network training process. The interactive nature of the application helped my classmates grasp concepts that are often abstract in textbooks.

The project successfully served its original purpose - making backpropagation accessible and understandable in a single presentation. Since then, it has continued to serve as an educational resource for anyone wanting to understand neural networks through hands-on experimentation.
