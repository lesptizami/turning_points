# Turning Point Analyzer

## Overview
The Turning Point Analyzer is a Python package for analyzing turning points in trajectories and calculating transition probabilities.

## Installation
You can install the Turning Point Analyzer package using pip:
```bash
pip install turning_point_analyzer
```

## Usage
To use the Turning Point Analyzer in your Python code, you can import it as follows:

```pytho
from turning_points import TurningPointAnalyzer
```

Analyzing Turning Points
```python
from turning_points import TurningPointAnalyzer

# Example trajectory data
trajectories = [
    [(0, 0), (1, 1), (2, 2), (3, 1), (4, 0)],  # Example trajectory 1
    [(0, 0), (1, -1), (2, -2), (3, -1), (4, 0)]  # Example trajectory 2
]

# Create a TurningPointAnalyzer instance
analyzer = TurningPointAnalyzer(trajectories)

# Plot transition directions
analyzer.plot_transition()

# Plot frequency distribution of time between turns
analyzer.plot_frequency()

# Get the probability of same turn direction
probability = analyzer.get_proba_same_turn()
print("Probability of same turn direction:", probability)

# Get the results
results = analyzer.get_results()
print("Results:", results)
```

## Testing
To run tests for the Turning Point Analyzer, you can use the following command:

```bash
pytest
```

## Contributing
Contributions are welcome! Please feel free to submit issues or pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citing
If you use this package for your publication, don't hesitate to cite it.
example provided in citations.md


## thanks for the RDP algorythm

The package use the [Ramer–Douglas–Peucker](http://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm) algorithm.
It was obtained from: https://github.com/fhirschmann/rdp