# Extractor Classes

This document describes the new extractor classes for the PEST library.

## Overview

The extractor classes provide a standardized interface for extracting data from different astronomical simulations and surveys. The base `Extractor` class defines the common interface, while specific implementations like `IllustrisExtractor` handle the details of data extraction from particular sources.

## Base Class: Extractor

The `Extractor` class serves as an abstract base class that defines the interface all extractors must implement.

### Methods

- `extract()`: Extract data based on the configured parameters. Returns a dictionary containing extracted data organized by components.
- `get_available_fields(component)`: Get available fields for a given component (e.g., 'stars', 'gas', 'dark_matter').
- `validate_configuration()`: Validate the current configuration. Returns True if valid, False otherwise.

## IllustrisExtractor

The `IllustrisExtractor` is a concrete implementation for extracting data from IllustrisTNG simulation data.

### Features

- Supports extraction from local IllustrisTNG simulation files
- Handles multiple particle types: stars, gas, dark_matter, black_holes
- Configurable field selection per component
- Mass-based selection criteria
- Object type filtering (centrals, satellites, all)

### Configuration

The extractor is configured through initialization parameters:

```python
extractor = IllustrusExtractor(
    simulation_path="/path/to/simulation",
    simulation="TNG50",
    snapshot=99,
    objects="centrals",
    component=[
        {
            "name": "stars",
            "fields": ["masses", "positions", "velocities", "ages", "metallicities"],
            "selector": {
                "type": "stellar mass",
                "min": 5.0e+10,
                "max": 5.2e+10
            }
        },
        {
            "name": "gas",
            "fields": ["masses", "positions"]
        }
    ]
)
```

### Available Fields

#### Stars
- masses, positions, velocities, ages, metallicities, formation_times, initial_masses, gfm_stellar_photometrics

#### Gas
- masses, positions, velocities, densities, temperatures, internal_energies, smoothing_lengths, electron_abundances, neutral_hydrogen_abundances, star_formation_rates

#### Dark Matter
- masses, positions, velocities, particle_ids

#### Black Holes
- masses, positions, velocities, mdot, masses_bh

### Selection Criteria

The `selector` field in component configuration supports the following types:
- "stellar mass": Selection based on stellar mass
- "total mass": Selection based on total subhalo mass
- "gas mass": Selection based on gas mass
- "dark_matter_mass": Selection based on dark matter mass

### Example Usage

See `examples/illustris_extractor_example.py` for a complete example.

### Integration with Configuration Files

The extractor classes are designed to work with YAML configuration files like the one in `tests/config.yaml`. The configuration structure maps directly to the extractor initialization parameters.

```yaml
source:
  class_path: pest.IllustrisExtractor
  init_args:
    simulation_path: /path/to/simulation
    simulation: TNG50
    snapshot: 99
    objects: centrals
    component:
      - name: stars
        fields: [masses, positions, velocities, ages, metallicities]
        selector:
          type: stellar mass
          min: 5.0e+10
          max: 5.2e+10
```

This allows for flexible configuration-driven data extraction workflows.
