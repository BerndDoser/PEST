#!/usr/bin/env python3
"""Example usage of the IllustrisExtractor class."""

from pest import IllustrisExtractor


def main():
    """Demonstrate IllustrisExtractor usage."""

    print("IllustrisExtractor Example")
    print("=" * 30)

    # Configuration matching the config.yaml structure
    extractor_config = {
        "simulation_path": "/home/doserbd/data/illustris/Illustris-3",
        "simulation": "TNG50",
        "snapshot": 99,
        "objects": "centrals",
        "component": [
            {
                "name": "stars",
                "fields": ["masses", "positions", "velocities", "ages", "metallicities"],
                "selector": {"type": "stellar mass", "min": 5.0e10, "max": 5.2e10},
            },
            {"name": "gas", "fields": ["masses", "positions"]},
            {"name": "dark_matter", "fields": ["masses", "positions"]},
        ],
    }

    # Create the extractor
    print("Creating IllustrisExtractor...")
    extractor = IllustrisExtractor(**extractor_config)
    print(f"Extractor created: {extractor}")

    # Show available fields for each component
    print("\nAvailable fields by component:")
    for component in ["stars", "gas", "dark_matter", "black_holes"]:
        fields = extractor.get_available_fields(component)
        print(f"  {component}: {fields}")

    # Validate configuration
    print(f"\nConfiguration valid: {extractor.validate_configuration()}")

    # Extract data
    data = extractor.extract()
    print(f"Extracted data keys: {list(data.keys())}")


if __name__ == "__main__":
    main()
