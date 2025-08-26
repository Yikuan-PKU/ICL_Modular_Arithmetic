import json
import typing as t
from pathlib import Path

import numpy as np

#######################################################
# json file tools


class JsonProcessor:
    """Class for handling JSON serialization with NumPy type conversion."""

    @staticmethod
    def convert_numpy_types(obj: t.Any, _seen: set[int] | None = None) -> t.Any:
        """Recursively convert NumPy types and custom objects in a nested structure to standard Python types."""
        # Initialize seen objects set for cycle detection
        if _seen is None:
            _seen = set()

        # Check for cycles using object ID
        obj_id = id(obj)
        if obj_id in _seen:
            # Return a safe representation for circular references
            return f"<circular_reference_to_{type(obj).__name__}>"

        # Handle None
        if obj is None:
            return None

        # Early detection of NetworkX objects to prevent any issues
        if hasattr(obj, "__module__") and obj.__module__ and "networkx" in obj.__module__:
            class_name = obj.__class__.__name__
            if "Graph" in class_name:
                return {
                    "type": "networkx_graph",
                    "graph_type": class_name,
                    "nodes": list(obj.nodes()) if hasattr(obj, "nodes") else [],
                    "edges": list(obj.edges()) if hasattr(obj, "edges") else [],
                    "number_of_nodes": obj.number_of_nodes() if hasattr(obj, "number_of_nodes") else 0,
                    "number_of_edges": obj.number_of_edges() if hasattr(obj, "number_of_edges") else 0,
                }
            return f"<networkx_{class_name}>"

        # Handle SearchResult objects
        if hasattr(obj, "__class__") and obj.__class__.__name__ == "SearchResult":
            # Add to seen set before processing
            _seen.add(obj_id)
            try:
                # Convert SearchResult to dict
                result_dict = {
                    "neurons": JsonProcessor.convert_numpy_types(obj.neurons, _seen),
                    "delta_loss": JsonProcessor.convert_numpy_types(obj.delta_loss, _seen),
                }
                if hasattr(obj, "is_target_size"):
                    result_dict["is_target_size"] = obj.is_target_size
                return result_dict
            finally:
                _seen.discard(obj_id)

        # Handle NumPy arrays
        if isinstance(obj, np.ndarray):
            return JsonProcessor.convert_numpy_types(obj.tolist(), _seen)

        # Handle NumPy scalars
        if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(
            obj, (np.integer, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
        ):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.complex128):
            return complex(obj)

        # Handle Path objects
        if isinstance(obj, Path):
            return str(obj)

        # Handle NetworkX objects specifically
        if hasattr(obj, "__class__"):
            class_name = obj.__class__.__name__
            # Handle NetworkX graph objects
            if class_name in ["Graph", "DiGraph", "MultiGraph", "MultiDiGraph"]:
                return {
                    "type": "networkx_graph",
                    "graph_type": class_name,
                    "nodes": list(obj.nodes()),
                    "edges": list(obj.edges(data=True)) if hasattr(obj, "edges") else [],
                    "number_of_nodes": obj.number_of_nodes() if hasattr(obj, "number_of_nodes") else 0,
                    "number_of_edges": obj.number_of_edges() if hasattr(obj, "number_of_edges") else 0,
                }
            # Handle NetworkX views (AdjacencyView, etc.)
            if class_name in ["AdjacencyView", "AtlasView", "NodeView", "EdgeView", "DegreeView"]:
                try:
                    return list(obj) if hasattr(obj, "__iter__") else str(obj)
                except:
                    return f"<{class_name}_object>"

        # Add to seen set before processing complex objects
        _seen.add(obj_id)

        try:
            # Handle dictionaries
            if isinstance(obj, dict):
                return {
                    JsonProcessor.convert_numpy_types(k, _seen): JsonProcessor.convert_numpy_types(v, _seen)
                    for k, v in obj.items()
                }

            # Handle lists and tuples
            if isinstance(obj, (list, tuple)):
                converted = [JsonProcessor.convert_numpy_types(item, _seen) for item in obj]
                return converted if isinstance(obj, list) else tuple(converted)

            # Handle sets by converting to list
            if isinstance(obj, set):
                return [JsonProcessor.convert_numpy_types(item, _seen) for item in obj]

            # Handle objects with a to_dict method
            if hasattr(obj, "to_dict") and callable(obj.to_dict):
                return JsonProcessor.convert_numpy_types(obj.to_dict(), _seen)

            # Handle objects with __dict__ attribute
            if hasattr(obj, "__dict__") and not isinstance(obj, type):
                # Special handling for known problematic classes
                class_name = obj.__class__.__name__

                # Skip NetworkX objects entirely (already handled above)
                if any(nx_type in class_name for nx_type in ["Graph", "View", "Atlas", "Node", "Edge", "Degree"]):
                    return f"<{class_name}_skipped>"

                # Handle statistical validation objects carefully
                if class_name in ["ValidationResult", "StatisticalTest", "StatisticalValidator", "BootstrapEstimator"]:
                    # For these classes, extract only essential attributes
                    safe_dict = {}
                    for key, value in obj.__dict__.items():
                        if not key.startswith("_") and key not in [
                            "graph",
                            "data",
                            "samples",
                            "_seen",
                        ]:  # Skip problematic attributes
                            try:
                                safe_dict[key] = JsonProcessor.convert_numpy_types(value, _seen)
                            except (RecursionError, TypeError, AttributeError, ValueError):
                                # Fallback for problematic attributes
                                if isinstance(value, (int, float, str, bool, type(None))):
                                    safe_dict[key] = value
                                else:
                                    safe_dict[key] = str(type(value).__name__)
                    return safe_dict
                return JsonProcessor.convert_numpy_types(obj.__dict__, _seen)

            # Return the object as is if no conversion is needed
            return obj

        finally:
            # Remove from seen set when done processing this object
            _seen.discard(obj_id)

    @classmethod
    def save_json(cls, data: dict, filepath: Path) -> None:
        """Save a nested dictionary with float values to a file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        converted_data = cls.convert_numpy_types(data)
        with open(filepath, "w") as f:
            json.dump(converted_data, f, indent=2)

    @staticmethod
    def load_json(filepath: Path) -> dict:
        """Load a JSON file into a dictionary."""
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
