import json
import csv
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Union, Dict


@dataclass
class Range:
    """
    Represents a numeric range for a named parameter with a probability distribution.

    Attributes:
        min (float): Lower bound or mean, depending on the distribution type.
        max (float): Upper bound or standard deviation, depending on the distribution type.
        dist (str): Distribution type: 'uniform', 'normal', or 'lognormal'. Defaults to 'uniform'.
    """
    min: float
    max: float
    dist: str = "uniform"

    def validate(self):
        """
        Validates the range configuration based on the distribution type.

        Raises:
            ValueError: If the range is improperly configured.
        """
        if self.dist == "uniform" and self.min > self.max:
            raise ValueError(f"Invalid uniform range: min={self.min} > max={self.max}")
        if self.dist in {"normal", "lognormal"} and self.max <= 0:
            raise ValueError(f"Standard deviation must be > 0 for {self.dist} distribution.")

    def sample(self, size: Optional[Union[int, tuple]] = None) -> Union[float, np.ndarray]:
        """
        Generate samples from the defined probability distribution.

        Parameters:
            size (int or tuple, optional): Number of samples or shape of sample array.

        Returns:
            float or np.ndarray: Sampled value(s).
        """
        self.validate()
        if self.dist == "uniform":
            return np.random.uniform(self.min, self.max, size=size)
        elif self.dist == "normal":
            return np.random.normal(loc=self.min, scale=self.max, size=size)
        elif self.dist == "lognormal":
            return np.random.lognormal(mean=self.min, sigma=self.max, size=size)
        else:
            raise ValueError(f"Unsupported distribution type: {self.dist}")


class RangeDict:
    """
    Dictionary-like container for named probability ranges.

    Provides both dictionary-style and attribute-style access to `Range` instances.

    Example:
        ranges.activation.min
        ranges["activation"].sample(10)
    """

    def __init__(self):
        self._data = {}

    def add(self, name: str, min_val: float, max_val: float, dist: str = "uniform"):
        """
        Add a new named range.

        Parameters:
            name (str): Name of the parameter.
            min_val (float): Minimum value or mean.
            max_val (float): Maximum value or standard deviation.
            dist (str): Distribution type. Defaults to 'uniform'.
        """
        rng = Range(min=min_val, max=max_val, dist=dist)
        rng.validate()
        self._data[name] = rng
        setattr(self, name, rng)

    def __getitem__(self, key): return self._data[key]
    def __contains__(self, key): return key in self._data
    def keys(self): return self._data.keys()
    def items(self): return self._data.items()
    def values(self): return self._data.values()
    def __repr__(self): return f"RangeDict({self._data})"

    def to_dict(self) -> Dict[str, dict]:
        """
        Convert the internal data to a plain dictionary.

        Returns:
            dict: Dictionary with keys and serializable Range definitions.
        """
        return {k: asdict(v) for k, v in self._data.items()}


class Probability:
    """
    General-purpose class for managing and sampling from parameter ranges.

    Supports:
        - Uniform, normal, and lognormal distributions
        - Loading from JSON, CSV, or Python dict
        - Dot-style and dict-style access
        - Export to JSON and CSV
    """

    def __init__(self):
        self.values = RangeDict()

    def from_dict(self, data: Dict[str, dict]):
        """
        Load probability ranges from a Python dictionary.

        Format:
            {
                "param_name": {"min": 0.1, "max": 0.9, "dist": "uniform"},
                ...
            }

        Parameters:
            data (dict): Dictionary with named parameter configurations.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        for name, entry in data.items():
            if 'min' not in entry or 'max' not in entry:
                raise ValueError(f"Missing 'min' or 'max' in entry for '{name}'")
            self.values.add(
                name=name,
                min_val=entry['min'],
                max_val=entry['max'],
                dist=entry.get('dist', 'uniform')
            )

    def from_json(self, filename: str):
        """
        Load probability ranges from a JSON file and store them in `self.values`.

        The JSON file must contain a dictionary where each key is the name of a probability
        parameter, and the value is a dictionary with at least:
            - min (float): Minimum value or mean (for normal/lognormal)
            - max (float): Maximum value or std dev (for normal/lognormal)
            - dist (str, optional): Distribution type ('uniform', 'normal', 'lognormal').
            Defaults to 'uniform' if not provided.

        Example JSON:
            {
                "activation": {"min": 0.5, "max": 0.65},
                "clearance": {"min": 1.3, "max": 1.5, "dist": "normal"}
            }

        Parameters:
            filename (str): Path to the JSON file.

        Populates:
            self.values (RangeDict): A dictionary-like object mapping each parameter name
            to a validated Range object that supports sampling and access via dot/dict notation.

        Raises:
            FileNotFoundError: If the file does not exist or cannot be opened.
            json.JSONDecodeError: If the file content is not valid JSON.
            ValueError: If required fields are missing or cannot be parsed.
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            if not isinstance(data, dict):
                raise ValueError("Top-level JSON structure must be a dictionary.")

            for name, entry in data.items():
                if not isinstance(entry, dict):
                    raise ValueError(f"Invalid structure for parameter '{name}': must be a dictionary.")
                if 'min' not in entry or 'max' not in entry:
                    raise ValueError(f"Missing 'min' or 'max' for parameter '{name}'.")

                try:
                    min_val = float(entry['min'])
                    max_val = float(entry['max'])
                    dist = entry.get('dist', 'uniform')
                    self.values.add(name=name, min_val=min_val, max_val=max_val, dist=dist)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid values for '{name}': {entry} — {e}")

        except FileNotFoundError as fnf:
            raise FileNotFoundError(f"JSON file not found: {filename}") from fnf
        except json.JSONDecodeError as jde:
            raise json.JSONDecodeError(f"Failed to decode JSON in file '{filename}': {jde.msg}", jde.doc, jde.pos)


    def from_csv(self, filename: str):
        """
        Load probability ranges from a CSV file and store them in `self.values`.

        The CSV file must contain a header row with the following columns:
            - name (str): Identifier for each probability (e.g., 'activation').
            - min (float): Minimum value or mean (for normal/lognormal).
            - max (float): Maximum value or std dev (for normal/lognormal).
            - dist (str, optional): Distribution type ('uniform', 'normal', 'lognormal').
            Defaults to 'uniform' if not provided.

        Parameters:
            filename (str): Path to the input CSV file.

        Populates:
            self.values (RangeDict): A dictionary-like object mapping each parameter name
            to a validated Range object that supports sampling and access via dot/dict notation.

        Raises:
            FileNotFoundError: If the file is not found or cannot be opened.
            ValueError: If the CSV is missing required columns or contains invalid data.
            csv.Error: If the file format is malformed and cannot be parsed.
        """
        required_columns = {'name', 'min', 'max'}

        try:
            with open(filename, newline='') as f:
                reader = csv.DictReader(f)

                if not reader.fieldnames:
                    raise ValueError("CSV file is empty or missing a header row.")

                missing = required_columns - set(reader.fieldnames)
                if missing:
                    raise ValueError(f"CSV file is missing required columns: {missing}")

                for i, row in enumerate(reader, start=2):  # Line 2 = first data row
                    try:
                        name = row['name'].strip()
                        min_val = float(row['min'])
                        max_val = float(row['max'])
                        dist = row.get('dist', 'uniform')
                        self.values.add(name=name, min_val=min_val, max_val=max_val, dist=dist)
                    except (ValueError, TypeError) as e:
                        raise ValueError(f"Error parsing row {i}: {row} — {e}")

        except FileNotFoundError as fnf:
            raise FileNotFoundError(f"CSV file not found: {filename}") from fnf
        except csv.Error as ce:
            raise csv.Error(f"Malformed CSV file: {ce}") from ce
        

    def sample(self, name: str, size: Optional[Union[int, tuple]] = None) -> Union[float, np.ndarray]:
        """
        Sample from the distribution for a specific parameter.

        Parameters:
            name (str): Parameter name to sample.
            size (int or tuple, optional): Shape of the sampled output.

        Returns:
            float or np.ndarray: Sampled value(s).

        Raises:
            KeyError: If the parameter is not found.
        """
        if name not in self.values:
            raise KeyError(f"Parameter '{name}' not found.")
        return self.values[name].sample(size=size)

    def to_json(self, filename: str):
        """
        Export current parameter ranges to a JSON file.

        Parameters:
            filename (str): Output JSON file path.
        """
        with open(filename, 'w') as f:
            json.dump(self.values.to_dict(), f, indent=2)

    def to_csv(self, filename: str):
        """
        Export current parameter ranges to a CSV file.

        Parameters:
            filename (str): Output CSV file path.
        """
        fieldnames = ['name', 'min', 'max', 'dist']
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, rng in self.values.items():
                writer.writerow({
                    'name': name,
                    'min': rng.min,
                    'max': rng.max,
                    'dist': rng.dist
                })
