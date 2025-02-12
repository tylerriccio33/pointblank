from __future__ import annotations

import json
from dataclasses import dataclass, field

import narwhals as nw
from narwhals.typing import FrameT
from typing import Any


@dataclass
class DataProfiler:
    data: FrameT | Any
    tbl_name: str | None = None
    data_native: Any = field(init=False)
    profile: dict = field(init=False)

    def __post_init__(self):
        self.data_native = self.data
        self.data = nw.from_native(self.data)
        self.profile = self._generate_profile()

    def _generate_profile(self) -> dict:

        profile = {}

        if self.tbl_name:
            profile["tbl_name"] = self.tbl_name

        profile.update(
            {
                "dimensions": {"rows": self.data.shape[0], "columns": self.data.shape[1]},
                "columns": [],
            }
        )

        for column in self.data.columns:

            col_data = self.data[column]
            native_dtype = str(self.data_native[column].dtype)

            col_profile = {
                "column_name": column,
                "column_type": native_dtype,
                "missing_values": int(col_data.is_null().sum()),
                "unique_values": col_data.n_unique(),
                "sample_data": str(col_data.drop_nulls().head(5).cast(nw.String).to_list()),
                "statistics": {},
            }

            if "int" in str(col_data.dtype).lower() or "float" in str(col_data.dtype).lower():
                col_profile["statistics"]["numerical"] = {
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "std_dev": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "percentiles": {
                        "25th": float(col_data.quantile(0.25, interpolation="nearest")),
                        "50th": float(col_data.quantile(0.50, interpolation="nearest")),
                        "75th": float(col_data.quantile(0.75, interpolation="nearest")),
                    },
                }

            elif (
                "string" in str(col_data.dtype).lower()
                or "categorical" in str(col_data.dtype).lower()
            ):
                mode = col_data.mode()
                col_profile["statistics"]["categorical"] = {
                    "mode": mode[0],
                }

            elif "date" in str(col_data.dtype).lower():

                min_date = str(col_data.min())
                max_date = str(col_data.max())

                col_profile["statistics"]["datetime"] = {
                    "min_date": min_date,
                    "max_date": max_date,
                }

            profile["columns"].append(col_profile)

        return profile

    def get_profile(self) -> dict:
        return self.profile

    def to_json(self) -> str:
        return json.dumps(self.profile, indent=4)

    def save_to_json(self, output_file: str):
        with open(output_file, "w") as f:
            json.dump(self.profile, f, indent=4)
