from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator
import numpy as np

Activity = Literal[
    "standing", "sitting", "marching", "squats", "jumping_jacks", "waving"
]


class Frame(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    t: float = Field(ge=0, le=86400)
    landmarks: list[list[float]] = Field(min_length=33, max_length=33)

    @model_validator(mode="after")
    def validate_landmarks(self):
        if any(len(p) != 4 for p in self.landmarks):
            raise ValueError("Each landmark must contain x, y, z and visibility.")
        a = np.asarray(self.landmarks, dtype=float)
        if not np.isfinite(a).all() or np.abs(a[:, :3]).max() > 10:
            raise ValueError(
                "Landmark coordinates must be finite and within [-10, 10]."
            )
        if ((a[:, 3] < 0) | (a[:, 3] > 1)).any():
            raise ValueError("Visibility must be in [0, 1].")
        return self


class Sequence(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    frames: list[Frame] = Field(min_length=1, max_length=900)
    aspect: float = Field(default=1, ge=0.2, le=5)

    @model_validator(mode="after")
    def chronological(self):
        ts = [f.t for f in self.frames]
        if any(b <= a for a, b in zip(ts, ts[1:])):
            raise ValueError("Frame timestamps must be strictly increasing.")
        if ts[-1] - ts[0] > 60:
            raise ValueError("A sequence may span at most 60 seconds.")
        return self


class PredictRequest(Sequence):
    engine: Literal["rules", "trained"] = "rules"


class Clip(Sequence):
    subject: str = Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")
    label: Activity
    source: Literal["camera", "video", "synthetic"]
    consent: bool


class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    validation_subjects: list[str] = Field(min_length=1)
    test_subjects: list[str] = Field(min_length=1)
