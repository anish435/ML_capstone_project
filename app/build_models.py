"""Pre-train track artifacts. Run from repo root: python app/build_models.py"""
from app.core import load_all

if __name__ == "__main__":
    for track, payload in load_all(force=True).items():
        print(f"{track}: saved model artifact ({payload['fit_seconds']:.1f}s)")
