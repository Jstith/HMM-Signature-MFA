import json

import numpy as np


def generate_impostor_library(gesture_length=120):
    library = []

    # 1. Pure directional lines (8 directions)
    for direction in range(8):
        for _ in range(5):
            noise = [
                direction
                if np.random.random() > 0.1
                else (direction + np.random.choice([-1, 1])) % 8
                for _ in range(gesture_length)
            ]
            library.append(noise)

    # 2. Simple shapes
    patterns = [
        [2, 2, 4, 4, 6, 6, 0, 0],  # Square
        [3, 3, 0, 0, 5, 5],  # Triangle
        [2, 2, 2, 6, 6, 6],  # Z-shape
        [0, 1, 2, 3, 4, 5, 6, 7],  # Clockwise spiral
        [0, 7, 6, 5, 4, 3, 2, 1],  # Counter-clockwise spiral
    ]

    for pattern in patterns:
        for _ in range(5):
            gesture = []
            segment_len = gesture_length // len(pattern)
            for d in pattern:
                gesture.extend([d] * segment_len)
                # Add transition noise
                if len(gesture) < gesture_length:
                    gesture.append((d + 1) % 8)
            library.append(gesture[:gesture_length])

    # 3. Random walks (catches models that accept noise)
    for _ in range(10):
        gesture = []
        d = np.random.randint(0, 8)
        for _ in range(gesture_length):
            gesture.append(d)
            if np.random.random() < 0.15:
                d = (d + np.random.choice([-1, 0, 1])) % 8
        library.append(gesture)

    return library


if __name__ == "__main__":
    np.random.seed(42)  # Reproducible
    library = generate_impostor_library()
    library = [[int(x) for x in gesture] for gesture in library]
    with open("impostor_library.json", "w") as f:
        json.dump(library, f)

    print(f"Generated {len(library)} impostor gestures")
