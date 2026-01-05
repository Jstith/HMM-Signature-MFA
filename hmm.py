import json
import math
from copy import deepcopy

import numpy as np
from config import (
    HMM_N_FEATURES,
    HMM_RANDOM_STATE,
    HMM_STATES,
    HMM_TRAINING_ITERATIONS,
    SAMPLE_DISTANCE,
)
from hmmlearn import hmm


def quantize_gesture(points, sample_distance=SAMPLE_DISTANCE):
    """
    Convert raw gesture points to quantized directional sequence.

    Args:
        points: List of gesture points with x, y coordinates
        sample_distance: Distance between samples in pixels

    Returns:
        List of direction integers (0-7 representing 8 compass directions)
    """
    sample_rate = int(sample_distance / SAMPLE_DISTANCE)
    sampled = []
    for i in range(0, len(points), sample_rate):
        sampled.append(points[i])

    print("After sampling, user data looks like:")
    print(sampled)

    # Convert to directions (0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE)
    directions = []
    for i in range(len(sampled) - 1):
        dx = sampled[i + 1]["x"] - sampled[i]["x"]
        dy = sampled[i + 1]["y"] - sampled[i]["y"]
        angle = math.atan2(dy, dx)
        direction = int((angle + math.pi) / (math.pi / 4)) % 8
        directions.append(direction)

    return directions


def verify_gesture(gesture_points, model, threshold):
    """
    Verify a single gesture against a trained model.

    Args:
        gesture_points: Raw gesture points from frontend
        model: Trained HMM model
        threshold: Acceptance threshold

    Returns:
        Normalized log probability score
    """
    print(f"Verifying gesture with {len(gesture_points)} points")

    # Quantize the gesture
    direction_array = quantize_gesture(gesture_points)
    print(f"Quantized to {len(direction_array)} directions")

    # Use test_hmm for consistent scoring
    normalized_log_prob = test_hmm([direction_array], model)

    print(f"Gesture scored {normalized_log_prob:.3f} against threshold {threshold:.3f}")
    return normalized_log_prob


## Separation between web app and CLI testing


def build_hmm(train_data, config):
    observations = np.concatenate(train_data)
    observation_lengths = [len(seq) for seq in train_data]
    observations_column_vector = observations.reshape(-1, 1).astype(int)

    model = hmm.CategoricalHMM(
        n_components=config["HMM_STATES"],
        n_features=config["HMM_N_FEATURES"],
        n_iter=config["HMM_TRAINING_ITERATIONS"],
        random_state=config["HMM_RANDOM_STATE"],
    )

    model.fit(observations_column_vector, observation_lengths)

    # Smoothing to prevent zero probability
    eps = 1e-3
    model.emissionprob_ = (model.emissionprob_ + eps) / (model.emissionprob_ + eps).sum(
        axis=1, keepdims=True
    )
    model.transmat_ = (model.transmat_ + eps) / (model.transmat_ + eps).sum(
        axis=1, keepdims=True
    )
    model.startprob_ = (model.startprob_ + eps) / (model.startprob_ + eps).sum()

    return model


def test_hmm(test_data, model):
    observations = np.concatenate(test_data)
    observation_lengths = [len(seq) for seq in test_data]
    observations_column_vector = observations.reshape(-1, 1).astype(int)
    test_log_prob = model.score(observations_column_vector, observation_lengths)
    normalized_log_prob = test_log_prob / sum(observation_lengths)
    return normalized_log_prob


def k_folds_test(user_data, config, k=5):
    np_user_data = np.array(user_data, dtype=object)
    indices = np.arange(len(np_user_data))
    np.random.shuffle(indices)
    np_user_data = np_user_data[indices]

    folds = np.array_split(np_user_data, k)
    log_probs = []
    for i in range(k):
        test_fold = folds[i]
        train_fold = np.concatenate([folds[j] for j in range(k) if j != i])
        fold_model = build_hmm(train_fold, config)
        log_probs.append(test_hmm(test_fold, fold_model))

    return {"per_fold": log_probs, "mean": np.mean(log_probs), "std": np.std(log_probs)}


def find_optimal_states(user_data, config, state_range=range(4, 22, 2)):
    results = []
    for n_states in state_range:
        test_config = deepcopy(config)
        test_config["HMM_STATES"] = n_states
        k_folds_probabilities = k_folds_test(user_data, test_config)
        results.append(
            {
                "states": n_states,
                "mean": k_folds_probabilities["mean"],
                "std": k_folds_probabilities["std"],
            }
        )

    stable_results = [result for result in results if result["std"] < 0.1]
    if not stable_results:
        print("Warning: No stable results found in optimal state search!")
        return min(results, key=lambda x: x["std"])

    return max(stable_results, key=lambda x: x["mean"])


def calculate_error_rate(genuine_scores, impostor_scores, threshold):
    far = sum(1 for s in impostor_scores if s >= threshold) / len(impostor_scores)
    frr = sum(1 for s in genuine_scores if s < threshold) / len(genuine_scores)
    return far, frr


def find_eer_threshold(genuine_scores, impostor_scores, steps=1000):
    all_scores = genuine_scores + impostor_scores
    min_score = min(all_scores)
    max_score = max(all_scores)

    best_threshold = None
    best_eer = float("inf")
    best_diff = float("inf")

    for i in range(steps):
        threshold = min_score + (max_score - min_score) * i / steps
        far, frr = calculate_error_rate(genuine_scores, impostor_scores, threshold)
        diff = abs(far - frr)
        eer = (far + frr) / 2

        if diff < best_diff:
            best_diff = diff
            best_eer = eer
            best_threshold = threshold
            best_far = far
            best_frr = frr

    return {
        "threshold": best_threshold,
        "eer": best_eer,
        "far": best_far,
        "frr": best_frr,
    }


def analyze_threshold(user_data, impostor_data, config):
    optimal_states = find_optimal_states(user_data, config)
    test_config = {**config, "HMM_STATES": optimal_states["states"]}
    model = build_hmm(user_data, test_config)

    genuine_scores = []
    for gesture in user_data:
        score = test_hmm([gesture], model)
        genuine_scores.append(score)

    impostor_scores = []
    for gesture in impostor_data:
        score = test_hmm([gesture], model)
        impostor_scores.append(score)

    result = find_eer_threshold(genuine_scores, impostor_scores)

    print(
        f"Genuine:  mean={np.mean(genuine_scores):.3f}, std={np.std(genuine_scores):.3f}"
    )
    print(
        f"Impostor: mean={np.mean(impostor_scores):.3f}, std={np.std(impostor_scores):.3f}"
    )
    print(f"Threshold: {result['threshold']:.3f}")
    print(f"EER: {result['eer'] * 100:.1f}%")
    print(f"FAR: {result['far'] * 100:.1f}%, FRR: {result['frr'] * 100:.1f}%")

    # Include model and config in result
    result["model"] = model
    result["config"] = test_config

    return result


def save_model(model, threshold, config, filepath="models/active_model.pkl"):
    """Save the trained model, threshold, and config to a file."""
    import os
    import pickle

    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    model_data = {"model": model, "threshold": threshold, "config": config}

    with open(filepath, "wb") as f:
        pickle.dump(model_data, f)

    print(f"Model saved to {filepath}")


def load_model(filepath="models/active_model.pkl"):
    """Load a trained model from file."""
    import pickle

    with open(filepath, "rb") as f:
        model_data = pickle.load(f)

    return model_data


if __name__ == "__main__":
    from helper.train_data import sample_1, sample_2, sample_3

    test_drawings = [sample_1, sample_2, sample_3]
    # Hidden Markov Model architecture
    hmm_config = {
        "HMM_STATES": 6,
        "HMM_TRAINING_ITERATIONS": 100,
        "HMM_RANDOM_STATE": 1337,
        "HMM_N_FEATURES": 8,
    }

    with open("helper/impostor_library.json", "r") as f:
        impostor_data = json.load(f)

    for i, user_data in enumerate(test_drawings):
        print(f"Tuning gesture {i}...")
        analyze_threshold(user_data, impostor_data, hmm_config)
