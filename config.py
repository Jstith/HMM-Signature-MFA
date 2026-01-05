"""
Global configuration for HMM gesture authentication system.
All tunable parameters are centralized here for easy adjustment.
"""

# ============================================================================
# GESTURE CAPTURE PARAMETERS (Frontend & Backend Validation)
# ============================================================================

# Canvas settings
CANVAS_SIZE = 500

# Sampling configuration
SAMPLE_DISTANCE = 15  # Pixels per sample unit from client side
SAMPLES = 120  # Number of samples required

# Path length constraints (in pixels)
MIN_PATH_LENGTH = 500
MAX_PATH_LENGTH = 2000

# Training session settings
TRAINING_ITERATIONS = 15  # Total number of training gestures required

# ============================================================================
# HMM MODEL PARAMETERS
# ============================================================================

# Hidden Markov Model architecture
HMM_STATES = 6  # Number of hidden states in the HMM
HMM_TRAINING_ITERATIONS = 100  # Maximum iterations for HMM training (EM algorithm)
HMM_RANDOM_STATE = 1337  # Random seed for reproducibility

# Feature encoding
HMM_N_FEATURES = 8  # Number of directional features (8 compass directions)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_frontend_config():
    """Returns configuration dictionary for frontend JavaScript."""
    return {
        "CANVAS_SIZE": CANVAS_SIZE,
        "SAMPLE_DISTANCE": SAMPLE_DISTANCE,
        "SAMPLES": SAMPLES,
        "MIN_PATH_LENGTH": MIN_PATH_LENGTH,
        "MAX_PATH_LENGTH": MAX_PATH_LENGTH,
        "TRAINING_ITERATIONS": TRAINING_ITERATIONS,
    }
