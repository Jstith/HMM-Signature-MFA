"""
Simple user management for gesture authentication system.
In production, replace with proper database.
"""

import json
import os
from hashlib import sha256

USERS_FILE = "data/users.json"


def hash_password(password):
    """Simple password hashing (use bcrypt in production)"""
    return sha256(password.encode()).hexdigest()


def load_users():
    """Load users from JSON file"""
    if not os.path.exists(USERS_FILE):
        os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
        return {}

    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(users):
    """Save users to JSON file"""
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def create_user(username, password):
    """Create a new user"""
    users = load_users()

    if username in users:
        return False, "Username already exists"

    users[username] = {
        "password_hash": hash_password(password),
        "mfa_trained": False,
        "model_path": f"models/{username}_model.pkl",
    }

    save_users(users)
    return True, "User created successfully"


def verify_credentials(username, password):
    """Verify username and password"""
    users = load_users()

    if username not in users:
        return False

    return users[username]["password_hash"] == hash_password(password)


def user_exists(username):
    """Check if user exists"""
    users = load_users()
    return username in users


def get_user(username):
    """Get user data"""
    users = load_users()
    return users.get(username)


def mark_mfa_trained(username):
    """Mark user as having completed MFA training"""
    users = load_users()
    if username in users:
        users[username]["mfa_trained"] = True
        save_users(users)


def is_mfa_trained(username):
    """Check if user has completed MFA training"""
    users = load_users()
    if username in users:
        return users[username].get("mfa_trained", False)
    return False
