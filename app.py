import json
import os

import users
from config import (
    HMM_N_FEATURES,
    HMM_RANDOM_STATE,
    HMM_STATES,
    HMM_TRAINING_ITERATIONS,
    MAX_PATH_LENGTH,
    MIN_PATH_LENGTH,
    SAMPLES,
    TRAINING_ITERATIONS,
    get_frontend_config,
)
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from flask_sock import Sock
from hmm import (
    analyze_threshold,
    load_model,
    quantize_gesture,
    save_model,
    verify_gesture,
)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Simple session secret (use env var in production)
sock = Sock(app)

# Store training data per session (in production, use proper session management)
training_sessions = {}


@app.route("/")
def index():
    return redirect(url_for("landing"))


@app.route("/landing")
def landing():
    return render_template("landing.html")


@app.route("/register-page")
def register_page():
    return render_template("register.html")


@app.route("/register", methods=["POST"])
def register():
    """Handle user registration"""
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    success, message = users.create_user(username, password)

    if success:
        # Set session for training
        session["username"] = username
        session["auth_stage"] = "training"
        return jsonify({"message": message}), 200
    else:
        return jsonify({"error": message}), 400


@app.route("/login-page")
def login_page():
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login():
    """Handle user login (step 1)"""
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    if not users.verify_credentials(username, password):
        return jsonify({"error": "Invalid credentials"}), 401

    if not users.is_mfa_trained(username):
        return jsonify({"error": "MFA not set up. Please register."}), 400

    # Set session for verification
    session["username"] = username
    session["auth_stage"] = "verifying"

    return jsonify({"message": "Credentials verified. Please complete MFA."}), 200


@app.route("/train.html")
def train_page():
    # Require active session in training stage
    if "username" not in session or session.get("auth_stage") != "training":
        return redirect(url_for("landing"))
    return render_template("train.html")


@app.route("/api/config")
def get_config():
    """API endpoint to provide frontend configuration"""
    return jsonify(get_frontend_config())


@app.route("/verify.html")
def verify_page():
    # Require active session in verifying stage
    if "username" not in session or session.get("auth_stage") != "verifying":
        return redirect(url_for("landing"))
    return render_template("verify.html")


@sock.route("/ws/train")
def train_websocket(ws):
    """
    WebSocket endpoint for training gesture data
    Receives multiple gesture iterations from the client
    """
    # Get username from Flask session (accessed via request context)
    from flask import copy_current_request_context

    username = session.get("username")
    if not username or session.get("auth_stage") != "training":
        ws.send(
            json.dumps({"success": False, "error": "Not authenticated for training"})
        )
        return

    session_id = id(ws)  # WebSocket session tracking
    training_sessions[session_id] = {"username": username, "gestures": []}

    try:
        while True:
            message = ws.receive()
            if message is None:
                break

            data = json.loads(message)
            iteration = data.get("iteration", 0)
            gesture = data.get("gesture", [])
            path_length = data.get("pathLength", 0)

            # Validate gesture
            if not gesture or len(gesture) != SAMPLES:
                ws.send(
                    json.dumps(
                        {
                            "success": False,
                            "error": f"Gesture must have exactly {SAMPLES} samples (got {len(gesture)})",
                        }
                    )
                )
                continue

            if path_length < MIN_PATH_LENGTH:
                ws.send(
                    json.dumps(
                        {
                            "success": False,
                            "error": f"Path too short ({int(path_length)} pixels, minimum {MIN_PATH_LENGTH})",
                        }
                    )
                )
                continue

            if path_length > MAX_PATH_LENGTH:
                ws.send(
                    json.dumps(
                        {
                            "success": False,
                            "error": f"Path too long ({int(path_length)} pixels, maximum {MAX_PATH_LENGTH})",
                        }
                    )
                )
                continue

            # Store the gesture data
            training_sessions[session_id]["gestures"].append(
                {"iteration": iteration, "gesture": gesture, "pathLength": path_length}
            )

            print(
                f"User {username}: Received iteration {iteration}/{TRAINING_ITERATIONS} with {len(gesture)} samples, {path_length:.0f} pixels"
            )

            # Check if training is complete
            if len(training_sessions[session_id]["gestures"]) >= TRAINING_ITERATIONS:
                print(f"Training complete for session {session_id}. Analyzing...")

                # Quantize user gestures
                quantized_user_data = []
                for train_iteration in training_sessions[session_id]["gestures"]:
                    direction_array = quantize_gesture(train_iteration["gesture"])
                    quantized_user_data.append(direction_array)

                print(f"Quantized {len(quantized_user_data)} gestures")

                # Load impostor library and analyze
                try:
                    with open("helper/impostor_library.json", "r") as f:
                        impostor_data = json.load(f)

                    # Create HMM config
                    hmm_config = {
                        "HMM_STATES": HMM_STATES,
                        "HMM_TRAINING_ITERATIONS": HMM_TRAINING_ITERATIONS,
                        "HMM_RANDOM_STATE": HMM_RANDOM_STATE,
                        "HMM_N_FEATURES": HMM_N_FEATURES,
                    }

                    # Analyze threshold and print statistics
                    print("\n=== Threshold Analysis ===")
                    threshold_result = analyze_threshold(
                        quantized_user_data, impostor_data, hmm_config
                    )
                    print(f"Analysis complete: EER = {threshold_result['eer']:.3f}")
                    print("==========================\n")

                    # Check if EER is acceptable
                    if threshold_result["eer"] >= 0.20:
                        # Model not secure enough
                        ws.send(
                            json.dumps(
                                {
                                    "success": False,
                                    "training_failed": True,
                                    "error": f"Gesture pattern not secure enough (EER: {threshold_result['eer'] * 100:.1f}%). Please try again using straighter, more distinct lines.",
                                    "eer": threshold_result["eer"],
                                }
                            )
                        )
                        print(
                            f"Training rejected: EER {threshold_result['eer']:.3f} >= 0.20"
                        )
                    else:
                        # Model is secure, save it with username
                        user_data = users.get_user(username)
                        model_path = user_data["model_path"]

                        save_model(
                            threshold_result["model"],
                            threshold_result["threshold"],
                            threshold_result["config"],
                            filepath=model_path,
                        )

                        # Mark MFA as trained
                        users.mark_mfa_trained(username)

                        ws.send(
                            json.dumps(
                                {
                                    "success": True,
                                    "training_complete": True,
                                    "message": f"Training successful! Model saved (EER: {threshold_result['eer'] * 100:.1f}%)",
                                    "eer": threshold_result["eer"],
                                    "threshold": threshold_result["threshold"],
                                }
                            )
                        )
                        print(
                            f"Training successful: Model saved with EER {threshold_result['eer']:.3f}"
                        )

                except Exception as e:
                    print(f"Error during threshold analysis: {e}")
                    import traceback

                    traceback.print_exc()
                    ws.send(
                        json.dumps(
                            {
                                "success": False,
                                "error": f"Training analysis failed: {str(e)}",
                            }
                        )
                    )
            else:
                # Send success response for receiving gesture
                ws.send(
                    json.dumps(
                        {
                            "success": True,
                            "message": f"Gesture {iteration} received successfully",
                            "iteration": iteration,
                            "total": len(training_sessions[session_id]["gestures"]),
                        }
                    )
                )

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Cleanup when connection closes
        if session_id in training_sessions:
            print(f"Session {session_id} disconnected")
            del training_sessions[session_id]


@app.route("/verify", methods=["POST"])
def verify():
    """
    Endpoint to receive gesture verification data
    Expected JSON: {
        'gesture': [{'x': float, 'y': float, 'timestamp': int}, ...],
        'pathLength': float
    }
    Requirements:
    - 60-120 samples (taken every 25 pixels)
    - 500-2000 pixels total path length
    """
    try:
        # Check session
        username = session.get("username")
        if not username or session.get("auth_stage") != "verifying":
            return jsonify({"error": "Not authenticated. Please log in."}), 401

        data = request.get_json()
        gesture = data.get("gesture", [])
        path_length = data.get("pathLength", 0)

        if not gesture:
            return jsonify({"error": "No gesture data provided"}), 400

        if len(gesture) != SAMPLES:
            return jsonify(
                {
                    "error": f"Gesture must have exactly {SAMPLES} samples (got {len(gesture)})"
                }
            ), 400

        if path_length < MIN_PATH_LENGTH:
            return jsonify(
                {
                    "error": f"Path too short ({path_length:.0f} pixels, minimum {MIN_PATH_LENGTH})"
                }
            ), 400

        if path_length > MAX_PATH_LENGTH:
            return jsonify(
                {
                    "error": f"Path too long ({path_length:.0f} pixels, maximum {MAX_PATH_LENGTH})"
                }
            ), 400

        print(
            f"User {username}: Received verification gesture: {len(gesture)} samples, {path_length:.0f} pixels"
        )

        # Load the user's saved model
        try:
            user_data = users.get_user(username)
            model_path = user_data["model_path"]
            model_data = load_model(filepath=model_path)
            model = model_data["model"]
            threshold = model_data["threshold"]
        except FileNotFoundError:
            return jsonify(
                {"error": "No trained model found. Please complete training first."}
            ), 400
        except Exception as e:
            return jsonify({"error": f"Failed to load model: {str(e)}"}), 500

        # Verify the gesture
        log_prob = verify_gesture(gesture, model=model, threshold=threshold)
        print(f"Gesture log probability: {log_prob:.3f}, Threshold: {threshold:.3f}")

        # Check if gesture passes threshold
        verified = log_prob >= threshold

        if verified:
            # Mark session as fully authenticated
            session["authenticated"] = True
            print(f"User {username} successfully authenticated!")

        return jsonify(
            {
                "verified": verified,
                "confidence": log_prob,
                "threshold": threshold,
                "samples": len(gesture),
                "pathLength": path_length,
            }
        ), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/success")
def success():
    """Success page after full authentication"""
    if not session.get("authenticated"):
        return redirect(url_for("landing"))

    username = session.get("username", "User")
    return render_template("success.html", username=username)


@app.route("/logout")
def logout():
    """Clear session and logout"""
    session.clear()
    return redirect(url_for("landing"))


if __name__ == "__main__":
    app.run(debug=True)
