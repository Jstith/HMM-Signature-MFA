from flask import session, request, url_for
from flask_socketio import disconnect, emit
import math

user_states = {}
sid_to_user = {}
MAX_INK = 800

def register_socket_events(socketio, db, User, train_model, query_user):

    @socketio.on('connect')
    def handle_connect():
        sid = request.sid
        user_id = session.get("_user_id")
        print(f" socket for user {user_id} opened")
        if not user_id:
            disconnect()
            return

        sid_to_user[sid] = user_id
        user_states[sid] = {"ink_left": MAX_INK, "last_x": None, "last_y": None, "last_time": None}
        emit('status_message', {"message": "Draw MFA Pattern."}, to=sid)
        print(f"{sid} (user {user_id}) connected")

        user_states[sid]['tokens'] = 0
        user_states[sid]['token_data'] = []

    @socketio.on('disconnect')
    def handle_disconnect():
        sid = request.sid
        user_states.pop(sid, None)
        sid_to_user.pop(sid, None)
        socketio.emit('disconnect', to=sid)
        print(f"{sid} disconnected")

    @socketio.on('update_counter')
    def handle_counter(tokens):
        entries_remaining = 3 - tokens
        button_disabled = True
        if(entries_remaining == 0):
            button_disabled = False
        socketio.emit('update_content', {'new_html': entries_remaining, 'button_disabled': button_disabled}, to=request.sid)

    @socketio.on('mouse_data')
    def handle_mouse(data):
        sid = request.sid
        state = user_states.get(sid)
        if state is None:
            return

        updated_this_iter = False
        x, y, t = data["x"], data["y"], data["t"]
        last_x, last_y, last_time = state['last_x'], state['last_y'], state['last_time']

        if last_x is not None:
            dx = x - last_x
            dy = y - last_y
            distance = math.hypot(dx, dy)
            velocity = distance / max(1, (t - last_time))

            state.setdefault('data', []).append([dx, dy, velocity])

            state['ink_left'] = max(0, state['ink_left'] - distance)
            percent = (state['ink_left'] / MAX_INK) * 100
            emit('ink_status', {"percent": percent}, to=sid)

            if state['ink_left'] <= 0:
                emit('ink_out', {}, to=sid)
                emit('status_message', {"message": "Ink has run out!"}, to=sid)

                # We've got data to play with.
                # Store in session until we have all 3. THEN,
                # pass to the model.

                user_id = sid_to_user.get(sid)
                user = db.session.get(User, user_id)
                if user:

                    # Verifying MFA logic
                    if user.mfa_enabled:
                        model_b64 = user.mfa_model
                        threshold = user.mfa_threshold
                        mfa_passed = query_user(model_b64, threshold, state['data'])
                        if(mfa_passed):
                            user.mfa_validated = True
                            db.session.commit()
                            print("Passed MFA.")
                            emit('mfa-passed', to=sid)
                        else:
                            emit('mfa-failed', to=sid)

                        disconnect()
                        return

                    # Enabling MFA logic
                    else:

                        tokens = user_states[sid]['tokens']
                        user_states[sid]['token_data'].append(state['data'])
                        user_states[sid]['tokens'] += 1

                        if(tokens >= 2):
                            tokens = user_states[sid]['tokens']
                            emit('reset_canvas', to=sid)
                            handle_counter(tokens)
                            print("Training model")
                            model_b64, threshold = train_model(user_id, user_states[sid]['token_data']) # May need to reformat

                            print(f"Got model with threshold {threshold}")
                            print(len(user_states[sid]['token_data']))

                            if(threshold >= -600 and threshold <= -100):
                                user.mfa_model = model_b64
                                user.mfa_threshold = threshold
                                user.mfa_enabled = True
                                db.session.commit()
                                emit('status_message', {"message": "MFA model created!"}, to=sid)
                            else:

                                emit('status_message', {"message": "Failed to build MFA model, refresh to try again."}, to=sid)
                                emit('update_content', {"new_html": 0, "button_disabled": "true"}, to=sid)
                            disconnect()
                            return

                        else:
                            print("Trying to update counter in HTML")
                            tokens = user_states[sid]['tokens']
                            handle_counter(tokens)
                            # user_states[sid] = {"ink_left": MAX_INK, "last_x": None, "last_y": None, "last_time": None}
                            user_states[sid]['ink_left'] = MAX_INK
                            user_states[sid]['last_x'] = None
                            user_states[sid]['last_y'] = None
                            user_states[sid]['last_time'] = None
                            updated_this_iter = True
                            emit('reset_canvas', to=sid)
                            emit('status_message', {"message": "Draw MFA Pattern."}, to=sid)
        if(not updated_this_iter):
            state['last_x'] = x
            state['last_y'] = y
            state['last_time'] = t
