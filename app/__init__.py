from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect
from flask_socketio import SocketIO
from app.mfa_func.sockets import register_socket_events
from app.mfa_func.ai import train_model, query_user
import os

db = SQLAlchemy()
login_manager = LoginManager()
csrf = CSRFProtect()
socketio = SocketIO()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.urandom(32)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production
    app.config['WTF_CSRF_TIME_LIMIT'] = None

    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)
    socketio.init_app(app)

    from .routes import main
    from .auth import auth
    from .mfa import mfa

    app.register_blueprint(main)
    app.register_blueprint(auth)
    app.register_blueprint(mfa)

    from .models import User

    with app.app_context():
        db.create_all()

    register_socket_events(socketio, db, User, train_model, query_user)

    return app
