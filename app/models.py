from app import db
from flask_login import UserMixin

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    mfa_model = db.Column(db.String(8192), nullable=True)
    mfa_threshold = db.Column(db.Float, nullable=True)
    mfa_enabled = db.Column(db.Boolean, default=False)
    mfa_validated = db.Column(db.Boolean, default=False)
