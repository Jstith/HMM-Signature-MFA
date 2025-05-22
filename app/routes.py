from flask import Blueprint, render_template
from flask_login import login_required, current_user

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('home.html')

@main.route('/dashboard')
@login_required
def dashboard():
    if not current_user.mfa_enabled or not current_user.mfa_validated:
        return "MFA not verified", 403

    # from app import db
    # from models import User
    # user_id = current_user.get_id()
    # query_user = db.session.get(User, user_id)

    # print(query_user.username)
    # print(query_user.mfa_model)

    return render_template('dashboard.html')
