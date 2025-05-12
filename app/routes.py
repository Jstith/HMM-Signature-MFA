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
    return render_template('dashboard.html')
