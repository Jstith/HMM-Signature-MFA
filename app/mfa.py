from flask import Blueprint, render_template, request, redirect, url_for, session
from flask_login import login_required, current_user

mfa = Blueprint('mfa', __name__)

@mfa.route('/mfa/setup', methods=['GET'])
@login_required
def setup_mfa():
    return render_template('mfa_setup.html')

@mfa.route('/mfa/verify', methods=['GET', 'POST'])
@login_required
def verify_mfa():
    if not current_user.mfa_enabled:
        return redirect(url_for('mfa.setup_mfa'))

    if request.method == 'POST':
        # Replace with your TOTP validation logic
        # If valid:
        current_user.mfa_validated = True
        from app import db
        db.session.commit()
        return redirect(url_for('main.dashboard'))
    return render_template('mfa_verify.html')
