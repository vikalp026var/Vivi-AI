from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from authlib.integrations.flask_client import OAuth
from Vivi_AI.logger import logging
from Vivi_AI.constant import SUPABASE_URL, SUPABASE_KEY, OAUTH2_CLIENT_ID, OAUTH_SECRET_ID
from supabase import create_client, Client
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import httpx

auth_bp = Blueprint('auth', __name__)
oauth = OAuth()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


timeout_settings = httpx.Timeout(timeout=10)
# supabase.postgrest.timeout = 99999

def retry_supabase_operation(func):
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(2),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@auth_bp.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('username')
        password = request.form.get('password')
       
        try:
            @retry_supabase_operation
            def sign_in():
                return supabase.auth.sign_in_with_password({"email": email, "password": password})

            response = sign_in()
            logging.info(f"Login attempt for email: {email}")
            user = response.user
            if user:
                session['user'] = user.email
                flash('Login successful!', 'success')
                return redirect(url_for('index'))  
            else:
                flash('Invalid credentials', 'error')
        except Exception as e:
            logging.error(f"Login error: {str(e)}")
            flash(f'Login error: {str(e)}', 'error')
   
    return render_template('loginform.html')

@auth_bp.route("/signup", methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
       
        try:
            @retry_supabase_operation
            def sign_up():
                return supabase.auth.sign_up(
                    credentials={
                    "email": email, 
                    "password": password
                })
            response = sign_up()
            logging.info(f"Signup attempt for email: {email} and {password}" )
            if response.user:
                flash('Signup successful! Please check your email for verification.', 'success')
                return redirect(url_for('auth.login'))
            else:
                flash('Signup failed. Please try again.', 'error')
        except Exception as e:
            logging.error(f"Signup error: {str(e)}")
            flash(f'Signup error: {str(e)}', 'error')
   
    return render_template('loginform.html')

@auth_bp.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        logging.info(f"Attempting to send reset email to: {email}")
        try:
            @retry_supabase_operation
            def reset_password_email():
                return supabase.auth.reset_password_email(email)
            response = reset_password_email()
            if response.error:
                logging.error(f"Supabase error: {response.error}")
                flash('There was an error sending the password reset email. Please try again.', 'error')
            else:
                flash('A password reset link has been sent to your email address. Please check your inbox.', 'success')
        except Exception as e:
            logging.error(f"Exception occurred: {str(e)}")
            flash('An unexpected error occurred. Please try again later.', 'error')
        return redirect(url_for('auth.forgot_password'))
    return render_template('forgot_password.html')

@auth_bp.route('/reset_password/<access_token>', methods=['GET', 'POST'])
def reset_password(access_token):
    if request.method == 'GET':
        # Render the reset password form
        return render_template('reset_password.html', access_token=access_token)
    
    elif request.method == 'POST':
        new_password = request.form['password']
        try:
            @retry_supabase_operation
            def update_password():
                return supabase.auth.api.update_user_by_access_token(access_token, {"password": new_password})
            response = update_password()
            if response.get('error'):
                logging.error(f"Error updating password: {response.get('error')}")
                flash('There was an error updating your password. Please try again.', 'error')
            else:
                flash('Your password has been updated successfully.', 'success')
                return redirect(url_for('auth.login'))
        except Exception as e:
            logging.error(f"Exception occurred during password reset: {str(e)}")
            flash('An unexpected error occurred. Please try again later.', 'error')
    
    # If there's an error, re-render the form
    return render_template('reset_password.html', access_token=access_token)

# Google OAuth configuration
google = oauth.register(
    name='google',
    client_id=OAUTH2_CLIENT_ID,
    client_secret=OAUTH_SECRET_ID,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

@auth_bp.route('/logingoogle')
def login_google():
    redirect_uri = url_for('auth.authorize', _external=True)
    return google.authorize_redirect(redirect_uri)

@auth_bp.route('/authorize')
def authorize():
    token = google.authorize_access_token()
    session['google_token'] = token
    flash('Google login successful!', 'success')
    return redirect(url_for('index'))

@auth_bp.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('google_token', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))