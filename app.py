import os
import logging
from flask import Flask, render_template, request, redirect, url_for, session,jsonify
from Vivi_AI.utils.main_utils import final_feature_Extr, list_
from auth import auth_bp, oauth
from chat_app import chatbot_bp
import uuid
from Model1 import model1, predict1
from Model2 import model2, predict2

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
oauth.init_app(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Register the blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(chatbot_bp)
app.register_blueprint(model1)
app.register_blueprint(model2)

@app.before_request
def start_user_session():
    """Initialize a session for each user."""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        logging.info(f"New session started with user_id: {session['user_id']}")

@app.route("/")
def index():
    app.logger.info('Index route called')
    print("Debugging: Index function called")
    return render_template("index.html")

@app.route('/research', methods=['GET', 'POST'])
def research():
    if 'google_token' not in session and 'user' not in session:
        app.logger.info('User not authenticated, redirecting to login')
        return redirect(url_for('auth.login'))
    
    if request.method == 'POST':
        try:
            img_file = request.files.get('file')
            if not img_file:
                return render_template('research.html', result="No file uploaded")
            if not allowed_file(img_file.filename):
                return render_template('research.html', result="Invalid file type")
            
            # Perform the prediction and get results
            result, plot_paths, fd, en, lc, SRE, LRE, GLU = predict1(img_file)
            
            # Store the results in the session for future access
            session['result'] = result
            session['plot_paths'] = plot_paths
            session['fd'] = fd
            session['en'] = en
            session['lc'] = lc
            session['SRE'] = SRE
            session['LRE'] = LRE
            session['GLU'] = GLU

            # Render the research page with the result and stored session data
            return render_template('research.html', 
                                   result=result,
                                   plot_paths=plot_paths, 
                                   fd=fd, en=en, lc=lc, 
                                   SRE=SRE, LRE=LRE, GLU=GLU)
        except Exception as e:
            logging.error(f"Error occurred: {str(e)}")
            if 'model' in str(e).lower():
                return render_template('research.html', result=f"An error occurred while loading the model: {str(e)}")
            elif 'file' in str(e).lower():
                return render_template('research.html', result=f"An error occurred while processing the file: {str(e)}")
            else:
                return render_template('research.html', result=f"An unexpected error occurred: {str(e)}")
    
    else:  # GET request
        # Retrieve data from the session if exists
        result = session.get('result')
        plot_paths = session.get('plot_paths')
        fd = session.get('fd')
        en = session.get('en')
        lc = session.get('lc')
        SRE = session.get('SRE')
        LRE = session.get('LRE')
        GLU = session.get('GLU')

        # Render the research page with the session data
        return render_template('research.html', 
                               result=result,
                               plot_paths=plot_paths, 
                               fd=fd, en=en, lc=lc, 
                               SRE=SRE, LRE=LRE, GLU=GLU)


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/reset', methods=['POST'])
def reset_session():
    # Clear specific session data to reset for new prediction
    session.pop('result',None)
    session.pop('plot_paths', None)
    session.pop('fd', None)
    session.pop('en', None)
    session.pop('lc', None)
    session.pop('SRE', None)
    session.pop('LRE', None)
    session.pop('GLU', None)
    
    return jsonify({'status': 'Session cleared'}), 200


@app.route('/graphs')
def graphs():
    if 'google_token' not in session and 'user' not in session:
        app.logger.info('User not authenticated, redirecting to login')
        return redirect(url_for('auth.login'))
    
    plot_paths = session.get('plot_paths')
    fd = session.get('fd')
    en = session.get('en')
    lc = session.get('lc')
    SRE =session.get('SRE')
    LRE =session.get('LRE')
    GLU =session.get('GLU')
    
    if not plot_paths or not fd or not en or not lc or not SRE or not LRE or not GLU:
        return render_template('graph.html', error="No graph data available.")
    
    return render_template('graph.html', plot_paths=plot_paths, fd=fd, en=en, lc=lc,SRE=SRE,LRE=LRE,GLU=GLU)

@app.route('/contact')
def contact():
    # if 'google_token' in session or 'user' in session:
    #     app.logger.info('User authenticated, rendering contact.html')
    return render_template("contact.html")
    # app.logger.info('No authenticated user found, redirecting to login')
    # return redirect(url_for('auth.login'))

@app.route('/about')
def about():
    # if 'google_token' in session or 'user' in session:
    #     app.logger.info('User authenticated, rendering contact.html')
    return render_template("about.html")
    # app.logger.info('No authenticated user found, redirecting to login')
    # return redirect(url_for('auth.login'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)