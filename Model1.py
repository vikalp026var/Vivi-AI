import os
import cv2
from flask import request, render_template, url_for, Flask, redirect, Blueprint, send_file, session
import pickle
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import xgboost
import seaborn as sns
from datetime import datetime
from io import BytesIO
from Vivi_AI.utils.main_utils import fractal_dimension, lacunarity, entropy,process_images_and_extract_features_glrm,load_model
from Vivi_AI.logger import logging
from Vivi_AI.exception import CustomException
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import time
import matplotlib
import uuid
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing, Line
from reportlab.graphics.charts.barcharts import VerticalBarChart
matplotlib.use('Agg')

app = Flask(__name__)
df1 = pd.read_csv(os.path.join("data_transformation", "feature.csv"))
df1['standard_label'] = df1['filename'].apply(lambda x: 'Normal' if 'Normal' in x else 'GBM-Grade IV')

model1 = Blueprint(__name__, "model1")
# Set the upload folder
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def clean_upload_folder_periodically(folder, expiry_time=3600):
    """Periodically cleans up files older than a specified time."""
    current_time = time.time()
    logging.info("Running periodic cleanup of upload folder.")
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        file_mtime = os.path.getmtime(file_path)
        if current_time - file_mtime > expiry_time:  # If the file is older than `expiry_time` (default 1 hour)
            try:
                os.remove(file_path)
                logging.info(f"Removed old file: {file_path}")
            except Exception as e:
                logging.error(f"Error cleaning file {file_path}: {e}")

def generate_unique_filename(filename):
    """Generate a unique filename using UUID."""
    unique_filename = f"{uuid.uuid4().hex}.png"
    logging.info(f"Generated unique filename: {unique_filename}")
    return unique_filename

def create_feature_plot(df1, fd, en, lc,SRE,LRE,GLU):
    features = ['fractal_dimension', 'lacunarity', 'entropy','SRE','LRE','GLU']
    test_values = [fd, lc, en,SRE,LRE,GLU]
    titles = ['Fractal Dimension', 'Lacunarity', 'Entropy','SRE','LRE','GLU']
    plot_paths = []

    for feature, test_value, title in zip(features, test_values, titles):
        fig, ax = plt.subplots(figsize=(18, 12))  # Adjusted figure size (width, height)
        sns.histplot(data=df1, x=feature, hue='standard_label', element='step', stat='density', common_norm=False,
                    palette={'Normal': 'blue', 'GBM-Grade IV': 'red'}, ax=ax)
        ax.axvline(x=test_value, color='green', linestyle='--', label='Test Image')
        ax.set_title(f'Distribution of {title} by Category (Normal and GBM)', fontsize=20)  # Increased font size for title
        ax.set_xlabel(title, fontsize=20)  # Increased font size for x-axis label
        ax.set_ylabel('Density', fontsize=20)  # Increased font size for y-axis label
        ax.legend(fontsize=20)  # Increased font size for legend
        plt.xticks(fontsize=20)  # Increase x-axis tick labels font size
        plt.yticks(fontsize=20)  # Increase y-axis tick labels font size
        plt.show()

        # Custom legend
        handles, labels = ax.get_legend_handles_labels()
        custom_handles = [
            plt.Line2D([0], [0], color='blue', lw=4, label='Normal'),
            plt.Line2D([0], [0], color='red', lw=4, label='GBM-Grade IV'),
            plt.Line2D([0], [0], color='green', linestyle='--', lw=2, label='Test Image')
        ]
        ax.legend(handles=custom_handles, title='Category', loc='upper right')
        unique_file_name = generate_unique_filename(feature)

        plot_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_file_name}")
        fig.savefig(plot_path)
        plt.close(fig)
        plot_paths.append(plot_path)

    return plot_paths

def predict1(img_file):
    try:
        # Clean old files periodically
        clean_upload_folder_periodically(app.config['UPLOAD_FOLDER'])

        # Load the model
        model_path = os.path.join("trained_model", "model.pkl")
        model = load_model(model_path)   # define in utils file 
        logging.info(f"Model loaded from {model_path}")

        # Get the file from the form
        if not img_file:
            logging.warning("No file uploaded.")
            return render_template('research.html', result="No file uploaded",
                                   fractal_dimension=None, entropy=None, lacunarity=None)

        # Input validation
        allowed_extensions = {'png', 'jpg', 'jpeg', 'tiff'}
        if '.' not in img_file.filename or img_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            logging.warning("Unsupported file format.")
            return render_template('research.html', result="Unsupported file format", 
                                   fractal_dimension=None, entropy=None, lacunarity=None)

        # Generate unique file name
        unique_filename = generate_unique_filename(img_file.filename)

        # Save the uploaded file with the unique name
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        img_file.save(file_path)
        if not os.path.exists(file_path):
            logging.error(f"Failed to save uploaded file at: {file_path}")
            return render_template('research.html', result="File upload failed", 
                                   fractal_dimension=None, entropy=None, lacunarity=None)
        logging.info(f"File uploaded and saved at: {file_path}")

        # Read the image
        img = Image.open(file_path)
        img = np.array(img.convert('L'))  # Convert image to grayscale
        logging.info("Image converted to grayscale for processing.")

        # Compute features
        fd = fractal_dimension(img)
        en = entropy(img)
        lc = lacunarity(img, window_size=8)
        SRE,LRE,GLU=process_images_and_extract_features_glrm(img)
        
       
        logging.info(f"Computed features - Fractal Dimension: {fd}, Entropy: {en}, Lacunarity: {lc}")

        # Ensure features are single numerical values
        fd = fd[0] if isinstance(fd, tuple) else fd
        en = en[0] if isinstance(en, tuple) else en
        lc = lc[0] if isinstance(lc, tuple) else lc
        fd=round(fd,3)
        en=round(en,3)
        lc=round(lc,3)
        logging.info(f'After roundeing the fd,en and lc is {fd},{en},{lc}')

        
        # Create DataFrame and ensure correct types
        test = pd.DataFrame(data={
            'fractal_dimension': [fd],
            'entropy': [en],
            'lacunarity': [lc]
        }).astype(float)
        logging.info("Created test DataFrame for model prediction.")

        # Predict
        prediction = model.predict(test.values)
        logging.info(f"Model prediction result: {prediction[0]}")

        # Interpret the result
        result = "GBM" if prediction[0] == 1 else "Normal"
        logging.info(f"Prediction interpreted as: {result}")

        # Create the feature plot
        plot_paths = create_feature_plot(df1, fd, en, lc,SRE,LRE,GLU)
        logging.info(f"Feature plots saved at: {plot_paths}")

        # Optionally clean up the uploaded file after processing (if no longer needed)
        try:
            os.remove(file_path)
            logging.info(f"Uploaded file removed after processing: {file_path}")
        except Exception as e:
            logging.error(f"Error removing file {file_path}: {e}")

        return result, plot_paths, fd, en, lc,SRE,LRE,GLU

    except ValueError as ve:
        logging.error(f"ValueError in predict1: {ve}")
        raise CustomException("ValueError in prediction", ve)
    except IOError as ioe:
        logging.error(f"IOError in predict1: {ioe}")
        raise CustomException("IOError in prediction", ioe)
    except Exception as e:
        logging.error(f"Unexpected error in predict1: {e}")
        raise CustomException("Unexpected prediction error", e)



@model1.route('/Model1/download_report', methods=['GET'])
def download_report():
    result = request.args.get('result')
    buffer = BytesIO()
    
    # Create the document
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    elements = []
    styles = getSampleStyleSheet()

    # Add custom styles
    if 'Centered' not in styles:
        styles.add(ParagraphStyle(name='Centered', alignment=1, fontSize=20, spaceAfter=20))
    if 'Heading' not in styles:
        styles.add(ParagraphStyle(name='Heading', fontSize=14, spaceAfter=10, textColor=colors.black))
    if 'BodyText' not in styles:
        styles.add(ParagraphStyle(name='BodyText', fontSize=12, spaceAfter=10))
    if 'Result' not in styles:
        styles.add(ParagraphStyle(name='Result', fontSize=18, spaceAfter=20, textColor=colors.red, fontName="Helvetica-Bold"))



    # Header
    def header(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica-Bold', 16)
        canvas.drawString(inch, doc.height + doc.topMargin - 0.5*inch, "Cancer Detection Report")
        canvas.setFont('Helvetica', 9)
        canvas.drawRightString(doc.width + doc.leftMargin, doc.height + doc.topMargin - 0.5*inch, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        canvas.restoreState()

    # Footer
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 9)
        canvas.drawString(inch, 0.5*inch, "Confidential Medical Document")
        canvas.drawRightString(doc.width + doc.leftMargin, 0.5*inch, f"Page {doc.page}")
        canvas.restoreState()

    # Add logos
    vivi_logo_path = 'static/vivi.png'
    elements.append(RLImage(vivi_logo_path, width=50, height=50))
    elements.append(Spacer(1, 12))  # Increased space after logo for better separation

    # Analysis Results
    elements.append(Paragraph("Analysis Results", styles['Heading']))
    
    # Retrieve values from session
    fractal_dimension = session.get('fd', 'N/A')
    lacunarity = session.get('lc', 'N/A')
    entropy = session.get('en', 'N/A')
    SRE = session.get('SRE', 'N/A')
    LRE = session.get('LRE', 'N/A')
    GLU = session.get('GLU', 'N/A')

    data = [
        ['Metric', 'Value', 'Reference Range for GBM'],
        ['Fractal Dimension', fractal_dimension, '>1.5'],
        ['Lacunarity', lacunarity, '>0.02'],
        ['Entropy', entropy, '>5.8'],
        ['SRE (Short Run Emphasis)', SRE, '>2.35'],
        ['LRE (Long Run Emphasis)', LRE, '< 58'],
        ['GLU (Gray Level Uniformity)', GLU, '<120']
    ]

    table = Table(data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    elements.append(table)
    elements.append(Spacer(1, 12))  # Space after the table

    # Interpretation
    elements.append(Paragraph("Interpretation", styles['Heading']))
    elements.append(Paragraph("The analysis results provide insights into the texture and complexity of the brain tumor image. "
                              "Fractal dimension indicates the complexity of the tumor's structure. "
                              "Lacunarity represents the heterogeneity of the tumor tissue. "
                              "Entropy measures the randomness or unpredictability in the image texture.", styles['BodyText']))
    elements.append(Spacer(1,12))

    # Conclusion
    elements.append(Paragraph("Conclusion", styles['Heading']))
    elements.append(Paragraph(f"Based on the analysis,  result is: {result}", styles['Result']))
    elements.append(Spacer(1,20))
    elements.append(Paragraph("**Note:This result should be interpreted in conjunction with clinical findings and other diagnostic tests. "
                              "Further evaluation and follow-up may be necessary.", styles['BodyText']))
    elements.append(Spacer(1,13))
    elements.append(Paragraph("----------------------------------------------------------------------------------------------------------------------------------"))

    # Build PDF
    doc.build(elements, onFirstPage=header, onLaterPages=footer)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='GBM_Cancer_Detection_Report.pdf', mimetype='application/pdf')


@app.teardown_appcontext
def cleanup(response_or_exc):
    """Clean up user-specific files after each request."""
    user_id = session.get('user_id')
    logging.info(f"Tearing down session for user_id: {user_id}")
    if user_id:
        # Remove files specific to the user after the request
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if user_id in filename:  # Assuming user-specific filenames include the user_id
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    os.remove(file_path)
                    logging.info(f"Cleaned up file for user {user_id}: {file_path}")
                except Exception as e:
                    logging.error(f"Error cleaning up file {file_path}: {e}")

    return response_or_exc

app.register_blueprint(model1, url_prefix='/model1')
