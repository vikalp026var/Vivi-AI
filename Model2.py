import os
import cv2
from flask import request, render_template, url_for, Flask, redirect, Blueprint, send_file,jsonify
import pickle
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from Vivi_AI.utils.main_utils import fractal_dimension, lacunarity, entropy,list_,final_feature_Extr
import tensorflow as tf 
from Vivi_AI.logger import logging
from Vivi_AI.exception import CustomException
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from datetime import datetime
import time
import threading



app=Flask(__name__)
model2=Blueprint(__name__,"model2")
power_transformation = os.path.join('data_transformation', 'power_transformer.pkl')
standard_scale_path = os.path.join('data_transformation', 'standard.pkl')

UPLOAD_FOLDER = "static/uploads/"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def clean_upload_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            logging.error(f"Error cleaning upload folder: {str(e)}")
            raise CustomException(e)

progress = 0

def long_running_task():
    global progress
    for i in range(11):
        time.sleep(1)  # Simulate work
        progress = i * 10
        print(f"Progress updated to: {progress}")  

@app.route('/start')
def start():
    global progress
    progress = 0
    thread = threading.Thread(target=long_running_task)
    thread.start()
    return jsonify({'status': 'started'})

@app.route('/progress')
def get_progress():
    global progress
    return jsonify({'progress': progress})


def predict2(img_file):
    try:
        clean_upload_folder(app.config['UPLOAD_FOLDER'])
        

        if img_file and allowed_file(img_file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
            app.logger.info(f"Attempting to save file to: {file_path}")
            img_file.save(file_path)
            app.logger.info(f"File saved successfully to: {file_path}")
            
            model = tf.keras.models.load_model(os.path.join('trained_model','best_brain_tumor_model_tuned.h5'))
            logging.info("model loaded successfully")
            
            img = cv2.imread(file_path)
            data = final_feature_Extr(img)
            data.drop(columns=['Energy1', 'Correlation1', 'Dissimilarity1', 'Homogeneity1', 'Asm1', 'Contrast1'], axis=1, inplace=True)
            with open(power_transformation, 'rb') as file:
                pt_loaded = pickle.load(file)
            with open(standard_scale_path, 'rb') as file:
                st_loaded = pickle.load(file)
            data[list_]=pt_loaded.transform(data[list_])
            data=st_loaded.transform(data)
            logging.info(f"Test data transform is {data}")
            y_pred_proba = model.predict(data)
            logging.info(f"Prediction is {y_pred_proba} ")
            prediction = np.argmax(y_pred_proba, axis=1)
            
            logging.info(f"Prediction variable is {prediction}")
        
            if prediction[0]==0:
                result="GBM"
            elif prediction[0]==1:
                result="Normal"
            elif prediction[0]==2:
                result="Astrocytoma"
            else:
                result="Oligodendroglima"
            return result
        
    except Exception as e:
        logging.info(e)
def allowed_file(filename):
    """Check if the file is allowed."""
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@model2.route('/Model2/download_report', methods=['GET'])
def download_report():
    result = request.args.get('result')
    # Create PDF
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Check if the style exists before adding it
    if 'Centered' not in styles:
        styles.add(ParagraphStyle(name='Centered', alignment=1, fontSize=20, spaceAfter=20))
    if 'Heading' not in styles:
        styles.add(ParagraphStyle(name='Heading', fontSize=14, spaceAfter=10, textColor=colors.black))
    if 'BodyText' not in styles:
        styles.add(ParagraphStyle(name='BodyText', fontSize=12, spaceAfter=10))
    if 'Result' not in styles:
        styles.add(ParagraphStyle(name='Result', fontSize=18, spaceAfter=20, textColor=colors.red, fontName="Helvetica-Bold"))

    # Add logos
    vivi_logo_path = 'static/vivi.png'
    elements.append(RLImage(vivi_logo_path, width=200, height=100))
    elements.append(Spacer(5, 12))

    # Add title
    elements.append(Paragraph("GBM Cancer Detection Report", styles['Centered']))

    # Add Introduction
    elements.append(Paragraph("Introduction", styles['Heading']))
    elements.append(Paragraph("This report provides an analysis of GBM cancer detection and patient outcomes.", styles['BodyText']))

    # Add Conclusion
    elements.append(Paragraph("Conclusion:", styles['Heading']))
    elements.append(Paragraph(f"The analysis indicates the result as: {result}", styles['Result']))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name='GBM_Cancer_Detection_Report.pdf', mimetype='application/pdf')
app.register_blueprint(model2, url_prefix='/model2')