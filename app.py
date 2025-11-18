from flask import Flask, render_template, request, jsonify, Response
import torch
from PIL import Image
import numpy as np
import cv2
import sys
import os
import pathlib
import platform
import base64
from io import BytesIO
import traceback
import time

# --- FIX 1: CROSS-PLATFORM MODEL LOADING ---
# This allows a model trained on Windows to run on Linux (Render)
plat = platform.system()
if plat == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath
elif plat == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

# Add yolov5 to path
sys.path.insert(0, 'yolov5')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Global model variable
model = None
model_info = {}
camera = None

def load_model():
    """Load YOLOv5 model"""
    global model, model_info
    try:
        from models.experimental import attempt_load
        
        device = torch.device('cpu')
        # --- FIX 2: RENAMED MODEL FILE REFERENCE ---
        # Make sure you rename 'best (1).pt' to 'best.pt' in your repo!
        model_path = 'best.pt' 
        
        # Fallback check if user didn't rename it
        if not os.path.exists(model_path) and os.path.exists('best (1).pt'):
            model_path = 'best (1).pt'

        model = attempt_load(model_path, device=device, fuse=False)
        model.eval()
        
        model_info = {
            'loaded': True,
            'device': str(device),
            'classes': list(model.names.values()) if isinstance(model.names, dict) else model.names,
            'stride': int(model.stride.max()) if hasattr(model, 'stride') else 32
        }
        print("✓ Model loaded successfully")
        return True
        
    except Exception as e:
        model_info = {
            'loaded': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        print(f"✗ Model loading failed: {e}")
        return False

# --- FIX 3: LOAD MODEL IMMEDIATELY FOR GUNICORN ---
print("Loading model during startup...")
load_model()

def process_image(image_path):
    """Process image and run detection"""
    if model is None:
        return {'success': False, 'error': 'Model not loaded'}
        
    try:
        from utils.general import non_max_suppression, scale_boxes
        from utils.plots import Annotator, colors
        from utils.augmentations import letterbox
        
        image = Image.open(image_path)
        img_array = np.array(image)
        
        if img_array.shape[-1] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        img_original = img_array.copy()
        
        stride = model_info.get('stride', 32)
        img = letterbox(img_array, 640, stride=stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to('cpu')
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        with torch.no_grad():
            pred = model(img, augment=False, visualize=False)[0]
        
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
        
        detections = []
        annotator = Annotator(img_original.copy(), line_width=3, example=str(model.names))
        
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img_original.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{model.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    detections.append({
                        'class': model.names[c],
                        'confidence': float(conf),
                        'bbox': [float(x) for x in xyxy]
                    })
        
        annotated_img = annotator.result()
        return {
            'success': True,
            'detections': detections,
            'num_detections': len(detections),
            'annotated_image': annotated_img
        }
    except Exception as e:
        return {'success': False, 'error': str(e), 'traceback': traceback.format_exc()}

def process_frame(frame):
    """Process a single frame for live detection"""
    if model is None:
        return frame, []
        
    try:
        from utils.general import non_max_suppression, scale_boxes
        from utils.plots import Annotator, colors
        from utils.augmentations import letterbox
        
        img_original = frame.copy()
        stride = model_info.get('stride', 32)
        img = letterbox(frame, 640, stride=stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to('cpu')
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        with torch.no_grad():
            pred = model(img, augment=False, visualize=False)[0]
        
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
        
        detections = []
        annotator = Annotator(img_original, line_width=2, example=str(model.names))
        
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img_original.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{model.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    detections.append({'class': model.names[c], 'confidence': float(conf)})
        
        annotated_frame = annotator.result()
        
        has_rat = len(detections) > 0
        if has_rat:
            cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 60), (0, 0, 255), -1)
            cv2.putText(annotated_frame, 'HYGIENE ALERT: RAT DETECTED!', (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        else:
            cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 60), (0, 255, 0), -1)
            cv2.putText(annotated_frame, 'HYGIENE STATUS: CLEAR', (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        return annotated_frame, detections
    except Exception as e:
        print(f"Frame processing error: {e}")
        return frame, []

def generate_frames():
    global camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        return
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            annotated_frame, detections = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except GeneratorExit:
        pass
    finally:
        if camera is not None:
            camera.release()

def image_to_base64(img_array):
    img = Image.fromarray(img_array)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

@app.route('/')
def index():
    return render_template('index.html', model_info=model_info)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        filename = 'uploaded_' + file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = process_image(filepath)
        
        if result['success']:
            original_img = Image.open(filepath)
            original_base64 = image_to_base64(np.array(original_img))
            annotated_base64 = image_to_base64(result['annotated_image'])
            return jsonify({
                'success': True,
                'original_image': original_base64,
                'annotated_image': annotated_base64,
                'detections': result['detections'],
                'num_detections': result['num_detections'],
                'has_rat': result['num_detections'] > 0,
                'status': 'UNHYGIENIC' if result['num_detections'] > 0 else 'CLEAR'
            })
        else:
            return jsonify({'success': False, 'error': result['error']}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({'success': True})

@app.route('/model-info')
def get_model_info():
    return jsonify(model_info)

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model_info.get('loaded', False)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)