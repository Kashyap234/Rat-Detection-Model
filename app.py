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

# Fix for Windows - MUST be before any YOLOv5 imports
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

# Add yolov5 to path
sys.path.insert(0, 'yolov5')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Create necessary folders
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
        model = attempt_load('best (1).pt', device=device, fuse=False)
        model.eval()
        
        # Store model info
        model_info = {
            'loaded': True,
            'device': str(device),
            'classes': list(model.names.values()) if isinstance(model.names, dict) else model.names,
            'stride': int(model.stride.max()) if hasattr(model, 'stride') else 32
        }
        
        print("✓ Model loaded successfully")
        print(f"  Classes: {model_info['classes']}")
        return True
        
    except Exception as e:
        model_info = {
            'loaded': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        print(f"✗ Model loading failed: {e}")
        return False

def process_image(image_path):
    """Process image and run detection"""
    try:
        from utils.general import non_max_suppression, scale_boxes
        from utils.plots import Annotator, colors
        from utils.augmentations import letterbox
        
        # Load image
        image = Image.open(image_path)
        img_array = np.array(image)
        
        # Convert RGBA to RGB if necessary
        if img_array.shape[-1] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        img_original = img_array.copy()
        
        # Preprocess
        stride = model_info.get('stride', 32)
        img = letterbox(img_array, 640, stride=stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to('cpu')
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            pred = model(img, augment=False, visualize=False)[0]
        
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
        
        # Process predictions
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
        
        # Get annotated image
        annotated_img = annotator.result()
        
        return {
            'success': True,
            'detections': detections,
            'num_detections': len(detections),
            'annotated_image': annotated_img
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def process_frame(frame):
    """Process a single frame for live detection"""
    try:
        from utils.general import non_max_suppression, scale_boxes
        from utils.plots import Annotator, colors
        from utils.augmentations import letterbox
        
        img_original = frame.copy()
        
        # Preprocess
        stride = model_info.get('stride', 32)
        img = letterbox(frame, 640, stride=stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to('cpu')
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            pred = model(img, augment=False, visualize=False)[0]
        
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
        
        # Process predictions
        detections = []
        annotator = Annotator(img_original, line_width=2, example=str(model.names))
        
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img_original.shape).round()
                
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{model.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    
                    detections.append({
                        'class': model.names[c],
                        'confidence': float(conf)
                    })
        
        # Get annotated image
        annotated_frame = annotator.result()
        
        # Add status banner
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
    """Generate frames for live video stream"""
    global camera
    
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties for better performance
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    try:
        while True:
            success, frame = camera.read()
            
            if not success:
                break
            
            # Process frame
            annotated_frame, detections = process_frame(frame)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
    except GeneratorExit:
        print("Client disconnected from video stream")
    finally:
        if camera is not None:
            camera.release()

def image_to_base64(img_array):
    """Convert numpy array to base64 string"""
    img = Image.fromarray(img_array)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', model_info=model_info)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and detection"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({'error': 'Invalid file type. Please upload JPG, JPEG, or PNG'}), 400
    
    try:
        # Save uploaded file
        filename = 'uploaded_' + file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        result = process_image(filepath)
        
        if result['success']:
            # Convert images to base64
            original_img = Image.open(filepath)
            original_base64 = image_to_base64(np.array(original_img))
            annotated_base64 = image_to_base64(result['annotated_image'])
            
            # Determine hygiene status
            has_rat = result['num_detections'] > 0
            
            return jsonify({
                'success': True,
                'original_image': original_base64,
                'annotated_image': annotated_base64,
                'detections': result['detections'],
                'num_detections': result['num_detections'],
                'has_rat': has_rat,
                'status': 'UNHYGIENIC' if has_rat else 'CLEAR'
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera stream"""
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({'success': True})

@app.route('/model-info')
def get_model_info():
    """Get model information"""
    return jsonify(model_info)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_info.get('loaded', False)
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🔍 Hygiene Monitoring System - Rat Detection")
    print("=" * 60)
    print("\nLoading YOLOv5 model...")
    
    if load_model():
        print("\n✓ Server ready!")
        print("\n📍 Access the application at:")
        print("   http://127.0.0.1:5000")
        print("\n" + "=" * 60)
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    else:
        print("\n✗ Failed to load model. Cannot start server.")
        print("\nError details:")
        print(model_info.get('error', 'Unknown error'))
        print("\nPlease check:")
        print("1. YOLOv5 version matches your training version")
        print("2. 'best (1).pt' file exists in the current directory")
        print("3. All dependencies are installed")