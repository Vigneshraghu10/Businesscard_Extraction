from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import requests
import json
import pandas as pd
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import time
from datetime import datetime
import gc
import PIL.Image
from io import BytesIO, StringIO
import google.generativeai as genai
import base64
import re

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload
app.config['BATCH_SIZE'] = 50

# API Configuration
OCR_API_KEY = 'K85893325188957'  # Replace with your OCR.space API key
GEMINI_API_KEY = 'AIzaSyBCNgUZT4687UaA1cL9n-4vbDoiKP4BGtI'  # Replace with your Gemini API key
OCR_API_URL = 'https://api.ocr.space/parse/image'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_WORKERS = 4

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('business_cards.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def optimize_image(image_path):
    """Optimize image size before processing"""
    try:
        with PIL.Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            max_size = 1600
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, PIL.Image.Resampling.LANCZOS)
            
            buffer = BytesIO()
            img.save(buffer, format='JPEG', optimize=True, quality=85)
            with open(image_path, 'wb') as f:
                f.write(buffer.getvalue())
            
            return True
    except Exception as e:
        logger.error(f"Image optimization failed for {image_path}: {e}")
        return False

def extract_business_card_data(text):
    """Extract structured data from OCR text"""
    data = {
        'person_name': None,
        'company_name': None,
        'email': None,
        'contact_number': None,
        'designation': None,
        'website': None,
        'address': None
    }
    
    # Split text into lines and clean
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    for line in lines:
        email_match = re.search(email_pattern, line)
        if email_match:
            data['email'] = email_match.group()
            break
    
    # Extract phone number
    phone_pattern = r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    for line in lines:
        phone_match = re.search(phone_pattern, line)
        if phone_match:
            data['contact_number'] = phone_match.group()
            break
    
    # Extract website
    website_pattern = r'(?:https?://)?(?:www\.)?[A-Za-z0-9-]+\.[A-Za-z]{2,}(?:\.[A-Za-z]{2,})?(?:/\S*)?'
    for line in lines:
        website_match = re.search(website_pattern, line)
        if website_match and 'email' not in line.lower():
            data['website'] = website_match.group()
            break
    
    # Extract name and designation (typically in first few lines)
    for i, line in enumerate(lines[:3]):
        if not any(field in line.lower() for field in ['www', '@', '.com', 'tel:', 'fax:', 'phone']):
            if not data['person_name']:
                data['person_name'] = line
            elif not data['designation']:
                data['designation'] = line
                break
    
    # Extract company name
    company_start_idx = 2 if data['designation'] else 1
    for line in lines[company_start_idx:]:
        if not any(field in line.lower() for field in ['www', '@', '.com', 'tel:', 'fax:', 'phone']):
            if not data['company_name']:
                data['company_name'] = line
                break
    
    # Extract address
    address_lines = []
    for line in reversed(lines):
        if not any(field in line.lower() for field in ['www', '@', '.com', 'tel:', 'fax:', 'phone']) and \
           line not in [data['person_name'], data['designation'], data['company_name']]:
            address_lines.insert(0, line)
        if len(address_lines) >= 3:
            break
    if address_lines:
        data['address'] = ', '.join(address_lines)
    
    return data

def process_with_gemini(image_path):
    """Process image using Gemini Vision API"""
    try:
        with PIL.Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
        prompt = """
        Extract the following information from this business card image and return ONLY a JSON object with these exact keys:
        {
            "person_name": "Full name of the person",
            "company_name": "Name of the company",
            "email": "Email address",
            "contact_number": "Phone number",
            "designation": "Job title",
            "website": "Company website",
            "address": "Physical address"
        }
        
        Return ONLY the JSON object, no additional text.
        If a field is not found, use null as the value.
        """
        
        response = gemini_model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": base64_image}
        ])
        
        try:
            response_text = response.text.strip()
            response_text = re.sub(r'^```json\s*|\s*```$', '', response_text)
            result = json.loads(response_text)
            
            required_keys = ['person_name', 'company_name', 'email', 'contact_number', 
                           'designation', 'website', 'address']
            for key in required_keys:
                if key not in result:
                    result[key] = None
                    
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            return None
            
    except Exception as e:
        logger.error(f"Gemini processing failed for {image_path}: {e}")
        return None

def process_with_ocr_space(image_path):
    """Process image using OCR.space API"""
    try:
        with open(image_path, 'rb') as image_file:
            payload = {
                'apikey': OCR_API_KEY,
                'language': 'eng',
                'isOverlayRequired': False,
                'detectOrientation': True,
                'scale': True,
                'OCREngine': 2
            }
            files = {
                'file': image_file
            }
            
            response = requests.post(OCR_API_URL, files=files, data=payload)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get('ParsedResults'):
                extracted_text = result['ParsedResults'][0]['ParsedText']
                if not extracted_text:
                    return None
                    
                return extract_business_card_data(extracted_text)
            
            return None
            
    except Exception as e:
        logger.error(f"OCR.space processing failed for {image_path}: {e}")
        return None

def validate_field(value, field_type):
    """Validate and clean extracted fields"""
    if not value:
        return None
        
    value = str(value).strip()
    if not value:
        return None
        
    if field_type == 'email':
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(email_pattern, value)
        return match.group() if match else None
        
    elif field_type == 'phone':
        cleaned = re.sub(r'[^\d+]', '', value)
        if len(cleaned) >= 10:
            return cleaned
        return None
        
    elif field_type == 'website':
        if not value.startswith(('http://', 'https://')):
            value = 'http://' + value
        website_pattern = r'https?://(?:www\.)?[A-Za-z0-9-]+\.[A-Za-z]{2,}(?:\.[A-Za-z]{2,})?(?:/\S*)?'
        match = re.search(website_pattern, value)
        return match.group() if match else None
        
    return value

def merge_results(gemini_result, ocr_result):
    """Merge and validate results from both APIs"""
    if not gemini_result and not ocr_result:
        return None
        
    merged = {}
    field_types = {
        'person_name': 'text',
        'company_name': 'text',
        'email': 'email',
        'contact_number': 'phone',
        'designation': 'text',
        'website': 'website',
        'address': 'text'
    }
    
    for field, field_type in field_types.items():
        gemini_value = gemini_result.get(field) if gemini_result else None
        ocr_value = ocr_result.get(field) if ocr_result else None
        
        gemini_value = validate_field(gemini_value, field_type)
        ocr_value = validate_field(ocr_value, field_type)
        
        # Prefer OCR results for structured data like email, website, phone
        if field in ['email', 'website', 'contact_number']:
            merged[field] = ocr_value or gemini_value
        else:
            # Prefer Gemini results for unstructured text
            merged[field] = gemini_value or ocr_value
            
    return merged

def process_single_image(image_path):
    """Process a single image using both APIs"""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            if not optimize_image(image_path):
                return None
                
            gemini_result = process_with_gemini(image_path)
            ocr_result = process_with_ocr_space(image_path)
            
            merged_result = merge_results(gemini_result, ocr_result)
            
            if merged_result:
                merged_result.update({
                    'source_image': os.path.basename(image_path),
                    'processed_timestamp': datetime.now().isoformat(),
                    'processing_status': 'success'
                })
                return merged_result
                
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
                
            return {
                'source_image': os.path.basename(image_path),
                'processed_timestamp': datetime.now().isoformat(),
                'processing_status': 'error',
                'error_message': 'Failed to extract data from image'
            }
            
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Retry {attempt + 1} for {image_path}: {e}")
                time.sleep(retry_delay)
            else:
                logger.error(f"Final error processing {image_path}: {e}")
                return {
                    'source_image': os.path.basename(image_path),
                    'processed_timestamp': datetime.now().isoformat(),
                    'processing_status': 'error',
                    'error_message': str(e)
                }

def process_batch(image_batch):
    """Process a batch of images in parallel"""
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_single_image, image_path)
            for image_path in image_batch
        ]
        
        results = []
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
    
    return results

def process_all_images(image_files):
    """Process all images in batches with progress tracking"""
    all_extracted_data = []
    total_batches = (len(image_files) + app.config['BATCH_SIZE'] - 1) // app.config['BATCH_SIZE']
    
    for i in tqdm(range(0, len(image_files), app.config['BATCH_SIZE']), total=total_batches, desc="Processing batches"):
        batch = image_files[i:i + app.config['BATCH_SIZE']]
        batch_paths = [
            os.path.join(app.config['UPLOAD_FOLDER'], f)
            for f in batch if allowed_file(f)
        ]
        
        if batch_paths:
            batch_results = process_batch(batch_paths)
            all_extracted_data.extend(batch_results)
        
        gc.collect()

    return all_extracted_data

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test', methods=['GET'])
def test_connection():
    return jsonify({"status": "success", "message": "Backend server is running"})

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({"status": "error", "message": "No files provided"}), 400
    
    files = request.files.getlist('files')
    processed_data = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                # Process the image using both OCR and Gemini
                result = process_single_image(file_path)
                if result:
                    processed_data.append(result)
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                processed_data.append({
                    'source_image': filename,
                    'processed_timestamp': datetime.now().isoformat(),
                    'processing_status': 'error',
                    'error_message': str(e)
                })
            
            # Clean up the uploaded file
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {e}")
    
    if not processed_data:
        return jsonify({
            "status": "error",
            "message": "No valid files were processed"
        }), 400
    
    return jsonify({
        "status": "success",
        "data": processed_data
    })

@app.route('/batch', methods=['POST'])
def process_batch_upload():
    if 'files' not in request.files:
        return jsonify({"status": "error", "message": "No files provided"}), 400
    
    files = request.files.getlist('files')
    saved_files = []
    
    # Save all files first
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            saved_files.append(filename)
    
    if not saved_files:
        return jsonify({
            "status": "error",
            "message": "No valid files were uploaded"
        }), 400
    
    try:
        # Process all saved files
        results = process_all_images(saved_files)
        
        # Clean up
        for filename in saved_files:
            try:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {filename}: {e}")
        
        return jsonify({
            "status": "success",
            "data": results
        })
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/download', methods=['POST'])
def download_results():
    try:
        data = request.get_json()
        if not data or 'results' not in data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        results = data['results']
        if not results:
            return jsonify({"status": "error", "message": "Empty results"}), 400
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Remove processing metadata columns for the CSV
        metadata_cols = ['processed_timestamp', 'processing_status', 'error_message']
        export_cols = [col for col in df.columns if col not in metadata_cols]
        df_export = df[export_cols]
        
        # Convert to CSV
        csv_buffer = StringIO()
        df_export.to_csv(csv_buffer, index=False)
        
        # Create response
        output = BytesIO()
        output.write(csv_buffer.getvalue().encode('utf-8'))
        output.seek(0)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'business_cards_{timestamp}.csv'
        )
        
    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to generate download file"
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "status": "error",
        "message": "File too large. Maximum size is 32MB."
    }), 413

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)