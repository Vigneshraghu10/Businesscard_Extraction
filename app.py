from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import requests
import json
import threading
import pandas as pd
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
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
from functools import lru_cache
import asyncio
import aiohttp
from cachetools import TTLCache
import numpy as np

app = Flask(__name__)
CORS(app)

# Enhanced Configuration
app.config.update(
    UPLOAD_FOLDER='uploads',
    MAX_CONTENT_LENGTH=1024* 1024 * 1024,  # 32MB max upload
    BATCH_SIZE=100,  # Increased batch size
    IMAGE_CACHE_TTL=3600,  # 1 hour cache for processed images
    MAX_WORKERS=min(os.cpu_count() * 2, 16),  # Optimal number of workers
    CHUNK_SIZE=5  # Number of images to process in parallel within each process
)

# API Configuration
OCR_API_KEY = 'K85893325188957'
GEMINI_API_KEY = 'AIzaSyDeEh7-NY3elAZHIRILzaxT2zo4JyAH56c'
OCR_API_URL = 'https://api.ocr.space/parse/image'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

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

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize caches
result_cache = TTLCache(maxsize=1000, ttl=3600)
image_cache = TTLCache(maxsize=100, ttl=300)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@lru_cache(maxsize=128)
def get_optimized_image(image_path):
    """Cache and optimize images for reuse"""
    try:
        with PIL.Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Optimize size while maintaining quality
            max_size = 1200  # Reduced from 1600 for faster processing
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, PIL.Image.Resampling.LANCZOS)
            
            buffer = BytesIO()
            img.save(buffer, format='JPEG', optimize=True, quality=80)
            return buffer.getvalue()
    except Exception as e:
        logger.error(f"Image optimization failed: {e}")
        return None

async def process_with_ocr_space_async(session, image_data):
    """Asynchronous OCR processing"""
    try:
        payload = {
            'apikey': OCR_API_KEY,
            'language': 'eng',
            'isOverlayRequired': False,
            'detectOrientation': True,
            'scale': True,
            'OCREngine': 2
        }
        
        data = aiohttp.FormData()
        data.add_field('file', image_data, filename='image.jpg')
        for key, value in payload.items():
            data.add_field(key, str(value))
        
        async with session.post(OCR_API_URL, data=data) as response:
            result = await response.json()
            if result.get('ParsedResults'):
                return result['ParsedResults'][0]['ParsedText']
            return None
    except Exception as e:
        logger.error(f"Async OCR processing failed: {e}")
        return None

@lru_cache(maxsize=256)
def extract_business_card_data(text):
    """Optimized data extraction with caching"""
    if not text:
        return None
        
    data = {
        'person_name': None,
        'company_name': None,
        'email': None,
        'contact_number': None,
        'designation': None,
        'website': None,
        'address': None
    }
    
    # Compile regex patterns once
    patterns = {
        'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        'phone': re.compile(r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
        'website': re.compile(r'(?:https?://)?(?:www\.)?[A-Za-z0-9-]+\.[A-Za-z]{2,}(?:\.[A-Za-z]{2,})?(?:/\S*)?')
    }
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Extract using compiled patterns
    for line in lines:
        if not data['email']:
            email_match = patterns['email'].search(line)
            if email_match:
                data['email'] = email_match.group()
        
        if not data['contact_number']:
            phone_match = patterns['phone'].search(line)
            if phone_match:
                data['contact_number'] = phone_match.group()
        
        if not data['website']:
            website_match = patterns['website'].search(line)
            if website_match and 'email' not in line.lower():
                data['website'] = website_match.group()
    
    # Optimized name and designation extraction
    for i, line in enumerate(lines[:3]):
        if not any(field in line.lower() for field in ['www', '@', '.com', 'tel:', 'fax:', 'phone']):
            if not data['person_name']:
                data['person_name'] = line
            elif not data['designation']:
                data['designation'] = line
                break
    
    # Optimized company name extraction
    company_start_idx = 2 if data['designation'] else 1
    for line in lines[company_start_idx:]:
        if not any(field in line.lower() for field in ['www', '@', '.com', 'tel:', 'fax:', 'phone']):
            if not data['company_name']:
                data['company_name'] = line
                break
    
    # Optimized address extraction
    address_lines = []
    excluded_values = {data['person_name'], data['designation'], data['company_name']}
    for line in reversed(lines):
        if not any(field in line.lower() for field in ['www', '@', '.com', 'tel:', 'fax:', 'phone']) and \
           line not in excluded_values:
            address_lines.insert(0, line)
            if len(address_lines) >= 3:
                break
    if address_lines:
        data['address'] = ', '.join(address_lines)
    
    return data

async def process_image_chunk(chunk, semaphore):
    """Process a chunk of images concurrently"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for image_path in chunk:
            if image_path in result_cache:
                continue
            
            image_data = get_optimized_image(image_path)
            if not image_data:
                continue
            
            async with semaphore:
                tasks.append(asyncio.create_task(process_single_image_async(session, image_path, image_data)))
        
        return await asyncio.gather(*tasks)

async def process_single_image_async(session, image_path, image_data):
    """Asynchronous single image processing"""
    try:
        # Parallel OCR and Gemini processing
        ocr_task = asyncio.create_task(process_with_ocr_space_async(session, image_data))
        gemini_task = asyncio.create_task(process_with_gemini_async(image_data))
        
        ocr_result, gemini_result = await asyncio.gather(ocr_task, gemini_task)
        
        if ocr_result:
            ocr_data = extract_business_card_data(ocr_result)
        else:
            ocr_data = None
        
        merged_result = merge_results(gemini_result, ocr_data)
        
        if merged_result:
            result = {
                **merged_result,
                'source_image': os.path.basename(image_path),
                'processed_timestamp': datetime.now().isoformat(),
                'processing_status': 'success'
            }
        else:
            result = {
                'source_image': os.path.basename(image_path),
                'processed_timestamp': datetime.now().isoformat(),
                'processing_status': 'error',
                'error_message': 'Failed to extract data'
            }
        
        result_cache[image_path] = result
        return result
        
    except Exception as e:
        logger.error(f"Async processing failed for {image_path}: {e}")
        return {
            'source_image': os.path.basename(image_path),
            'processed_timestamp': datetime.now().isoformat(),
            'processing_status': 'error',
            'error_message': str(e)
        }

async def process_with_gemini_async(image_data):
    """Asynchronous Gemini processing"""
    try:
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
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
        
        response = await asyncio.to_thread(
            gemini_model.generate_content,
            [prompt, {"mime_type": "image/jpeg", "data": base64_image}]
        )
        
        try:
            response_text = response.text.strip()
            response_text = re.sub(r'^```json\s*|\s*```$', '', response_text)
            return json.loads(response_text)
        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            return None
            
    except Exception as e:
        logger.error(f"Gemini processing failed: {e}")
        return None

def process_all_images(image_files):
    """Process all images with enhanced parallelization"""
    if not image_files:
        return []
    
    # Split images into chunks for batch processing
    chunks = np.array_split(image_files, max(1, len(image_files) // app.config['CHUNK_SIZE']))
    
    async def process_all():
        semaphore = asyncio.Semaphore(app.config['MAX_WORKERS'])
        tasks = [process_image_chunk(chunk, semaphore) for chunk in chunks]
        results = await asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist if item]
    
    return asyncio.run(process_all())

# Routes
@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({"status": "error", "message": "No files provided"}), 400
    
    files = request.files.getlist('files')
    saved_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            saved_files.append(file_path)
    
    if not saved_files:
        return jsonify({"status": "error", "message": "No valid files uploaded"}), 400
    
    try:
        results = process_all_images(saved_files)
        
        # Cleanup
        for file_path in saved_files:
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {e}")
        
        return jsonify({
            "status": "success",
            "data": results
        })
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Remaining routes (index, test, batch, download) remain unchanged
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test', methods=['GET'])
def test_connection():
    return jsonify({"status": "success", "message": "Backend server is running"})

@app.route('/download', methods=['POST'])
def download_results():
    try:
        data = request.get_json()
        if not data or 'results' not in data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        results = data['results']
        if not results:
            return jsonify({"status": "error", "message": "Empty results"}), 400
        
        df = pd.DataFrame(results)
        metadata_cols = ['processed_timestamp', 'processing_status', 'error_message']
        export_cols = [col for col in df.columns if col not in metadata_cols]
        df_export = df[export_cols]
        
        csv_buffer = StringIO()
        df_export.to_csv(csv_buffer, index=False)
        
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

def validate_field(value, field_type):
    """Validate and clean extracted fields with improved validation"""
    if not value:
        return None
        
    value = str(value).strip()
    if not value:
        return None
    
    # Cached regex patterns
    patterns = {
        'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        'phone': re.compile(r'[^\d+]'),
        'website': re.compile(r'https?://(?:www\.)?[A-Za-z0-9-]+\.[A-Za-z]{2,}(?:\.[A-Za-z]{2,})?(?:/\S*)?')
    }
        
    if field_type == 'email':
        match = patterns['email'].search(value)
        return match.group() if match else None
        
    elif field_type == 'phone':
        cleaned = patterns['phone'].sub('', value)
        return cleaned if len(cleaned) >= 10 else None
        
    elif field_type == 'website':
        if not value.startswith(('http://', 'https://')):
            value = 'http://' + value
        match = patterns['website'].search(value)
        return match.group() if match else None
        
    return value

def merge_results(gemini_result, ocr_result):
    """Merge and validate results from both APIs with improved accuracy"""
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
    
    # Confidence weights for different sources
    weights = {
        'gemini': {
            'text': 0.7,
            'email': 0.3,
            'phone': 0.3,
            'website': 0.3
        },
        'ocr': {
            'text': 0.3,
            'email': 0.7,
            'phone': 0.7,
            'website': 0.7
        }
    }
    
    for field, field_type in field_types.items():
        gemini_value = gemini_result.get(field) if gemini_result else None
        ocr_value = ocr_result.get(field) if ocr_result else None
        
        gemini_value = validate_field(gemini_value, field_type)
        ocr_value = validate_field(ocr_value, field_type)
        
        # Use weighted selection based on field type
        if gemini_value and ocr_value:
            if weights['gemini'][field_type if field_type in weights['gemini'] else 'text'] > \
               weights['ocr'][field_type if field_type in weights['ocr'] else 'text']:
                merged[field] = gemini_value
            else:
                merged[field] = ocr_value
        else:
            merged[field] = gemini_value or ocr_value
            
    return merged

def cleanup_old_files():
    """Periodically clean up old temporary files"""
    try:
        current_time = time.time()
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > 3600:  # Remove files older than 1 hour
                    os.remove(file_path)
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

@app.route('/batch', methods=['POST'])
def process_batch_upload():
    """Enhanced batch processing endpoint with progress tracking"""
    if 'files' not in request.files:
        return jsonify({"status": "error", "message": "No files provided"}), 400
    
    files = request.files.getlist('files')
    saved_files = []
    
    # Save files with validation
    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Quick validation of saved file
                if os.path.getsize(file_path) > 0:
                    saved_files.append(file_path)
                else:
                    os.remove(file_path)
                    logger.warning(f"Skipped empty file: {filename}")
            except Exception as e:
                logger.error(f"Error saving file {file.filename}: {e}")
                continue
    
    if not saved_files:
        return jsonify({
            "status": "error",
            "message": "No valid files were uploaded"
        }), 400
    
    try:
        # Process files in optimized batches
        results = process_all_images(saved_files)
        
        # Clean up
        for file_path in saved_files:
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {e}")
        
        # Run background cleanup
        cleanup_old_files()
        
        successful = [r for r in results if r.get('processing_status') == 'success']
        failed = [r for r in results if r.get('processing_status') == 'error']
        
        return jsonify({
            "status": "success",
            "data": results,
            "summary": {
                "total_processed": len(results),
                "successful": len(successful),
                "failed": len(failed)
            }
        })
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

def cleanup_old_files():
    """Periodically clean up old temporary files"""
    try:
        current_time = time.time()
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > 3600:  # Remove files older than 1 hour
                    os.remove(file_path)
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

def start_background_tasks():
    """Initialize background tasks"""
    def run_cleanup():
        while True:
            try:
                cleanup_old_files()
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
            time.sleep(3600)  # Run every hour
    
    # Create and start the cleanup thread
    cleanup_thread = threading.Thread(target=run_cleanup, daemon=True)
    cleanup_thread.start()
    logger.info("Background cleanup task started")

if __name__ == '__main__':
    try:
        # Initialize background tasks
        start_background_tasks()
        
        # Run app with optimized settings
        app.run(
            debug=False,  # Disable debug mode in production
            host='0.0.0.0',
            port=5000,
            threaded=True,
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"Application startup error: {e}")