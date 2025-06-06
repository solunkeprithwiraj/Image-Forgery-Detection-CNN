from fastapi import FastAPI,APIRouter, File, UploadFile, HTTPException, Depends
from typing import List
import os
import shutil
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor
from src.api.services.prediction_service import predict_image
from src.api.utils.logger import logger
from src.api.utils.dependencies import inject_models
from fastapi.responses import StreamingResponse
from PIL import Image
from src.api.services.ela import generate_ela_image
import io
from src.api.models.models import PredictionResult, MultiPredictionResult
from src.api.services.heatmap import generate_forgery_heatmap

# Create router
router = APIRouter()

@router.post("/predict/", response_model=PredictionResult)
async def predict_single_image(file: UploadFile = File(...), models: dict = Depends(inject_models)):
    """
    Predict whether a single image is tampered or authentic
    :param file: The uploaded image file
    :param cnn_model: The pre-trained CNN model (injected)
    :param svm_model: The pre-trained SVM model (injected)
    :returns: Prediction result with confidence
    """
    logger.info(f"Received single image prediction request: {file.filename}")
    # Save uploaded file temporarily
    with NamedTemporaryFile(delete=False) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name
    
    try:
        # Make prediction
        result = predict_image(temp_path, models['cnn_model'], models['svm_model'])
        return result
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temp file
        os.unlink(temp_path)
        logger.debug(f"Temporary file removed: {temp_path}")

@router.post("/predict-batch/", response_model=MultiPredictionResult)
async def predict_multiple_images(files: List[UploadFile] = File(...), models: dict = Depends(inject_models)):
    """
    Predict whether multiple images are tampered or authentic
    :param files: The uploaded image files
    :param cnn_model: The pre-trained CNN model (injected)
    :param svm_model: The pre-trained SVM model (injected)
    :returns: Prediction results for all images
    """
    logger.info(f"Received batch prediction request with {len(files)} images")
    results = []
    temp_files = []
    
    try:
        # Save all uploaded files temporarily
        for file in files:
            with NamedTemporaryFile(delete=False) as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_files.append((temp_file.name, file.filename))
        
        # Process images in parallel using ThreadPoolExecutor
        def process_image(temp_path_and_filename):
            temp_path, original_filename = temp_path_and_filename
            try:
                result = predict_image(temp_path, models['cnn_model'], models['svm_model'])
                # Replace the temporary filename with the original one in the result
                result["filename"] = original_filename
                return result
            except Exception as e:
                logger.error(f"Error processing {original_filename}: {str(e)}")
                return {
                    "filename": original_filename,
                    "error": str(e)
                }
        
        # Use a maximum of 4 workers or the number of files, whichever is smaller
        max_workers = min(4, len(temp_files))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_image, temp_files))
        
        # Filter out any results with errors
        valid_results = [r for r in results if "error" not in r]
        error_results = [r for r in results if "error" in r]
        
        if error_results:
            logger.warning(f"Encountered errors in {len(error_results)} out of {len(files)} images")
        
        return {
            "predictions": valid_results,
            "total": len(valid_results),
            "errors": len(error_results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up all temp files
        for temp_path, _ in temp_files:
            try:
                os.unlink(temp_path)
                logger.debug(f"Temporary file removed: {temp_path}")
            except Exception as e:
                logger.error(f"Error removing temporary file {temp_path}: {str(e)}")

@router.get("/test-all/")
async def test_all_images(models: dict = Depends(inject_models)):
    """
    Test the API with all test images in the data/test_images directory
    :param cnn_model: The pre-trained CNN model (injected)
    :param svm_model: The pre-trained SVM model (injected)
    :returns: Prediction results for all test images
    """
    logger.info("Testing all images in the test directory")
    
    try:
        # Get the absolute path to the project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        test_images_dir = os.path.join(project_root, 'data', 'test_images')
        
        # Get all image files in the test directory
        image_files = []
        for filename in os.listdir(test_images_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                image_files.append(os.path.join(test_images_dir, filename))
        
        logger.info(f"Found {len(image_files)} test images")
        
        # Process images in parallel using ThreadPoolExecutor
        def process_test_image(image_path):
            try:
                return predict_image(image_path, models['cnn_model'], models['svm_model'])
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                return {
                    "filename": os.path.basename(image_path),
                    "error": str(e)
                }
        
        # Use a maximum of 4 workers
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_test_image, image_files))
        
        # Filter out any results with errors
        valid_results = [r for r in results if "error" not in r]
        error_results = [r for r in results if "error" in r]
        
        if error_results:
            logger.warning(f"Encountered errors in {len(error_results)} out of {len(image_files)} images")
        
        return {
            "predictions": valid_results,
            "total": len(valid_results),
            "errors": len(error_results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/ela")
async def perform_ela(image: UploadFile = File(...)):
    # Read and open the image
    image_data = await image.read()
    original_image = Image.open(io.BytesIO(image_data))

    # Generate ELA
    ela_image = generate_ela_image(original_image)

    # Prepare response
    buffer = io.BytesIO()
    ela_image.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")



# @router.post("/heatmap/forgery")
# async def create_forgery_heatmap(file: UploadFile = File(...), models: dict = Depends(inject_models)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image format")

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Generate heatmap prediction mask
    prediction_mask = get_prediction_mask(image, models["cnn_model"], models["svm_model"])

    # Overlay heatmap on image
    from src.api.services.heatmap import generate_forgery_heatmap
    heatmap_image = generate_forgery_heatmap(image, prediction_mask)

    buffer = io.BytesIO()
    heatmap_image.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")