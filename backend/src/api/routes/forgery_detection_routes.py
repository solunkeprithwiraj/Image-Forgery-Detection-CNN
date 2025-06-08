from fastapi import APIRouter, File, UploadFile, Form, Query, HTTPException, Depends
from typing import Optional, List
import os
import shutil
from tempfile import NamedTemporaryFile
from PIL import Image
import io
from src.api.utils.logger import logger
from fastapi.responses import StreamingResponse
import numpy as np

router = APIRouter(prefix='/forgery', tags=['forgery_detection'])

@router.post('/copy-move')
async def detect_copy_move(
    file: UploadFile = File(...),
    method: str = Form("orb"),
    max_size: int = Form(1200)
):
    """
    Detect copy-move forgery in an image
    """
    logger.info(f"Received copy-move detection request using {method} method")
    
    temp_path = None
    img = None
    
    try:
        # Save uploaded file temporarily
        with NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Open and process the image
        img = Image.open(temp_path)
        
        # Generate ELA image
        from src.api.services.prediction_service import generate_ela_image
        ela_image_bytes = generate_ela_image(temp_path)
        if ela_image_bytes:
            # Convert to base64 for including in header
            import base64
            ela_base64 = base64.b64encode(ela_image_bytes).decode('utf-8')
            ela_data_uri = f"data:image/png;base64,{ela_base64}"
        else:
            ela_data_uri = None
        
        # Import the service
        if method.lower() == "orb":
            from src.api.services.copy_move import detect_copy_move_orb
            result_img, confidence, regions = detect_copy_move_orb(img)
        else:
            from src.api.services.copy_move import detect_copy_move_dct
            result_img, confidence, regions = detect_copy_move_dct(img)
        
        # Determine if image is tampered based on confidence
        is_tampered = confidence > 0.5
        
        # Prepare prediction result
        prediction = {
            "prediction": "tampered" if is_tampered else "authentic",
            "confidence": float(confidence),
            "detected_regions": regions if is_tampered else [],
            "method": method,
            "forgery_type": "copy-move",
            "ela_image_url": ela_data_uri
        }
        
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        # Create response
        return StreamingResponse(
            io.BytesIO(img_bytes),
            media_type="image/png",
            headers={
                "X-Forgery-Prediction": prediction["prediction"],
                "X-Forgery-Confidence": str(confidence),
                "X-Forgery-Method": method,
                "X-Forgery-Type": "copy-move",
                "X-Forgery-Details": str(prediction)
            }
        )
    
    except Exception as e:
        logger.error(f"Error in copy-move detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")
    
    finally:
        # Close the image if it was opened
        if img:
            img.close()
        
        # Clean up temporary file if it was created
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.debug(f"Temporary file removed: {temp_path}")
            except Exception as e:
                logger.error(f"Failed to remove temporary file: {str(e)}")

@router.post('/splicing')
async def detect_splicing(
    file: UploadFile = File(...),
    method: str = Form("combined"),
    max_size: int = Form(1200)
):
    """
    Detect splicing forgery in an image
    """
    logger.info(f"Received splicing detection request using {method} method")
    
    temp_path = None
    img = None
    
    try:
        # Save uploaded file temporarily
        with NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Open and process the image
        img = Image.open(temp_path)
        
        # Generate ELA image
        from src.api.services.prediction_service import generate_ela_image
        ela_image_bytes = generate_ela_image(temp_path)
        if ela_image_bytes:
            # Convert to base64 for including in header
            import base64
            ela_base64 = base64.b64encode(ela_image_bytes).decode('utf-8')
            ela_data_uri = f"data:image/png;base64,{ela_base64}"
        else:
            ela_data_uri = None
        
        # Import the service and run detection
        if method.lower() == "edge":
            from src.api.services.splicing import detect_splicing_edge_inconsistencies
            result_img, confidence = detect_splicing_edge_inconsistencies(img)
        elif method.lower() == "lighting":
            from src.api.services.splicing import detect_splicing_lighting_inconsistency
            result_img, confidence = detect_splicing_lighting_inconsistency(img)
        else:
            from src.api.services.splicing import detect_splicing_combined
            result_img, confidence = detect_splicing_combined(img)
        
        # Determine if image is tampered based on confidence
        is_tampered = confidence > 0.5
        
        # Prepare prediction result
        prediction = {
            "prediction": "tampered" if is_tampered else "authentic",
            "confidence": float(confidence),
            "method": method,
            "forgery_type": "splicing",
            "ela_image_url": ela_data_uri
        }
        
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        # Create response
        return StreamingResponse(
            io.BytesIO(img_bytes),
            media_type="image/png",
            headers={
                "X-Forgery-Prediction": prediction["prediction"],
                "X-Forgery-Confidence": str(confidence),
                "X-Forgery-Method": method,
                "X-Forgery-Type": "splicing",
                "X-Forgery-Details": str(prediction)
            }
        )
    
    except Exception as e:
        logger.error(f"Error in splicing detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")
    
    finally:
        # Close the image if it was opened
        if img:
            img.close()
        
        # Clean up temporary file if it was created
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.debug(f"Temporary file removed: {temp_path}")
            except Exception as e:
                logger.error(f"Failed to remove temporary file: {str(e)}")

@router.post('/inpainting')
async def detect_inpainting(
    file: UploadFile = File(...),
    method: str = Form("combined"),
    max_size: int = Form(1200)
):
    """
    Detect inpainting forgery in an image
    """
    logger.info(f"Received inpainting detection request using {method} method")
    
    temp_path = None
    img = None
    
    try:
        # Save uploaded file temporarily
        with NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Open and process the image
        img = Image.open(temp_path)
        
        # Generate ELA image
        from src.api.services.prediction_service import generate_ela_image
        ela_image_bytes = generate_ela_image(temp_path)
        if ela_image_bytes:
            # Convert to base64 for including in header
            import base64
            ela_base64 = base64.b64encode(ela_image_bytes).decode('utf-8')
            ela_data_uri = f"data:image/png;base64,{ela_base64}"
        else:
            ela_data_uri = None
        
        # Import the service and run detection
        if method.lower() == "texture":
            from src.api.services.inpainting import detect_inpainting_texture
            result_img, confidence = detect_inpainting_texture(img)
        elif method.lower() == "noise":
            from src.api.services.inpainting import detect_inpainting_noise
            result_img, confidence = detect_inpainting_noise(img)
        else:
            from src.api.services.inpainting import detect_inpainting_combined
            result_img, confidence = detect_inpainting_combined(img)
        
        # Determine if image is tampered based on confidence
        is_tampered = confidence > 0.5
        
        # Prepare prediction result
        prediction = {
            "prediction": "tampered" if is_tampered else "authentic",
            "confidence": float(confidence),
            "method": method,
            "forgery_type": "inpainting",
            "ela_image_url": ela_data_uri
        }
        
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        # Create response
        return StreamingResponse(
            io.BytesIO(img_bytes),
            media_type="image/png",
            headers={
                "X-Forgery-Prediction": prediction["prediction"],
                "X-Forgery-Confidence": str(confidence),
                "X-Forgery-Method": method,
                "X-Forgery-Type": "inpainting",
                "X-Forgery-Details": str(prediction)
            }
        )
    
    except Exception as e:
        logger.error(f"Error in inpainting detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")
    
    finally:
        # Close the image if it was opened
        if img:
            img.close()
        
        # Clean up temporary file if it was created
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.debug(f"Temporary file removed: {temp_path}")
            except Exception as e:
                logger.error(f"Failed to remove temporary file: {str(e)}")

@router.post('/double-jpeg')
async def detect_double_jpeg_compression(
    file: UploadFile = File(...),
    method: str = Form("combined"),
    max_size: int = Form(1200)
):
    """
    Detect double JPEG compression in an image
    """
    logger.info(f"Received double JPEG detection request using {method} method")
    
    temp_path = None
    img = None
    
    try:
        # Save uploaded file temporarily
        with NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Open and process the image
        img = Image.open(temp_path)
        
        # Generate ELA image
        from src.api.services.prediction_service import generate_ela_image
        ela_image_bytes = generate_ela_image(temp_path)
        if ela_image_bytes:
            # Convert to base64 for including in header
            import base64
            ela_base64 = base64.b64encode(ela_image_bytes).decode('utf-8')
            ela_data_uri = f"data:image/png;base64,{ela_base64}"
        else:
            ela_data_uri = None
        
        # Import the service and run detection
        if method.lower() == "histogram":
            from src.api.services.double_jpeg import detect_double_jpeg_histogram
            result_img, confidence, details = detect_double_jpeg_histogram(img)
        elif method.lower() == "ela":
            from src.api.services.double_jpeg import detect_double_jpeg_ela
            result_img, confidence, details = detect_double_jpeg_ela(img)
        else:
            from src.api.services.double_jpeg import detect_double_jpeg
            result_img, confidence, details = detect_double_jpeg(img)
        
        # Determine if image is tampered based on confidence
        is_tampered = confidence > 0.5
        
        # Prepare prediction result
        prediction = {
            "prediction": "tampered" if is_tampered else "authentic",
            "confidence": float(confidence),
            "method": method,
            "forgery_type": "double_jpeg",
            "details": details,
            "ela_image_url": ela_data_uri
        }
        
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        # Create response
        return StreamingResponse(
            io.BytesIO(img_bytes),
            media_type="image/png",
            headers={
                "X-Forgery-Prediction": prediction["prediction"],
                "X-Forgery-Confidence": str(confidence),
                "X-Forgery-Method": method,
                "X-Forgery-Type": "double-jpeg",
                "X-Forgery-Details": str(prediction)
            }
        )
    
    except Exception as e:
        logger.error(f"Error in double JPEG detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")
    
    finally:
        # Close the image if it was opened
        if img:
            img.close()
        
        # Clean up temporary file if it was created
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.debug(f"Temporary file removed: {temp_path}")
            except Exception as e:
                logger.error(f"Failed to remove temporary file: {str(e)}")

@router.post('/metadata')
async def analyze_metadata(
    file: UploadFile = File(...),
    detailed: bool = Form(False)
):
    """
    Analyze image metadata for signs of tampering
    """
    logger.info(f"Received metadata analysis request")
    
    temp_path = None
    img = None
    
    try:
        # Save uploaded file temporarily
        with NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Open the image
        img = Image.open(temp_path)
        
        # Import the service and run detection
        from src.api.services.metadata import check_metadata_tampering, generate_metadata_report
        
        if detailed:
            report, confidence = generate_metadata_report(img)
            is_tampered = confidence > 0.5
            report["prediction"] = "tampered" if is_tampered else "authentic"
            report["confidence"] = float(confidence)
            return report
        else:
            tampering_results, confidence = check_metadata_tampering(img)
            is_tampered = confidence > 0.5
            
            result = {
                "prediction": "tampered" if is_tampered else "authentic",
                "confidence": float(confidence),
                "tampering_analysis": tampering_results,
                "forgery_type": "metadata_tampering"
            }
            
            return result
    
    except Exception as e:
        logger.error(f"Error in metadata analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
    
    finally:
        # Close the image if it was opened
        if img:
            img.close()
        
        # Clean up temporary file if it was created
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.debug(f"Temporary file removed: {temp_path}")
            except Exception as e:
                logger.error(f"Failed to remove temporary file: {str(e)}")

@router.post('/comprehensive')
async def comprehensive_forgery_detection(
    file: UploadFile = File(...),
    max_size: int = Form(1200)
):
    """
    Perform comprehensive forgery detection using multiple methods
    """
    logger.info("Received comprehensive forgery detection request")
    
    img = None
    image_id = None
    
    try:
        # Read the file into memory
        file_content = await file.read()
        
        # Import the image storage utility
        from src.api.utils.image_storage import image_storage
        
        # Store the image in memory
        image_id = image_storage.store_image(file_content, filename=file.filename)
        
        # Open and process the image
        img = image_storage.get_image(image_id)
        
        # Import the services
        from src.api.services.copy_move import detect_copy_move_orb
        from src.api.services.splicing import detect_splicing_combined
        from src.api.services.inpainting import detect_inpainting_combined
        from src.api.services.metadata import check_metadata_tampering
        from src.api.services.prediction_service import generate_ela_image, predict_image
        from src.api.services.noise_analysis import detect_noise_forgery
        from src.api.services.frequency_analysis import detect_frequency_artifacts
        
        # Import models for CNN prediction
        from src.api.utils.model_loader import get_models
        cnn_model, svm_model = get_models()
        
        # Get direct CNN prediction
        cnn_prediction = None
        try:
            # Get image as bytes
            img_bytes = image_storage.get_image_as_bytes(image_id)
            # Create a temporary buffer
            img_buffer = io.BytesIO(img_bytes)
            # Make prediction using the buffer
            cnn_result = predict_image(img_buffer, cnn_model, svm_model)
            cnn_prediction = {
                "prediction": "tampered" if cnn_result["prediction"] == 1 else "authentic",
                "confidence": float(cnn_result["confidence"]),
                "processing_time": float(cnn_result["processing_time"])
            }
            logger.debug(f"CNN prediction: {cnn_prediction}")
        except Exception as e:
            logger.error(f"Error getting CNN prediction: {str(e)}")
            cnn_prediction = None
        
        # Generate ELA image
        ela_data_uri = None
        try:
            # Get image as bytes again
            img_bytes = image_storage.get_image_as_bytes(image_id)
            # Create a temporary buffer
            img_buffer = io.BytesIO(img_bytes)
            # Generate ELA
            ela_image_bytes = generate_ela_image(img_buffer)
            if ela_image_bytes:
                # Convert to base64 for including in JSON response
                import base64
                ela_base64 = base64.b64encode(ela_image_bytes).decode('utf-8')
                ela_data_uri = f"data:image/png;base64,{ela_base64}"
        except Exception as e:
            logger.error(f"Error generating ELA image: {str(e)}")
        
        # Run noise analysis (new method)
        noise_prediction = None
        noise_visualization_uri = None
        try:
            # Get image object
            img_obj = image_storage.get_image(image_id)
            # Run noise analysis
            _, noise_confidence, is_noise_tampered, noise_regions = detect_noise_forgery(img_obj)
            
            # Convert result image to base64 for including in JSON response
            noise_visualization_uri = f"data:image/png;base64,{base64.b64encode(noise_regions[0]).decode('utf-8')}"
            
            noise_prediction = {
                "prediction": "tampered" if is_noise_tampered else "authentic",
                "confidence": float(noise_confidence),
                "detected_regions": noise_regions
            }
            logger.debug(f"Noise analysis prediction: {noise_prediction['prediction']} with {noise_confidence:.2f} confidence")
        except Exception as e:
            logger.error(f"Error in noise analysis: {str(e)}")
        
        # Run frequency domain analysis (new method)
        freq_prediction = None
        freq_visualization_uri = None
        try:
            # Get image object
            img_obj = image_storage.get_image(image_id)
            # Run frequency domain analysis
            is_freq_tampered, freq_confidence, freq_base64, freq_regions = detect_frequency_artifacts(img_obj)
            
            # Use the base64 result directly
            freq_visualization_uri = f"data:image/png;base64,{freq_base64}"
            
            freq_prediction = {
                "prediction": "tampered" if is_freq_tampered else "authentic",
                "confidence": float(freq_confidence),
                "detected_regions": freq_regions
            }
            logger.debug(f"Frequency analysis prediction: {freq_prediction['prediction']} with {freq_confidence:.2f} confidence")
        except Exception as e:
            logger.error(f"Error in frequency analysis: {str(e)}")
        
        # Run all detection methods
        _, copy_move_confidence, copy_move_regions = detect_copy_move_orb(img)
        _, splicing_confidence = detect_splicing_combined(img)
        _, inpainting_confidence = detect_inpainting_combined(img)
        metadata_results, metadata_confidence = check_metadata_tampering(img)
        
        # Determine if image is tampered based on highest confidence
        confidences = [
            copy_move_confidence,
            splicing_confidence,
            inpainting_confidence,
            metadata_confidence
        ]
        
        # Add new method confidences
        if noise_prediction:
            confidences.append(noise_prediction["confidence"] if noise_prediction["prediction"] == "tampered" else 0)
        
        if freq_prediction:
            confidences.append(freq_prediction["confidence"] if freq_prediction["prediction"] == "tampered" else 0)
        
        # If we have CNN prediction, add it to confidences
        if cnn_prediction:
            confidences.append(cnn_prediction["confidence"] if cnn_prediction["prediction"] == "tampered" else 0)
        
        max_confidence = max(confidences)
        is_tampered = max_confidence > 0.5
        
        # Get most likely forgery type
        forgery_types = ["copy-move", "splicing", "inpainting", "metadata"]
        
        # Add new method types
        if noise_prediction:
            forgery_types.append("noise-pattern")
        
        if freq_prediction:
            forgery_types.append("frequency-artifact")
        
        # Add CNN as a forgery type if available
        if cnn_prediction:
            forgery_types.append("cnn-direct")
        
        most_likely_type = forgery_types[confidences.index(max_confidence)]
        
        # Initialize results dictionary
        results = {
            "prediction": "tampered" if is_tampered else "authentic",
            "overall_confidence": float(max_confidence),
            "most_likely_forgery_type": most_likely_type if is_tampered else None,
            "filename": file.filename,
            "original_size": {"width": img.width, "height": img.height},
            "ela_image_url": ela_data_uri,  # Include ELA image URL in the response
            "results": {
                "copy_move": {
                    "confidence": float(copy_move_confidence),
                    "prediction": "tampered" if copy_move_confidence > 0.5 else "authentic",
                    "detected_regions": copy_move_regions if copy_move_confidence > 0.5 else []
                },
                "splicing": {
                    "confidence": float(splicing_confidence),
                    "prediction": "tampered" if splicing_confidence > 0.5 else "authentic"
                },
                "inpainting": {
                    "confidence": float(inpainting_confidence),
                    "prediction": "tampered" if inpainting_confidence > 0.5 else "authentic"
                },
                "metadata": {
                    "confidence": float(metadata_confidence),
                    "prediction": "tampered" if metadata_confidence > 0.5 else "authentic",
                    "analysis": metadata_results
                }
            }
        }
        
        # Add CNN direct prediction if available
        if cnn_prediction:
            results["results"]["cnn_direct"] = {
                "confidence": cnn_prediction["confidence"],
                "prediction": cnn_prediction["prediction"],
                "processing_time": cnn_prediction["processing_time"]
            }
        
        # Add noise analysis if available
        if noise_prediction:
            results["results"]["noise_analysis"] = {
                "confidence": noise_prediction["confidence"],
                "prediction": noise_prediction["prediction"],
                "visualization_url": noise_visualization_uri,
                "detected_regions": noise_prediction["detected_regions"] if noise_prediction["prediction"] == "tampered" else []
            }
        
        # Add frequency analysis if available
        if freq_prediction:
            results["results"]["frequency_analysis"] = {
                "confidence": freq_prediction["confidence"],
                "prediction": freq_prediction["prediction"],
                "visualization_url": freq_visualization_uri,
                "detected_regions": freq_prediction["detected_regions"] if freq_prediction["prediction"] == "tampered" else []
            }
        
        return results
    
    except Exception as e:
        logger.error(f"Error in comprehensive forgery detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")
    
    finally:
        # Close the image if it was opened
        if img:
            img.close()
        
        # Clean up the stored image if necessary
        if image_id:
            try:
                from src.api.utils.image_storage import image_storage
                image_storage.delete_image(image_id)
                logger.debug(f"Deleted image with ID: {image_id}")
            except Exception as e:
                logger.error(f"Failed to delete stored image: {str(e)}")
