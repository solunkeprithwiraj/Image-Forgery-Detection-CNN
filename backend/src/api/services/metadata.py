import os
import time
from PIL import Image
import piexif
from piexif.helper import UserComment
import io
import json
import re
import datetime
from src.api.utils.logger import logger

def resize_if_needed(image, max_size=1200):
    """Resize image if any dimension is larger than max_size while preserving aspect ratio"""
    width, height = image.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        return image.resize((new_width, new_height), Image.LANCZOS)
    return image

def check_metadata_tampering(image):
    """
    Check image metadata for signs of tampering
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        Tuple[dict, float]: Dictionary of results and confidence score
    """
    start_time = time.time()
    logger.info("Starting metadata tampering analysis")
    
    # Initialize results
    results = {
        "has_exif": False,
        "software_used": None,
        "creation_time": None,
        "modification_time": None,
        "camera_make": None,
        "camera_model": None,
        "suspicious_tags": [],
        "missing_essential_tags": [],
        "inconsistencies": []
    }
    
    # Check if the image has exif data
    try:
        exif_dict = piexif.load(image.info.get("exif", b""))
        results["has_exif"] = True
    except:
        logger.info("No valid EXIF data found")
        results["has_exif"] = False
        return results, 0.7  # High confidence of tampering if no EXIF data
    
    # Essential tags for different EXIF IFD sections
    essential_tags = {
        "0th": [piexif.ImageIFD.Make, piexif.ImageIFD.Model, piexif.ImageIFD.Software],
        "Exif": [piexif.ExifIFD.DateTimeOriginal, piexif.ExifIFD.DateTimeDigitized]
    }
    
    # Check for essential tags
    for ifd in essential_tags:
        if ifd in exif_dict:
            for tag in essential_tags[ifd]:
                if tag not in exif_dict[ifd]:
                    tag_name = get_tag_name(ifd, tag)
                    results["missing_essential_tags"].append(tag_name)
    
    # Extract key information
    if "0th" in exif_dict:
        # Software information
        if piexif.ImageIFD.Software in exif_dict["0th"]:
            software = exif_dict["0th"][piexif.ImageIFD.Software]
            if isinstance(software, bytes):
                software = software.decode('utf-8', errors='replace')
            results["software_used"] = software
            
            # Check for known editing software
            editing_software_patterns = ["photoshop", "gimp", "lightroom", "affinity", "pixlr", "paintshop"]
            for pattern in editing_software_patterns:
                if pattern.lower() in software.lower():
                    results["suspicious_tags"].append(f"Edited with {software}")
        
        # Camera make and model
        if piexif.ImageIFD.Make in exif_dict["0th"]:
            make = exif_dict["0th"][piexif.ImageIFD.Make]
            if isinstance(make, bytes):
                make = make.decode('utf-8', errors='replace')
            results["camera_make"] = make
        
        if piexif.ImageIFD.Model in exif_dict["0th"]:
            model = exif_dict["0th"][piexif.ImageIFD.Model]
            if isinstance(model, bytes):
                model = model.decode('utf-8', errors='replace')
            results["camera_model"] = model
    
    # Extract timestamps
    if "Exif" in exif_dict:
        # Original creation time
        if piexif.ExifIFD.DateTimeOriginal in exif_dict["Exif"]:
            datetime_original = exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal]
            if isinstance(datetime_original, bytes):
                datetime_original = datetime_original.decode('utf-8', errors='replace')
            results["creation_time"] = datetime_original
        
        # Digitization time
        if piexif.ExifIFD.DateTimeDigitized in exif_dict["Exif"]:
            datetime_digitized = exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized]
            if isinstance(datetime_digitized, bytes):
                datetime_digitized = datetime_digitized.decode('utf-8', errors='replace')
            results["modification_time"] = datetime_digitized
    
    # Check timestamp inconsistencies
    if results["creation_time"] and results["modification_time"]:
        try:
            # Parse timestamps
            creation_dt = datetime.datetime.strptime(results["creation_time"], "%Y:%m:%d %H:%M:%S")
            modification_dt = datetime.datetime.strptime(results["modification_time"], "%Y:%m:%d %H:%M:%S")
            
            # Check if modification time is before creation time
            if modification_dt < creation_dt:
                results["inconsistencies"].append("Modification time is earlier than creation time")
            
            # Check if timestamps are in the future
            current_time = datetime.datetime.now()
            if creation_dt > current_time:
                results["inconsistencies"].append("Creation time is in the future")
            if modification_dt > current_time:
                results["inconsistencies"].append("Modification time is in the future")
        except:
            results["inconsistencies"].append("Invalid timestamp format")
    
    # Check for thumbnail inconsistencies
    if "thumbnail" in exif_dict and exif_dict["thumbnail"]:
        try:
            # Load the thumbnail image
            thumb = Image.open(io.BytesIO(exif_dict["thumbnail"]))
            
            # Check if thumbnail dimensions match the aspect ratio of the main image
            main_aspect_ratio = image.width / image.height
            thumb_aspect_ratio = thumb.width / thumb.height
            
            # Allow for small rounding differences
            if abs(main_aspect_ratio - thumb_aspect_ratio) > 0.1:
                results["inconsistencies"].append("Thumbnail has different aspect ratio than the main image")
        except:
            results["inconsistencies"].append("Invalid thumbnail data")
    
    # Check GPS data if available
    if "GPS" in exif_dict and exif_dict["GPS"]:
        gps_tags = [piexif.GPSIFD.GPSLatitude, piexif.GPSIFD.GPSLongitude]
        
        # Check for incomplete GPS data
        missing_gps = False
        for tag in gps_tags:
            if tag not in exif_dict["GPS"]:
                missing_gps = True
        
        if missing_gps and len(exif_dict["GPS"]) > 0:
            results["suspicious_tags"].append("Incomplete GPS metadata")
    
    # Look for unusual or modified tags
    if "0th" in exif_dict:
        for tag, value in exif_dict["0th"].items():
            if tag not in piexif.ImageIFD.__dict__.values():
                results["suspicious_tags"].append(f"Unknown tag in IFD0: {tag}")
    
    if "Exif" in exif_dict:
        for tag, value in exif_dict["Exif"].items():
            if tag not in piexif.ExifIFD.__dict__.values():
                results["suspicious_tags"].append(f"Unknown tag in EXIF: {tag}")
    
    # Calculate suspicion score based on findings
    suspicion_score = 0.0
    
    # No EXIF is highly suspicious
    if not results["has_exif"]:
        suspicion_score = 0.7
    else:
        # Missing essential tags
        suspicion_score += len(results["missing_essential_tags"]) * 0.1
        
        # Inconsistencies are very suspicious
        suspicion_score += len(results["inconsistencies"]) * 0.15
        
        # Suspicious tags
        suspicion_score += len(results["suspicious_tags"]) * 0.1
        
        # Evidence of editing software
        if results["software_used"] and any(pattern in results["software_used"].lower() 
                                           for pattern in ["photoshop", "gimp", "lightroom"]):
            suspicion_score += 0.2
    
    # Cap the confidence score
    confidence = min(0.95, suspicion_score)
    
    # Add a base confidence level - even clean images might have some suspicion
    confidence = max(0.05, confidence)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Metadata tampering analysis completed in {elapsed_time:.2f} seconds with confidence {confidence:.2f}")
    
    return results, confidence

def get_tag_name(ifd, tag):
    """Get the human-readable name of an EXIF tag"""
    if ifd == "0th":
        for key, value in piexif.ImageIFD.__dict__.items():
            if value == tag and not key.startswith("_"):
                return key
    elif ifd == "Exif":
        for key, value in piexif.ExifIFD.__dict__.items():
            if value == tag and not key.startswith("_"):
                return key
    elif ifd == "GPS":
        for key, value in piexif.GPSIFD.__dict__.items():
            if value == tag and not key.startswith("_"):
                return key
    return f"Unknown ({tag})"

def generate_metadata_report(image):
    """
    Generate a comprehensive report of image metadata and tampering analysis
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        Tuple[dict, float]: Detailed report and confidence score
    """
    start_time = time.time()
    logger.info("Generating comprehensive metadata report")
    
    # Get the tampering analysis
    tampering_results, confidence = check_metadata_tampering(image)
    
    # Create a comprehensive report
    report = {
        "tampering_analysis": tampering_results,
        "confidence": confidence,
        "metadata": extract_all_metadata(image),
        "analysis_summary": ""
    }
    
    # Generate a summary based on the analysis
    if confidence > 0.7:
        report["analysis_summary"] = "High probability of metadata tampering detected."
        if tampering_results["inconsistencies"]:
            report["analysis_summary"] += f" Found {len(tampering_results['inconsistencies'])} inconsistencies."
    elif confidence > 0.4:
        report["analysis_summary"] = "Moderate signs of metadata tampering or manipulation."
    else:
        report["analysis_summary"] = "Low probability of metadata tampering."
    
    elapsed_time = time.time() - start_time
    logger.info(f"Metadata report generation completed in {elapsed_time:.2f} seconds")
    
    return report, confidence

def extract_all_metadata(image):
    """
    Extract all available metadata from an image
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        dict: Dictionary of all metadata
    """
    metadata = {
        "basic": {
            "format": image.format,
            "mode": image.mode,
            "size": {"width": image.width, "height": image.height}
        },
        "exif": {},
        "iptc": {},
        "xmp": {}
    }
    
    # Extract EXIF data
    try:
        exif_dict = piexif.load(image.info.get("exif", b""))
        
        # Process each IFD
        for ifd in exif_dict:
            if ifd == "thumbnail":
                metadata["exif"]["thumbnail"] = "Present" if exif_dict["thumbnail"] else "Not present"
                continue
                
            if ifd not in metadata["exif"]:
                metadata["exif"][ifd] = {}
            
            for tag, value in exif_dict[ifd].items():
                tag_name = get_tag_name(ifd, tag)
                
                # Handle different value types
                if isinstance(value, bytes):
                    try:
                        value = value.decode('utf-8', errors='replace')
                    except:
                        value = str(value)
                elif isinstance(value, tuple) and ifd == "GPS":
                    # Convert GPS coordinates to a more readable format
                    if tag in [piexif.GPSIFD.GPSLatitude, piexif.GPSIFD.GPSLongitude]:
                        try:
                            value = f"{value[0][0]/value[0][1]}° {value[1][0]/value[1][1]}' {value[2][0]/value[2][1]}\""
                        except:
                            pass
                
                metadata["exif"][ifd][tag_name] = value
    except:
        metadata["exif"] = {"error": "No valid EXIF data found"}
    
    # Extract other metadata from image.info
    for key, value in image.info.items():
        if key.lower() not in ["exif"]:
            try:
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='replace')
                metadata["other"] = metadata.get("other", {})
                metadata["other"][key] = value
            except:
                pass
    
    return metadata 