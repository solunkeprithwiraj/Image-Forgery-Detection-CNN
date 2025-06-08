import io
import uuid
from PIL import Image
from src.api.utils.logger import logger

class ImageStorage:
    """
    Class for in-memory image storage to avoid using temporary files
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageStorage, cls).__new__(cls)
            cls._instance._images = {}
            cls._instance._max_cache_size = 100  # Max number of images to keep in cache
        return cls._instance
    
    def store_image(self, image_data, filename=None):
        """
        Store an image in memory
        
        :param image_data: The image data (bytes, file-like object, or PIL.Image)
        :param filename: Optional filename to associate with the image
        :return: A unique ID for the stored image
        """
        try:
            # Generate a unique ID for the image
            image_id = str(uuid.uuid4())
            
            # Convert image_data to PIL.Image if it's not already
            if isinstance(image_data, bytes):
                img = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, io.BytesIO):
                img = Image.open(image_data)
            elif hasattr(image_data, 'read'):
                # If it's a file-like object
                img = Image.open(image_data)
            elif isinstance(image_data, Image.Image):
                img = image_data
            else:
                raise ValueError(f"Unsupported image data type: {type(image_data)}")
            
            # Store the image and metadata
            self._images[image_id] = {
                'image': img,
                'filename': filename or f"{image_id}.jpg",
                'timestamp': uuid.uuid1().time
            }
            
            # Clean up if we have too many images
            if len(self._images) > self._max_cache_size:
                self._cleanup()
            
            logger.debug(f"Stored image with ID: {image_id}")
            return image_id
        
        except Exception as e:
            logger.error(f"Error storing image: {str(e)}")
            raise
    
    def get_image(self, image_id):
        """
        Retrieve an image from storage
        
        :param image_id: The ID of the image to retrieve
        :return: The PIL.Image object
        :raises: KeyError if the image_id is not found
        """
        if image_id not in self._images:
            raise KeyError(f"Image with ID {image_id} not found in storage")
        
        return self._images[image_id]['image']
    
    def get_image_as_bytes(self, image_id, format='JPEG', **save_kwargs):
        """
        Retrieve an image as bytes
        
        :param image_id: The ID of the image to retrieve
        :param format: The format to save the image as (e.g., 'JPEG', 'PNG')
        :param save_kwargs: Additional arguments to pass to Image.save()
        :return: The image as bytes
        :raises: KeyError if the image_id is not found
        """
        img = self.get_image(image_id)
        
        # Convert the image to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format=format, **save_kwargs)
        img_bytes.seek(0)
        
        return img_bytes.getvalue()
    
    def delete_image(self, image_id):
        """
        Delete an image from storage
        
        :param image_id: The ID of the image to delete
        :return: True if the image was deleted, False if it wasn't found
        """
        if image_id in self._images:
            del self._images[image_id]
            logger.debug(f"Deleted image with ID: {image_id}")
            return True
        
        logger.debug(f"Image with ID {image_id} not found for deletion")
        return False
    
    def _cleanup(self):
        """
        Clean up old images to prevent memory leaks
        """
        # Sort images by timestamp (oldest first)
        sorted_images = sorted(
            self._images.items(),
            key=lambda x: x[1]['timestamp']
        )
        
        # Remove the oldest third of images
        images_to_remove = len(sorted_images) // 3
        for i in range(images_to_remove):
            image_id = sorted_images[i][0]
            del self._images[image_id]
        
        logger.debug(f"Cleaned up {images_to_remove} old images from storage")

# Create a singleton instance
image_storage = ImageStorage() 