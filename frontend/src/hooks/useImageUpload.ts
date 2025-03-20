import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";

interface UseImageUploadProps {
  maxSizeInMB?: number;
  acceptedFileTypes?: string[];
  onImageSelected?: (file: File) => void;
}

export const useImageUpload = ({
  maxSizeInMB = 10,
  acceptedFileTypes = ["image/jpeg", "image/png", "image/gif", "image/tiff"],
  onImageSelected,
}: UseImageUploadProps = {}) => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const maxSizeInBytes = maxSizeInMB * 1024 * 1024;

  const onDrop = useCallback(
    (acceptedFiles: File[], rejectedFiles: any[]) => {
      // Clear previous errors
      setError(null);

      // Handle rejected files
      if (rejectedFiles && rejectedFiles.length > 0) {
        const { errors } = rejectedFiles[0];
        if (errors[0]?.code === "file-too-large") {
          setError(`File is too large. Max size is ${maxSizeInMB}MB.`);
        } else if (errors[0]?.code === "file-invalid-type") {
          setError("File type is not supported. Please upload an image file.");
        } else {
          setError("Error uploading file. Please try again.");
        }
        return;
      }

      // Handle accepted files
      if (acceptedFiles && acceptedFiles.length > 0) {
        const selectedFile = acceptedFiles[0];

        // Check file size manually (extra validation)
        if (selectedFile.size > maxSizeInBytes) {
          setError(`File is too large. Max size is ${maxSizeInMB}MB.`);
          return;
        }

        // Set file and create preview
        setFile(selectedFile);

        // Create preview URL
        const objectUrl = URL.createObjectURL(selectedFile);
        setPreview(objectUrl);

        // Call the callback if provided
        if (onImageSelected) {
          onImageSelected(selectedFile);
        }
      }
    },
    [maxSizeInBytes, maxSizeInMB, onImageSelected]
  );

  // Configure react-dropzone
  const {
    getRootProps,
    getInputProps,
    isDragActive,
    isDragAccept,
    isDragReject,
  } = useDropzone({
    onDrop,
    accept: {
      "image/*": acceptedFileTypes.map((type) => type.replace("image/", ".")),
    },
    maxSize: maxSizeInBytes,
    multiple: false,
  });

  // Function to clear the selected image
  const clearImage = useCallback(() => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    setFile(null);
    setPreview(null);
    setError(null);
  }, [preview]);

  return {
    file,
    preview,
    error,
    clearImage,
    getRootProps,
    getInputProps,
    isDragActive,
    isDragAccept,
    isDragReject,
  };
};

export default useImageUpload;
