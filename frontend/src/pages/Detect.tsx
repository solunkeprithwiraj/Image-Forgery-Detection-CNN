import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  FaInfoCircle,
  FaUpload,
  FaImage,
  FaTimes,
  FaLayerGroup,
  FaSearchLocation,
  FaBrain,
  FaHeatmap,
} from "react-icons/fa";
import AnalysisResult from "../components/ui/AnalysisResult";
import {
  analyzeElaImage,
  analyzeImage,
  analyzeImageEnsemble,
  generateForgeryHeatmap,
  AnalysisResult as ApiAnalysisResult,
  LocalizationMethod,
} from "../services/api";
import useImageUpload from "../hooks/useImageUpload";

const Detect: React.FC = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<ApiAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showLocalization, setShowLocalization] = useState(true);
  const [showEla, setShowEla] = useState(true);
  const [useEnsemble, setUseEnsemble] = useState(true);
  const [heatmapUrl, setHeatmapUrl] = useState<string | null>(null);

  const {
    file,
    preview,
    clearImage,
    getRootProps,
    getInputProps,
    isDragActive,
    isDragAccept,
    isDragReject,
    error: uploadError,
  } = useImageUpload({
    maxSizeInMB: 10,
    onImageSelected: () => {
      // Clear any previous errors when a new image is selected
      setError(null);
    },
  });
  const handlePredict = async () => {
    if (!file) {
      setError("Please select an image first.");
      return;
    }

    setError(null);
    setIsProcessing(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://localhost:8000/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Prediction failed");
      }

      const data = await response.json();

      setResult({
        filename: file.name,
        prediction: data.prediction,
        prediction_label: data.prediction_label,
        confidence: data.confidence,
        processing_time: data.processing_time,
        ela_image_url: null, // Not used in prediction
      });

      console.log("Prediction result:", data);
    } catch (err) {
      console.error("Prediction error:", err);
      setError("An error occurred during prediction. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };
  
  const handleElaAnalysis = async () => {
    if (!file) {
      setError("Please select an image first.");
      return;
    }

    setError(null);
    setIsProcessing(true);

    try {
      const elaImageUrl = await analyzeElaImage(file);

      setResult({
        filename: file.name,
        prediction: 0,
        prediction_label: "Unknown",
        confidence: 0,
        processing_time: 0,
        ela_image_url: elaImageUrl,
      });

      console.log("ELA Image URL:", elaImageUrl);
    } catch (err) {
      console.error("ELA analysis failed:", err);
      setError("An error occurred during ELA analysis. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };
  
  const handleHeatmapGeneration = async () => {
    if (!file) {
      setError("Please select an image first.");
      return;
    }

    setError(null);
    setIsProcessing(true);

    try {
      const heatmapImageUrl = await generateForgeryHeatmap(file);

      setResult({
        filename: file.name,
        prediction: 0,
        prediction_label: "Heatmap Analysis",
        confidence: 0,
        processing_time: 0,
        heatmap_path: heatmapImageUrl,
      });

      console.log("Heatmap Image URL:", heatmapImageUrl);
    } catch (err) {
      console.error("Heatmap generation failed:", err);
      setError("An error occurred during heatmap generation. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };
  
  const handleReset = () => {
    clearImage();
    setResult(null);
    setError(null);
    setHeatmapUrl(null);
  };

  return (
    <div className="bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 min-h-screen py-12 relative overflow-hidden">
      {/* Animated gradient orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-gradient-to-r from-cyan-400/20 to-blue-600/20 rounded-full blur-3xl animate-pulse z-0"></div>
      <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-gradient-to-r from-purple-400/20 to-pink-600/20 rounded-full blur-3xl animate-pulse animation-delay-1000 z-0"></div>
      <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-gradient-to-r from-emerald-400/15 to-teal-600/15 rounded-full blur-3xl animate-bounce z-0"></div>

      {/* Dark overlay for better text readability */}
      <div className="absolute inset-0 bg-black/40 z-0"></div>

      <div className="container mx-auto px-4 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="max-w-5xl mx-auto"
        >
          <h1 className="text-3xl md:text-4xl font-bold text-center mb-4 text-white">
            Image Forgery Detection
          </h1>
          <p className="text-xl text-center text-gray-300 mb-12 max-w-3xl mx-auto">
            Upload an image to analyze it for potential manipulation or forgery
            using our advanced CNN model.
          </p>

          <div className="bg-white/10 backdrop-blur-xl rounded-xl shadow-lg overflow-hidden border border-white/20 hover:border-white/30 transition-all duration-300">
            <div className="p-6 md:p-8">
              {!result ? (
                <>
                  <div className="mb-8 p-4 bg-blue-900/30 backdrop-blur-sm rounded-lg border border-blue-500/30">
                    <div className="flex items-start">
                      <FaInfoCircle className="text-blue-400 mt-1 mr-3 flex-shrink-0" />
                      <div>
                        <h4 className="font-medium text-blue-300 mb-1">
                          Supported Image Formats
                        </h4>
                        <p className="text-blue-400 text-sm">
                          You can upload images in JPG, JPEG, PNG, and BMP
                          formats. Maximum file size is 10MB.
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="mb-6">
                    <h3 className="text-lg font-medium text-white mb-3">
                      Detection Options
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="flex items-center">
                        <input
                          type="checkbox"
                          id="showLocalization"
                          checked={showLocalization}
                          onChange={(e) =>
                            setShowLocalization(e.target.checked)
                          }
                          className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                        />
                        <label
                          htmlFor="showLocalization"
                          className="ml-2 block text-sm text-gray-300"
                        >
                          <div className="flex items-center">
                            <FaSearchLocation className="mr-1" />
                            Localize Tampering
                          </div>
                        </label>
                      </div>

                      <div className="flex items-center">
                        <input
                          type="checkbox"
                          id="showEla"
                          checked={showEla}
                          onChange={(e) => setShowEla(e.target.checked)}
                          className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                        />
                        <label
                          htmlFor="showEla"
                          className="ml-2 block text-sm text-gray-300"
                        >
                          <div className="flex items-center">
                            <FaLayerGroup className="mr-1" />
                            Error Level Analysis
                          </div>
                        </label>
                      </div>
                    </div>

                    {/* {useEnsemble && (
                      <div className="mt-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                        <div className="flex items-start">
                          <FaBrain className="text-primary-500 dark:text-primary-400 mt-1 mr-2 flex-shrink-0" />
                          <div className="text-sm text-gray-600 dark:text-gray-300">
                            Ensemble mode uses multiple AI models to analyze your
                            image, providing higher accuracy and better detection
                            of various tampering techniques.
                          </div>
                        </div>
                      </div>
                    )} */}
                  </div>

                  <div className="w-full">
                    {!preview ? (
                      <div
                        {...getRootProps()}
                        className={`
                          w-full border-2 border-dashed rounded-lg p-8 transition-colors
                          ${
                            isDragActive
                              ? "bg-primary-900/30 border-primary-400"
                              : "border-gray-300/50"
                          }
                          ${
                            isDragAccept
                              ? "bg-green-900/30 border-green-400"
                              : ""
                          }
                          ${
                            isDragReject || uploadError
                              ? "bg-red-900/30 border-red-400"
                              : ""
                          }
                          ${
                            isProcessing
                              ? "pointer-events-none opacity-60"
                              : "cursor-pointer hover:bg-white/5"
                          }
                        `}
                      >
                        <input {...getInputProps()} />
                        <div className="flex flex-col items-center justify-center text-center">
                          <FaUpload className="text-4xl text-gray-300 mb-4" />
                          <p className="text-lg text-gray-300 mb-2">
                            {isDragActive
                              ? "Drop the image here..."
                              : "Drag & drop an image here, or click to select"}
                          </p>
                          <p className="text-sm text-gray-400">
                            JPG, JPEG, PNG, BMP (max 10MB)
                          </p>
                          {uploadError && (
                            <p className="mt-4 text-red-400 text-sm">
                              {uploadError}
                            </p>
                          )}
                        </div>
                      </div>
                    ) : (
                      <div className="relative">
                        <div className="relative rounded-lg overflow-hidden border border-white/20 shadow-xl">
                          <img
                            src={preview}
                            alt="Preview"
                            className="w-full h-auto max-h-[500px] object-contain bg-black/50 backdrop-blur-sm"
                          />
                          <button
                            onClick={handleReset}
                            className="absolute top-2 right-2 bg-red-500/80 hover:bg-red-600/80 text-white rounded-full p-2 transition-colors backdrop-blur-sm"
                            disabled={isProcessing}
                          >
                            <FaTimes />
                          </button>
                        </div>

                        <div className="mt-6 flex justify-center">
                          <div className="flex flex-wrap gap-3 mt-6">
                            <button
                              onClick={handlePredict}
                              disabled={isProcessing || !file}
                              className="flex-1 min-w-[120px] bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-medium py-2 px-4 rounded-lg flex items-center justify-center transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                              {isProcessing ? (
                                <span className="flex items-center">
                                  <svg
                                    className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                                    xmlns="http://www.w3.org/2000/svg"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                  >
                                    <circle
                                      className="opacity-25"
                                      cx="12"
                                      cy="12"
                                      r="10"
                                      stroke="currentColor"
                                      strokeWidth="4"
                                    ></circle>
                                    <path
                                      className="opacity-75"
                                      fill="currentColor"
                                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                    ></path>
                                  </svg>
                                  Processing...
                                </span>
                              ) : (
                                <>
                                  <FaImage className="mr-2" /> Analyze Image
                                </>
                              )}
                            </button>

                            <button
                              onClick={handleElaAnalysis}
                              disabled={isProcessing || !file}
                              className="flex-1 min-w-[120px] bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white font-medium py-2 px-4 rounded-lg flex items-center justify-center transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                              {isProcessing ? (
                                <span className="flex items-center">
                                  <svg
                                    className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                                    xmlns="http://www.w3.org/2000/svg"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                  >
                                    <circle
                                      className="opacity-25"
                                      cx="12"
                                      cy="12"
                                      r="10"
                                      stroke="currentColor"
                                      strokeWidth="4"
                                    ></circle>
                                    <path
                                      className="opacity-75"
                                      fill="currentColor"
                                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                    ></path>
                                  </svg>
                                  Processing...
                                </span>
                              ) : (
                                <>
                                  <FaLayerGroup className="mr-2" /> ELA Analysis
                                </>
                              )}
                            </button>
                            
                            <button
                              onClick={handleHeatmapGeneration}
                              disabled={isProcessing || !file}
                              className="flex-1 min-w-[120px] bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 text-white font-medium py-2 px-4 rounded-lg flex items-center justify-center transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                              {isProcessing ? (
                                <span className="flex items-center">
                                  <svg
                                    className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                                    xmlns="http://www.w3.org/2000/svg"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                  >
                                    <circle
                                      className="opacity-25"
                                      cx="12"
                                      cy="12"
                                      r="10"
                                      stroke="currentColor"
                                      strokeWidth="4"
                                    ></circle>
                                    <path
                                      className="opacity-75"
                                      fill="currentColor"
                                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                    ></path>
                                  </svg>
                                  Processing...
                                </span>
                              ) : (
                                <>
                                  <FaHeatmap className="mr-2" /> Generate Heatmap
                                </>
                              )}
                            </button>
                          </div>
                        </div>
                        {error && (
                          <div className="mt-4 p-4 bg-red-900/30 backdrop-blur-sm text-red-300 rounded-lg border border-red-500/30 text-center">
                            {error}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <AnalysisResult
                  result={result}
                  originalImage={preview || ""}
                  onReset={handleReset}
                  showLocalization={showLocalization}
                  showEla={showEla}
                />
              )}
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Detect;
