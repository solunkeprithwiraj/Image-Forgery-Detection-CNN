import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  FaInfoCircle,
  FaUpload,
  FaImage,
  FaTimes,
  FaLayerGroup,
  FaSearchLocation,
  FaChevronDown,
  FaChevronUp,
  FaEye,
  FaPalette,
  FaMagic,
  FaRegObjectGroup,
  FaCrosshairs,
  FaExchangeAlt,
  FaExclamationTriangle
} from "react-icons/fa";
import AnalysisResult from "../components/ui/AnalysisResult";
import {
  analyzeElaImage,
  analyzeImage,
  analyzeImageEnsemble,
  generateForgeryHeatmap,
  AnalysisResult as ApiAnalysisResult,
  LocalizationMethod,
  ELAMode,
  HeatmapMode
} from "../services/api";
import useImageUpload from "../hooks/useImageUpload";
import ThreeDModel from "../components/3D_Model/3DModel";

const Detect: React.FC = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<ApiAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showLocalization, setShowLocalization] = useState(true);
  const [showEla, setShowEla] = useState(true);
  const [useEnsemble, setUseEnsemble] = useState(true);
  const [heatmapUrl, setHeatmapUrl] = useState<string | null>(null);
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  
  // Advanced ELA options
  const [elaMode, setElaMode] = useState<ELAMode>("enhanced");
  const [elaQuality, setElaQuality] = useState(85);
  const [elaEnhanceContrast, setElaEnhanceContrast] = useState(true);
  const [elaColorize, setElaColorize] = useState(true);
  
  // Advanced heatmap options
  const [heatmapMode, setHeatmapMode] = useState<HeatmapMode>("basic");
  const [heatmapThreshold, setHeatmapThreshold] = useState(0.5);
  const [heatmapColormap, setHeatmapColormap] = useState("jet");
  
  // Show 3D model
  const [showModel, setShowModel] = useState(false);

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
      const elaImageUrl = await analyzeElaImage(
        file,
        elaMode,
        elaQuality,
        elaEnhanceContrast,
        elaColorize
      );

      setResult({
        filename: file.name,
        prediction: 0,
        prediction_label: `ELA Analysis (${elaMode})`,
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
      const heatmapImageUrl = await generateForgeryHeatmap(
        file,
        heatmapMode,
        heatmapThreshold,
        heatmapColormap
      );

      setResult({
        filename: file.name,
        prediction: 0,
        prediction_label: `Heatmap Analysis (${heatmapMode})`,
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
    <div className="relative min-h-screen py-12 overflow-hidden">
      {/* 3D Background - Always visible */}
      <div className="absolute inset-0 z-0">
        <ThreeDModel />
      </div>

      {/* Dark overlay for better text readability */}
      <div className="absolute inset-0 bg-black/50 z-10"></div>

      {/* Animated gradient orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-gradient-to-r from-cyan-400/20 to-blue-600/20 rounded-full blur-3xl animate-pulse z-20"></div>
      <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-gradient-to-r from-purple-400/20 to-pink-600/20 rounded-full blur-3xl animate-pulse animation-delay-1000 z-20"></div>
      <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-gradient-to-r from-emerald-400/15 to-teal-600/15 rounded-full blur-3xl animate-bounce z-20"></div>

      <div className="container mx-auto px-4 relative z-30">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="max-w-5xl mx-auto"
        >
          <h1 className="text-3xl md:text-5xl font-bold text-center mb-4 text-white">
            <span className="bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
              Image Forgery Detection
            </span>
          </h1>
          <p className="text-xl text-center text-gray-300 mb-8 max-w-3xl mx-auto">
            Upload an image to analyze it for potential manipulation or forgery
            using our advanced CNN model.
          </p>

          <div className="bg-white/10 backdrop-blur-xl rounded-2xl shadow-2xl overflow-hidden border border-white/20 hover:border-white/30 transition-all duration-300">
            <div className="p-6 md:p-8">
              {!result ? (
                <>
                  <div className="mb-8 p-6 bg-blue-900/30 backdrop-blur-sm rounded-xl border border-blue-500/30">
                    <div className="flex items-start">
                      <FaInfoCircle className="text-blue-400 text-xl mt-0.5 mr-3 flex-shrink-0" />
                      <div>
                        <h3 className="text-xl font-medium text-blue-300 mb-2">
                          How It Works
                        </h3>
                        <p className="text-gray-300">
                          Our AI-powered system analyzes your image using a convolutional neural network
                          trained on thousands of authentic and manipulated images. The model identifies
                          telltale signs of forgery that are often invisible to the human eye.
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Upload Area */}
                  <div className="mb-8">
                    <div
                      {...getRootProps()}
                      className={`bg-black/30 backdrop-blur-md border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200 ${
                        isDragActive
                          ? "border-blue-500 bg-blue-500/10"
                          : "border-gray-500 hover:border-blue-400 hover:bg-blue-900/20"
                      } ${isDragReject ? "border-red-500 bg-red-500/10" : ""}`}
                    >
                      <input {...getInputProps()} />
                      <div className="flex flex-col items-center justify-center py-4">
                        {!preview ? (
                          <>
                            <FaUpload className="text-blue-400 text-4xl mb-4" />
                            <p className="text-xl font-medium text-gray-300 mb-2">
                              Drag & drop an image here, or click to select
                            </p>
                            <p className="text-gray-400 text-sm">
                              Supports JPG, PNG, BMP, TIFF (Max: 10MB)
                            </p>
                          </>
                        ) : (
                          <div className="relative">
                            <img
                              src={preview}
                              alt="Preview"
                              className="max-h-64 max-w-full rounded-lg shadow-lg"
                            />
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                clearImage();
                              }}
                              className="absolute -top-3 -right-3 bg-red-500 hover:bg-red-600 text-white rounded-full p-1 shadow-md transition-colors"
                            >
                              <FaTimes />
                            </button>
                          </div>
                        )}
                      </div>
                    </div>

                    {uploadError && (
                      <div className="mt-2 text-red-500 text-sm">
                        {uploadError}
                      </div>
                    )}
                  </div>

                  {/* Analysis Options */}
                  <div className="flex flex-col gap-4 mb-4">
                    {/* Analysis Type Selection */}
                    <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-xl p-4 hover:bg-white/10 transition-colors duration-300">
                      <h3 className="text-gray-200 font-medium mb-3 flex items-center">
                        <FaLayerGroup className="mr-2 text-blue-400" /> Analysis Type
                      </h3>
                      <div className="flex flex-col space-y-2">
                        <label className="inline-flex items-center text-gray-300 hover:text-white cursor-pointer">
                          <input
                            type="checkbox"
                            className="form-checkbox rounded text-blue-500 focus:ring-blue-500 focus:ring-opacity-50"
                            checked={useEnsemble}
                            onChange={(e) => setUseEnsemble(e.target.checked)}
                          />
                          <span className="ml-2">Use Ensemble Model</span>
                        </label>

                        <label className="inline-flex items-center text-gray-300 hover:text-white cursor-pointer">
                          <input
                            type="checkbox"
                            className="form-checkbox rounded text-blue-500 focus:ring-blue-500 focus:ring-opacity-50"
                            checked={showLocalization}
                            onChange={(e) => setShowLocalization(e.target.checked)}
                          />
                          <span className="ml-2">Show Forgery Heatmap</span>
                        </label>

                        <label className="inline-flex items-center text-gray-300 hover:text-white cursor-pointer">
                          <input
                            type="checkbox"
                            className="form-checkbox rounded text-blue-500 focus:ring-blue-500 focus:ring-opacity-50"
                            checked={showEla}
                            onChange={(e) => setShowEla(e.target.checked)}
                          />
                          <span className="ml-2">Error Level Analysis</span>
                        </label>
                      </div>
                    </div>

                    {/* Advanced Options Button */}
                    <div className="md:col-span-2 bg-white/5 backdrop-blur-md border border-white/10 rounded-xl p-4 hover:bg-white/10 transition-colors duration-300">
                      <div className="flex justify-between items-center mb-3">
                        <h3 className="text-gray-200 font-medium flex items-center">
                          <FaSearchLocation className="mr-2 text-blue-400" /> Advanced Analysis Options
                        </h3>
                        <button
                          onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
                          className="text-blue-400 hover:text-blue-300 transition-colors"
                        >
                          {showAdvancedOptions ? <FaChevronUp /> : <FaChevronDown />}
                        </button>
                      </div>

                      {showAdvancedOptions && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                          {/* ELA Options */}
                          <div>
                            <h4 className="text-gray-300 font-medium mb-2 flex items-center">
                              <FaEye className="mr-2 text-blue-400" /> ELA Options
                            </h4>
                            <div className="space-y-3">
                              <div>
                                <label className="block text-gray-400 text-sm mb-1">
                                  Mode
                                </label>
                                <select
                                  value={elaMode}
                                  onChange={(e) => setElaMode(e.target.value as ELAMode)}
                                  className="w-full bg-black/30 border border-gray-700 rounded-md py-2 px-3 text-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                >
                                  <option value="basic">Basic</option>
                                  <option value="enhanced">Enhanced</option>
                                  <option value="comparison">Comparison</option>
                                  <option value="zoom">Zoom on Suspicious</option>
                                </select>
                              </div>
                              <div>
                                <label className="block text-gray-400 text-sm mb-1">
                                  Quality ({elaQuality})
                                </label>
                                <input
                                  type="range"
                                  min="50"
                                  max="95"
                                  value={elaQuality}
                                  onChange={(e) => setElaQuality(parseInt(e.target.value))}
                                  className="w-full accent-blue-500"
                                />
                              </div>
                              <div className="flex justify-between">
                                <label className="inline-flex items-center text-gray-300 hover:text-white cursor-pointer">
                                  <input
                                    type="checkbox"
                                    className="form-checkbox rounded text-blue-500 focus:ring-blue-500 focus:ring-opacity-50"
                                    checked={elaEnhanceContrast}
                                    onChange={(e) => setElaEnhanceContrast(e.target.checked)}
                                  />
                                  <span className="ml-2">Enhance Contrast</span>
                                </label>
                                <label className="inline-flex items-center text-gray-300 hover:text-white cursor-pointer">
                                  <input
                                    type="checkbox"
                                    className="form-checkbox rounded text-blue-500 focus:ring-blue-500 focus:ring-opacity-50"
                                    checked={elaColorize}
                                    onChange={(e) => setElaColorize(e.target.checked)}
                                  />
                                  <span className="ml-2">Colorize</span>
                                </label>
                              </div>
                            </div>
                          </div>

                          {/* Heatmap Options */}
                          <div>
                            <h4 className="text-gray-300 font-medium mb-2 flex items-center">
                              <FaPalette className="mr-2 text-blue-400" /> Heatmap Options
                            </h4>
                            <div className="space-y-3">
                              <div>
                                <label className="block text-gray-400 text-sm mb-1">
                                  Mode
                                </label>
                                <select
                                  value={heatmapMode}
                                  onChange={(e) => setHeatmapMode(e.target.value as HeatmapMode)}
                                  className="w-full bg-black/30 border border-gray-700 rounded-md py-2 px-3 text-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                >
                                  <option value="basic">Basic</option>
                                  <option value="detail">Detail View</option>
                                  <option value="multi">Multi-Colormap</option>
                                  <option value="composite">Composite</option>
                                </select>
                              </div>
                              <div>
                                <label className="block text-gray-400 text-sm mb-1">
                                  Threshold ({heatmapThreshold.toFixed(2)})
                                </label>
                                <input
                                  type="range"
                                  min="0.2"
                                  max="0.8"
                                  step="0.01"
                                  value={heatmapThreshold}
                                  onChange={(e) => setHeatmapThreshold(parseFloat(e.target.value))}
                                  className="w-full accent-blue-500"
                                />
                              </div>
                              <div>
                                <label className="block text-gray-400 text-sm mb-1">
                                  Colormap
                                </label>
                                <select
                                  value={heatmapColormap}
                                  onChange={(e) => setHeatmapColormap(e.target.value)}
                                  className="w-full bg-black/30 border border-gray-700 rounded-md py-2 px-3 text-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                >
                                  <option value="jet">Jet</option>
                                  <option value="viridis">Viridis</option>
                                  <option value="plasma">Plasma</option>
                                  <option value="inferno">Inferno</option>
                                  <option value="rainbow">Rainbow</option>
                                  <option value="hot">Hot</option>
                                </select>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Analysis Buttons */}
                  <div className="flex flex-col sm:flex-row gap-4">
                    <motion.button
                      onClick={handlePredict}
                      disabled={!file || isProcessing}
                      className={`flex-1 py-3 px-6 rounded-xl font-medium flex items-center justify-center ${
                        !file || isProcessing
                          ? "bg-gray-700 text-gray-400 cursor-not-allowed"
                          : "bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white shadow-lg shadow-blue-500/20 transform hover:scale-105 transition-all duration-300"
                      }`}
                      whileHover={file && !isProcessing ? { scale: 1.05 } : {}}
                      whileTap={file && !isProcessing ? { scale: 0.98 } : {}}
                    >
                      {isProcessing ? (
                        <>
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
                        </>
                      ) : (
                        <>
                          <FaRegObjectGroup className="mr-2" />
                          Analyze with CNN
                        </>
                      )}
                    </motion.button>

                    {showEla && (
                      <motion.button
                        onClick={handleElaAnalysis}
                        disabled={!file || isProcessing}
                        className={`flex-1 py-3 px-6 rounded-xl font-medium flex items-center justify-center ${
                          !file || isProcessing
                            ? "bg-gray-700 text-gray-400 cursor-not-allowed"
                            : "bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 text-white shadow-lg shadow-purple-500/20 transform hover:scale-105 transition-all duration-300"
                        }`}
                        whileHover={file && !isProcessing ? { scale: 1.05 } : {}}
                        whileTap={file && !isProcessing ? { scale: 0.98 } : {}}
                      >
                        <FaMagic className="mr-2" />
                        Error Level Analysis
                      </motion.button>
                    )}

                    {showLocalization && (
                      <motion.button
                        onClick={handleHeatmapGeneration}
                        disabled={!file || isProcessing}
                        className={`flex-1 py-3 px-6 rounded-xl font-medium flex items-center justify-center ${
                          !file || isProcessing
                            ? "bg-gray-700 text-gray-400 cursor-not-allowed"
                            : "bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 text-white shadow-lg shadow-emerald-500/20 transform hover:scale-105 transition-all duration-300"
                        }`}
                        whileHover={file && !isProcessing ? { scale: 1.05 } : {}}
                        whileTap={file && !isProcessing ? { scale: 0.98 } : {}}
                      >
                        <FaCrosshairs className="mr-2" />
                        Generate Heatmap
                      </motion.button>
                    )}
                  </div>

                  {error && (
                    <div className="mt-4 p-3 bg-red-900/50 backdrop-blur-sm border border-red-800 rounded-lg text-red-300">
                      <div className="flex items-center">
                        <FaExclamationTriangle className="mr-2 flex-shrink-0" />
                        <span>{error}</span>
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <>
                  <AnalysisResult
                    result={result}
                    apiBaseUrl="http://localhost:8000"
                    originalImage={preview || undefined}
                    onReset={handleReset}
                    showLocalization={showLocalization}
                    showEla={showEla}
                  />

                  <div className="flex justify-center mt-6">
                    <motion.button
                      onClick={handleReset}
                      className="py-3 px-6 bg-white/10 backdrop-blur-md hover:bg-white/20 text-white rounded-xl font-medium flex items-center justify-center transition-all duration-300"
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <FaExchangeAlt className="mr-2" />
                      Analyze Another Image
                    </motion.button>
                  </div>
                </>
              )}
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Detect;
