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
  FaExclamationTriangle,
  FaCopy,
  FaCut as FaScissors,
  FaEraser,
  FaFileImage
} from "react-icons/fa";
import AnalysisResult from "../components/ui/AnalysisResult";
import {
  analyzeElaImage,
  detectCopyMove,
  detectSplicing,
  detectInpainting,
  analyzeMetadata,
  comprehensiveAnalysis,
  ForgeryType,
  API_BASE_URL
} from "../services/api";
import useImageUpload from "../hooks/useImageUpload";
import ThreeDModel from "../components/3D_Model/3DModel";

const Detect: React.FC = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedMethod, setSelectedMethod] = useState<ForgeryType>("comprehensive");
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [resultImageUrl, setResultImageUrl] = useState<string | null>(null);
  
  // Method-specific options
  const [copyMoveMethod, setCopyMoveMethod] = useState("orb");
  const [splicingMethod, setSplicingMethod] = useState("combined");
  const [inpaintingMethod, setInpaintingMethod] = useState("combined");
  const [metadataDetailed, setMetadataDetailed] = useState(false);
  
  // ELA options
  const [elaQuality, setElaQuality] = useState(85);
  
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
  
  const handleAnalysis = async () => {
    if (!file) {
      setError("Please select an image first.");
      return;
    }

    setError(null);
    setIsProcessing(true);

    try {
      let analysisResult: any = null;
      let imageUrl = null;
      let elaImageUrl = null;

      switch (selectedMethod) {
        case "copy-move":
          const copyMoveResult = await detectCopyMove(file, copyMoveMethod);
          analysisResult = copyMoveResult.result;
          imageUrl = copyMoveResult.imageUrl;
          elaImageUrl = analysisResult.ela_image_url;
          break;
        
        case "splicing":
          const splicingResult = await detectSplicing(file, splicingMethod);
          analysisResult = splicingResult.result;
          imageUrl = splicingResult.imageUrl;
          elaImageUrl = analysisResult.ela_image_url;
          break;
        
        case "inpainting":
          const inpaintingResult = await detectInpainting(file, inpaintingMethod);
          analysisResult = inpaintingResult.result;
          imageUrl = inpaintingResult.imageUrl;
          elaImageUrl = analysisResult.ela_image_url;
          break;
        
        case "metadata":
          analysisResult = await analyzeMetadata(file, metadataDetailed);
          break;
        
        case "comprehensive":
        default:
          const compResult = await comprehensiveAnalysis(file);
          analysisResult = compResult;
          elaImageUrl = compResult.ela_image_url;
          break;
      }

      // Format the result to include all necessary fields
      const formattedResult = {
        filename: file.name,
        prediction: analysisResult.prediction === "tampered" ? 1 : 0,
        prediction_label: analysisResult.prediction,
        confidence: analysisResult.confidence || analysisResult.overall_confidence,
        method: selectedMethod,
        ela_image_url: elaImageUrl,
        processing_time: analysisResult.processing_time || 0
      };

      setResult(formattedResult);
      if (imageUrl) {
        setResultImageUrl(imageUrl);
      }

      console.log("Analysis result:", formattedResult);
    } catch (err: any) {
      console.error("Analysis error:", err);
      setError(`An error occurred during analysis: ${err.message || "Unknown error"}`);
    } finally {
      setIsProcessing(false);
    }
  };
  
  const handleReset = () => {
    clearImage();
    setResult(null);
    setError(null);
    setResultImageUrl(null);
  };

  const renderMethodIcon = (method: ForgeryType) => {
    switch (method) {
      case "copy-move": return <FaCopy />;
      case "splicing": return <FaScissors />;
      case "inpainting": return <FaEraser />;
      case "metadata": return <FaFileImage />;
      case "comprehensive": return <FaLayerGroup />;
      default: return <FaEye />;
    }
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
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                    {/* Analysis Type Selection */}
                    <div className="md:col-span-2 bg-white/5 backdrop-blur-md border border-white/10 rounded-xl p-4 hover:bg-white/10 transition-colors duration-300">
                      <h3 className="text-gray-200 font-medium mb-3 flex items-center">
                        <FaLayerGroup className="mr-2 text-blue-400" /> Detection Method
                      </h3>
                      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2">
                        {[
                          { id: "comprehensive", label: "All Methods" },
                          { id: "copy-move", label: "Copy-Move" },
                          { id: "splicing", label: "Splicing" },
                          { id: "inpainting", label: "Inpainting" },
                          { id: "metadata", label: "Metadata" },
                        ].map((method) => (
                          <button
                            key={method.id}
                            onClick={() => setSelectedMethod(method.id as ForgeryType)}
                            className={`p-2 rounded-lg text-center text-sm flex flex-col items-center justify-center transition-all duration-300 ${
                              selectedMethod === method.id
                                ? "bg-blue-600 text-white"
                                : "bg-white/5 text-gray-300 hover:bg-white/10"
                            }`}
                          >
                            <span className="text-xl mb-1">
                              {renderMethodIcon(method.id as ForgeryType)}
                            </span>
                            <span>{method.label}</span>
                          </button>
                        ))}
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
                        <div className="mt-4 space-y-4">
                          {selectedMethod === "copy-move" && (
                            <div>
                              <label className="block text-gray-400 text-sm mb-1">
                                Algorithm
                              </label>
                              <select
                                value={copyMoveMethod}
                                onChange={(e) => setCopyMoveMethod(e.target.value)}
                                className="w-full bg-black/30 border border-gray-700 rounded-md py-2 px-3 text-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                              >
                                <option value="orb">ORB Keypoints</option>
                                <option value="dct">DCT Blocks</option>
                              </select>
                            </div>
                          )}

                          {selectedMethod === "splicing" && (
                            <div>
                              <label className="block text-gray-400 text-sm mb-1">
                                Algorithm
                              </label>
                              <select
                                value={splicingMethod}
                                onChange={(e) => setSplicingMethod(e.target.value)}
                                className="w-full bg-black/30 border border-gray-700 rounded-md py-2 px-3 text-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                              >
                                <option value="edge">Edge Inconsistency</option>
                                <option value="lighting">Lighting Analysis</option>
                                <option value="combined">Combined</option>
                              </select>
                            </div>
                          )}

                          {selectedMethod === "inpainting" && (
                            <div>
                              <label className="block text-gray-400 text-sm mb-1">
                                Algorithm
                              </label>
                              <select
                                value={inpaintingMethod}
                                onChange={(e) => setInpaintingMethod(e.target.value)}
                                className="w-full bg-black/30 border border-gray-700 rounded-md py-2 px-3 text-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                              >
                                <option value="texture">Texture Analysis</option>
                                <option value="noise">Noise Analysis</option>
                                <option value="combined">Combined</option>
                              </select>
                            </div>
                          )}

                          {selectedMethod === "metadata" && (
                            <div>
                              <label className="inline-flex items-center text-gray-300 hover:text-white cursor-pointer">
                                <input
                                  type="checkbox"
                                  className="form-checkbox rounded text-blue-500 focus:ring-blue-500 focus:ring-opacity-50"
                                  checked={metadataDetailed}
                                  onChange={(e) => setMetadataDetailed(e.target.checked)}
                                />
                                <span className="ml-2">Detailed Analysis</span>
                              </label>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Analysis Button */}
                  <div className="flex flex-col sm:flex-row gap-4">
                    <motion.button
                      onClick={handleAnalysis}
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
                          {renderMethodIcon(selectedMethod)}
                          <span className="ml-2">
                            Analyze with {selectedMethod === "comprehensive" ? "All Methods" : selectedMethod}
                          </span>
                        </>
                      )}
                    </motion.button>
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
                  {/* Results Display */}
                  <AnalysisResult 
                    result={result}
                    apiBaseUrl={API_BASE_URL}
                    originalImage={preview || undefined}
                    onReset={handleReset}
                    showLocalization={true}
                    showEla={true}
                  />
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
