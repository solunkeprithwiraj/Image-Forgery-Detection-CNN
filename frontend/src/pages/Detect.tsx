import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  FaInfoCircle,
  FaUpload,
  FaTimes,
  FaLayerGroup,
  FaExclamationTriangle,
  FaCheckCircle,
  FaRegLightbulb,
  FaWaveSquare
} from "react-icons/fa";
import {
  comprehensiveAnalysis
} from "../services/api";
import useImageUpload from "../hooks/useImageUpload";
import ThreeDModel from "../components/3D_Model/3DModel";

// Define the tab types
type AnalysisTab = "ela" | "noise" | "frequency";

const Detect: React.FC = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [resultImageUrl, setResultImageUrl] = useState<string | null>(null);
  
  // Show 3D model
  
  // Track the active analysis tab
  const [activeTab, setActiveTab] = useState<AnalysisTab>("ela");

  const {
    file,
    preview,
    clearImage,
    getRootProps,
    getInputProps,
    isDragActive,
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
      // Always use comprehensive analysis
      const compResult = await comprehensiveAnalysis(file);
      
      // Format the result to include all necessary fields
      const formattedResult = {
        filename: file.name,
        prediction: compResult.prediction === "tampered" ? 1 : 0,
        prediction_label: compResult.prediction,
        confidence: compResult.overall_confidence || compResult.confidence || 0.5,
        method: "comprehensive",
        ela_image_url: compResult.ela_image_url,
        processing_time: compResult.processing_time || 0,
        results: compResult.results || {},
        most_likely_forgery_type: compResult.most_likely_forgery_type
      };

      setResult(formattedResult);
      
      // Debug log for visualizations
      console.log("Analysis result:", formattedResult);
      console.log("Noise Analysis URL:", formattedResult.results?.noise_analysis?.visualization_url);
      console.log("Frequency Analysis URL:", formattedResult.results?.frequency_analysis?.visualization_url);
      
      // Default to ela tab initially
      setActiveTab("ela");
      
      // If ELA is not available, try other tabs
      if (!formattedResult.ela_image_url) {
        if (formattedResult.results?.noise_analysis?.visualization_url) {
          setActiveTab("noise");
        } else if (formattedResult.results?.frequency_analysis?.visualization_url) {
          setActiveTab("frequency");
        }
      }
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
    setActiveTab("ela"); // Reset to default tab
  };

  // Function to safely change tabs with validation
  const changeTab = (newTab: AnalysisTab) => {
    console.log(`Attempting to change tab from ${activeTab} to ${newTab}`);
    
    // Validate if the tab is available before switching
    if (newTab === "noise" && !result?.results?.noise_analysis?.visualization_url) {
      console.log("Noise tab not available - visualization URL missing");
      return;
    }
    
    if (newTab === "frequency" && !result?.results?.frequency_analysis?.visualization_url) {
      console.log("Frequency tab not available - visualization URL missing");
      return;
    }
    
    if (newTab === "ela" && !result?.ela_image_url) {
      console.log("ELA tab not available - visualization URL missing");
      return;
    }
    
    console.log(`Tab changed to: ${newTab}`);
    setActiveTab(newTab);
  };

  // Function to render the content based on active tab
  const renderTabContent = () => {
    switch (activeTab) {
      case "noise":
        if (!result.results?.noise_analysis?.visualization_url) {
          // Fallback to ELA if noise analysis is not available
          console.log("Noise visualization URL missing, falling back to ELA");
          // Use setTimeout to avoid React state update during render
          setTimeout(() => changeTab("ela"), 0);
          return null;
        }
        return (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            <div className="flex flex-col md:flex-row gap-4">
              <div className="w-full md:w-1/2">
                <h3 className="text-lg font-medium text-white mb-2">Original Image</h3>
                <div className="bg-black/30 rounded-lg overflow-hidden aspect-square">
                  <img 
                    src={preview || ""} 
                    alt="Original" 
                    className="w-full h-full object-contain"
                  />
                </div>
              </div>
              <div className="w-full md:w-1/2">
                <h3 className="text-lg font-medium text-white mb-2">Noise Analysis</h3>
                <div className="bg-black/30 rounded-lg overflow-hidden aspect-square">
                  {result.results.noise_analysis.visualization_url ? (
                    <img 
                      src={result.results.noise_analysis.visualization_url} 
                      alt="Noise Analysis" 
                      className="w-full h-full object-contain"
                      onError={(e) => {
                        console.error("Error loading noise analysis image");
                        e.currentTarget.src = preview || "";  // Fallback to original image
                      }}
                    />
                  ) : (
                    <div className="flex items-center justify-center h-full text-gray-400">
                      Noise analysis visualization not available
                    </div>
                  )}
                </div>
              </div>
            </div>
            <div className="mt-4 p-3 bg-blue-900/20 backdrop-blur-sm rounded-lg text-blue-300">
              <p className="text-sm">
                <strong>Noise Analysis:</strong> This technique examines the image's noise patterns to detect inconsistencies that may indicate manipulation. Areas with inconsistent noise patterns often suggest forgery.
              </p>
            </div>
          </motion.div>
        );
      
      case "frequency":
        if (!result.results?.frequency_analysis?.visualization_url) {
          // Fallback to ELA if frequency analysis is not available
          console.log("Frequency visualization URL missing, falling back to ELA");
          // Use setTimeout to avoid React state update during render
          setTimeout(() => changeTab("ela"), 0);
          return null;
        }
        return (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            <div className="flex flex-col md:flex-row gap-4">
              <div className="w-full md:w-1/2">
                <h3 className="text-lg font-medium text-white mb-2">Original Image</h3>
                <div className="bg-black/30 rounded-lg overflow-hidden aspect-square">
                  <img 
                    src={preview || ""} 
                    alt="Original" 
                    className="w-full h-full object-contain"
                  />
                </div>
              </div>
              <div className="w-full md:w-1/2">
                <h3 className="text-lg font-medium text-white mb-2">Frequency Analysis</h3>
                <div className="bg-black/30 rounded-lg overflow-hidden aspect-square">
                  {result.results.frequency_analysis.visualization_url ? (
                    <img 
                      src={result.results.frequency_analysis.visualization_url} 
                      alt="Frequency Analysis" 
                      className="w-full h-full object-contain"
                      onError={(e) => {
                        console.error("Error loading frequency analysis image");
                        e.currentTarget.src = preview || "";  // Fallback to original image
                      }}
                    />
                  ) : (
                    <div className="flex items-center justify-center h-full text-gray-400">
                      Frequency analysis visualization not available
                    </div>
                  )}
                </div>
              </div>
            </div>
            <div className="mt-4 p-3 bg-blue-900/20 backdrop-blur-sm rounded-lg text-blue-300">
              <p className="text-sm">
                <strong>Frequency Analysis:</strong> This method analyzes the image in the frequency domain to detect anomalies that may indicate manipulation. Inconsistencies in the frequency spectrum can reveal tampering that's not visible in the spatial domain.
              </p>
            </div>
          </motion.div>
        );
      
      case "ela":
      default:
        if (!result.ela_image_url) {
          // Check if we should switch to another available tab
          if (result.results?.noise_analysis?.visualization_url) {
            console.log("ELA visualization URL missing, switching to noise tab");
            // Use setTimeout to avoid React state update during render
            setTimeout(() => changeTab("noise"), 0);
            return null;
          }
          if (result.results?.frequency_analysis?.visualization_url) {
            console.log("ELA visualization URL missing, switching to frequency tab");
            // Use setTimeout to avoid React state update during render
            setTimeout(() => changeTab("frequency"), 0);
            return null;
          }
        }
        return (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            <div className="flex flex-col md:flex-row gap-4">
              <div className="w-full md:w-1/2">
                <h3 className="text-lg font-medium text-white mb-2">Original Image</h3>
                <div className="bg-black/30 rounded-lg overflow-hidden aspect-square">
                  <img 
                    src={preview || ""} 
                    alt="Original" 
                    className="w-full h-full object-contain"
                  />
                </div>
              </div>
              <div className="w-full md:w-1/2">
                <h3 className="text-lg font-medium text-white mb-2">Error Level Analysis</h3>
                <div className="bg-black/30 rounded-lg overflow-hidden aspect-square">
                  {result.ela_image_url ? (
                    <img 
                      src={result.ela_image_url} 
                      alt="ELA" 
                      className="w-full h-full object-contain"
                      onError={(e) => {
                        console.error("Error loading ELA image");
                        e.currentTarget.src = preview || "";  // Fallback to original image
                      }}
                    />
                  ) : (
                    <div className="flex items-center justify-center h-full text-gray-400">
                      ELA visualization not available
                    </div>
                  )}
                </div>
              </div>
            </div>
            <div className="mt-4 p-3 bg-blue-900/20 backdrop-blur-sm rounded-lg text-blue-300">
              <p className="text-sm">
                <strong>Error Level Analysis (ELA):</strong> This technique identifies areas in the image that have different compression levels, which may indicate manipulation. Brighter areas in the ELA image often reveal edited regions.
              </p>
            </div>
          </motion.div>
        );
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
                          <FaLayerGroup className="mr-2" />
                          <span>Analyze Image</span>
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
                  {/* Results Display with Tab Cards */}
                  <div className="bg-white/10 backdrop-blur-xl rounded-2xl p-6 border border-white/20">
                    {/* Result Header */}
                    <div className="flex items-center mb-6">
                      {result.prediction === 1 || result.prediction_label === "tampered" ? (
                        <div className="flex items-center text-red-500">
                          <FaExclamationTriangle className="text-3xl mr-3" />
                          <h2 className="text-2xl font-bold text-white">Manipulation Detected</h2>
                        </div>
                      ) : (
                        <div className="flex items-center text-green-500">
                          <FaCheckCircle className="text-3xl mr-3" />
                          <h2 className="text-2xl font-bold text-white">Image Appears Authentic</h2>
                        </div>
                      )}
                    </div>
                    
                    {/* Confidence Bar */}
                    <div className="mb-6">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-medium text-gray-300">
                          Confidence
                        </span>
                        <span className="text-sm font-medium text-gray-300">
                          {Math.round(result.confidence * 100)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-700/50 backdrop-blur-sm rounded-full h-2.5 overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${Math.round(result.confidence * 100)}%` }}
                          transition={{ duration: 0.8, ease: "easeOut" }}
                          className={`h-2.5 rounded-full ${
                            result.prediction === 1 || result.prediction_label === "tampered"
                              ? "bg-gradient-to-r from-red-500 to-orange-500"
                              : "bg-gradient-to-r from-green-400 to-emerald-500"
                          }`}
                        ></motion.div>
                      </div>
                    </div>
                    
                    {/* Tab Cards for Different Analysis Types */}
                    <div className="space-y-4">
                      {/* Tabs Header - Card Style Navigation */}
                      <div className="flex flex-wrap gap-2 mb-4">
                        <button 
                          onClick={() => changeTab("ela")}
                          className={`cursor-pointer flex items-center gap-1 text-xs sm:gap-2 sm:text-sm px-4 py-2 rounded-lg text-white 
                            ${activeTab === "ela" 
                              ? "bg-blue-600 shadow-lg shadow-blue-500/30" 
                              : "bg-white/10 hover:bg-white/20"}`}
                          disabled={!result.ela_image_url}
                          title={!result.ela_image_url ? "ELA visualization not available" : ""}
                        >
                          <FaRegLightbulb className="h-3 w-3 sm:h-4 sm:w-4" /> 
                          ELA Analysis
                        </button>
                        
                        {result.results?.noise_analysis?.visualization_url && (
                          <button 
                            onClick={() => changeTab("noise")}
                            className={`cursor-pointer flex items-center gap-1 text-xs sm:gap-2 sm:text-sm px-4 py-2 rounded-lg text-white 
                              ${activeTab === "noise" 
                                ? "bg-blue-600 shadow-lg shadow-blue-500/30" 
                                : "bg-white/10 hover:bg-white/20"}`}
                          >
                            <FaWaveSquare className="h-3 w-3 sm:h-4 sm:w-4" /> 
                            Noise Analysis
                          </button>
                        )}
                        
                        {result.results?.frequency_analysis?.visualization_url && (
                          <button 
                            onClick={() => changeTab("frequency")}
                            className={`cursor-pointer flex items-center gap-1 text-xs sm:gap-2 sm:text-sm px-4 py-2 rounded-lg text-white 
                              ${activeTab === "frequency" 
                                ? "bg-blue-600 shadow-lg shadow-blue-500/30" 
                                : "bg-white/10 hover:bg-white/20"}`}
                          >
                            <FaWaveSquare className="h-3 w-3 sm:h-4 sm:w-4" /> 
                            Frequency Analysis
                          </button>
                        )}
                      </div>
                      
                      {/* Content Area for Tab Cards */}
                      <div className="bg-black/20 backdrop-blur-md rounded-xl p-4">
                        {/* Render content based on active tab */}
                        {renderTabContent()}
                      </div>
                      
                      {/* Analysis Summary */}
                      {result.results && Object.keys(result.results).length > 0 && (
                        <div className="mt-4 bg-black/20 backdrop-blur-md rounded-xl p-4">
                          <h3 className="text-lg font-medium text-white mb-3">Analysis Summary</h3>
                          
                          {/* Display voting summary if available */}
                          {result.voting_summary && (
                            <div className="mb-4 p-3 bg-blue-900/30 backdrop-blur-sm rounded-lg">
                              <h4 className="text-white font-medium mb-2">Voting Results</h4>
                              <div className="grid grid-cols-2 gap-2 mb-2">
                                <div className="bg-green-900/20 rounded p-2 flex items-center justify-between">
                                  <span className="text-green-300">Authentic Votes:</span>
                                  <span className="font-bold text-green-300">{result.voting_summary.authentic_votes}</span>
                                </div>
                                <div className="bg-red-900/20 rounded p-2 flex items-center justify-between">
                                  <span className="text-red-300">Tampered Votes:</span>
                                  <span className="font-bold text-red-300">{result.voting_summary.tampered_votes}</span>
                                </div>
                              </div>
                              <div className="text-xs text-gray-300 italic">
                                {result.voting_summary.majority === "tie_as_authentic" ? 
                                  "The votes were tied, so the image is considered authentic by default." :
                                  `Majority prediction: ${result.voting_summary.majority}`
                                }
                                {result.results?.cnn_direct?.confidence > 0.95 && 
                                  result.results.cnn_direct.prediction === "authentic" && 
                                  " (CNN prediction given extra weight due to high confidence)"}
                              </div>
                            </div>
                          )}
                          
                          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                            {Object.entries(result.results).map(([method, data]: [string, any]) => (
                              <div 
                                key={method}
                                className={`p-3 rounded-lg border ${
                                  data.prediction === "tampered" 
                                    ? "border-red-500/50 bg-red-900/20" 
                                    : "border-green-500/50 bg-green-900/20"
                                }`}
                              >
                                <div className="flex justify-between items-center mb-1">
                                  <span className="font-medium text-white capitalize">
                                    {method.replace("_", " ")}
                                  </span>
                                  <span className={`text-xs px-2 py-0.5 rounded-full ${
                                    data.prediction === "tampered"
                                      ? "bg-red-500/30 text-red-300"
                                      : "bg-green-500/30 text-green-300"
                                  }`}>
                                    {data.prediction}
                                  </span>
                                </div>
                                <div className="w-full bg-black/30 h-1.5 rounded-full overflow-hidden">
                                  <div 
                                    className={`h-1.5 rounded-full ${
                                      data.prediction === "tampered"
                                        ? "bg-red-500"
                                        : "bg-green-500"
                                    }`}
                                    style={{ width: `${Math.round(data.confidence * 100)}%` }}
                                  ></div>
                                </div>
                                <div className="mt-1 text-right text-xs text-gray-400">
                                  {Math.round(data.confidence * 100)}% confidence
                                </div>
                              </div>
                            ))}
                          </div>
                          
                          {result.most_likely_forgery_type && (
                            <div className="mt-4 p-3 bg-red-900/20 backdrop-blur-sm rounded-lg text-red-300">
                              <p className="text-sm">
                                <strong>Most Likely Forgery Type:</strong> {result.most_likely_forgery_type.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}
                              </p>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                    
                    {/* Action Buttons */}
                    <div className="flex justify-between mt-6">
                      <button
                        onClick={handleReset}
                        className="px-4 py-2 bg-gray-700/50 hover:bg-gray-700/70 text-white rounded-lg transition-colors"
                      >
                        Analyze Another Image
                      </button>
                    </div>
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
