import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  FaCheckCircle,
  FaExclamationTriangle,
  FaInfoCircle,
  FaBrain,
  FaDownload,
  FaEye,
  FaCompressAlt,
  FaExpandAlt,
  FaSync,
  FaChevronLeft,
  FaChevronRight,
} from "react-icons/fa";
import dayjs from "dayjs";

interface AnalysisResult {
  is_tampered?: boolean;
  prediction: number;
  prediction_label: string;
  confidence: number;
  processing_time?: number;
  message?: string;
  method?: string;
  timestamp?: string;
  input_image_path?: string;
  filename: string;
  ela_path?: string;
  ela_image_url?: string;
  heatmap_path?: string;
  ensemble_detail?: {
    ensemble_size: number;
    tampered_votes: number;
    authentic_votes: number;
    consensus_level: string;
    model_predictions: Array<{
      model_name: string;
      prediction: number;
      confidence: number;
    }>;
  };
}

interface AnalysisResultProps {
  result: AnalysisResult;
  apiBaseUrl: string;
  originalImage?: string;
  onReset: () => void;
  showLocalization?: boolean;
  showEla?: boolean;
}

const AnalysisResult: React.FC<AnalysisResultProps> = ({
  result,
  apiBaseUrl,
  originalImage,
  onReset,
  showLocalization = true,
  showEla = true,
}) => {
  const [imageLoadError, setImageLoadError] = useState({
    original: false,
    ela: false,
    heatmap: false,
  });
  
  const [activeView, setActiveView] = useState<'split' | 'original' | 'analysis'>('split');
  const [imageZoomed, setImageZoomed] = useState(false);

  // Get the ELA image URL (prefer ela_image_url over ela_path)
  const elaImageUrl = result.ela_image_url || result.ela_path;
  
  // Get the heatmap image URL
  const heatmapImageUrl = result.heatmap_path;

  // Get original image URL
  const originalImageUrl = originalImage || (result.input_image_path
    ? result.input_image_path.startsWith("blob:")
      ? result.input_image_path
      : `${apiBaseUrl}${result.input_image_path.startsWith("/") ? "" : "/"}${
          result.input_image_path
        }`
    : null);

  // Function to download image
  const downloadImage = async (imageUrl: string, filename: string) => {
    try {
      const response = await fetch(imageUrl);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error("Error downloading image:", error);
      alert("Failed to download image");
    }
  };

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        when: "beforeChildren",
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { type: "spring", stiffness: 100 },
    },
  };
  
  // Get the analysis image URL (ELA or Heatmap)
  const analysisImageUrl = elaImageUrl || heatmapImageUrl;
  
  // Helper to determine which analysis is being shown
  const analysisType = elaImageUrl ? "ELA" : heatmapImageUrl ? "Heatmap" : "";
  
  // Toggle view mode
  const toggleViewMode = () => {
    if (activeView === 'split') {
      setActiveView('original');
    } else if (activeView === 'original') {
      setActiveView('analysis');
    } else {
      setActiveView('split');
    }
  };

  return (
    <motion.div
      className="bg-white/10 backdrop-blur-xl rounded-2xl shadow-2xl p-6 mb-8 border border-white/20"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* Result Header with Glassmorphism */}
      <motion.div className="flex items-center mb-6" variants={itemVariants}>
        {result.prediction == 1 ? (
          <div className="flex items-center text-red-500 dark:text-red-400">
            <FaExclamationTriangle className="text-3xl mr-3" />
            <h2 className="text-2xl font-bold text-white">Manipulation Detected</h2>
          </div>
        ) : elaImageUrl ? (
          <div className="flex items-center text-blue-500 dark:text-blue-400">
            <FaInfoCircle className="text-3xl mr-3" />
            <h2 className="text-2xl font-bold text-white">{result.prediction_label}</h2>
          </div>
        ) : heatmapImageUrl ? (
          <div className="flex items-center text-red-500 dark:text-red-400">
            <FaExclamationTriangle className="text-3xl mr-3" />
            <h2 className="text-2xl font-bold text-white">{result.prediction_label}</h2>
          </div>
        ) : (
          <div className="flex items-center text-green-500 dark:text-green-400">
            <FaCheckCircle className="text-3xl mr-3" />
            <h2 className="text-2xl font-bold text-white">Image Appears Authentic</h2>
          </div>
        )}
      </motion.div>

      {/* Confidence Bar with Enhanced Styling */}
      {result.confidence > 0 && result.prediction !== 0 && (
        <motion.div className="mb-6" variants={itemVariants}>
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
                result.is_tampered
                  ? "bg-gradient-to-r from-red-500 to-orange-500"
                  : "bg-gradient-to-r from-green-400 to-emerald-500"
              }`}
            ></motion.div>
          </div>
        </motion.div>
      )}

      {/* Image Comparison Section with Glassmorphism */}
      {(elaImageUrl || heatmapImageUrl) && (
        <motion.div className="mb-6" variants={itemVariants}>
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl font-medium text-gray-200">
              {activeView === 'split' ? 'Analysis Comparison' : 
               activeView === 'original' ? 'Original Image' : 
               `${analysisType} Analysis`}
            </h3>
            
            <div className="flex items-center gap-2">
              {/* View Controls */}
              <div className="flex bg-black/30 backdrop-blur-md rounded-lg p-1 border border-white/10">
                <button 
                  onClick={() => setActiveView('original')}
                  className={`px-2 py-1 rounded-md text-xs ${
                    activeView === 'original' ? 'bg-white/20 text-white' : 'text-gray-400 hover:text-white'
                  }`}
                  title="View Original"
                >
                  Original
                </button>
                <button 
                  onClick={() => setActiveView('split')}
                  className={`px-2 py-1 rounded-md text-xs ${
                    activeView === 'split' ? 'bg-white/20 text-white' : 'text-gray-400 hover:text-white'
                  }`}
                  title="View Side by Side"
                >
                  Split
                </button>
                <button 
                  onClick={() => setActiveView('analysis')}
                  className={`px-2 py-1 rounded-md text-xs ${
                    activeView === 'analysis' ? 'bg-white/20 text-white' : 'text-gray-400 hover:text-white'
                  }`}
                  title="View Analysis"
                >
                  {analysisType}
                </button>
              </div>
              
              {/* Zoom Toggle */}
              <button
                onClick={() => setImageZoomed(!imageZoomed)}
                className="p-1 bg-black/30 backdrop-blur-md rounded-lg border border-white/10 text-gray-300 hover:text-white"
                title={imageZoomed ? "Exit Fullscreen" : "Fullscreen View"}
              >
                {imageZoomed ? <FaCompressAlt size={14} /> : <FaExpandAlt size={14} />}
              </button>
              
              {/* Cycle Views */}
              <button
                onClick={toggleViewMode}
                className="p-1 bg-black/30 backdrop-blur-md rounded-lg border border-white/10 text-gray-300 hover:text-white"
                title="Cycle Views"
              >
                <FaSync size={14} />
              </button>
              
              {/* Download Button */}
              {analysisImageUrl && (
                <button
                  onClick={() => downloadImage(analysisImageUrl, `${analysisType.toLowerCase()}_${result.filename}`)}
                  className="p-1 bg-black/30 backdrop-blur-md rounded-lg border border-white/10 text-gray-300 hover:text-white"
                  title={`Download ${analysisType}`}
                >
                  <FaDownload size={14} />
                </button>
              )}
            </div>
          </div>
          
          {/* Image Container with Enhanced Glassmorphism */}
          <div 
            className={`
              relative overflow-hidden transition-all duration-300 ease-in-out
              ${imageZoomed ? 'fixed inset-0 z-50 p-4 bg-black/80 flex items-center justify-center' : 'rounded-xl bg-black/20 backdrop-blur-sm border border-white/10'}
            `}
          >
            {/* Close Button for Fullscreen */}
            {imageZoomed && (
              <button
                onClick={() => setImageZoomed(false)}
                className="absolute top-4 right-4 bg-black/50 text-white p-2 rounded-full z-10"
              >
                <FaCompressAlt />
              </button>
            )}
            
            {/* Image View */}
            <div className={`
              w-full h-full flex 
              ${activeView === 'split' ? 'flex-row' : 'flex-col'} 
              ${activeView === 'split' ? 'divide-x divide-white/20' : 'divide-y divide-white/20'} 
              overflow-hidden
            `}>
              {/* Original Image Section */}
              {(activeView === 'original' || activeView === 'split') && originalImageUrl && (
                <div className={`
                  relative 
                  ${activeView === 'split' ? 'w-1/2' : 'w-full'} 
                  ${activeView === 'split' ? 'h-full' : 'h-full'} 
                  bg-neutral-900/30 backdrop-blur-sm
                  overflow-hidden
                `}>
                  <div className="absolute top-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">
                    Original
                  </div>
                  <img 
                    src={originalImageUrl} 
                    alt="Original" 
                    className="w-full h-full object-contain"
                    onError={() => setImageLoadError({...imageLoadError, original: true})}
                  />
                  {imageLoadError.original && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/50 text-red-400">
                      Failed to load original image
                    </div>
                  )}
                </div>
              )}
              
              {/* Analysis Image Section */}
              {(activeView === 'analysis' || activeView === 'split') && analysisImageUrl && (
                <div className={`
                  relative 
                  ${activeView === 'split' ? 'w-1/2' : 'w-full'} 
                  ${activeView === 'split' ? 'h-full' : 'h-full'} 
                  bg-neutral-900/30 backdrop-blur-sm
                  overflow-hidden
                `}>
                  <div className="absolute top-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">
                    {analysisType} Analysis
                  </div>
                  <img 
                    src={analysisImageUrl} 
                    alt={`${analysisType} Analysis`} 
                    className="w-full h-full object-contain"
                    onError={() => setImageLoadError({
                      ...imageLoadError, 
                      ela: elaImageUrl ? true : false,
                      heatmap: heatmapImageUrl ? true : false
                    })}
                  />
                  {(elaImageUrl && imageLoadError.ela) || (heatmapImageUrl && imageLoadError.heatmap) ? (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/50 text-red-400">
                      Failed to load analysis image
                    </div>
                  ) : null}
                </div>
              )}
            </div>
          </div>
        </motion.div>
      )}

      {/* Analysis Details with Glassmorphism */}
      <motion.div variants={itemVariants} className="space-y-4">
        {/* Basic Details */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
          <div className="bg-white/5 backdrop-blur-md rounded-xl p-4 border border-white/10">
            <div className="text-sm text-gray-400 mb-1">Filename</div>
            <div className="text-gray-200 truncate">{result.filename}</div>
          </div>
          
          {result.processing_time && (
            <div className="bg-white/5 backdrop-blur-md rounded-xl p-4 border border-white/10">
              <div className="text-sm text-gray-400 mb-1">Processing Time</div>
              <div className="text-gray-200">{result.processing_time.toFixed(2)}s</div>
            </div>
          )}
          
          {result.timestamp && (
            <div className="bg-white/5 backdrop-blur-md rounded-xl p-4 border border-white/10">
              <div className="text-sm text-gray-400 mb-1">Analysis Time</div>
              <div className="text-gray-200">{dayjs(result.timestamp).format('YYYY-MM-DD HH:mm:ss')}</div>
            </div>
          )}
          
          {result.method && (
            <div className="bg-white/5 backdrop-blur-md rounded-xl p-4 border border-white/10">
              <div className="text-sm text-gray-400 mb-1">Detection Method</div>
              <div className="text-gray-200">{result.method}</div>
            </div>
          )}
        </div>
        
        {/* Ensemble Details if available */}
        {result.ensemble_detail && (
          <motion.div 
            variants={itemVariants}
            className="bg-white/5 backdrop-blur-md rounded-xl p-4 border border-white/10 mt-4"
          >
            <h3 className="text-lg font-medium text-gray-200 mb-3 flex items-center">
              <FaBrain className="mr-2 text-purple-400" /> Ensemble Model Details
            </h3>
            
            <div className="grid grid-cols-3 gap-3 mb-4">
              <div className="bg-black/30 backdrop-blur-md rounded-lg p-3 border border-white/10">
                <div className="text-sm text-gray-400">Models</div>
                <div className="text-xl font-medium text-white">{result.ensemble_detail.ensemble_size}</div>
              </div>
              
              <div className="bg-black/30 backdrop-blur-md rounded-lg p-3 border border-white/10">
                <div className="text-sm text-gray-400">Tampered Votes</div>
                <div className="text-xl font-medium text-red-400">{result.ensemble_detail.tampered_votes}</div>
              </div>
              
              <div className="bg-black/30 backdrop-blur-md rounded-lg p-3 border border-white/10">
                <div className="text-sm text-gray-400">Authentic Votes</div>
                <div className="text-xl font-medium text-green-400">{result.ensemble_detail.authentic_votes}</div>
              </div>
            </div>
            
            <div className="mb-4">
              <div className="text-sm text-gray-400 mb-1">Consensus Level</div>
              <div className={`text-lg font-medium ${
                result.ensemble_detail.consensus_level === 'Strong' ? 'text-green-400' :
                result.ensemble_detail.consensus_level === 'Moderate' ? 'text-yellow-400' :
                'text-red-400'
              }`}>
                {result.ensemble_detail.consensus_level} Consensus
              </div>
            </div>
            
            <div>
              <div className="text-sm text-gray-400 mb-2">Individual Model Predictions</div>
              <div className="space-y-2 max-h-40 overflow-y-auto pr-2 custom-scrollbar">
                {result.ensemble_detail.model_predictions.map((model, idx) => (
                  <div 
                    key={idx} 
                    className="bg-black/20 backdrop-blur-md rounded-lg p-2 border border-white/10 flex justify-between items-center"
                  >
                    <div className="text-gray-300 text-sm">{model.model_name}</div>
                    <div className="flex items-center">
                      <div className={`text-sm font-medium ${model.prediction === 1 ? 'text-red-400' : 'text-green-400'}`}>
                        {model.prediction === 1 ? 'Tampered' : 'Authentic'}
                      </div>
                      <div className="text-xs text-gray-400 ml-2">
                        {Math.round(model.confidence * 100)}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
        
        {/* Analysis Message if any */}
        {result.message && (
          <motion.div
            variants={itemVariants}
            className="bg-white/5 backdrop-blur-md rounded-xl p-4 border border-white/10 mt-4"
          >
            <h3 className="text-lg font-medium text-gray-200 mb-2 flex items-center">
              <FaInfoCircle className="mr-2 text-blue-400" /> Analysis Note
            </h3>
            <p className="text-gray-300">{result.message}</p>
          </motion.div>
        )}
      </motion.div>
    </motion.div>
  );
};

export default AnalysisResult;
