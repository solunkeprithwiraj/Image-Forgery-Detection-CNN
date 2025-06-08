import React, { useState, useEffect } from "react";
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
  FaWaveSquare,
  FaRegLightbulb,
} from "react-icons/fa";
import dayjs from "dayjs";

interface AnalysisResult {
  is_tampered?: boolean;
  prediction: number | string;
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
  results?: {
    copy_move?: {
      confidence: number;
      prediction: string;
      detected_regions: any[];
    };
    splicing?: {
      confidence: number;
      prediction: string;
    };
    inpainting?: {
      confidence: number;
      prediction: string;
    };
    metadata?: {
      confidence: number;
      prediction: string;
      analysis: any;
    };
    cnn_direct?: {
      confidence: number;
      prediction: string;
      processing_time: number;
    };
    noise_analysis?: {
      confidence: number;
      prediction: string;
      visualization_url: string | null;
      detected_regions: any[];
    };
    frequency_analysis?: {
      confidence: number;
      prediction: string;
      visualization_url: string | null;
      detected_regions: any[];
    };
  };
  most_likely_forgery_type?: string | null;
  overall_confidence?: number;
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
    noise: false,
    frequency: false,
  });
  
  const [activeView, setActiveView] = useState<'split' | 'original' | 'analysis'>('split');
  const [imageZoomed, setImageZoomed] = useState(false);
  const [activeAnalysis, setActiveAnalysis] = useState<'ela' | 'noise' | 'frequency'>('ela');

  // Get the ELA image URL (prefer ela_image_url over ela_path)
  const elaImageUrl = result.ela_image_url || result.ela_path;

  
  // Get noise analysis image URL
  const noiseImageUrl = result.results?.noise_analysis?.visualization_url || null;
  
  // Get frequency analysis image URL
  const frequencyImageUrl = result.results?.frequency_analysis?.visualization_url || null;

  // Get original image URL
  const originalImageUrl = originalImage || (result.input_image_path
    ? result.input_image_path.startsWith("blob:")
      ? result.input_image_path
      : `${apiBaseUrl}${result.input_image_path.startsWith("/") ? "" : "/"}${
          result.input_image_path
        }`
    : null);


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
  
  // Get the analysis image URL based on active analysis
  const getAnalysisImageUrl = () => {
    switch (activeAnalysis) {
      case 'ela': 
        return elaImageUrl;
      case 'noise':
        return noiseImageUrl;
      case 'frequency':
        return frequencyImageUrl;
      default:
        return elaImageUrl || noiseImageUrl || frequencyImageUrl;
    }
  };
  
  const analysisImageUrl = getAnalysisImageUrl();
  
  // Helper to determine which analysis is being shown
  const getAnalysisType = () => {
    switch (activeAnalysis) {
      case 'ela': 
        return "Error Level Analysis";
      case 'noise':
        return "Noise Pattern Analysis";
      case 'frequency':
        return "Frequency Domain Analysis";
      default:
        return "Analysis";
    }
  };
  
  const analysisType = getAnalysisType();
  
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
  

  // Auto switch to split view if analysis image is available
  useEffect(() => {
    if (analysisImageUrl) {
      setActiveView('split');
    }
    
    // Set initial active analysis based on available visualizations
    if (elaImageUrl) {
      setActiveAnalysis('ela');
    } else if (noiseImageUrl) {
      setActiveAnalysis('noise');
    } else if (frequencyImageUrl) {
      setActiveAnalysis('frequency');
    }
  }, [elaImageUrl, noiseImageUrl, frequencyImageUrl]);
  
  // Determine if we have comprehensive results
  
  // Determine which prediction to show
  const finalPrediction = typeof result.prediction === 'string' 
    ? result.prediction === 'tampered' ? 1 : 0
    : result.prediction;
    
  // Use overall confidence if available, otherwise use the confidence directly
  const finalConfidence = result.overall_confidence !== undefined 
    ? result.overall_confidence 
    : result.confidence;

  return (
    <motion.div
      className="bg-white/10 backdrop-blur-xl rounded-2xl shadow-2xl p-6 mb-8 border border-white/20"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* Result Header with Glassmorphism */}
      <motion.div className="flex items-center mb-6" variants={itemVariants}>
        {finalPrediction === 1 || result.prediction_label === "tampered" ? (
          <div className="flex items-center text-red-500 dark:text-red-400">
            <FaExclamationTriangle className="text-3xl mr-3" />
            <h2 className="text-2xl font-bold text-white">Manipulation Detected</h2>
          </div>
        ) : (
          <div className="flex items-center text-green-500 dark:text-green-400">
            <FaCheckCircle className="text-3xl mr-3" />
            <h2 className="text-2xl font-bold text-white">Image Appears Authentic</h2>
          </div>
        )}
      </motion.div>

      {/* Confidence Bar with Enhanced Styling */}
      {finalConfidence > 0 && (
        <motion.div className="mb-6" variants={itemVariants}>
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium text-gray-300">
              Confidence
            </span>
            <span className="text-sm font-medium text-gray-300">
              {Math.round(finalConfidence * 100)}%
            </span>
          </div>
          <div className="w-full bg-gray-700/50 backdrop-blur-sm rounded-full h-2.5 overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.round(finalConfidence * 100)}%` }}
              transition={{ duration: 0.8, ease: "easeOut" }}
              className={`h-2.5 rounded-full ${
                finalPrediction === 1 || result.prediction_label === "tampered"
                  ? "bg-gradient-to-r from-red-500 to-orange-500"
                  : "bg-gradient-to-r from-green-400 to-emerald-500"
              }`}
            ></motion.div>
          </div>
        </motion.div>
      )}

      {/* Most Likely Forgery Type */}
      {result.most_likely_forgery_type && finalPrediction === 1 && (
        <motion.div className="mb-6 p-3 bg-black/30 rounded-lg" variants={itemVariants}>
          <div className="flex items-start">
            <FaInfoCircle className="text-blue-400 mt-1 mr-2 flex-shrink-0" />
            <div>
              <p className="text-gray-200 font-medium">Detected Forgery Type</p>
              <p className="text-blue-300">
                {result.most_likely_forgery_type.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </p>
            </div>
          </div>
        </motion.div>
      )}

      {/* Analysis Selection */}
      {(elaImageUrl || noiseImageUrl || frequencyImageUrl) && (
        <motion.div className="mb-4 flex items-center justify-between" variants={itemVariants}>
          <div className="flex items-center space-x-3">
            {elaImageUrl && (
              <button 
                onClick={() => setActiveAnalysis('ela')}
                className={`px-3 py-1.5 rounded-lg text-sm flex items-center ${
                  activeAnalysis === 'ela' 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-black/20 text-gray-300 hover:bg-black/30'
                }`}
              >
                <FaRegLightbulb className="mr-1.5" />
                ELA
              </button>
            )}
            
            {noiseImageUrl && (
              <button 
                onClick={() => setActiveAnalysis('noise')}
                className={`px-3 py-1.5 rounded-lg text-sm flex items-center ${
                  activeAnalysis === 'noise' 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-black/20 text-gray-300 hover:bg-black/30'
                }`}
              >
                <FaWaveSquare className="mr-1.5" />
                Noise
              </button>
            )}
            
            {frequencyImageUrl && (
              <button 
                onClick={() => setActiveAnalysis('frequency')}
                className={`px-3 py-1.5 rounded-lg text-sm flex items-center ${
                  activeAnalysis === 'frequency' 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-black/20 text-gray-300 hover:bg-black/30'
                }`}
              >
                <FaWaveSquare className="mr-1.5" />
                Frequency
              </button>
            )}
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={toggleViewMode}
              className="p-2 bg-black/30 hover:bg-black/40 rounded-full text-gray-300 hover:text-white transition-colors"
              title={`Switch to ${
                activeView === 'split'
                  ? 'original only'
                  : activeView === 'original'
                  ? 'analysis only'
                  : 'split view'
              }`}
            >
              {activeView === 'split' ? (
                <FaChevronLeft />
              ) : activeView === 'original' ? (
                <FaChevronRight />
              ) : (
                <FaSync />
              )}
            </button>
            
            <button
              onClick={() => setImageZoomed(!imageZoomed)}
              className="p-2 bg-black/30 hover:bg-black/40 rounded-full text-gray-300 hover:text-white transition-colors"
              title={imageZoomed ? 'Zoom out' : 'Zoom in'}
            >
              {imageZoomed ? <FaCompressAlt /> : <FaExpandAlt />}
            </button>
          </div>
        </motion.div>
      )}

      {/* Image Comparison Section with Glassmorphism */}
      {analysisImageUrl && (
        <motion.div 
          className={`mb-6 overflow-hidden rounded-xl ${
            imageZoomed ? 'max-h-full' : 'max-h-96'
          }`} 
          variants={itemVariants}
        >
          <div className="relative">
            <div 
              className={`w-full ${
                imageZoomed ? 'h-auto' : 'h-96'
              } bg-black/50 backdrop-blur-sm flex overflow-hidden`}
            >
              {activeView === 'split' && originalImageUrl && (
                <>
                  <div className="w-1/2 h-full overflow-hidden relative">
                    <img
                      src={originalImageUrl}
                      alt="Original"
                      className="w-full h-full object-contain"
                      onError={() => setImageLoadError({ ...imageLoadError, original: true })}
                    />
                    <div className="absolute bottom-2 left-2 text-xs bg-black/50 text-white px-2 py-1 rounded">
                      Original
                    </div>
                  </div>
                  <div className="w-1/2 h-full overflow-hidden relative">
                    <img
                      src={analysisImageUrl}
                      alt={analysisType}
                      className="w-full h-full object-contain"
                      onError={() => {
                        if (activeAnalysis === 'ela') {
                          setImageLoadError({ ...imageLoadError, ela: true });
                        } else if (activeAnalysis === 'noise') {
                          setImageLoadError({ ...imageLoadError, noise: true });
                        } else if (activeAnalysis === 'frequency') {
                          setImageLoadError({ ...imageLoadError, frequency: true });
                        }
                      }}
                    />
                    <div className="absolute bottom-2 left-2 text-xs bg-black/50 text-white px-2 py-1 rounded">
                      {analysisType}
                    </div>
                  </div>
                </>
              )}
              
              {activeView === 'original' && originalImageUrl && (
                <div className="w-full h-full overflow-hidden relative">
                  <img
                    src={originalImageUrl}
                    alt="Original"
                    className="w-full h-full object-contain"
                    onError={() => setImageLoadError({ ...imageLoadError, original: true })}
                  />
                  <div className="absolute bottom-2 left-2 text-xs bg-black/50 text-white px-2 py-1 rounded">
                    Original
                  </div>
                </div>
              )}
              
              {activeView === 'analysis' && analysisImageUrl && (
                <div className="w-full h-full overflow-hidden relative">
                  <img
                    src={analysisImageUrl}
                    alt={analysisType}
                    className="w-full h-full object-contain"
                    onError={() => {
                      if (activeAnalysis === 'ela') {
                        setImageLoadError({ ...imageLoadError, ela: true });
                      } else if (activeAnalysis === 'noise') {
                        setImageLoadError({ ...imageLoadError, noise: true });
                      } else if (activeAnalysis === 'frequency') {
                        setImageLoadError({ ...imageLoadError, frequency: true });
                      }
                    }}
                  />
                  <div className="absolute bottom-2 left-2 text-xs bg-black/50 text-white px-2 py-1 rounded">
                    {analysisType}
                  </div>
                </div>
              )}
              
              {/* Error states */}
              {(imageLoadError.original && activeView !== 'analysis') && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/30 text-white">
                  <div className="text-center p-4">
                    <FaExclamationTriangle className="mx-auto mb-2 text-yellow-500" />
                    <p>Failed to load original image</p>
                  </div>
                </div>
              )}
              
              {((activeAnalysis === 'ela' && imageLoadError.ela) ||
                 (activeAnalysis === 'noise' && imageLoadError.noise) ||
                 (activeAnalysis === 'frequency' && imageLoadError.frequency)) && 
                activeView !== 'original' && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/30 text-white">
                  <div className="text-center p-4">
                    <FaExclamationTriangle className="mx-auto mb-2 text-yellow-500" />
                    <p>Failed to load {analysisType} image</p>
                  </div>
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

      {/* Reset Button */}
      <motion.div 
        variants={itemVariants} 
        className="mt-6 flex justify-center"
      >
        <button
          onClick={onReset}
          className="py-3 px-6 bg-white/10 backdrop-blur-md hover:bg-white/20 text-white rounded-xl font-medium flex items-center justify-center transition-all duration-300"
        >
          Analyze Another Image
        </button>
      </motion.div>
    </motion.div>
  );
};

export default AnalysisResult;
