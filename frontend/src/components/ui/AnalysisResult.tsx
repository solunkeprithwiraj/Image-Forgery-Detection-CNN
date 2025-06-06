import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  FaCheckCircle,
  FaExclamationTriangle,
  FaInfoCircle,
  FaBrain,
  FaDownload,
  FaEye,
  // FaHeatmap,
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
}

const AnalysisResult: React.FC<AnalysisResultProps> = ({
  result,
  apiBaseUrl,
  originalImage,
  onReset,
}) => {
  const [imageLoadError, setImageLoadError] = useState({
    original: false,
    ela: false,
    heatmap: false,
  });

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

  return (
    <motion.div
      className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 mb-8"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* Result Header */}
      <motion.div className="flex items-center mb-6" variants={itemVariants}>
        {result.prediction == 1 ? (
          <div className="flex items-center text-red-500 dark:text-red-400">
            <FaExclamationTriangle className="text-3xl mr-3" />
            <h2 className="text-2xl font-bold">Manipulation Detected</h2>
          </div>
        ) : elaImageUrl ? (
          <div className="flex items-center text-blue-500 dark:text-blue-400">
            <FaInfoCircle className="text-3xl mr-3" />
            <h2 className="text-2xl font-bold">ELA Analysis Complete</h2>
          </div>
        ) : heatmapImageUrl ? (
          <div className="flex items-center text-red-500 dark:text-red-400">
            <FaExclamationTriangle className="text-3xl mr-3" />
            <h2 className="text-2xl font-bold">Forgery Heatmap Generated</h2>
          </div>
        ) : (
          <div className="flex items-center text-green-500 dark:text-green-400">
            <FaCheckCircle className="text-3xl mr-3" />
            <h2 className="text-2xl font-bold">Image Appears Authentic</h2>
          </div>
        )}
      </motion.div>

      {/* Confidence Bar */}
      {result.confidence > 0 && result.prediction !== 0 && (
        <motion.div className="mb-6" variants={itemVariants}>
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Confidence
            </span>
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              {Math.round(result.confidence * 100)}%
            </span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
            <div
              className={`h-2.5 rounded-full ${
                result.is_tampered
                  ? "bg-red-500 dark:bg-red-400"
                  : "bg-green-500 dark:bg-green-400"
              }`}
              style={{ width: `${Math.round(result.confidence * 100)}%` }}
            ></div>
          </div>
        </motion.div>
      )}

      {/* Image Comparison Section */}
      {(elaImageUrl || heatmapImageUrl) && (
        <motion.div className="mb-6" variants={itemVariants}>
          <h3 className="text-xl font-medium mb-4 text-gray-800 dark:text-gray-200">
            Analysis Visualization
          </h3>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Original Image */}
            <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
              <div className="flex justify-between items-center p-2 bg-gray-50 dark:bg-gray-900">
                <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Original Image
                </h5>
                {originalImageUrl && !imageLoadError.original && (
                  <div className="flex gap-1">
                    <button
                      onClick={() => window.open(originalImageUrl, "_blank")}
                      className="p-1 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                      title="Open in new tab"
                    >
                      <FaEye className="text-xs" />
                    </button>
                    <button
                      onClick={() =>
                        downloadImage(
                          originalImageUrl,
                          `original_${result.filename}`
                        )
                      }
                      className="p-1 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                      title="Download image"
                    >
                      <FaDownload className="text-xs" />
                    </button>
                  </div>
                )}
              </div>

              {originalImageUrl && !imageLoadError.original ? (
                <img
                  src={originalImageUrl}
                  alt="Original uploaded image"
                  className="w-full h-auto object-contain bg-gray-100 dark:bg-gray-800 max-h-[400px]"
                  onError={() =>
                    setImageLoadError((prev) => ({ ...prev, original: true }))
                  }
                />
              ) : (
                <div className="flex items-center justify-center h-[300px] bg-gray-100 dark:bg-gray-800 p-4">
                  <p className="text-gray-500 dark:text-gray-400 text-center">
                    {imageLoadError.original
                      ? "Failed to load original image"
                      : result.filename}
                  </p>
                </div>
              )}
            </div>

            {/* ELA or Heatmap Visualization */}
            {elaImageUrl ? (
              <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
                <div className="flex justify-between items-center p-2 bg-gray-50 dark:bg-gray-900">
                  <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Error Level Analysis
                  </h5>
                  {!imageLoadError.ela && (
                    <div className="flex gap-1">
                      <button
                        onClick={() => window.open(elaImageUrl, "_blank")}
                        className="p-1 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                        title="Open in new tab"
                      >
                        <FaEye className="text-xs" />
                      </button>
                      <button
                        onClick={() =>
                          downloadImage(elaImageUrl, `ela_${result.filename}`)
                        }
                        className="p-1 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                        title="Download ELA image"
                      >
                        <FaDownload className="text-xs" />
                      </button>
                    </div>
                  )}
                </div>

                {!imageLoadError.ela ? (
                  <img
                    src={elaImageUrl}
                    alt="Error Level Analysis visualization"
                    className="w-full h-auto object-contain bg-gray-100 dark:bg-gray-800 max-h-[400px]"
                    onError={() =>
                      setImageLoadError((prev) => ({ ...prev, ela: true }))
                    }
                  />
                ) : (
                  <div className="flex items-center justify-center h-[300px] bg-gray-100 dark:bg-gray-800 p-4">
                    <p className="text-gray-500 dark:text-gray-400">
                      Failed to load ELA visualization
                    </p>
                  </div>
                )}
              </div>
            ) : heatmapImageUrl ? (
              <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
                <div className="flex justify-between items-center p-2 bg-gray-50 dark:bg-gray-900">
                  <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Forgery Heatmap
                  </h5>
                  {!imageLoadError.heatmap && (
                    <div className="flex gap-1">
                      <button
                        onClick={() => window.open(heatmapImageUrl, "_blank")}
                        className="p-1 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                        title="Open in new tab"
                      >
                        <FaEye className="text-xs" />
                      </button>
                      <button
                        onClick={() =>
                          downloadImage(heatmapImageUrl, `heatmap_${result.filename}`)
                        }
                        className="p-1 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                        title="Download heatmap image"
                      >
                        <FaDownload className="text-xs" />
                      </button>
                    </div>
                  )}
                </div>

                {!imageLoadError.heatmap ? (
                  <img
                    src={heatmapImageUrl}
                    alt="Forgery heatmap visualization"
                    className="w-full h-auto object-contain bg-gray-100 dark:bg-gray-800 max-h-[400px]"
                    onError={() =>
                      setImageLoadError((prev) => ({ ...prev, heatmap: true }))
                    }
                  />
                ) : (
                  <div className="flex items-center justify-center h-[300px] bg-gray-100 dark:bg-gray-800 p-4">
                    <p className="text-gray-500 dark:text-gray-400">
                      Failed to load heatmap visualization
                    </p>
                  </div>
                )}
              </div>
            ) : null}
          </div>
        </motion.div>
      )}

      {/* Reset Button */}
      <motion.div className="mt-6 text-center" variants={itemVariants}>
        <button
          onClick={onReset}
          className="bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-800 dark:text-gray-200 font-medium py-2 px-6 rounded-lg transition-colors"
        >
          Analyze Another Image
        </button>
      </motion.div>

      {/* Technical Details */}
      <motion.div
        className="rounded-lg bg-gray-50 dark:bg-gray-700 p-4"
        variants={itemVariants}
      >
        <h3 className="text-lg font-medium mb-3 text-gray-800 dark:text-gray-200">
          Technical Details
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
              <strong>Detection Method:</strong>{" "}
              {result.method || "ELA Analysis"}
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
              <strong>Confidence Score:</strong>{" "}
              {result.confidence > 0
                ? (result.confidence * 100).toFixed(2) + "%"
                : "N/A"}
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
              <strong>Image Status:</strong>{" "}
              {result.prediction == 1
                ? "Potentially Manipulated"
                : "Analysis Complete"}
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              <strong>Analysis Date:</strong>{" "}
              {dayjs().format("YYYY-MM-DD HH:mm")}
            </p>
          </div>
        </div>

        {/* Ensemble Details */}
        {result.ensemble_detail && (
          <div className="mt-4 border-t border-gray-200 dark:border-gray-600 pt-4">
            <h4 className="text-md font-medium mb-2 text-gray-700 dark:text-gray-300 flex items-center">
              <FaBrain className="mr-2 text-blue-500 dark:text-blue-400" />
              Ensemble Analysis Details
            </h4>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-3">
              <div className="bg-white dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700 text-center">
                <div className="text-sm text-gray-500 dark:text-gray-400">
                  Models Used
                </div>
                <div className="font-semibold text-gray-900 dark:text-white">
                  {result.ensemble_detail.ensemble_size}
                </div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700 text-center">
                <div className="text-sm text-gray-500 dark:text-gray-400">
                  Tampered Votes
                </div>
                <div className="font-semibold text-gray-900 dark:text-white">
                  {result.ensemble_detail.tampered_votes}
                </div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700 text-center">
                <div className="text-sm text-gray-500 dark:text-gray-400">
                  Authentic Votes
                </div>
                <div className="font-semibold text-gray-900 dark:text-white">
                  {result.ensemble_detail.authentic_votes}
                </div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700 text-center">
                <div className="text-sm text-gray-500 dark:text-gray-400">
                  Consensus
                </div>
                <div className="font-semibold text-gray-900 dark:text-white">
                  {result.ensemble_detail.consensus_level}
                </div>
              </div>
            </div>

            <details className="text-sm">
              <summary className="cursor-pointer text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 font-medium">
                View Model Predictions
              </summary>
              <div className="mt-2 overflow-auto max-h-60 bg-white dark:bg-gray-800 p-3 rounded border border-gray-200 dark:border-gray-700">
                <table className="w-full text-xs">
                  <thead className="bg-gray-50 dark:bg-gray-900">
                    <tr>
                      <th className="px-2 py-1 text-left">Model</th>
                      <th className="px-2 py-1 text-center">Prediction</th>
                      <th className="px-2 py-1 text-right">Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.ensemble_detail.model_predictions?.map(
                      (prediction, index) => (
                        <tr
                          key={index}
                          className="border-t border-gray-100 dark:border-gray-800"
                        >
                          <td className="px-2 py-1 text-gray-600 dark:text-gray-400 text-left">
                            {prediction.model_name}
                          </td>
                          <td className="px-2 py-1 text-center">
                            <span
                              className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                                prediction.prediction === 1
                                  ? "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300"
                                  : "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300"
                              }`}
                            >
                              {prediction.prediction === 1
                                ? "Tampered"
                                : "Authentic"}
                            </span>
                          </td>
                          <td className="px-2 py-1 text-gray-600 dark:text-gray-400 text-right">
                            {Math.round(prediction.confidence * 100)}%
                          </td>
                        </tr>
                      )
                    )}
                  </tbody>
                </table>
              </div>
            </details>
          </div>
        )}
      </motion.div>
    </motion.div>
  );
};

export default AnalysisResult;
