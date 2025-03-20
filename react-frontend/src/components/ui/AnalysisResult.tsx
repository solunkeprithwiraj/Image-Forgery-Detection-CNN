import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  FaCheckCircle,
  FaExclamationTriangle,
  FaInfoCircle,
  FaChartArea,
  FaImage,
  FaDrawPolygon,
  FaCameraRetro,
  FaMask,
  FaHighlighter,
  FaBorderAll,
} from "react-icons/fa";

// Define visualization types that might be available
type VisualizationType =
  | "heatmap"
  | "overlay"
  | "contour"
  | "ela"
  | "mask"
  | "edge"
  | "highlight";

// Map visualization types to their icons
const visualizationIcons: Record<VisualizationType, React.ReactNode> = {
  heatmap: <FaChartArea className="mr-2" />,
  overlay: <FaImage className="mr-2" />,
  contour: <FaDrawPolygon className="mr-2" />,
  ela: <FaCameraRetro className="mr-2" />,
  mask: <FaMask className="mr-2" />,
  edge: <FaBorderAll className="mr-2" />,
  highlight: <FaHighlighter className="mr-2" />,
};

// Map visualization types to their display names
const visualizationLabels: Record<VisualizationType, string> = {
  heatmap: "Heatmap",
  overlay: "Overlay",
  contour: "Contour",
  ela: "Error Level Analysis",
  mask: "Mask",
  edge: "Edge Detection",
  highlight: "Highlight",
};

// Define the result structure
interface AnalysisResult {
  is_tampered: boolean;
  confidence: number;
  message: string;
  method: string;
  timestamp: string;
  input_image_path: string;
  heatmap_path?: string;
  overlay_path?: string;
  contour_path?: string;
  ela_path?: string;
  mask_path?: string;
  edge_path?: string;
  highlight_path?: string;
}

interface AnalysisResultProps {
  result: AnalysisResult;
  apiBaseUrl: string;
}

const AnalysisResult: React.FC<AnalysisResultProps> = ({
  result,
  apiBaseUrl,
}) => {
  // Determine which visualizations are available
  const getAvailableVisualizations = (): VisualizationType[] => {
    const visualizations: VisualizationType[] = [];

    if (result.heatmap_path) visualizations.push("heatmap");
    if (result.overlay_path) visualizations.push("overlay");
    if (result.contour_path) visualizations.push("contour");
    if (result.ela_path) visualizations.push("ela");
    if (result.mask_path) visualizations.push("mask");
    if (result.edge_path) visualizations.push("edge");
    if (result.highlight_path) visualizations.push("highlight");

    return visualizations;
  };

  const availableVisualizations = getAvailableVisualizations();

  // Debug logs
  console.log("Available visualization paths:", {
    heatmap: result.heatmap_path || null,
    overlay: result.overlay_path || null,
    contour: result.contour_path || null,
    ela: result.ela_path || null,
    mask: result.mask_path || null,
    edge: result.edge_path || null,
    highlight: result.highlight_path || null,
  });
  console.log("Available visualization types:", availableVisualizations);

  // Select the first available visualization by default
  const [activeTab, setActiveTab] = useState<VisualizationType | null>(
    availableVisualizations.length > 0 ? availableVisualizations[0] : null
  );

  // Track image loading errors
  const [imageLoadError, setImageLoadError] = useState<Record<string, boolean>>(
    {}
  );

  // Update active tab when available visualizations change
  useEffect(() => {
    if (
      availableVisualizations.length > 0 &&
      activeTab &&
      !availableVisualizations.includes(activeTab)
    ) {
      setActiveTab(availableVisualizations[0]);
    }
  }, [availableVisualizations, activeTab]);

  // Get the path for the active visualization
  const getActiveVisualizationPath = (): string | undefined => {
    if (!activeTab) return undefined;

    switch (activeTab) {
      case "heatmap":
        return result.heatmap_path;
      case "overlay":
        return result.overlay_path;
      case "contour":
        return result.contour_path;
      case "ela":
        return result.ela_path;
      case "mask":
        return result.mask_path;
      case "edge":
        return result.edge_path;
      case "highlight":
        return result.highlight_path;
      default:
        return undefined;
    }
  };

  const activeVisualizationPath = getActiveVisualizationPath();

  // Format the image URL
  const formatImageUrl = (path?: string): string => {
    if (!path) return "";

    // Replace backslashes with forward slashes for URL compatibility
    const normalizedPath = path.replace(/\\/g, "/");

    // Make sure the path doesn't have any truncation issues
    if (normalizedPath.endsWith("_")) {
      console.warn("Path appears to be truncated:", normalizedPath);
    }

    console.log(`Formatting image URL for path: ${normalizedPath}`);

    // Check if the path already starts with http
    if (normalizedPath.startsWith("http")) {
      return normalizedPath;
    }

    // If the path already contains /static/ or starts with /static/
    if (
      normalizedPath.includes("/static/") ||
      normalizedPath.startsWith("/static/")
    ) {
      // Make sure we have the full URL with apiBaseUrl
      const staticPath = normalizedPath.includes("/static/")
        ? normalizedPath
        : normalizedPath.replace("/static", "");
      return `${apiBaseUrl}${staticPath}`;
    }

    // For paths that don't have /static/ but might start with /
    if (normalizedPath.startsWith("/")) {
      return `${apiBaseUrl}${normalizedPath}`;
    }

    // For paths with no leading slash, add the /static/ prefix
    return `${apiBaseUrl}/static/${normalizedPath}`;
  };

  // Function to check if an image exists
  const checkImageExists = async (path: string): Promise<boolean> => {
    try {
      // Remove /static/ if present
      const cleanPath = path.startsWith("/static/") ? path.substring(8) : path;

      // Use the debug endpoint to check if the image exists
      const response = await fetch(
        `${apiBaseUrl}/api/debug-image-path?path=${encodeURIComponent(
          cleanPath
        )}`
      );

      if (!response.ok) {
        console.error(
          `Server returned ${response.status}: ${response.statusText}`
        );
        return false;
      }

      const data = await response.json();
      console.log("Server response for image check:", data);
      return data.exists && data.is_file && data.size > 0;
    } catch (error) {
      console.error("Error checking image:", error);
      return false;
    }
  };

  // Function to check parent directory contents
  const checkDirectoryContents = async () => {
    try {
      // Get the directory where the image should be
      const directory = "uploads";

      const response = await fetch(
        `${apiBaseUrl}/api/list-directory?path=${encodeURIComponent(directory)}`
      );

      if (!response.ok) {
        console.error(
          `Server returned ${response.status}: ${response.statusText}`
        );
        alert(`Failed to list directory: ${response.statusText}`);
        return;
      }

      const data = await response.json();
      console.log("Directory contents:", data);
      alert(`Found ${data.files.length} files in uploads directory`);
    } catch (error) {
      console.error("Error listing directory:", error);
      alert(`Error: ${error.message}`);
    }
  };

  // Get visualization description
  const getVisualizationDescription = (type: VisualizationType): string => {
    switch (type) {
      case "heatmap":
        return "Shows probability of forgery with color intensity. Red areas are most likely to be tampered with.";
      case "overlay":
        return "Combines original with heatmap to highlight tampered regions while preserving the visual context.";
      case "contour":
        return "Draws green outlines around specific regions identified as tampered, providing precise localization.";
      case "ela":
        return "Error Level Analysis identifies areas with different compression levels, which can indicate manipulation.";
      case "mask":
        return "Highlights tampered areas with a semi-transparent overlay, making it easy to identify affected areas.";
      case "edge":
        return "Highlights the boundaries of tampered regions, showing precisely where manipulations occur.";
      case "highlight":
        return "Places highlights around tampered regions and displays the percentage of image affected by each manipulation.";
      default:
        return "Visualization showing potential regions of image manipulation.";
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
        {result.is_tampered ? (
          <div className="flex items-center text-red-500 dark:text-red-400">
            <FaExclamationTriangle className="text-3xl mr-3" />
            <h2 className="text-2xl font-bold">Manipulation Detected</h2>
          </div>
        ) : (
          <div className="flex items-center text-green-500 dark:text-green-400">
            <FaCheckCircle className="text-3xl mr-3" />
            <h2 className="text-2xl font-bold">Image Appears Authentic</h2>
          </div>
        )}
      </motion.div>

      {/* Confidence Bar */}
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

      {/* Visualization Section */}
      <motion.div className="mb-6" variants={itemVariants}>
        <div className="flex flex-col mb-4">
          <h3 className="text-xl font-medium mb-2 text-gray-800 dark:text-gray-200">
            Visualization
          </h3>

          {/* Message about visualizations */}
          <div className="mb-4">
            {availableVisualizations.length === 0 && (
              <div className="text-gray-600 dark:text-gray-400 mb-2">
                No visualizations available for this analysis.
              </div>
            )}

            {availableVisualizations.length === 1 && (
              <div className="text-gray-600 dark:text-gray-400 mb-2">
                Showing {visualizationLabels[availableVisualizations[0]]}{" "}
                visualization.
              </div>
            )}

            {availableVisualizations.length > 1 && (
              <div className="text-gray-600 dark:text-gray-400 mb-2">
                Select a visualization method:
              </div>
            )}
          </div>

          {/* Visualization type buttons */}
          {availableVisualizations.length > 1 && (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 mb-4">
              {availableVisualizations.map((type) => (
                <button
                  key={type}
                  onClick={() => setActiveTab(type)}
                  className={`flex items-center justify-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    activeTab === type
                      ? "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-200"
                      : "bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600"
                  }`}
                >
                  {visualizationIcons[type]}
                  {visualizationLabels[type]}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Visualization Image */}
        {activeTab && activeVisualizationPath && (
          <div className="mt-4">
            <h4 className="text-md font-medium mb-2 text-gray-700 dark:text-gray-300">
              Side-by-side Comparison
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Original uploaded image */}
              <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
                <h5 className="text-sm font-medium p-2 bg-gray-50 dark:bg-gray-900 text-gray-700 dark:text-gray-300">
                  Original Image
                </h5>
                <img
                  src={formatImageUrl(result.input_image_path)}
                  alt="Original uploaded image"
                  className="w-full h-auto object-contain bg-gray-100 dark:bg-gray-800 max-h-[300px]"
                />
              </div>

              {/* Visualization image */}
              <div className="relative border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
                <h5 className="text-sm font-medium p-2 bg-gray-50 dark:bg-gray-900 text-gray-700 dark:text-gray-300">
                  {visualizationLabels[activeTab]} Visualization
                </h5>
                <img
                  src={formatImageUrl(activeVisualizationPath)}
                  alt={`${visualizationLabels[activeTab]} visualization of the image analysis`}
                  className="w-full h-auto object-contain bg-gray-100 dark:bg-gray-800 max-h-[300px]"
                  onError={async (e) => {
                    // Log the error
                    console.error(
                      `Failed to load image: ${activeVisualizationPath}`
                    );

                    // Check if the image exists on the server using our endpoint
                    const exists = await checkImageExists(
                      activeVisualizationPath
                    );
                    console.log(
                      `Image ${activeVisualizationPath} exists: ${exists}`
                    );

                    // Update the error state
                    setImageLoadError({ ...imageLoadError, [activeTab]: true });
                  }}
                />
                {imageLoadError[activeTab] && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center p-4 bg-gray-50 dark:bg-gray-800">
                    <h3 className="text-lg font-medium text-gray-800 dark:text-gray-200 mb-2">
                      Image Load Error
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400 mb-4">
                      Failed to load the {visualizationLabels[activeTab]}{" "}
                      visualization.
                    </p>
                    <div className="text-xs text-gray-500 dark:text-gray-400 p-2 bg-gray-100 dark:bg-gray-700 rounded-md">
                      Path: {activeVisualizationPath}
                    </div>
                    <div className="mt-4 flex flex-col sm:flex-row gap-2">
                      <button
                        onClick={async () => {
                          // Try to check if the image exists
                          const exists = await checkImageExists(
                            activeVisualizationPath
                          );
                          alert(
                            `Image check result: ${
                              exists
                                ? "Image exists on server"
                                : "Image does not exist on server"
                            }`
                          );
                        }}
                        className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm"
                      >
                        Check Image Availability
                      </button>
                      <button
                        onClick={checkDirectoryContents}
                        className="px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600 text-sm"
                      >
                        Check Uploads Directory
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Visualization explanation */}
        {activeTab && !imageLoadError[activeTab] && (
          <div className="mt-4">
            <div className="flex items-start text-gray-600 dark:text-gray-400">
              <FaInfoCircle className="text-blue-500 dark:text-blue-400 mt-1 mr-2" />
              <p className="text-sm">
                {getVisualizationDescription(activeTab)}
              </p>
            </div>
          </div>
        )}
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
              <strong>Detection Method:</strong> {result.method}
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
              <strong>Confidence Score:</strong>{" "}
              {Math.round(result.confidence * 100)}%
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
              <strong>Image Status:</strong>{" "}
              {result.is_tampered ? "Manipulated" : "Authentic"}
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              <strong>Analysis Date:</strong>{" "}
              {new Date(result.timestamp).toLocaleString()}
            </p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default AnalysisResult;
