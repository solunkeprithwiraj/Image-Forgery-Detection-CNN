import axios from "axios";

// Get the API URL from environment variables
 const API_BASE_URL = "http://localhost:8000";

// Create an axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "multipart/form-data",
  },
});

export interface AnalysisResult {
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
  ensemble_detail?: {
    ensemble_size: number;
    tampered_votes: number;
    authentic_votes: number;
    consensus_level: string;
    model_predictions: any[];
  };
}

export type LocalizationMethod =
  | "heatmap"
  | "overlay"
  | "contour"
  | "mask"
  | "edge"
  | "highlight";

export type ELAMode = "basic" | "enhanced" | "comparison" | "zoom";
export type HeatmapMode = "basic" | "detail" | "multi" | "composite";

export const analyzeElaImage = async (
  file: File, 
  mode: ELAMode = "basic",
  quality: number = 85,
  enhanceContrast: boolean = true,
  colorize: boolean = false
): Promise<string> => {
  const formData = new FormData();
  formData.append("image", file);
  formData.append("mode", mode);
  formData.append("quality", quality.toString());
  formData.append("enhance_contrast", enhanceContrast.toString());
  formData.append("colorize", colorize.toString());

  try {
    const response = await fetch("http://localhost:8000/api/ela", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(
        `ELA analysis failed: ${response.status} ${response.statusText}`
      );
    }

    // Convert the response blob to a data URL
    const blob = await response.blob();
    return URL.createObjectURL(blob);
  } catch (error) {
    console.error("Error in ELA analysis:", error);
    throw error;
  }
};

export const generateForgeryHeatmap = async (
  file: File,
  mode: HeatmapMode = "basic",
  threshold: number = 0.5,
  colormap: string = "jet"
): Promise<string> => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("mode", mode);
  formData.append("threshold", threshold.toString());
  formData.append("colormap", colormap);

  try {
    const response = await fetch("http://localhost:8000/api/heatmap/forgery", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(
        `Forgery heatmap generation failed: ${response.status} ${response.statusText}`
      );
    }

    // Convert the response blob to a data URL
    const blob = await response.blob();
    return URL.createObjectURL(blob);
  } catch (error) {
    console.error("Error generating forgery heatmap:", error);
    throw error;
  }
};

export const analyzeImage = async (
  imageFile: File,
  showLocalization: boolean = true,
  showEla: boolean = true,
  localizationMethods: LocalizationMethod[] = [
    "heatmap",
    "overlay",
    "contour",
    "mask",
    "edge",
    "highlight",
  ]
): Promise<AnalysisResult> => {
  try {
    const formData = new FormData();
    formData.append("file", imageFile);
    formData.append("show_localization", showLocalization ? "true" : "false");
    formData.append("show_ela", showEla ? "true" : "false");

    // Add each localization method separately
    if (showLocalization && localizationMethods.length > 0) {
      localizationMethods.forEach((method) => {
        formData.append("localization_methods[]", method);
      });
    }

    // const response = await api.post<AnalysisResult>("/api/analyze", formData);
    const response = await api.post<AnalysisResult>("/api/predict/", formData);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error) && error.response) {
      throw new Error(
        `Analysis failed: ${error.response.data.error || "Unknown error"}`
      );
    }
    throw new Error("Failed to connect to the server. Please try again later.");
  }
};

export const analyzeImageEnsemble = async (
  imageFile: File,
  showLocalization: boolean = true,
  showEla: boolean = true,
  localizationMethods: LocalizationMethod[] = [
    "heatmap",
    "overlay",
    "contour",
    "mask",
    "edge",
    "highlight",
  ]
): Promise<AnalysisResult> => {
  try {
    const formData = new FormData();
    formData.append("file", imageFile);
    formData.append("show_localization", showLocalization ? "true" : "false");
    formData.append("show_ela", showEla ? "true" : "false");

    // Add each localization method separately
    if (showLocalization && localizationMethods.length > 0) {
      localizationMethods.forEach((method) => {
        formData.append("localization_methods[]", method);
      });
    }

    const response = await api.post<AnalysisResult>(
      "/api/analyze/ensemble",
      formData
    );
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error) && error.response) {
      throw new Error(
        `Ensemble analysis failed: ${
          error.response.data.error || "Unknown error"
        }`
      );
    }
    throw new Error("Failed to connect to the server. Please try again later.");
  }
};

export const convertTiffToJpeg = async (file: File): Promise<string> => {
  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await api.post("/api/convert-tiff", formData);
    return response.data.preview_url;
  } catch (error) {
    console.error("Error converting TIFF:", error);
    throw error;
  }
};

export const viewTiffFile = async (tiffPath: string): Promise<string> => {
  try {
    const response = await api.get(`/api/view-tiff/${tiffPath}`);
    return response.data.preview_url;
  } catch (error) {
    console.error("Error viewing TIFF:", error);
    throw error;
  }
};

export type ForgeryType = "comprehensive" | "copy-move" | "splicing" | "inpainting" | "metadata";

export const detectCopyMove = async (
  file: File,
  method: string = "orb"
): Promise<{ result: any; imageUrl: string | null }> => {
  try {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("method", method);

    const response = await api.post("/api/detect/copy-move", formData);
    return {
      result: response.data,
      imageUrl: response.data.visualization_url || null
    };
  } catch (error) {
    if (axios.isAxiosError(error) && error.response) {
      throw new Error(`Copy-move detection failed: ${error.response.data.error || "Unknown error"}`);
    }
    throw new Error("Failed to connect to the server. Please try again later.");
  }
};

export const detectSplicing = async (
  file: File,
  method: string = "combined"
): Promise<{ result: any; imageUrl: string | null }> => {
  try {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("method", method);

    const response = await api.post("/api/detect/splicing", formData);
    return {
      result: response.data,
      imageUrl: response.data.visualization_url || null
    };
  } catch (error) {
    if (axios.isAxiosError(error) && error.response) {
      throw new Error(`Splicing detection failed: ${error.response.data.error || "Unknown error"}`);
    }
    throw new Error("Failed to connect to the server. Please try again later.");
  }
};

export const detectInpainting = async (
  file: File,
  method: string = "combined"
): Promise<{ result: any; imageUrl: string | null }> => {
  try {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("method", method);

    const response = await api.post("/api/detect/inpainting", formData);
    return {
      result: response.data,
      imageUrl: response.data.visualization_url || null
    };
  } catch (error) {
    if (axios.isAxiosError(error) && error.response) {
      throw new Error(`Inpainting detection failed: ${error.response.data.error || "Unknown error"}`);
    }
    throw new Error("Failed to connect to the server. Please try again later.");
  }
};

export const analyzeMetadata = async (
  file: File,
  detailed: boolean = false
): Promise<any> => {
  try {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("detailed", detailed.toString());

    const response = await api.post("/api/analyze/metadata", formData);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error) && error.response) {
      throw new Error(`Metadata analysis failed: ${error.response.data.error || "Unknown error"}`);
    }
    throw new Error("Failed to connect to the server. Please try again later.");
  }
};

export const comprehensiveAnalysis = async (
  file: File
): Promise<any> => {
  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await api.post("/api/forgery/comprehensive", formData);
    
    // Get the response data - should already be properly processed by the backend
    const result = response.data;
    
    // Double-check the consistency of the prediction with the voting results
    // This ensures frontend sync with backend's authentic-biased logic
    if (result.voting_summary) {
      const { tampered_votes, authentic_votes } = result.voting_summary;
      
      // If votes are equal or authentic votes are more, ensure prediction is authentic
      if (tampered_votes <= authentic_votes && result.prediction !== "authentic") {
        console.warn("Fixing inconsistent prediction from backend: defaulting to authentic on tied vote");
        result.prediction = "authentic";
        
        // Recalculate average confidence for authentic predictions
        let authenticConfidence = 0;
        let authenticCount = 0;
        
        if (result.results) {
          Object.values(result.results).forEach((analysis: any) => {
            if (analysis.prediction === "authentic") {
              authenticConfidence += analysis.confidence;
              authenticCount++;
            }
          });
          
          if (authenticCount > 0) {
            result.overall_confidence = authenticConfidence / authenticCount;
          }
        }
        
        // Reset forgery type when authentic
        result.most_likely_forgery_type = null;
      }
    }
    
    return result;
  } catch (error) {
    if (axios.isAxiosError(error) && error.response) {
      throw new Error(`Comprehensive analysis failed: ${error.response.data.error || "Unknown error"}`);
    }
    throw new Error("Failed to connect to the server. Please try again later.");
  }
};

export default api;
