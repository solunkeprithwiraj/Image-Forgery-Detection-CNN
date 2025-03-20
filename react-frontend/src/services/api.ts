import axios from "axios";

// Get the API URL from environment variables
const API_BASE_URL = "http://localhost:5000";

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
}

export type LocalizationMethod =
  | "heatmap"
  | "overlay"
  | "contour"
  | "mask"
  | "edge"
  | "highlight";

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

    const response = await api.post<AnalysisResult>("/api/analyze", formData);
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

export default api;
