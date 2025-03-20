import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

// Initialize dark mode based on user preference or saved setting
const setInitialDarkMode = () => {
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const savedTheme = localStorage.getItem("theme");

  if (savedTheme === "dark" || (!savedTheme && prefersDark)) {
    document.documentElement.classList.add("dark");
  } else {
    document.documentElement.classList.remove("dark");
  }
};

// Call the function to set the initial mode
setInitialDarkMode();

const root = ReactDOM.createRoot(
  document.getElementById("root") as HTMLElement
);

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
