# Image Forgery Detection Frontend

This is the frontend application for the Image Forgery Detection system, built with React, TypeScript, and Tailwind CSS.

## Features

- Upload images for forgery detection
- View analysis results with visualizations
- User-friendly interface with dark mode support
- Responsive design for all devices

## Setup and Installation

### Prerequisites

- Node.js (v14+)
- npm or yarn

### Installation

1. Clone the repository
2. Navigate to the project directory
3. Install dependencies:

```bash
cd react-frontend
npm install
```

### Development

To start the development server:

```bash
npm run dev
```

This will start the development server at http://localhost:3000 (or another port if 3000 is in use).

### Production Build

To create a production build:

```bash
npm run build
```

The build output will be in the `dist` directory.

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```
VITE_API_URL=http://localhost:8000
VITE_APP_NAME=ForgeDetect
VITE_APP_DESCRIPTION=Advanced Image Forgery Detection with CNN
```

Adjust the `VITE_API_URL` to match your backend API endpoint.

## Project Structure

- `/src/components/`: UI components
  - `/ui/`: Reusable UI components
  - `/layout/`: Layout components (Header, Footer)
- `/src/hooks/`: Custom React hooks
- `/src/pages/`: Page components
- `/src/services/`: API service functions
- `/src/assets/`: Static assets

## Technologies Used

- React 18
- TypeScript
- Framer Motion for animations
- Tailwind CSS for styling
- Vite for build tooling

## License

This project is part of the Image Forgery Detection system for academic research.
