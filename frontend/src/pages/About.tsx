import React, { useRef, useEffect } from "react";
import { motion } from "framer-motion";
import * as THREE from "three";
import {
  FaGithub,
  FaLinkedin,
  FaEnvelope,
  FaChevronRight,
} from "react-icons/fa";

const About: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const threeContainerRef = useRef<HTMLDivElement>(null);

  // Setup THREE.js scene
  useEffect(() => {
    if (!threeContainerRef.current) return;

    // Setup scene
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    const renderer = new THREE.WebGLRenderer({ alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    threeContainerRef.current.appendChild(renderer.domElement);

    // Create DNA-like helix structure
    const particles: THREE.Mesh[] = [];
    const particleGeometry = new THREE.SphereGeometry(0.2, 8, 8);
    const particleMaterial = new THREE.MeshBasicMaterial({
      color: 0x3498db,
      transparent: true,
      opacity: 0.7,
    });

    // Create DNA double helix
    for (let i = 0; i < 100; i++) {
      // First strand
      const particle1 = new THREE.Mesh(particleGeometry, particleMaterial);
      const angle1 = i * 0.2;
      particle1.position.x = 5 * Math.cos(angle1);
      particle1.position.y = i * 0.5 - 25;
      particle1.position.z = 5 * Math.sin(angle1);
      scene.add(particle1);
      particles.push(particle1);

      // Second strand
      const particle2 = new THREE.Mesh(particleGeometry, particleMaterial);
      const angle2 = i * 0.2 + Math.PI;
      particle2.position.x = 5 * Math.cos(angle2);
      particle2.position.y = i * 0.5 - 25;
      particle2.position.z = 5 * Math.sin(angle2);
      scene.add(particle2);
      particles.push(particle2);

      // Cross-links (only add some)
      if (i % 5 === 0) {
        const linkGeometry = new THREE.BoxGeometry(
          Math.abs(particle1.position.x - particle2.position.x),
          0.05,
          0.05
        );
        const linkMaterial = new THREE.MeshBasicMaterial({
          color: 0x9b59b6,
          transparent: true,
          opacity: 0.5,
        });
        const link = new THREE.Mesh(linkGeometry, linkMaterial);
        link.position.x = (particle1.position.x + particle2.position.x) / 2;
        link.position.y = particle1.position.y;
        link.position.z = (particle1.position.z + particle2.position.z) / 2;
        link.rotation.z = Math.atan2(
          particle2.position.y - particle1.position.y,
          particle2.position.x - particle1.position.x
        );
        scene.add(link);
        particles.push(link);
      }
    }

    // Create neural network model
    const createNeuralNetwork = () => {
      const layers = [4, 6, 6, 3]; // Nodes per layer
      const layerDistance = 7;
      const nodes: THREE.Mesh[] = [];
      const connections: THREE.Line[] = [];

      // Create nodes
      for (let l = 0; l < layers.length; l++) {
        const nodeCount = layers[l];
        const layerX = l * layerDistance - 15;

        for (let n = 0; n < nodeCount; n++) {
          const y = (n - (nodeCount - 1) / 2) * 2;
          const nodeGeometry = new THREE.SphereGeometry(0.3, 16, 16);
          const nodeMaterial = new THREE.MeshBasicMaterial({
            color: 0x2ecc71,
            transparent: true,
            opacity: 0.8,
          });
          const node = new THREE.Mesh(nodeGeometry, nodeMaterial);
          node.position.set(layerX, y + 5, -5);
          scene.add(node);
          nodes.push(node);

          // Connect to previous layer
          if (l > 0) {
            for (let p = 0; p < layers[l - 1]; p++) {
              const prevY = (p - (layers[l - 1] - 1) / 2) * 2;
              const prevNodeIndex = nodes.length - nodeCount - layers[l - 1] + p;

              if (prevNodeIndex >= 0) {
                const lineMaterial = new THREE.LineBasicMaterial({
                  color: 0xe74c3c,
                  transparent: true,
                  opacity: 0.3,
                });
                const points = [
                  new THREE.Vector3(layerX - layerDistance, prevY + 5, -5),
                  new THREE.Vector3(layerX, y + 5, -5),
                ];
                const lineGeometry = new THREE.BufferGeometry().setFromPoints(
                  points
                );
                const line = new THREE.Line(lineGeometry, lineMaterial);
                scene.add(line);
                connections.push(line);
              }
            }
          }
        }
      }

      return { nodes, connections };
    };

    const neuralNetwork = createNeuralNetwork();

    // Add an image recognition model (simplified cube structure)
    const createImageModel = () => {
      const imageGeometry = new THREE.BoxGeometry(5, 5, 0.2);
      const imageMaterial = new THREE.MeshBasicMaterial({
        color: 0x3498db,
        transparent: true,
        opacity: 0.5,
      });
      const image = new THREE.Mesh(imageGeometry, imageMaterial);
      image.position.set(0, -10, 5);
      scene.add(image);

      // Add features as small cubes on the image
      for (let i = 0; i < 10; i++) {
        const featureGeometry = new THREE.BoxGeometry(0.5, 0.5, 0.3);
        const featureMaterial = new THREE.MeshBasicMaterial({
          color: 0xe74c3c,
          transparent: true,
          opacity: 0.8,
        });
        const feature = new THREE.Mesh(featureGeometry, featureMaterial);
        feature.position.set(
          Math.random() * 4 - 2,
          Math.random() * 4 - 2 - 10,
          5.3
        );
        scene.add(feature);
      }

      return image;
    };

    const imageModel = createImageModel();

    // Position camera
    camera.position.z = 20;

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);

      // Rotate DNA
      particles.forEach((particle) => {
        particle.rotation.x += 0.003;
        particle.rotation.y += 0.002;
      });

      // Pulse neural network nodes
      neuralNetwork.nodes.forEach((node, index) => {
        const scale = 1 + 0.2 * Math.sin(Date.now() * 0.001 + index * 0.5);
        node.scale.set(scale, scale, scale);
      });

      // Rotate image model
      imageModel.rotation.y = Math.sin(Date.now() * 0.0005) * 0.2;

      renderer.render(scene, camera);
    };

    animate();

    // Handle window resize
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    window.addEventListener("resize", handleResize);

    // Cleanup
    return () => {
      window.removeEventListener("resize", handleResize);
      if (threeContainerRef.current) {
        threeContainerRef.current.removeChild(renderer.domElement);
      }
      particles.forEach((particle) => {
        scene.remove(particle);
        particle.geometry.dispose();
        (particle.material as THREE.Material).dispose();
      });
    };
  }, []);

  return (
    <div ref={containerRef} className="relative min-h-screen py-12 overflow-hidden">
      {/* THREE.js container (absolute positioned) */}
      <div
        ref={threeContainerRef}
        className="absolute inset-0 z-0"
        style={{ pointerEvents: "none" }}
      />

      {/* Content */}
      <div className="container mx-auto px-4 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="max-w-4xl mx-auto"
        >
          <h1 className="text-3xl md:text-4xl font-bold text-center mb-8 text-white">
            About This Project
          </h1>

          <div className="bg-white/10 backdrop-blur-xl rounded-xl shadow-lg overflow-hidden mb-12 border border-white/20 hover:border-white/30 transition-all duration-300">
            <div className="p-6 md:p-8">
              <h2 className="text-2xl font-bold mb-4 text-white">
                Image Forgery Detection using CNN
              </h2>

              <div className="prose prose-invert max-w-none">
                <p className="text-gray-300">
                  This project implements a Convolutional Neural Network (CNN)
                  based approach for detecting image forgeries, particularly
                  focusing on identifying manipulated regions within digital
                  images. The system can detect various types of image
                  forgeries, including copy-move, splicing, inpainting, compression,
                  and metadata tampering.
                </p>

                <h3 className="text-xl font-semibold text-blue-300 mt-6 mb-3">Technical Overview</h3>
                <p className="text-gray-300">
                  The core of this system is based on a deep learning
                  architecture that analyzes patterns and inconsistencies within
                  images that may not be visible to the human eye. The CNN model
                  has been trained on a dataset of authentic and forged images
                  to learn the subtle artifacts introduced during manipulation.
                </p>

                <h3 className="text-xl font-semibold text-blue-300 mt-6 mb-3">Key Features</h3>
                <ul className="list-disc list-inside text-gray-300 space-y-1">
                  <li>
                    Multiple forgery detection techniques
                  </li>
                  <li>
                    Localization of tampered regions with heatmap visualization
                  </li>
                  <li>High accuracy on standard image forgery datasets</li>
                  <li>
                    User-friendly interface for uploading and analyzing images
                  </li>
                  <li>Detailed result visualization with confidence scoring</li>
                </ul>

                <h3 className="text-xl font-semibold text-blue-300 mt-6 mb-3">Methodology</h3>
                <p className="text-gray-300">The detection process involves several specialized techniques:</p>
                <ol className="list-decimal list-inside text-gray-300 space-y-1 mt-2">
                  <li>
                    <strong className="text-white">Copy-Move Detection:</strong> Identifies duplicated regions within the same image using ORB keypoints or DCT coefficients.
                  </li>
                  <li>
                    <strong className="text-white">Splicing Detection:</strong> Locates inconsistencies in edges and lighting that occur when content from one image is inserted into another.
                  </li>
                  <li>
                    <strong className="text-white">Inpainting Detection:</strong> Finds areas that have been filled in using AI or content-aware fill tools.
                  </li>
                  <li>
                    <strong className="text-white">Metadata Analysis:</strong> Examines EXIF data for inconsistencies that suggest tampering.
                  </li>
                </ol>

                <h3 className="text-xl font-semibold text-blue-300 mt-6 mb-3">Technologies Used</h3>
                <p className="text-gray-300">This project is built using several modern technologies:</p>
                <ul className="list-disc list-inside text-gray-300 space-y-1 mt-2">
                  <li>Python with PyTorch for the CNN model development</li>
                  <li>FastAPI for the backend API</li>
                  <li>React and Tailwind CSS for the frontend interface</li>
                  <li>OpenCV and scikit-image for image processing</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-white/10 backdrop-blur-xl rounded-xl shadow-lg overflow-hidden mb-12 border border-white/20 hover:border-white/30 transition-all duration-300">
            <div className="p-6 md:p-8">
              <h2 className="text-2xl font-bold mb-4 text-white">
                Research Background
              </h2>

              <div className="prose prose-invert max-w-none">
                <p className="text-gray-300">
                  Image forgery detection is an increasingly important field in
                  digital forensics as image manipulation becomes more
                  sophisticated and accessible. Traditional methods often rely
                  on statistical analysis and handcrafted features, but deep
                  learning approaches have shown superior performance in recent
                  years.
                </p>

                <p className="text-gray-300 mt-4">
                  This project builds upon several influential research papers:
                </p>

                <ul className="mt-4 space-y-4">
                  <li>
                    <strong className="text-white">CNN-based Image Forgery Detection (2018)</strong>
                    <p className="mt-1 text-gray-300">
                      Pioneering work on using CNN architectures for detecting
                      manipulation artifacts in images.
                    </p>
                  </li>
                  <li>
                    <strong className="text-white">
                      ManTra-Net: Manipulation Tracing Network (2019)
                    </strong>
                    <p className="mt-1 text-gray-300">
                      End-to-end solution for both detecting and localizing
                      image forgeries regardless of manipulation type.
                    </p>
                  </li>
                  <li>
                    <strong className="text-white">
                      SPAN: Spatial Pyramid Attention Network (2020)
                    </strong>
                    <p className="mt-1 text-gray-300">
                      Advanced architecture incorporating attention mechanisms
                      to focus on relevant image regions.
                    </p>
                  </li>
                </ul>

                <p className="text-gray-300 mt-4">
                  Our implementation incorporates specialized detection techniques for different types of forgeries, providing a comprehensive analysis system that can detect various manipulation methods.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white/10 backdrop-blur-xl rounded-xl shadow-lg overflow-hidden border border-white/20 hover:border-white/30 transition-all duration-300">
            <div className="p-6 md:p-8">
              <h2 className="text-2xl font-bold mb-4 text-white">
                Contact & Resources
              </h2>

              <div className="prose prose-invert max-w-none mb-6">
                <p className="text-gray-300">
                  This project was developed as a final year project. Feel free
                  to reach out for questions, contributions, or collaboration
                  opportunities.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <a
                  href="https://github.com/yourusername/image-forgery-detection-cnn"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center p-4 bg-white/5 backdrop-blur-sm rounded-lg hover:bg-white/10 transition duration-200 border border-white/10 hover:border-white/20"
                >
                  <FaGithub className="text-2xl mr-3 text-gray-300" />
                  <div>
                    <div className="font-medium text-white">
                      GitHub Repository
                    </div>
                    <div className="text-sm text-gray-400">
                      Source code and documentation
                    </div>
                  </div>
                  <FaChevronRight className="ml-auto text-gray-400" />
                </a>

                <a
                  href="https://linkedin.com/in/yourusername"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center p-4 bg-white/5 backdrop-blur-sm rounded-lg hover:bg-white/10 transition duration-200 border border-white/10 hover:border-white/20"
                >
                  <FaLinkedin className="text-2xl mr-3 text-blue-400" />
                  <div>
                    <div className="font-medium text-white">LinkedIn</div>
                    <div className="text-sm text-gray-400">
                      Connect professionally
                    </div>
                  </div>
                  <FaChevronRight className="ml-auto text-gray-400" />
                </a>

                <a
                  href="mailto:your.email@example.com"
                  className="flex items-center p-4 bg-white/5 backdrop-blur-sm rounded-lg hover:bg-white/10 transition duration-200 border border-white/10 hover:border-white/20"
                >
                  <FaEnvelope className="text-2xl mr-3 text-green-400" />
                  <div>
                    <div className="font-medium text-white">Email</div>
                    <div className="text-sm text-gray-400">
                      Send a direct message
                    </div>
                  </div>
                  <FaChevronRight className="ml-auto text-gray-400" />
                </a>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default About;
