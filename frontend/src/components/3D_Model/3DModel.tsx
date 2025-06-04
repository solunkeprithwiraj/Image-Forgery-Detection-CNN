import React, { useRef, useEffect } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";

const ThreeDModel = () => {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const animationIdRef = useRef(null);

  useEffect(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    scene.fog = new THREE.Fog(0x000000, 10, 50);

    const camera = new THREE.PerspectiveCamera(
      75,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 5, 15);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(
      mountRef.current.clientWidth,
      mountRef.current.clientHeight
    );
    renderer.shadowMap.enabled = true;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;

    mountRef.current.appendChild(renderer.domElement);
    sceneRef.current = scene;
    rendererRef.current = renderer;

    // Orbit controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.rotateSpeed = 0.5;
    controls.zoomSpeed = 0.5;

    // Lights
    scene.add(new THREE.AmbientLight(0x404040, 0.3));
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    const pointLight1 = new THREE.PointLight(0x00ff88, 1, 20);
    pointLight1.position.set(-10, 5, 5);
    scene.add(pointLight1);

    const pointLight2 = new THREE.PointLight(0xff0088, 0.8, 15);
    pointLight2.position.set(10, -5, -5);
    scene.add(pointLight2);

    const neuralNodes = [];
    const connections = [];

    for (let i = 0; i < 20; i++) {
      const geometry = new THREE.SphereGeometry(
        0.1 + Math.random() * 0.1,
        16,
        16
      );
      const material = new THREE.MeshPhongMaterial({
        color: new THREE.Color().setHSL(0.6 + Math.random() * 0.4, 0.8, 0.6),
        transparent: true,
        opacity: 0.8,
        emissive: new THREE.Color().setHSL(0.6 + Math.random() * 0.4, 0.3, 0.1),
      });
      const node = new THREE.Mesh(geometry, material);
      node.position.setFromSphericalCoords(
        3 + Math.random() * 4,
        Math.random() * Math.PI,
        Math.random() * Math.PI * 2
      );
      node.userData = {
        originalPosition: node.position.clone(),
        phase: Math.random() * Math.PI * 2,
        speed: 0.01 + Math.random() * 0.02,
        amplitude: 0.5 + Math.random() * 0.5,
        originalEmissive: material.emissive.getHex(),
      };
      scene.add(node);
      neuralNodes.push(node);
    }

    for (let i = 0; i < neuralNodes.length; i++) {
      for (let j = i + 1; j < neuralNodes.length; j++) {
        if (Math.random() < 0.3) {
          const points = [neuralNodes[i].position, neuralNodes[j].position];
          const geometry = new THREE.BufferGeometry().setFromPoints(points);
          const material = new THREE.LineBasicMaterial({
            color: 0x00ffff,
            transparent: true,
            opacity: 0.2,
          });
          const line = new THREE.Line(geometry, material);
          scene.add(line);
          connections.push({
            line,
            node1: neuralNodes[i],
            node2: neuralNodes[j],
          });
        }
      }
    }

    const brain = new THREE.Mesh(
      new THREE.IcosahedronGeometry(1.5, 2),
      new THREE.MeshPhongMaterial({
        color: 0x00ff88,
        transparent: true,
        opacity: 0.7,
        wireframe: true,
        emissive: 0x002200,
      })
    );
    scene.add(brain);

    const particleCount = 100;
    const particleGeometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 20;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 20;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 20;
      const color = new THREE.Color().setHSL(
        0.6 + Math.random() * 0.4,
        0.8,
        0.6
      );
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }
    particleGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(positions, 3)
    );
    particleGeometry.setAttribute(
      "color",
      new THREE.BufferAttribute(colors, 3)
    );

    const particleMaterial = new THREE.PointsMaterial({
      size: 0.05,
      vertexColors: true,
      transparent: true,
      opacity: 0.6,
    });

    const particles = new THREE.Points(particleGeometry, particleMaterial);
    scene.add(particles);

    // Interactivity setup
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let hoveredNode = null;

    const onMouseMove = (event) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    };

    renderer.domElement.addEventListener("mousemove", onMouseMove);

    // Animation loop
    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate);
      const time = Date.now() * 0.001;

      brain.rotation.x = time * 0.2;
      brain.rotation.y = time * 0.3;

      neuralNodes.forEach((node) => {
        const { originalPosition, phase, speed, amplitude } = node.userData;
        node.position.x =
          originalPosition.x + Math.sin(time * speed + phase) * amplitude;
        node.position.y =
          originalPosition.y +
          Math.cos(time * speed * 1.1 + phase) * amplitude * 0.5;
        node.position.z =
          originalPosition.z +
          Math.sin(time * speed * 0.8 + phase) * amplitude * 0.3;

        const scale = 1 + Math.sin(time * 2 + phase) * 0.2;
        node.scale.setScalar(scale);
      });

      connections.forEach(({ line, node1, node2 }) => {
        line.geometry.setFromPoints([node1.position, node2.position]);
        line.material.opacity = 0.1 + Math.sin(time * 2) * 0.1;
      });

      const particlePositions = particles.geometry.attributes.position.array;
      for (let i = 0; i < particleCount; i++) {
        particlePositions[i * 3 + 1] += Math.sin(time + i) * 0.01;
      }
      particles.geometry.attributes.position.needsUpdate = true;
      particles.rotation.y = time * 0.1;

      // Raycasting
      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(neuralNodes);

      if (intersects.length > 0) {
        if (hoveredNode !== intersects[0].object) {
          if (hoveredNode) {
            hoveredNode.material.emissive.setHex(
              hoveredNode.userData.originalEmissive
            );
          }
          hoveredNode = intersects[0].object;
          hoveredNode.userData.originalEmissive =
            hoveredNode.material.emissive.getHex();
          hoveredNode.material.emissive.setHex(0xffff00);
        }
      } else {
        if (hoveredNode) {
          hoveredNode.material.emissive.setHex(
            hoveredNode.userData.originalEmissive
          );
          hoveredNode = null;
        }
      }

      controls.update();
      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      if (mountRef.current && renderer && camera) {
        const width = mountRef.current.clientWidth;
        const height = mountRef.current.clientHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      renderer.domElement.removeEventListener("mousemove", onMouseMove);
      if (animationIdRef.current) cancelAnimationFrame(animationIdRef.current);
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      controls.dispose();
      renderer.dispose();
    };
  }, []);

  return (
    <div
      ref={mountRef}
      className="w-full h-full rounded-2xl overflow-hidden bg-gradient-to-br from-gray-900 to-black"
      style={{ minHeight: "400px" }}
    />
  );
};

export default ThreeDModel;
