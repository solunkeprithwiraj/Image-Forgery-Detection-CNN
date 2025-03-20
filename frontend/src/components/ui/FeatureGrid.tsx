import React from "react";
import { motion } from "framer-motion";
import {
  FaShieldAlt,
  FaSearchLocation,
  FaChartLine,
  FaRocket,
  FaMicrochip,
  FaLayerGroup,
} from "react-icons/fa";

const features = [
  {
    icon: <FaShieldAlt className="h-6 w-6" />,
    title: "High Accuracy Detection",
    description:
      "Our model achieves state-of-the-art accuracy in detecting various types of image manipulations.",
    gradient: "from-primary-600 to-primary-400",
  },
  {
    icon: <FaSearchLocation className="h-6 w-6" />,
    title: "Precise Localization",
    description:
      "Identify exactly where an image has been tampered with through our advanced localization techniques.",
    gradient: "from-blue-600 to-blue-400",
  },
  {
    icon: <FaChartLine className="h-6 w-6" />,
    title: "Detailed Analytics",
    description:
      "Get comprehensive analysis with confidence scores and detailed reports on detected forgeries.",
    gradient: "from-green-600 to-green-400",
  },
  {
    icon: <FaRocket className="h-6 w-6" />,
    title: "Fast Processing",
    description:
      "Our optimized algorithms provide quick results without compromising on accuracy.",
    gradient: "from-red-600 to-orange-400",
  },
  {
    icon: <FaMicrochip className="h-6 w-6" />,
    title: "Advanced CNN Architecture",
    description:
      "Powered by a sophisticated CNN with residual connections and attention mechanisms.",
    gradient: "from-indigo-600 to-indigo-400",
  },
  {
    icon: <FaLayerGroup className="h-6 w-6" />,
    title: "Multiple Visualizations",
    description:
      "View results through heatmaps, overlays, and contour detection for comprehensive understanding.",
    gradient: "from-purple-600 to-purple-400",
  },
];

const FeatureGrid: React.FC = () => {
  return (
    <section className="py-20 bg-gray-50 dark:bg-gray-900">
      <div className="container mx-auto px-4 sm:px-6">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <motion.span
            className="inline-block text-sm font-semibold text-primary-600 dark:text-primary-400 mb-2 px-3 py-1 bg-primary-50 dark:bg-primary-900/30 rounded-full"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            Powerful Features
          </motion.span>

          <motion.h2
            className="text-3xl lg:text-4xl font-bold mb-4 text-gray-900 dark:text-white"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            Advanced Detection Capabilities
          </motion.h2>

          <motion.p
            className="text-xl text-gray-600 dark:text-gray-400"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            Our CNN-based image forgery detection system offers a comprehensive
            set of features to help you identify and analyze manipulated images.
          </motion.p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              className="bg-white dark:bg-gray-800 rounded-xl overflow-hidden shadow-md hover:shadow-xl transition-shadow"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              whileHover={{ y: -5 }}
            >
              <div className="p-1">
                <div
                  className={`bg-gradient-to-r ${feature.gradient} rounded-t-xl p-4 flex items-center justify-center`}
                >
                  <div className="w-12 h-12 rounded-full bg-white/20 flex items-center justify-center text-white">
                    {feature.icon}
                  </div>
                </div>

                <div className="p-6">
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-3">
                    {feature.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    {feature.description}
                  </p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FeatureGrid;
