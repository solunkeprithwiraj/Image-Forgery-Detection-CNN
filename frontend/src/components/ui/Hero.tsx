import React from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";

const Hero: React.FC = () => {
  return (
    <section className="relative min-h-[90vh] flex items-center overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0 bg-gradient-to-br from-white to-gray-100 dark:from-gray-900 dark:to-gray-950"></div>
      <div className="absolute inset-0 bg-grid-pattern opacity-20"></div>

      {/* Animated circles */}
      <motion.div
        className="absolute top-20 right-20 w-64 h-64 rounded-full bg-primary-200/30 dark:bg-primary-900/20 blur-3xl"
        animate={{
          scale: [1, 1.2, 1],
          x: [0, 20, 0],
          y: [0, -20, 0],
        }}
        transition={{
          duration: 8,
          repeat: Infinity,
          repeatType: "reverse",
        }}
      />

      <motion.div
        className="absolute bottom-20 left-20 w-96 h-96 rounded-full bg-accent-200/20 dark:bg-accent-900/10 blur-3xl"
        animate={{
          scale: [1, 1.1, 1],
          x: [0, -30, 0],
          y: [0, 30, 0],
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          repeatType: "reverse",
        }}
      />

      <div className="container mx-auto px-4 md:px-6 z-10">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Text content */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
          >
            <motion.span
              className="inline-block text-sm font-semibold text-primary-600 dark:text-primary-400 mb-2 px-3 py-1 bg-primary-50 dark:bg-primary-900/30 rounded-full"
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.5 }}
            >
              Advanced Technology
            </motion.span>

            <motion.h1
              className="text-4xl lg:text-6xl font-bold mb-4 leading-tight bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-400"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5, duration: 0.8 }}
            >
              AI-Powered Image <br />
              Forgery Detection
            </motion.h1>

            <motion.p
              className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-xl"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.7, duration: 0.8 }}
            >
              Detect manipulated images with state-of-the-art CNN technology.
              Our system identifies tampered regions with precision and
              accuracy.
            </motion.p>

            <motion.div
              className="flex flex-col sm:flex-row gap-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.9, duration: 0.8 }}
            >
              <Link to="/detect">
                <motion.button
                  className="px-8 py-4 bg-primary-600 text-white rounded-lg font-medium shadow-lg hover:bg-primary-700 transition duration-300"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Try It Now
                </motion.button>
              </Link>

              <Link to="/about">
                <motion.button
                  className="px-8 py-4 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 text-gray-800 dark:text-white rounded-lg font-medium hover:bg-gray-50 dark:hover:bg-gray-700 transition duration-300"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Learn More
                </motion.button>
              </Link>
            </motion.div>
          </motion.div>

          {/* Image/Visualization */}
          <motion.div
            className="relative"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1 }}
          >
            <div className="relative rounded-2xl overflow-hidden shadow-intense bg-white dark:bg-gray-800">
              <div className="p-8 aspect-video relative flex items-center justify-center">
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-xl text-gray-400 dark:text-gray-600">
                    [Example Visualization]
                  </span>
                </div>

                {/* This would be replaced with actual image in production */}
                <div className="w-full h-full bg-gradient-to-br from-primary-100 to-accent-100 dark:from-primary-900/30 dark:to-accent-900/30 rounded-lg">
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="relative w-full h-full">
                      <div className="absolute top-1/4 left-1/4 w-24 h-24 rounded-lg bg-red-500/20 backdrop-blur-sm border border-red-500/30"></div>
                      <div className="absolute top-1/2 right-1/3 w-32 h-16 rounded-lg bg-red-500/30 backdrop-blur-sm border border-red-500/40"></div>

                      <div className="absolute inset-0 flex items-center justify-center flex-col">
                        <span className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                          Tampering Detected
                        </span>
                        <span className="px-3 py-1 bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 rounded-full text-sm">
                          2 regions identified
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Floating elements */}
            <motion.div
              className="absolute -top-5 -right-5 w-24 h-24 rounded-lg bg-white dark:bg-gray-800 shadow-lg overflow-hidden"
              animate={{
                y: [0, -10, 0],
                rotate: [0, 5, 0],
              }}
              transition={{
                duration: 4,
                repeat: Infinity,
              }}
            >
              <div className="h-full w-full bg-gradient-to-br from-green-100 to-green-200 dark:from-green-900/30 dark:to-green-800/30 p-3 flex items-center justify-center">
                <span className="text-xs font-semibold text-green-800 dark:text-green-300">
                  95% Accuracy
                </span>
              </div>
            </motion.div>

            <motion.div
              className="absolute -bottom-5 -left-5 w-28 h-28 rounded-lg bg-white dark:bg-gray-800 shadow-lg overflow-hidden"
              animate={{
                y: [0, 10, 0],
                rotate: [0, -5, 0],
              }}
              transition={{
                duration: 5,
                repeat: Infinity,
                delay: 1,
              }}
            >
              <div className="h-full w-full bg-gradient-to-br from-primary-100 to-primary-200 dark:from-primary-900/30 dark:to-primary-800/30 p-3 flex items-center justify-center">
                <span className="text-xs font-semibold text-primary-800 dark:text-primary-300">
                  Advanced CNN Technology
                </span>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
