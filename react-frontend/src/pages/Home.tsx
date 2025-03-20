import React from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  FaArrowRight,
  FaCheckCircle,
  FaChartBar,
  FaTachometerAlt,
  FaRegLightbulb,
} from "react-icons/fa";

const Home: React.FC = () => {
  return (
    <>
      {/* Hero Section */}
      <section className="py-16 md:py-24 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 overflow-hidden relative">
        <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>

        {/* Animated circles */}
        <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-primary-400/20 dark:bg-primary-600/10 rounded-full blur-3xl animate-float"></div>
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-primary-300/20 dark:bg-primary-500/10 rounded-full blur-3xl animate-float-slow"></div>

        <div className="container mx-auto px-4 sm:px-6 relative z-10">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
            >
              <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-6 leading-tight">
                <span className="text-primary-600 dark:text-primary-400">
                  AI-Powered
                </span>{" "}
                Image
                <br />
                Forgery Detection
              </h1>

              <p className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-lg">
                Detect manipulated images with high accuracy using our advanced
                CNN model. Identify and visualize tampered regions in seconds.
              </p>

              <div className="flex flex-col sm:flex-row gap-4">
                <Link to="/detect">
                  <motion.button
                    className="px-6 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium shadow-lg flex items-center justify-center"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    Try It Now <FaArrowRight className="ml-2" />
                  </motion.button>
                </Link>

                <Link to="/about">
                  <motion.button
                    className="px-6 py-3 bg-white dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg font-medium shadow border border-gray-200 dark:border-gray-700 flex items-center justify-center"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    Learn More
                  </motion.button>
                </Link>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="relative"
            >
              <div className="bg-white dark:bg-gray-800 p-4 rounded-2xl shadow-xl">
                <div className="rounded-lg overflow-hidden relative aspect-[4/3] bg-gray-100 dark:bg-gray-700">
                  <img
                    src="/demo-image.jpg"
                    alt="Image analysis visualization"
                    className="w-full h-full object-cover"
                    onError={(e) => {
                      e.currentTarget.src =
                        "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/cd6602ef-6f54-4a73-9570-2ef752696cf1/dejiebx-5a7f7d4d-2513-4e63-8909-6bd01d2b3b1a.jpg/v1/fill/w_1920,h_1280,q_75,strp/photoshop_manipulation_fish_by_furkankadran_dejiebx-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9MTI4MCIsInBhdGgiOiJcL2ZcL2NkNjYwMmVmLTZmNTQtNGE3My05NTcwLTJlZjc1MjY5NmNmMVwvZGVqaWVieC01YTdmN2Q0ZC0yNTEzLTRlNjMtODkwOS02YmQwMWQyYjNiMWEuanBnIiwid2lkdGgiOiI8PTE5MjAifV1dLCJhdWQiOlsidXJuOnNlcnZpY2U6aW1hZ2Uub3BlcmF0aW9ucyJdfQ.qunhcLD8O8sgqpkdKtE3zH3Osqdk80QEKIbWvInbZrE";
                    }}
                  />

                  <div className="absolute top-3 right-3 bg-red-600/80 text-white px-3 py-1 rounded-full text-sm backdrop-blur-sm">
                    Tampered
                  </div>
                </div>

                <div className="mt-4 p-2">
                  <div className="bg-red-50 dark:bg-red-900/20 p-3 rounded-lg border border-red-100 dark:border-red-800 mb-4">
                    <div className="flex items-start">
                      <FaCheckCircle className="text-red-600 dark:text-red-400 mt-0.5 mr-3 flex-shrink-0" />
                      <div>
                        <h4 className="font-medium text-red-900 dark:text-red-300">
                          Manipulation Detected
                        </h4>
                        <p className="text-red-700 dark:text-red-400 text-sm">
                          This image shows signs of tampering in the highlighted
                          regions.
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-500 dark:text-gray-400">
                      Confidence Score
                    </span>
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      94%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-1 mb-4">
                    <div
                      className="bg-red-600 h-2 rounded-full"
                      style={{ width: "94%" }}
                    ></div>
                  </div>
                </div>
              </div>

              <motion.div
                className="absolute -top-6 -right-6 bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700"
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.5, duration: 0.5 }}
              >
                <div className="flex items-center">
                  <div className="w-8 h-8 rounded-full bg-primary-100 dark:bg-primary-900/40 flex items-center justify-center mr-2">
                    <FaChartBar className="text-primary-600 dark:text-primary-400" />
                  </div>
                  <div>
                    <div className="font-medium text-gray-900 dark:text-white">
                      94% Accurate
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      Based on analysis
                    </div>
                  </div>
                </div>
              </motion.div>

              <motion.div
                className="absolute -bottom-6 -left-6 bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700"
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.7, duration: 0.5 }}
              >
                <div className="flex items-center">
                  <div className="w-8 h-8 rounded-full bg-primary-100 dark:bg-primary-900/40 flex items-center justify-center mr-2">
                    <FaTachometerAlt className="text-primary-600 dark:text-primary-400" />
                  </div>
                  <div>
                    <div className="font-medium text-gray-900 dark:text-white">
                      CNN Technology
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      Advanced AI detection
                    </div>
                  </div>
                </div>
              </motion.div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-white dark:bg-gray-900">
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
              Advanced Image Forgery Detection
            </motion.h2>

            <motion.p
              className="text-xl text-gray-600 dark:text-gray-400"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              Our system uses state-of-the-art convolutional neural networks to
              identify and localize manipulated regions in digital images.
            </motion.p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                icon: <FaRegLightbulb className="h-6 w-6" />,
                title: "High Accuracy Detection",
                description:
                  "Detect various types of image forgeries with high precision and minimal false positives.",
                gradient: "from-blue-500 to-purple-500",
              },
              {
                icon: <FaCheckCircle className="h-6 w-6" />,
                title: "Multiple Forgery Types",
                description:
                  "Identify splicing, copy-move, removal, and other common image manipulation techniques.",
                gradient: "from-green-500 to-teal-500",
              },
              {
                icon: <FaChartBar className="h-6 w-6" />,
                title: "Detailed Analysis",
                description:
                  "Get comprehensive reports with confidence scores and visual identification of tampered regions.",
                gradient: "from-orange-500 to-pink-500",
              },
              {
                icon: <FaTachometerAlt className="h-6 w-6" />,
                title: "Fast Processing",
                description:
                  "Get results in seconds with our optimized machine learning algorithms and backend.",
                gradient: "from-red-500 to-orange-500",
              },
              {
                icon: <FaCheckCircle className="h-6 w-6" />,
                title: "Visual Heatmaps",
                description:
                  "See exactly where manipulations occur with color-coded visualization overlays.",
                gradient: "from-purple-500 to-indigo-500",
              },
              {
                icon: <FaRegLightbulb className="h-6 w-6" />,
                title: "Easy to Use",
                description:
                  "Simple drag-and-drop interface makes it easy to analyze any image quickly.",
                gradient: "from-cyan-500 to-blue-500",
              },
            ].map((feature, index) => (
              <motion.div
                key={index}
                className="bg-gray-50 dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700 hover:shadow-xl transition-shadow duration-300"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <div
                  className={`w-12 h-12 rounded-lg bg-gradient-to-r ${feature.gradient} flex items-center justify-center text-white mb-6`}
                >
                  {feature.icon}
                </div>

                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-3">
                  {feature.title}
                </h3>

                <p className="text-gray-600 dark:text-gray-400">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How it works section */}
      <section className="py-20 bg-gray-50 dark:bg-gray-950">
        <div className="container mx-auto px-4 sm:px-6">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <motion.span
              className="inline-block text-sm font-semibold text-primary-600 dark:text-primary-400 mb-2 px-3 py-1 bg-primary-50 dark:bg-primary-900/30 rounded-full"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
            >
              Simple Process
            </motion.span>

            <motion.h2
              className="text-3xl lg:text-4xl font-bold mb-4 text-gray-900 dark:text-white"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              How It Works
            </motion.h2>

            <motion.p
              className="text-xl text-gray-600 dark:text-gray-400"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              Our system uses advanced deep learning techniques to analyze
              images and detect forgeries.
            </motion.p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            {[
              {
                number: 1,
                title: "Upload Image",
                description:
                  "Upload any image you want to analyze for potential forgery or manipulation.",
                icon: <FaRegLightbulb className="h-6 w-6" />,
              },
              {
                number: 2,
                title: "AI Analysis",
                description:
                  "Our CNN model processes the image, extracting features to identify manipulated regions.",
                icon: <FaTachometerAlt className="h-6 w-6" />,
              },
              {
                number: 3,
                title: "View Results",
                description:
                  "Get detailed results showing whether the image is authentic or forged, with visualizations.",
                icon: <FaChartBar className="h-6 w-6" />,
              },
            ].map((step, index) => (
              <motion.div
                key={index}
                className="bg-white dark:bg-gray-800 rounded-xl p-8 text-center relative shadow-lg border border-gray-200 dark:border-gray-700"
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.2 }}
              >
                <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2">
                  <div className="w-12 h-12 rounded-full bg-primary-600 dark:bg-primary-500 flex items-center justify-center text-white font-bold text-lg shadow-colored">
                    {step.number}
                  </div>
                </div>

                <div className="h-12 w-12 bg-primary-100 dark:bg-primary-900/30 rounded-full flex items-center justify-center text-primary-600 dark:text-primary-400 mx-auto mb-4">
                  {step.icon}
                </div>

                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-3">
                  {step.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  {step.description}
                </p>
              </motion.div>
            ))}
          </div>

          <div className="text-center mt-14">
            <Link to="/detect">
              <motion.button
                className="px-8 py-4 bg-primary-600 text-white rounded-lg font-medium shadow-lg hover:bg-primary-700 transition duration-300 flex items-center justify-center mx-auto"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                Try It Now <FaArrowRight className="ml-2" />
              </motion.button>
            </Link>
          </div>
        </div>
      </section>

      {/* Stats section */}
      <section className="py-16 bg-white dark:bg-gray-900">
        <div className="container mx-auto px-4 sm:px-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { value: "98%", label: "Accuracy on Test Cases" },
              { value: "2.5s", label: "Average Processing Time" },
              { value: "95%", label: "Localization Precision" },
              { value: "10+", label: "Forgery Types Detected" },
            ].map((stat, index) => (
              <motion.div
                key={index}
                className="bg-gray-50 dark:bg-gray-800 rounded-xl p-6 text-center shadow-md"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <div className="text-3xl font-bold text-primary-600 dark:text-primary-400 mb-2">
                  {stat.value}
                </div>
                <div className="text-gray-600 dark:text-gray-400">
                  {stat.label}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA section */}
      <section className="py-20 bg-gradient-to-r from-primary-600/10 via-primary-500/10 to-primary-400/10 dark:from-primary-900/30 dark:via-primary-800/20 dark:to-primary-700/10">
        <div className="container mx-auto px-4 sm:px-6 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="max-w-3xl mx-auto"
          >
            <h2 className="text-3xl lg:text-4xl font-bold mb-4 text-gray-900 dark:text-white">
              Ready to Detect Image Forgeries?
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-400 mb-8">
              Try our advanced image forgery detection system today and ensure
              the authenticity of your images.
            </p>
            <Link to="/detect">
              <motion.button
                className="px-8 py-4 bg-primary-600 text-white rounded-lg font-medium shadow-lg hover:bg-primary-700 transition duration-300"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                Start Detecting Now
              </motion.button>
            </Link>
          </motion.div>
        </div>
      </section>
    </>
  );
};

export default Home;
