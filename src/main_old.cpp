/**
 * @file main.cpp
 * @brief Entry point for YOLOv8n realtime inference system - OPTIMIZED
 * 
 * RADICAL OPTIMIZATIONS:
 * - Direct BGR path for video (no YUYV conversion)
 * - FP32 throughout (no FP16 overhead)
 * - Pre-allocated buffers
 * - Optimized thread configuration
 */

#include "common.h"
#include "input_pipeline.h"
#include "neon_preprocess.h"
#include "inference_engine.h"
#include "postprocess.h"
#include "benchmark.h"
#include "video_writer.h"
#include "drm_display.h"
#include "bytetrack/BYTETracker.h"
#include "ocsort/OCSort.hpp"
#include <Eigen/Dense>

#include <iostream>
#include <string>
#include <atomic>
#include <signal.h>
#include <getopt.h>
#include <set>
#include <sstream>
#include <algorithm>
#include <deque>
#include <map>
#include <algorithm> // For std::max, std::min

#include <zenoh-pico.h>

using namespace yolo;

// ============================================================================
// Zenoh session
// ============================================================================
z_owned_session_t z_session;

// ============================================================================
// GATE ROI FOR CAM 1
// ============================================================================
// Chỉ số Class từ model của bạn[cite: 2]
const int PERSON_ID = 0;
const int HELMET_SAFE = 1;
const int CLOTHES_VIOLATION = 2; // self_clothes
const int VEST_SAFE = 3;         // safety_clothes
const int HEAD_VIOLATION = 4;    // head (không mũ)
// 5: blur_head, 6: blur_clothes cũng tính là vi phạm[cite: 2]


struct WorkerState {
    int track_id;
    std::deque<int> violation_history; // 1: Vi phạm, 0: An toàn
    bool is_violating;
    int consecutive_misses;
    float last_ground_y = -1.0f; // Tọa độ y trên mặt sàn
    float last_cy = -1.0f;
    bool counted_in = false;
    bool counted_out = false;
};
inline cv::Point2f get_ground_coords(float x1, float y1, float x2, float y2, 
                                     const cv::Mat& H, 
				     int frame_w, int frame_h) {
    // 1. Chuyển tọa độ từ normalized [0,1] về pixel trên ảnh gốc (Original Space)
    // Detections từ postprocess.cpp đã được map ngược về ảnh gốc, nên chỉ cần nhân với resolution.
    float x_pixel_mid = ((x1 + x2) / 2.0f) * (float) frame_w;
    float y_pixel_bottom = y2 * (float) frame_h;

    // std::cout << "DEBUG - Tọa độ Video gốc: x=" << x_pixel_mid << ", y=" << y_pixel_bottom << std::endl;

    // 2. Áp dụng Homography trên tọa độ gốc
    std::vector<cv::Point2f> src_pts = { cv::Point2f(x_pixel_mid, y_pixel_bottom) };
    std::vector<cv::Point2f> dst_pts;
    
    // Kiểm tra ma trận H trước khi biến đổi
    if (H.empty()) return cv::Point2f(0, 0);

    cv::perspectiveTransform(src_pts, dst_pts, H);
    
    return dst_pts[0]; 
}

// Khai báo biến toàn cục
GateROI active_roi;
int factory_in_count = 0;
int factory_out_count = 0;
std::map<int, WorkerState> active_workers;
// ============================================================================
// HELPER FUNCTIONS FOR TRACKING
// ============================================================================
/**
 * @brief Converts a 2D std::vector of float data into an Eigen::Matrix for OC-SORT input.
 * * @param data A 2D vector where each row represents a bounding box: [x1, y1, x2, y2, conf, class_id]
 * @return Eigen::Matrix<float, Eigen::Dynamic, 6> The formatted matrix for OC-SORT update()
 */
inline Eigen::Matrix<float, Eigen::Dynamic, 6> Vector2Matrix(const std::vector<std::vector<float>>& data) {
    if (data.empty()) {
        return Eigen::Matrix<float, 0, 6>();
    }
    Eigen::Matrix<float, Eigen::Dynamic, 6> matrix(data.size(), 6);
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < 6; ++j) {
            matrix(i, j) = data[i][j];
        }
    }
    return matrix;
}

// Function to calculate IoA (Intersection over Area) of PPE relative to the Person bounding box
inline float calculate_ioa(float ppe_x1, float ppe_y1, float ppe_x2, float ppe_y2,
                           float person_x1, float person_y1, float person_x2, float person_y2) {
    float x_left = std::max(ppe_x1, person_x1);
    float y_top = std::max(ppe_y1, person_y1);
    float x_right = std::min(ppe_x2, person_x2);
    float y_bottom = std::min(ppe_y2, person_y2);

    // Check for non-overlapping bounding boxes
    if (x_right < x_left || y_bottom < y_top) return 0.0f;

    float intersection_area = (x_right - x_left) * (y_bottom - y_top);
    float ppe_area = (ppe_x2 - ppe_x1) * (ppe_y2 - ppe_y1);

    // Prevent division by zero
    if (ppe_area <= 0) return 0.0f;
    return intersection_area / ppe_area;
}

// ============================================================================
// Global State
// ============================================================================

std::atomic<bool> g_running{true};

void signal_handler(int sig) {
    g_running.store(false);
}

// ============================================================================
// Command Line Options
// ============================================================================

struct Options {
    std::string mode = "benchmark";      // benchmark, camera, video
    std::string device = "/dev/video0";  // Camera device or video file
    std::string param_path;              // Model param file
    std::string bin_path;                // Model bin file
    int frames = 1000;                   // Frames to process
    int warmup_frames = 30;              // Warmup frames
    bool verbose = false;                // Verbose output
    bool test_model = false;             // Test model loading only
    bool test_inference = false;         // Test single inference only
    bool test_camera = false;            // Test camera capture only
    std::string output_csv;              // Output CSV path
    std::string output_video;            // Output video path (with bbox)
    bool show_fps = true;                // Show FPS overlay in output video
    std::set<int> class_filter;          // Classes to detect (empty = all)
    std::string class_filter_str;        // Original class names string
    bool display_enabled = false;        // Enable display window (OpenCV)
    bool fb_display_enabled = false;      // Enable framebuffer display (no X11)
    bool use_vulkan = false;             // Use Vulkan GPU compute
    bool use_int8 = false;               // Use INT8 quantized model
    int gpu_device = 0;                  // Vulkan GPU device index
    
    bool use_rtsp = false; // Add flag
};

void print_usage(const char* program) {
    std::cout << "YOLOv8n Realtime Inference System for Raspberry Pi 5\n\n";
    std::cout << "Usage: " << program << " [options]\n\n";
    std::cout << "Modes:\n";
    std::cout << "  --benchmark          Run benchmark with video or synthetic data\n";
    std::cout << "  --camera DEVICE      Run with camera input\n";
    std::cout << "  --video FILE         Run with video file input\n\n";
    std::cout << "Model:\n";
    std::cout << "  --param FILE         Path to NCNN .param file\n";
    std::cout << "  --bin FILE           Path to NCNN .bin file\n\n";
    std::cout << "Options:\n";
    std::cout << "  --frames N           Number of frames to process (default: 1000)\n";
    std::cout << "  --warmup N           Warmup frames (default: 30)\n";
    std::cout << "  --output FILE        Export results to CSV\n";
    std::cout << "  --output-video FILE  Save video with bounding boxes\n";
    std::cout << "  --no-fps             Don't show FPS overlay in output video\n";
    std::cout << "  --class NAMES        Filter classes (comma-separated, e.g., 'person,car,dog')\n";
    std::cout << "  --display            Show detection results in window (auto DISPLAY=:0)\n";
    std::cout << "  --vulkan             Use Vulkan GPU (VideoCore VII) for inference\n";
    std::cout << "  --int8               Use INT8 quantized model (faster, similar accuracy)\n";
    std::cout << "  --gpu N              Vulkan GPU device index (default: 0)\n";
    std::cout << "  --verbose            Print per-frame results\n\n";
    std::cout << "Testing:\n";
    std::cout << "  --test-model         Test model loading\n";
    std::cout << "  --test-inference     Test single frame inference\n";
    std::cout << "  --test-camera        Test camera capture\n\n";
}

bool parse_options(int argc, char* argv[], Options& opts) {
    static struct option long_options[] = {
        {"benchmark", no_argument, 0, 'b'},
        {"camera", required_argument, 0, 'c'},
        {"video", required_argument, 0, 'v'},
        {"param", required_argument, 0, 'p'},
        {"bin", required_argument, 0, 'm'},
        {"frames", required_argument, 0, 'n'},
        {"warmup", required_argument, 0, 'w'},
        {"output", required_argument, 0, 'o'},
        {"output-video", required_argument, 0, 'O'},
        {"no-fps", no_argument, 0, 'F'},
        {"class", required_argument, 0, 'C'},
        {"display", no_argument, 0, 'D'},
        {"fb", no_argument, 0, 'B'},
        {"vulkan", no_argument, 0, 'G'},
        {"int8", no_argument, 0, 'I'},
        {"gpu", required_argument, 0, 'g'},
        {"verbose", no_argument, 0, 'V'},
        {"test-model", no_argument, 0, '1'},
        {"test-inference", no_argument, 0, '2'},
        {"test-camera", no_argument, 0, '3'},
        {"device", required_argument, 0, 'd'},
        {"help", no_argument, 0, 'h'},
        {"rtsp", required_argument, 0, 'r'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "bc:v:p:m:n:w:o:O:FC:DGIg:Vd:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'b':
                opts.mode = "benchmark";
                break;
            case 'c':
                opts.mode = "camera";
                opts.device = optarg;
                break;
            case 'v':
                opts.mode = "video";
                opts.device = optarg;
                break;
            case 'p':
                opts.param_path = optarg;
                break;
            case 'm':
                opts.bin_path = optarg;
                break;
            case 'n':
                opts.frames = std::stoi(optarg);
                break;
            case 'w':
                opts.warmup_frames = std::stoi(optarg);
                break;
            case 'o':
                opts.output_csv = optarg;
                break;
            case 'O':
                opts.output_video = optarg;
                break;
            case 'F':
                opts.show_fps = false;
                break;
            case 'C':
                opts.class_filter_str = optarg;
                break;
            case 'D':
                opts.display_enabled = true;
                break;
            case 'B':
                opts.fb_display_enabled = true;
                break;
            case 'G':
                opts.use_vulkan = true;
                break;
            case 'I':
                opts.use_int8 = true;
                break;
            case 'g':
                opts.gpu_device = std::stoi(optarg);
                break;
            case 'V':
                opts.verbose = true;
                break;
            case 'd':
                opts.device = optarg;
                break;
            case '1':
                opts.test_model = true;
                break;
            case '2':
                opts.test_inference = true;
                break;
            case '3':
                opts.test_camera = true;
                break;
            case 'h':
                print_usage(argv[0]);
                return false;
            case 'r':
                opts.mode = "rtsp";
                opts.device = optarg;
                break;
            default:
                print_usage(argv[0]);
                return false;
        }
    }

    // Validate required options
    if (!opts.test_camera && (opts.param_path.empty() || opts.bin_path.empty())) {
        std::cerr << "Error: --param and --bin are required\n";
        return false;
    }
    
    std::cout << "MAIN.CPP: LINE 249"; 

    // Parse class filter if specified
    if (!opts.class_filter_str.empty()) {
        std::stringstream ss(opts.class_filter_str);
        std::string class_name;
        while (std::getline(ss, class_name, ',')) {
            // Trim whitespace
            class_name.erase(0, class_name.find_first_not_of(" \t"));
            class_name.erase(class_name.find_last_not_of(" \t") + 1);
            
            // Convert to lowercase for comparison
            std::string class_lower = class_name;
            std::transform(class_lower.begin(), class_lower.end(), class_lower.begin(), ::tolower);
            
            bool found = false;
            for (int i = 0; i < NUM_CLASSES; i++) {
                std::string coco_lower = CLASS_NAMES[i];
                std::transform(coco_lower.begin(), coco_lower.end(), coco_lower.begin(), ::tolower);
                if (class_lower == coco_lower) {
                    opts.class_filter.insert(i);
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cerr << "Warning: Unknown class '" << class_name << "' ignored\n";
                std::cerr << "Available classes: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, etc.\n";
            }
        }
        
        if (!opts.class_filter.empty()) {
            std::cout << "Class filter: ";
            for (int id : opts.class_filter) {
                std::cout << CLASS_NAMES[id] << " ";
            }
            std::cout << "(" << opts.class_filter.size() << " classes)\n";
        }
    }

    return true;
}

// ============================================================================
// Test Functions
// ============================================================================

int test_model_loading(const Options& opts) {
    std::cout << "Testing model loading...\n";
    
    InferenceEngine engine;
    InferenceEngine::Config config;
    config.param_path = opts.param_path;
    config.bin_path = opts.bin_path;
    config.num_threads = NCNN_NUM_THREADS;
    config.use_fp16 = true;
    
    ErrorCode err = engine.initialize(config);
    if (err != ErrorCode::SUCCESS) {
        std::cerr << "Failed to load model: " << error_to_string(err) << "\n";
        return 1;
    }
    
    std::cout << "Model loaded successfully\n";
    return 0;
}

int test_single_inference(const Options& opts) {
    std::cout << "Testing single frame inference...\n";
    
    // Initialize engine
    InferenceEngine engine;
    InferenceEngine::Config config;
    config.param_path = opts.param_path;
    config.bin_path = opts.bin_path;
    config.num_threads = NCNN_NUM_THREADS;
    config.use_fp16 = true;
    
    ErrorCode err = engine.initialize(config);
    if (err != ErrorCode::SUCCESS) {
        std::cerr << "Failed to load model: " << error_to_string(err) << "\n";
        return 1;
    }
    
    // Warmup
    std::cout << "Warming up...\n";
    engine.warmup(3);
    
    // Create test input (all zeros)
    AlignedPtr<__fp16> input = make_aligned_buffer<__fp16>(MODEL_INPUT_SIZE);
    memset(input.get(), 0, MODEL_INPUT_SIZE * sizeof(__fp16));
    
    // Run inference
    DetectionResult result;
    int64_t inference_time;
    
    {
        ScopedTimer timer(inference_time);
        err = engine.infer(input.get(), result);
    }
    
    if (err != ErrorCode::SUCCESS) {
        std::cerr << "Inference failed: " << error_to_string(err) << "\n";
        return 1;
    }
    
    std::cout << "Inference completed in " << inference_time << " us\n";
    std::cout << "Detections: " << result.count << "\n";
    
    return 0;
}

int test_camera_capture(const Options& opts) {
    std::cout << "Testing camera capture from " << opts.device << "...\n";
    
    InputPipeline pipeline;
    InputPipeline::Config config;
    config.source = InputSource::CAMERA_V4L2;
    config.device_path = opts.device;
    config.width = INPUT_WIDTH;
    config.height = INPUT_HEIGHT;
    config.fps = 30;
    
    ErrorCode err = pipeline.initialize(config);
    if (err != ErrorCode::SUCCESS) {
        std::cerr << "Failed to initialize camera: " << error_to_string(err) << "\n";
        return 1;
    }
    
    int frames_captured = 0;
    int target_frames = opts.frames > 0 ? opts.frames : 30;
    
    err = pipeline.start([&](const FrameBuffer& frame) -> bool {
        frames_captured++;
        std::cout << "Frame " << frames_captured << ": " 
                  << frame.width << "x" << frame.height 
                  << ", " << frame.size << " bytes\n";
        return frames_captured < target_frames && g_running.load();
    });
    
    if (err != ErrorCode::SUCCESS) {
        std::cerr << "Camera capture failed: " << error_to_string(err) << "\n";
        return 1;
    }
    
    std::cout << "Captured " << frames_captured << " frames\n";
    return 0;
}

// ============================================================================
// Main Processing Loop
// ============================================================================

int run_inference_pipeline(const Options& opts) {
    std::cout << "Starting OPTIMIZED inference pipeline...\n";
    std::cout << "Mode: " << opts.mode << "\n";
    std::cout << "Device/File: " << opts.device << "\n";
    
    // Benchmark MOT
    std::ofstream mot_file("mot_results.txt");
    if (!opts.output_video.empty()) {
        std::cout << "Output video: " << opts.output_video << "\n";
    }
    
    // Initialize preprocessing buffers ONCE
    neon::init_preprocess_buffers();
    
    // Initialize inference engine
    InferenceEngine engine;
    InferenceEngine::Config engine_config;
    engine_config.param_path = opts.param_path;
    engine_config.bin_path = opts.bin_path;
    
    // Get thread count from environment or use default
    const char* omp_threads = getenv("OMP_NUM_THREADS");
    if (omp_threads) {
        engine_config.num_threads = std::atoi(omp_threads);
    } else {
        // OPTIMIZATION: Use fewer threads when display is active
        // This reduces CPU contention with X11/Qt display thread
        // Testing shows 3 threads + display = 26 FPS vs 4 threads + display = 22 FPS
        if (opts.display_enabled) {
            engine_config.num_threads = 3;  // Reserve 1 core for display
            std::cout << "Display mode: Using 3 NCNN threads (reduce contention)\n";
        } else {
            engine_config.num_threads = NCNN_NUM_THREADS;
        }
    }
    
    // INT8 and Vulkan options
    engine_config.use_int8 = opts.use_int8;
    engine_config.use_vulkan = opts.use_vulkan;
    engine_config.gpu_device = opts.gpu_device;
    
    // Don't use FP16 with INT8 mode
    engine_config.use_fp16 = !opts.use_int8;
    
    ErrorCode err = engine.initialize(engine_config);
    if (err != ErrorCode::SUCCESS) {
        std::cerr << "Failed to load model: " << error_to_string(err) << "\n";
        neon::cleanup_preprocess_buffers();
        return 1;
    }
    
    // Print acceleration status
    std::cout << "Model loaded (threads=" << engine_config.num_threads;
    if (engine.is_using_int8()) std::cout << ", INT8";
    if (engine.is_using_vulkan()) std::cout << ", Vulkan";
    std::cout << "), warming up...\n";
    engine.warmup(30);  // Extended warmup for JIT/cache priming and stability
    
    // Initialize input pipeline
    InputPipeline pipeline;
    InputPipeline::Config input_config;
    
    if (opts.mode == "camera") {
        input_config.source = InputSource::CAMERA_V4L2;
        input_config.device_path = opts.device;
    } else if (opts.mode == "rtsp") { // <--- Add check
        input_config.source = InputSource::RTSP_STREAM;
        input_config.device_path = opts.device;
    } else {
        input_config.source = InputSource::VIDEO_FILE;
        input_config.device_path = opts.device;
        input_config.loop_video = opts.output_video.empty();  // Don't loop if saving video
    }
    
    input_config.width = INPUT_WIDTH;
    input_config.height = INPUT_HEIGHT;
    input_config.fps = 30;
    
    err = pipeline.initialize(input_config);
    if (err != ErrorCode::SUCCESS) {
        std::cerr << "Failed to initialize input: " << error_to_string(err) << "\n";
        neon::cleanup_preprocess_buffers();
        return 1;
    }
    
    // Initialize async video writer (if output video specified)
    std::unique_ptr<AsyncVideoWriter> video_writer;
    int video_width = INPUT_WIDTH;   // Use pipeline resolution (already resized)
    int video_height = INPUT_HEIGHT;
    
    if (!opts.output_video.empty()) {
        video_writer = std::make_unique<AsyncVideoWriter>();
        AsyncVideoWriter::Config writer_config;
        writer_config.output_path = opts.output_video;
        writer_config.width = video_width;
        writer_config.height = video_height;
        writer_config.fps = pipeline.get_video_fps();  // Match input video FPS
        // Queue size: buffer all frames (inference faster than encoding)
        // For typical videos: ~30fps * 60sec = 1800 frames max (~500MB)
        writer_config.queue_size = 2000;
        writer_config.draw_fps = opts.show_fps;
        
        if (!video_writer->start(writer_config)) {
            std::cerr << "Failed to initialize video writer\n";
            neon::cleanup_preprocess_buffers();
            return 1;
        }
        
        std::cout << "Video writer initialized: " << video_width << "x" << video_height << "\n";
    }
    
    // Initialize async display (if enabled)
    std::unique_ptr<AsyncDisplay> display;
    if (opts.display_enabled) {
        display = std::make_unique<AsyncDisplay>();
        AsyncDisplay::Config display_config;
        display_config.window_name = "YOLOv8n Detection";
        display_config.queue_size = 3;  // Very small for low latency
        display_config.draw_fps = opts.show_fps;
        display_config.draw_bbox = true;
        display_config.max_screen_ratio = 0.75f;  // 75% of screen max
        
        if (!display->start(display_config, INPUT_WIDTH, INPUT_HEIGHT)) {
            std::cerr << "Failed to initialize display\n";
            neon::cleanup_preprocess_buffers();
            return 1;
        }
    }
    
    // Initialize framebuffer display (if enabled) - bypasses X11, faster!
    std::unique_ptr<FramebufferDisplay> fb_display;
    if (opts.fb_display_enabled) {
        fb_display = std::make_unique<FramebufferDisplay>();
        FramebufferDisplay::Config fb_config;
        fb_config.target_width = INPUT_WIDTH;
        fb_config.target_height = INPUT_HEIGHT;
        fb_config.draw_fps = opts.show_fps;
        fb_config.draw_bbox = true;
        
        if (!fb_display->start(fb_config)) {
            std::cerr << "Failed to initialize framebuffer display\n";
            std::cerr << "Try: sudo chmod 666 /dev/fb0\n";
            neon::cleanup_preprocess_buffers();
            return 1;
        }
        std::cout << "Framebuffer mode: No X11 overhead, max FPS!\n";
    }
    
    // Initialize benchmark
    Benchmark benchmark;
    BenchmarkConfig bench_config;
    bench_config.warmup_frames = opts.warmup_frames;
    bench_config.test_frames = opts.frames;
    bench_config.verbose = opts.verbose;
    benchmark.configure(bench_config);
    
    // Allocate FP32 model input buffer (pre-allocated, reused)
    AlignedPtr<float> model_input = make_aligned_buffer<float>(MODEL_INPUT_FLOATS);
    if (!model_input) {
        std::cerr << "Failed to allocate input buffer\n";
        neon::cleanup_preprocess_buffers();
        return 1;
    }

    z_owned_config_t z_config;
    z_config_default(&z_config);

    zp_config_insert(z_config_loan_mut(&z_config), Z_CONFIG_MODE_KEY, "peer");
    zp_config_insert(z_config_loan_mut(&z_config), Z_CONFIG_LISTEN_KEY, "tcp/0.0.0.0:7447");

    if (z_open(&z_session, z_config_move(&z_config), NULL) < 0) {
        std::cerr << "[ZENOH] Cannot open session!" << std::endl;
        return -1;
    }
    zp_start_read_task(z_session_loan_mut(&z_session), NULL);
    zp_start_lease_task(z_session_loan_mut(&z_session), NULL);
    std::cout << "[ZENOH] Listening on tcp/0.0.0.0:7447" << std::endl;

    // Khai báo publisher một lần duy nhất
    z_owned_publisher_t z_pub;
    z_view_keyexpr_t ke;
    z_view_keyexpr_from_str(&ke, "factory/cam1/stats");
    if (z_declare_publisher(z_session_loan(&z_session), &z_pub, z_view_keyexpr_loan(&ke), NULL) < 0) {
         std::cerr << "[ZENOH] Cannot declare publisher!" << std::endl;
         return -1;
    }
    std::cout << "[ZENOH] Publisher declared." << std::endl;


    // Initialize Tracker
    //int tracker_fps = pipeline.get_video_fps() > 0 ? (int)pipeline.get_video_fps() : 30;
    //BYTETracker tracker(tracker_fps, 30); // 30 is track_buffer
    
    // ========================================================================
    // INITIALIZE OC-SORT TRACKER
    // ========================================================================
    // Parameters configuration:
    // det_thresh: 0.45f    (Minimum confidence to consider a detection valid)
    // max_age: 30          (Maximum frames to keep a lost track alive)
    // min_hits: 1          (Minimum consecutive detections to confirm a track)
    // iou_threshold: 0.3f  (IoU threshold for matching)
    // delta_t: 3           (Time gap for velocity/momentum calculation)
    // asso_func: "giou"    (Association function: Generalized IoU)
    // inertia: 0.2f        (Weight of momentum/direction consistency)
    // use_byte: false      (Disable ByteTrack low-score association logic if fps is low)
    //ocsort::OCSort tracker(0.25f, 30, 1, 0.3f, 3, "giou", 0.2f, true);
    ocsort::OCSort tracker(0.25f, 30, 1, 0.4f, 3, "giou", 0.2f, true); // for gemini_1.mp4 (High quality, clear seperation)
    
    std::cout << "Starting frame processing...\n";
    std::cout << "Warmup: " << opts.warmup_frames << " frames\n";
    std::cout << "Test: " << opts.frames << " frames\n\n";
    
    //initial ROI feature
    // active_roi.H = (cv::Mat_<double>(3,3) << -10.612869946298655, -0.9971824110616158, 3373.3256419913027, -0.9529660530068571, -33.35381185523972, 6623.114068397604, 0.0007716650779543593, -0.021876598125090636, 1.0);
    active_roi.H = active_roi.H = (cv::Mat_<double>(3,3) << 1.25, 0.0, -200.0, 0.0, 2.857, -571.4, 0.0, 0.0, 1.0);  
    // active_roi.ground_crossing_line = 300.0;
    active_roi.ground_crossing_line = 400.0; // for gemini_1.mp4




    // Track FPS for overlay
    float rolling_fps = 0;
    float rolling_inference_ms = 0;
    
    // Frame processing callback
    auto process_frame = [&](const FrameBuffer& frame) -> bool {
        FrameTiming timing;
        timing.frame_index = frame.frame_index;
        
        int64_t total_start = get_timestamp_ns();
        
        timing.capture_time_us = 0;
        
        // Preprocessing - dispatch based on pixel format
        float scale;
        int pad_x, pad_y;
        int64_t preprocess_time = 0;
        
        {
            ScopedTimer timer(preprocess_time);
            
            if (frame.format == PixelFormat::BGR) {
                // OPTIMIZED: Direct BGR path (video files)
                neon::preprocess_bgr_direct(
                    frame.data,
                    model_input.get(),
                    frame.width,
                    frame.height,
                    frame.stride,
                    &scale, &pad_x, &pad_y
                );
            } else {
                // Camera path (YUYV)
                neon::preprocess_yuyv_to_fp32(
                    frame.data,
                    model_input.get(),
                    &scale, &pad_x, &pad_y
                );
            }
        }
        timing.preprocess_time_us = preprocess_time;
        
        // Set letterbox params for coordinate mapping
        engine.set_letterbox_params(scale, pad_x, pad_y);
        
        // Inference - DIRECT FP32
        DetectionResult result;
        int64_t inference_time = 0;
        
        {
            ScopedTimer timer(inference_time);
            
            ErrorCode infer_err = engine.infer_fp32(model_input.get(), result);
            if (infer_err != ErrorCode::SUCCESS) {
                std::cerr << "Inference failed on frame " << frame.frame_index << "\n";
            }
        }
        timing.inference_time_us = inference_time;
        
        timing.postprocess_time_us = 0;
        
        // Filter detections by class if specified
        if (!opts.class_filter.empty()) {
            int write_idx = 0;
            for (int i = 0; i < result.count; i++) {
                if (opts.class_filter.count(result.detections[i].class_id) > 0) {
                    if (write_idx != i) {
                        result.detections[write_idx] = result.detections[i];
                    }
                    write_idx++;
                }
            }
            result.count = write_idx;
        }
        
        int64_t tracking_time_us = 0; // Measure tracking time
        
        /*======================= OC-SORT TRACKING & ROI PPE LOGIC ===============================*/
	{
	    ScopedTimer timer(tracking_time_us);
	    std::vector<std::vector<float>> ocsort_input;
	    std::vector<yolo::Detection> ppe_detections;
	    std::vector<yolo::Detection> final_detections;

	    // 1. Phân loại Detection (Chạy trên Image Plane)[cite: 1]
	    for (int i = 0; i < result.count; i++) {
		auto& det = result.detections[i];
		if (det.class_id == PERSON_ID) {
		    ocsort_input.push_back({det.x1, det.y1, det.x2, det.y2, det.confidence, (float)det.class_id});
		} else {
		    ppe_detections.push_back(det);
		}
	    }

	    // 2. Cập nhật Tracker
	    std::vector<Eigen::RowVectorXf> output_stracks = tracker.update(Vector2Matrix(ocsort_input));
	    

	    // Export MOT for Benchmark
	    for (auto& track : output_stracks) {
		    int tid = static_cast<int>(track[4]);
		    //MOT format: <frame> <id>, <bb_left>, <bb_width>, <<bb_height>,<conf>,-1, -1, -1
		    mot_file<< frame.frame_index << "," << tid << ","
			    << track[0] <<"," << track[1] << ","
			    << track[2] <<"," << (track[3] - track[1])<< ","
			    << track[6] <<"-1,-1,-1"<<std::endl;
	    }
	    for (auto& track : output_stracks) {
		    int tid = static_cast<int>(track[4]);
		    yolo::Detection det_p;
		    det_p.x1 = track[0]; det_p.y1 = track[1]; det_p.x2 = track[2]; det_p.y2 = track[3];
		    det_p.confidence = track[6];

		    if (active_workers.find(tid) == active_workers.end()) {
			// Khởi tạo last_ground_y = 0 thay vì -1 để tránh lỗi logic khung hình đầu
			active_workers[tid] = {tid, {}, false, 0, -1.0f, -1.0f, false, false};
		    }
		    auto& state = active_workers[tid];
		    state.consecutive_misses = 0;

		    // Tính tọa độ sàn    
		    cv::Point2f ground_pos = get_ground_coords(det_p.x1, det_p.y1, det_p.x2, det_p.y2, active_roi.H, frame.width, frame.height);
		    float current_ground_y = ground_pos.y;

		    // [QUAN TRỌNG] Dòng Debug để bạn xem tọa độ thực trên console
		    if (frame.frame_index % 10 == 0) {
			std::cout << "ID: " << tid << " | Ground Y: " << current_ground_y << std::endl;
		    }

		    if (state.last_ground_y > 0 && current_ground_y >= 0) {
			// Logic đếm IN
			if (state.last_ground_y < active_roi.ground_crossing_line && current_ground_y >= active_roi.ground_crossing_line) {
			    if (!state.counted_in) { 
				factory_in_count++; 
				state.counted_in = true; 
				std::cout << ">>> COUNT IN! ID "<< tid << " Total: " << factory_in_count << std::endl;
				// Push to Zenoh 
				std::string payload = "{\"in\":" + std::to_string(factory_in_count) +
				      ",\"out\":" + std::to_string(factory_out_count) + "}";
				z_owned_bytes_t z_payload;
				z_bytes_copy_from_str(&z_payload, payload.c_str());
				z_publisher_put(z_publisher_loan(&z_pub), z_bytes_move(&z_payload), NULL);
			    }
			}
			// Logic đếm OUT
			else if (state.last_ground_y > active_roi.ground_crossing_line && current_ground_y <= active_roi.ground_crossing_line) {
			    if (!state.counted_out) { 
				factory_out_count++; 
				state.counted_out = true; 
				std::cout << "<<< COUNT OUT! ID "<< tid <<" Total: " << factory_out_count << std::endl;
				// Push to Zenoh 
				std::string payload = "{\"in\":" + std::to_string(factory_in_count) +
				      ",\"out\":" + std::to_string(factory_out_count) + "}";
				z_owned_bytes_t z_payload;
				z_bytes_copy_from_str(&z_payload, payload.c_str());
				z_publisher_put(z_publisher_loan(&z_pub), z_bytes_move(&z_payload), NULL);
			    }
			}

           	    }	
                state.last_ground_y = current_ground_y;   
	   	// --- HÀNH ĐỘNG 2: CHECK PPE (Image-Plane Logic) ---[cite: 1]
		// Mở rộng vùng đầu 15% trên ảnh gốc để check mũ[cite: 1]
		float head_y1 = std::max(0.0f, det_p.y1 - (det_p.height() * 0.15f));
		int current_frame_violation = 0; 
		
		for (const auto& ppe : ppe_detections) {
		    float ioa = calculate_ioa(ppe.x1, ppe.y1, ppe.x2, ppe.y2, det_p.x1, head_y1, det_p.x2, det_p.y2);
		    if (ioa > 0.6f && (ppe.class_id == HEAD_VIOLATION || ppe.class_id == CLOTHES_VIOLATION)) {
			current_frame_violation = 1;
			break;
		    }
		}

		// Temporal Voting (Cửa sổ 5 frame)[cite: 1]
		state.violation_history.push_back(current_frame_violation);
		if (state.violation_history.size() > 5) state.violation_history.pop_front();
		
		int votes = 0;
		for (int v : state.violation_history) votes += v;
		state.is_violating = (votes >= 3); // Vi phạm nếu >= 3/5 frame báo lỗi[cite: 1]

		// Mã hóa ID để vẽ: Số âm là vi phạm (Box đỏ)[cite: 1]
		det_p.class_id = state.is_violating ? -(tid + 1000) : (tid + 1000);
		final_detections.push_back(det_p);
	    }
	    for (auto it = active_workers.begin(); it != active_workers.end(); ) {
            	it->second.consecutive_misses++;
            	if (it->second.consecutive_misses > 15) { // Nếu mất dấu 15 frame (~1.5 giây)
                   it = active_workers.erase(it);
            	} else {
                   ++it;
            	}
	    }	
	    // 3. Đồng bộ kết quả cuối cùng
	    int write_idx = 0;
	    for (auto& d : final_detections) if (write_idx < MAX_DETECTIONS) result.detections[write_idx++] = d;
	    for (auto& p : ppe_detections) if (write_idx < MAX_DETECTIONS) result.detections[write_idx++] = p;
	    result.count = write_idx;
	}
	/*===========================================================================*/
	timing.postprocess_time_us = tracking_time_us;
        
        // Total time
        int64_t total_end = get_timestamp_ns();
        timing.total_time_us = (total_end - total_start) / 1000;
        timing.detection_count = result.count;
        
        // Update rolling stats for FPS overlay
        float current_fps = 1000000.0f / timing.total_time_us;
        float current_inference_ms = inference_time / 1000.0f;
        rolling_fps = rolling_fps * 0.9f + current_fps * 0.1f;
        rolling_inference_ms = rolling_inference_ms * 0.9f + current_inference_ms * 0.1f;
        
        // Push frame to async writer (non-blocking, done AFTER inference)
        if (video_writer && frame.format == PixelFormat::BGR) {
            // Create cv::Mat wrapper (no copy, just wrap existing data)
            cv::Mat bgr_frame(frame.height, frame.width, CV_8UC3, frame.data, frame.stride);
            if (!bgr_frame.empty()) {
	 	BBoxRenderer::draw_roi(bgr_frame, active_roi); 
                // --- VẼ VẠCH KẺ ĐẾM (LINE CROSSING) ---
                // Chúng ta lấy 2 điểm trên vạch 300 trong không gian BEV và map ngược về ảnh gốc
                cv::Mat H_inv = active_roi.H.inv();
                std::vector<cv::Point2f> bev_points = { 
                    cv::Point2f(0, active_roi.ground_crossing_line), 
                    cv::Point2f(400, active_roi.ground_crossing_line) 
                };
                std::vector<cv::Point2f> img_points;
                cv::perspectiveTransform(bev_points, img_points, H_inv);

                // Vẽ đường nối 2 điểm này trên ảnh gốc
                cv::line(bgr_frame, img_points[0], img_points[1], cv::Scalar(0, 255, 255), 3); // Màu vàng
                cv::putText(bgr_frame, "COUNTING LINE", img_points[0] + cv::Point2f(10, -10), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);   		 // Hiển thị bộ đếm In/Out
    		std::string stats = "IN: " + std::to_string(factory_in_count) + 
                       		" | OUT: " + std::to_string(factory_out_count);
   		cv::putText(bgr_frame, stats, cv::Point(20, 60), 
                		cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 3);     
	    }  
            // Push to async queue (bbox drawing happens in writer thread)
            video_writer->push(bgr_frame, result, frame.width, frame.height,
                             rolling_fps, rolling_inference_ms);
        }
        
        // Push frame to async display (non-blocking)
        if (display) {
            if (frame.format == PixelFormat::BGR) {
                cv::Mat bgr_frame(frame.height, frame.width, CV_8UC3, frame.data, frame.stride);
                display->push(bgr_frame, result, frame.width, frame.height,
                             rolling_fps, rolling_inference_ms);
            } else if (frame.format == PixelFormat::YUYV) {
                // Convert YUYV to BGR for display
                cv::Mat yuyv_frame(frame.height, frame.width, CV_8UC2, frame.data, frame.stride);
                cv::Mat bgr_frame;
                cv::cvtColor(yuyv_frame, bgr_frame, cv::COLOR_YUV2BGR_YUYV);
                display->push(bgr_frame, result, frame.width, frame.height,
                             rolling_fps, rolling_inference_ms);
            }
        }
        
        // Push frame to framebuffer display (direct, no X11 overhead)
        if (fb_display) {
            if (frame.format == PixelFormat::BGR) {
                fb_display->push_bgr(frame.data, frame.width, frame.height, frame.stride,
                                    result, rolling_fps, rolling_inference_ms);
            } else if (frame.format == PixelFormat::YUYV) {
                // Convert YUYV to BGR for framebuffer display
                cv::Mat yuyv_frame(frame.height, frame.width, CV_8UC2, frame.data, frame.stride);
                cv::Mat bgr_frame;
                cv::cvtColor(yuyv_frame, bgr_frame, cv::COLOR_YUV2BGR_YUYV);
                fb_display->push_bgr(bgr_frame.data, bgr_frame.cols, bgr_frame.rows, 
                                    bgr_frame.step, result, rolling_fps, rolling_inference_ms);
            }
        }
        
        // Record timing
        benchmark.record_frame(timing);
        
        // Progress indicator
        if (!opts.verbose && benchmark.current_frame() % 100 == 0) {
            std::cout << "Frame " << benchmark.current_frame() 
                      << " | " << timing.total_time_us << "us"
                      << " | " << current_fps << " FPS";
            if (video_writer) {
                std::cout << " | Queue: " << video_writer->queue_size();
            }
            std::cout << "\r" << std::flush;
        }
        
        return !benchmark.is_complete() && g_running.load();
    };
    
    // Run pipeline
    err = pipeline.start(process_frame);
    
    std::cout << "\n";
    
    // Stop video writer (flushes remaining frames)
    if (video_writer) {
        std::cout << "Flushing video writer...\n";
        video_writer->stop();
        std::cout << "Video saved: " << opts.output_video << "\n";
        std::cout << "  Frames written: " << video_writer->frames_written() << "\n";
        std::cout << "  Frames dropped: " << video_writer->frames_dropped() << "\n";
    }
    
    // Stop async display
    if (display) {
        display->stop();
    }
    
    // Print results
    benchmark.print_summary();
    
    // Export CSV if requested
    if (!opts.output_csv.empty()) {
        benchmark.export_csv(opts.output_csv);
        std::cout << "Results exported to " << opts.output_csv << "\n";
    }
    
    // Cleanup
    neon::cleanup_preprocess_buffers();
    
    // Determine exit code based on validation
    BenchmarkStats stats = benchmark.calculate_stats();
    
    if (stats.is_valid()) {
        std::cout << "\n✓ SYSTEM MEETS ALL PERFORMANCE REQUIREMENTS\n";
        std::cout << "  FPS (P99): " << stats.fps_p99 << " >= 20\n";
        return 0;
    } else {
        std::cout << "\n✗ SYSTEM DOES NOT MEET PERFORMANCE REQUIREMENTS\n";
        std::cout << "  FPS (P99): " << stats.fps_p99 << " < 20\n";
        return 1;
    }
    // clean up zenoh session
    z_publisher_drop(z_publisher_move(&z_pub));
    zp_stop_read_task(z_session_loan_mut(&z_session));
    zp_stop_lease_task(z_session_loan_mut(&z_session));
    z_session_drop(z_session_move(&z_session));
}

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char* argv[]) {
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Parse options
    Options opts;
    if (!parse_options(argc, argv, opts)) {
        return 1;
    }
    
    // Run appropriate test/mode
    if (opts.test_model) {
        return test_model_loading(opts);
    }
    
    if (opts.test_inference) {
        return test_single_inference(opts);
    }
    
    if (opts.test_camera) {
        return test_camera_capture(opts);
    }
    
    // Run full pipeline
    return run_inference_pipeline(opts);
}
