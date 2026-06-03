/**
 * @file main.cpp
 * @brief YOLOv8n Realtime PPE Inference System - Zenoh Integrated
 *
 * FIXES applied vs previous version:
 *  [FIX-1] pub_image: gửi raw JPEG binary thay vì base64-in-JSON để tránh
 *           Zenoh-pico drop payload khi size > internal transport buffer.
 *  [FIX-2] pub_events: tách metadata JSON và JPEG image ra 2 topic riêng
 *           (/events và /events/image) — tránh payload khổng lồ + lỗi
 *           z_bytes_copy_from_str dừng sớm nếu gặp null byte.
 *  [FIX-3] Dangling pointer trong keyexpr: lưu tất cả topic string vào
 *           biến có lifetime đủ dài trước khi gọi z_view_keyexpr_from_str.
 *  [FIX-4] Tất cả z_bytes dùng z_bytes_copy_from_buf thay vì _from_str
 *           để xử lý binary payload đúng cách.
 *  [FIX-5] consecutive_misses bị reset về 0 đúng chỗ; trước đây logic
 *           increment chạy sau khi erase nên worker miss không bao giờ bị xóa.
 */

#include "common.h"
#include "input_pipeline.h"
#include "neon_preprocess.h"
#include "inference_engine.h"
#include "postprocess.h"
#include "benchmark.h"
#include "video_writer.h"
#include "drm_display.h"
#include "zenoh_transport.h"
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
#include <chrono>
#include <vector>
#include <fstream>
#include <iomanip>

using namespace yolo;


// ============================================================================
// Logic Constants
// ============================================================================
constexpr int PERSON_ID        = 0;
constexpr int HELMET_SAFE      = 1;
constexpr int CLOTHES_VIOLATION = 2;
constexpr int VEST_SAFE        = 3;
constexpr int HEAD_VIOLATION   = 4;

constexpr int VIOL_NO_HELMET    = 1 << 0;
constexpr int VIOL_SELF_CLOTHES = 1 << 1;

// Cấu hình violation voting
constexpr int VIOLATION_HISTORY_LEN  = 7;   // số frame lưu lịch sử
constexpr int VIOLATION_VOTE_THRESH  = 5;   // bao nhiêu frame viol/5 thì coi là vi phạm
constexpr int WORKER_MAX_MISS        = 30;  // frame miss liên tiếp trước khi xóa worker
constexpr int VIOLATION_CLEAR_FRAMES = 10;  // safe frames required before closing an event episode
constexpr int VIOLATION_COOLDOWN_SEC = 30;  // suppress repeated events for the same worker/type
constexpr float IOA_THRESHOLD        = 0.6f;
constexpr float HEAD_REGION_RATIO    = 0.15f; // mở rộng lên trên bbox để check đầu

// Cấu hình JPEG snapshot
constexpr int SNAPSHOT_WIDTH   = 320;
constexpr int SNAPSHOT_HEIGHT  = 240;
constexpr int SNAPSHOT_QUALITY = 60;  // 0-100, thấp hơn = nhỏ hơn

// Interval publish
constexpr int STATS_INTERVAL_SEC  = 2;
constexpr int IMAGE_INTERVAL_SEC  = 5;
constexpr bool DEBUG_GROUND_LINE  = true;

// ============================================================================
// Worker State
// ============================================================================
struct WorkerState {
    int  track_id          = -1;
    std::deque<int> violation_history;
    bool is_violating      = false;
    int  consecutive_misses = 0;
    float last_ground_y    = -1.0f;
    bool counted_in        = false;
    bool counted_out       = false;
    bool event_sent        = false;
    int  safe_frames       = 0;
    int  last_violation_mask = 0;
    int  last_event_mask     = 0;
    std::chrono::steady_clock::time_point last_event_time{};
};

GateROI                 active_roi;
int                     factory_in_count  = 0;
int                     factory_out_count = 0;
int                     violation_event_count = 0;
std::map<int, WorkerState> active_workers;

// ============================================================================
// Global State & Signal Handling
// ============================================================================
std::atomic<bool> g_running{true};

static void draw_dashboard_snapshot_overlay(cv::Mat& frame,
                                            const DetectionResult& result,
                                            int frame_width,
                                            int frame_height,
                                            float rolling_fps,
                                            float rolling_inference_ms) {
    if (frame.empty()) return;

    BBoxRenderer::draw_roi(frame, active_roi);

    if (!active_roi.H.empty()) {
        cv::Mat H_inv = active_roi.H.inv();
        std::vector<cv::Point2f> bev_points = {
            cv::Point2f(0, active_roi.ground_crossing_line),
            cv::Point2f(400, active_roi.ground_crossing_line)
        };
        std::vector<cv::Point2f> img_points;
        cv::perspectiveTransform(bev_points, img_points, H_inv);
        if (img_points.size() >= 2) {
            cv::line(frame, img_points[0], img_points[1],
                     cv::Scalar(0, 255, 255), 3);
        }
    }

    double ui_scale = std::min(frame_width / 640.0, frame_height / 480.0);
    ui_scale = std::max(ui_scale, 0.5);
    int margin_x = static_cast<int>(20 * ui_scale);
    int margin_y = static_cast<int>(60 * ui_scale);
    double font_scale = 1.0 * ui_scale;
    int thickness = std::max(1, static_cast<int>(3 * ui_scale));

    std::string stats = "IN: " + std::to_string(factory_in_count) +
                        " | OUT: " + std::to_string(factory_out_count);
    cv::putText(frame, stats, cv::Point(margin_x, margin_y),
                cv::FONT_HERSHEY_SIMPLEX, font_scale,
                cv::Scalar(0, 255, 255), thickness);

    BBoxRenderer::draw(frame, result, frame_width, frame_height);
    BBoxRenderer::draw_fps(frame, rolling_fps, rolling_inference_ms);
}

void signal_handler(int /*sig*/) {
    g_running.store(false);
}

// ============================================================================
// CLI Options
// ============================================================================
struct Options {
    std::string mode           = "camera";
    std::string device         = "/dev/video0";
    std::string param_path;
    std::string bin_path;
    int         frames         = 1000;
    int         warmup_frames  = 30;
    bool        verbose        = false;
    std::string output_video;
    bool        display_enabled    = false;
    bool        fb_display_enabled = false;
    bool        use_vulkan         = false;
    bool        use_int8           = false;
    bool              use_zenoh          = false;
    bool show_fps = true;                // Show FPS overlay in output video
    int         gpu_device         = 0;
    std::string cam_id             = "cam1";
    std::string node_id            = "gsn";
    std::string router_ip          = "192.168.1.9";
    std::string zenoh_listen       = "tcp/0.0.0.0:7448";
    std::string asn_peer           = "tcp/192.168.1.197:7447";
};

void print_usage(const char* program) {
    std::cout
        << "YOLOv8n PPE System with Zenoh\n"
        << "Usage: " << program << " [options]\n\n"
        << "  --camera  DEV    Camera device path (default: /dev/video0)\n"
        << "  --video   FILE   Video file path\n"
        << "  --param   FILE   NCNN .param path\n"
        << "  --bin     FILE   NCNN .bin path\n"
        << "  --cam-id  ID     Camera node identifier (default: cam1)\n"
        << "  --node-id ID     Surveillance node identifier (default: gsn)\n"
        << "  --zenoh         Enable Zenoh cloud + P2P publishing\n"
        << "  --zenoh-router IP   Zenoh router IP for cloud/dashboard (default: 192.168.1.9)\n"
        << "  --zenoh-listen LOC  Local P2P listen locator (default: tcp/0.0.0.0:7448)\n"
        << "  --asn-peer LOC      ASN direct peer locator, e.g. tcp/192.168.1.9:7447\n"
        << "  --display        Enable DRM/KMS display\n"
        << "  --fb             Enable framebuffer display\n"
        << "  --vulkan         Use Vulkan GPU backend\n"
        << "  --int8           Use INT8 quantization\n"
        << "  --verbose        Verbose logging\n"
        << "  --rtsp    IP     Video stream through RTSP sever\n";       
}

bool parse_options(int argc, char* argv[], Options& opts) {
    static struct option long_options[] = {
        {"camera",       required_argument, 0, 'c'},
        {"video",        required_argument, 0, 'v'},
        {"param",        required_argument, 0, 'p'},
        {"bin",          required_argument, 0, 'm'},
        {"frames",       required_argument, 0, 'n'},
        {"warmup",       required_argument, 0, 'w'},
        {"output-video", required_argument, 0, 'O'},
        {"display",      no_argument,       0, 'D'},
        {"fb",           no_argument,       0, 'B'},
        {"vulkan",       no_argument,       0, 'G'},
        {"int8",         no_argument,       0, 'I'},
        {"gpu",          required_argument, 0, 'g'},
        {"verbose",      no_argument,       0, 'V'},
        {"cam-id",       required_argument, 0, 'k'},
        {"node-id",      required_argument, 0, 1000},
        {"zenoh",        no_argument,       0, 1001},
        {"zenoh-listen", required_argument, 0, 1002},
        {"asn-peer",     required_argument, 0, 1003},
        {"zenoh-peer",   required_argument, 0, 1003},
        {"zenoh-router", required_argument, 0, 'z'},
        {"help",         no_argument,       0, 'h'},
        {"rtsp",         required_argument, 0, 'r'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "c:v:p:m:n:w:O:DBGIg:Vk:z:h:r",
                              long_options, nullptr)) != -1) {
        switch (opt) {
            case 'c': opts.mode = "camera"; opts.device = optarg;   break;
            case 'v': opts.mode = "video";  opts.device = optarg;   break;
            case 'p': opts.param_path   = optarg;                   break;
            case 'm': opts.bin_path     = optarg;                   break;
            case 'n': opts.frames       = std::stoi(optarg);        break;
            case 'w': opts.warmup_frames = std::stoi(optarg);       break;
            case 'O': opts.output_video = optarg;                   break;
            case 'D': opts.display_enabled    = true;               break;
            case 'B': opts.fb_display_enabled = true;               break;
            case 'G': opts.use_vulkan   = true;                     break;
            case 'I': opts.use_int8     = true;                     break;
            case 'g': opts.gpu_device   = std::stoi(optarg);        break;
            case 'V': opts.verbose      = true;                     break;
            case 'k': opts.cam_id       = optarg;                   break;
            case 1000: opts.node_id     = optarg;                   break;
            case 1001: opts.use_zenoh   = true;                     break;
            case 1002: opts.zenoh_listen = optarg; opts.use_zenoh = true; break;
            case 1003: opts.asn_peer    = optarg; opts.use_zenoh = true; break;
            case 'z': opts.router_ip    = optarg; opts.use_zenoh = true; break;
            case 'r': opts.mode = "rtsp";opts.device = optarg;      break;
            case 'h': print_usage(argv[0]); exit(0);
            default: break;
        }
    }
    return true;
}

// ============================================================================
// Helpers
// ============================================================================

/**
 * Encode cv::Mat thành JPEG, trả về raw bytes.
 * Không dùng base64 — gửi thẳng binary qua Zenoh.
 */
static std::vector<uchar> encode_jpeg(const cv::Mat& img, int quality = 80) {
    std::vector<uchar> buf;
    if (img.empty()) return buf;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, quality};
    if (!cv::imencode(".jpg", img, buf, params)) {
        std::cerr << "[ERROR] OpenCV imencode failed!\n";
        buf.clear();
    }
    return buf;
}

static void publish_count_state(ZenohPublisher& zenoh, const Options& opts) {
    if (!zenoh.enabled()) return;

    zenoh.publish_count(opts.node_id, factory_in_count, factory_out_count);
}

/**
 * Chuyển tọa độ normalized bbox → tọa độ mặt phẳng BEV qua homography.
 */
static cv::Point2f get_ground_coords(float x1, float y1,
                                      float x2, float y2,
                                      const cv::Mat& H,
                                      int frame_w, int frame_h) {
    if (H.empty()) return {0.f, 0.f};
    float px = ((x1 + x2) / 2.f) * (float)frame_w;
    float py = y2 * (float)frame_h;
    std::vector<cv::Point2f> src = {{px, py}};
    std::vector<cv::Point2f> dst;
    cv::perspectiveTransform(src, dst, H);
    return dst[0];
}

/**
 * Tính tỉ lệ diện tích giao nhau / diện tích PPE bbox (IoA).
 * Dùng để xét PPE nằm bên trong bbox người hay không.
 */
static float calculate_ioa(float ppe_x1, float ppe_y1,
                             float ppe_x2, float ppe_y2,
                             float per_x1, float per_y1,
                             float per_x2, float per_y2) {
    float xl = std::max(ppe_x1, per_x1);
    float yt = std::max(ppe_y1, per_y1);
    float xr = std::min(ppe_x2, per_x2);
    float yb = std::min(ppe_y2, per_y2);
    if (xr < xl || yb < yt) return 0.f;
    float inter = (xr - xl) * (yb - yt);
    float area  = (ppe_x2 - ppe_x1) * (ppe_y2 - ppe_y1);
    return (area <= 0.f) ? 0.f : inter / area;
}

/**
 * Chuyển vector<vector<float>> sang Eigen matrix (6 cột) cho OCSort.
 */
static Eigen::Matrix<float, Eigen::Dynamic, 6>
to_eigen_matrix(const std::vector<std::vector<float>>& data) {
    if (data.empty()) return Eigen::Matrix<float, 0, 6>();
    Eigen::Matrix<float, Eigen::Dynamic, 6> mat(data.size(), 6);
    for (size_t i = 0; i < data.size(); ++i)
        for (size_t j = 0; j < 6; ++j)
            mat(i, j) = data[i][j];
    return mat;
}



static inline int64_t now_us() {
    return get_timestamp_ns() / 1000;
}

static const char* violation_type_name(int bit) {
    switch (bit) {
        case VIOL_NO_HELMET:    return "NO_HELMET";
        case VIOL_SELF_CLOTHES: return "SELF_CLOTHES";
        default:                return "UNKNOWN";
    }
}

static std::vector<std::string> violation_types_from_mask(int mask) {
    std::vector<std::string> types;
    if (mask & VIOL_NO_HELMET)
        types.emplace_back(violation_type_name(VIOL_NO_HELMET));
    if (mask & VIOL_SELF_CLOTHES)
        types.emplace_back(violation_type_name(VIOL_SELF_CLOTHES));
    if (types.empty())
        types.emplace_back("UNKNOWN");
    return types;
}

static std::string join_violation_types(int mask, const std::string& sep) {
    const auto types = violation_types_from_mask(mask);
    std::ostringstream oss;
    for (size_t i = 0; i < types.size(); ++i) {
        if (i > 0) oss << sep;
        oss << types[i];
    }
    return oss.str();
}

static std::string violation_types_json(int mask) {
    const auto types = violation_types_from_mask(mask);
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < types.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "\"" << types[i] << "\"";
    }
    oss << "]";
    return oss.str();
}


// Main Pipeline
// ============================================================================

int run_inference_pipeline(const Options& opts) {

    // ------------------------------------------------------------------
    // 1. NEON Preprocess buffers
    // ------------------------------------------------------------------
    std::cout << "[INIT] Initializing NEON buffers...\n";
    neon::init_preprocess_buffers();

    // ------------------------------------------------------------------
    // 2. Inference engine
    // ------------------------------------------------------------------
    std::cout << "[INIT] Setting up inference engine...\n";
    InferenceEngine engine;
    {
        InferenceEngine::Config cfg;
        cfg.param_path  = opts.param_path;
        cfg.bin_path    = opts.bin_path;
        cfg.num_threads = opts.display_enabled ? 3 : NCNN_NUM_THREADS;
        cfg.use_int8    = opts.use_int8;
        cfg.use_vulkan  = opts.use_vulkan;
        cfg.gpu_device  = opts.gpu_device;
        cfg.use_fp16    = !opts.use_int8;
        if (engine.initialize(cfg) != ErrorCode::SUCCESS) {
            std::cerr << "[FATAL] Failed to load model!\n";
            return 1;
        }
    }
    std::cout << "[INIT] Model loaded. Warming up...\n";
    engine.warmup(10);

    // ------------------------------------------------------------------
    // 3. Input pipeline
    // ------------------------------------------------------------------
    std::cout << "[INIT] Initializing input pipeline ("
              << opts.mode << " : " << opts.device << ")...\n";
    InputPipeline pipeline;
    {
        InputPipeline::Config cfg;
        if (opts.mode == "camera") {
            cfg.source = InputSource::CAMERA_V4L2;
            cfg.device_path = opts.device;
        } else if (opts.mode == "rtsp") { // <--- Add check
            cfg.source = InputSource::RTSP_STREAM;
            cfg.device_path = opts.device;
        } else {
            cfg.source = InputSource::VIDEO_FILE;
            cfg.device_path = opts.device;
            cfg.loop_video = opts.output_video.empty();  // Don't loop if saving video
        }

        cfg.width       = INPUT_WIDTH;
        cfg.height      = INPUT_HEIGHT;
        cfg.fps         = 30;
        if (pipeline.initialize(cfg) != ErrorCode::SUCCESS) {
            std::cerr << "[FATAL] Failed to initialize input pipeline!\n";
            return 1;
        }
    }
    std::cout << "[INIT] Input pipeline ready.\n";


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

    // ------------------------------------------------------------------
    // 4. Zenoh session
    // ------------------------------------------------------------------
    
    

    ZenohPublisher zenoh;
    ZenohConfig zenoh_cfg;
    zenoh_cfg.enabled = opts.use_zenoh;
    zenoh_cfg.cam_id = opts.cam_id;
    zenoh_cfg.router_ip = opts.router_ip;
    zenoh_cfg.p2p_listen = opts.zenoh_listen;
    zenoh_cfg.asn_peer = opts.asn_peer;
    if (!zenoh.initialize(zenoh_cfg)) return 1;

    // ------------------------------------------------------------------
    // 6. Tracker & ROI
    // ------------------------------------------------------------------
    ocsort::OCSort tracker(0.25f, 30, 3, 0.25f, 3, "giou", 0.3f, true);
    // det_thresh=0.25, max_age=30, min_hits=3 (tăng từ 1→3 để tránh ID giả),
    // iou_threshold=0.3 (giảm từ 0.4→0.3 để match tốt hơn khi đông người),
    // delta_t=3, asso_func="giou", inertia=0.2, use_byte=false (tắt để ổn định hơn)

    // Homography matrix (camera-specific, cấu hình theo thực tế)
 
  // FOR kling.mp4
//    active_roi.H = (cv::Mat_<double>(3,3) <<
//        -5.3064349731493275,  -0.6647882740410772,   3373.3256419913027,
//        -0.47648302650342855, -22.235874570159813,   6623.114068397604,
//         0.00038583253897717965, -0.014584398750060424, 1.0 );
//    active_roi.ground_crossing_line = 500.0f;

  // FOR gemini_1.mp4
    active_roi.H = (cv::Mat_<double>(3,3) <<
        0.625,      0.0,   -200.0,
          0.0,   1.9046,   -571.4,
          0.0,      0.0,      1.0);
    active_roi.ground_crossing_line = 550.0f;
    
    // ------------------------------------------------------------------
    // 7. Timing
    // ------------------------------------------------------------------
    auto last_stats_time = std::chrono::steady_clock::now();
    auto last_image_time = std::chrono::steady_clock::now();
    const auto stats_interval = std::chrono::seconds(STATS_INTERVAL_SEC);
    const auto image_interval = std::chrono::seconds(IMAGE_INTERVAL_SEC);

    // ------------------------------------------------------------------
    // 8. Model input buffer (aligned)
    // ------------------------------------------------------------------
    AlignedPtr<float> model_input = make_aligned_buffer<float>(MODEL_INPUT_FLOATS);

    std::cout << "[INIT] Starting frame loop...\n";
    

    Benchmark benchmark;
    
    BenchmarkConfig bench_cfg;
    
    bench_cfg.warmup_frames = 30;
    bench_cfg.test_frames   = 0;
    bench_cfg.verbose       = false;
    
    bench_cfg.output_path = "pipeline_benchmark.csv";
    
    benchmark.configure(bench_cfg);
    
    std::cout << "[BENCHMARK] Initialized\n";

    // Track FPS for overlay
    float rolling_fps = 0;
    float rolling_inference_ms = 0;
    
    // ------------------------------------------------------------------
    // 9. Per-frame callback
    // ------------------------------------------------------------------
    auto process_frame = [&](const FrameBuffer& frame) -> bool {



        FrameTiming timing = {};
        
        timing.frame_index = frame.frame_index;
        
        int64_t total_start = now_us();


        if (opts.verbose && frame.frame_index % 30 == 0)
            std::cout << "[FRAME] #" << frame.frame_index << "\n";

        auto now = std::chrono::steady_clock::now();

        // ---- Preprocess ------------------------------------------------
        int64_t t0 = now_us();
        
        float scale;
        int pad_x, pad_y;
        
        neon::preprocess_bgr_direct(
            frame.data,
            model_input.get(),
            frame.width,
            frame.height,
            frame.stride,
            &scale,
            &pad_x,
            &pad_y);
        
        timing.preprocess_time_us = now_us() - t0;
        engine.set_letterbox_params(scale, pad_x, pad_y, frame.width, frame.height);

        // ---- Inference -------------------------------------------------
        DetectionResult result;
        
        t0 = now_us();
        engine.infer_fp32(model_input.get(), result);
        timing.inference_time_us = now_us() - t0;
        
        if (opts.verbose && frame.frame_index % 30 == 0)
            std::cout << "[INFER] Detections: " << result.count << "\n";

        // ---- Split detections: person vs PPE ---------------------------
        std::vector<std::vector<float>> tracker_input;
        std::vector<yolo::Detection>    ppe_detections;
        std::vector<yolo::Detection>    final_detections;   // Detections với class_id đã mã hóa safe/violation
        tracker_input.reserve(result.count);
        ppe_detections.reserve(result.count);
        final_detections.reserve(result.count);

        for (int i = 0; i < result.count; ++i) {
            const auto& d = result.detections[i];
            if (d.class_id == PERSON_ID)
                tracker_input.push_back({d.x1, d.y1, d.x2, d.y2,
                                          d.confidence, 0.f});
            else
                ppe_detections.push_back(d);
        }

        // [DEBUG] In ra số lượng và điểm tự tin của người do YOLO bắt được
        if (!tracker_input.empty() || active_workers.size() > 0) {
            std::cout << "\n--- FRAME " << frame.frame_index << " ---\n";
            std::cout << "[DEBUG YOLO] Persons: " << tracker_input.size() << " | Confs: ";
            for (auto& d : tracker_input) std::cout << std::fixed << std::setprecision(2) << d[4] << " ";
            std::cout << "\n";
        }

        // ---- Track persons ---------------------------------------------
        auto tracks = tracker.update(to_eigen_matrix(tracker_input));
        int  current_violations = 0;

        // ---- Update worker states --------------------------------------
        // Mark all existing workers as potentially missing this frame
        for (auto& [id, ws] : active_workers)
            ws.consecutive_misses++;

        std::cout << "[DEBUG TRACKER] Active IDs returned: " << tracks.size() << " -> ";
        for (auto& track : tracks) std::cout << (int)track[4] << " ";
        std::cout << "\n";

        for (auto& track : tracks) {
            const int tid = static_cast<int>(track[4]);

            // Tạo state mới nếu chưa có
            if (active_workers.find(tid) == active_workers.end()) {
                WorkerState ws;
                ws.track_id = tid;
                active_workers[tid] = ws;
                std::cout << "[TRACK] New worker: " << tid << "\n";
            }

            auto& state = active_workers[tid];
            state.consecutive_misses = 0;  // [FIX-5] reset đúng chỗ

            // ---- Gate counting (BEV crossing) --------------------------
            cv::Point2f ground = get_ground_coords(
                track[0], track[1], track[2], track[3],
                active_roi.H, frame.width, frame.height);

            if (DEBUG_GROUND_LINE) {
                const float line_y = active_roi.ground_crossing_line;
                const char* side = ground.y >= line_y ? "BELOW_OR_ON" : "ABOVE";
                std::cout << "[DEBUG GROUND] frame=" << frame.frame_index
                          << " worker=" << tid
                          << " ground=(" << std::fixed << std::setprecision(2)
                          << ground.x << "," << ground.y << ")"
                          << " line_y=" << line_y
                          << " delta_y=" << (ground.y - line_y)
                          << " last_y=" << state.last_ground_y
                          << " side=" << side << "\n";
            }

            if (state.last_ground_y > 0.f
                && ground.y >= active_roi.ground_crossing_line
                && state.last_ground_y < active_roi.ground_crossing_line) {
                if (DEBUG_GROUND_LINE) {
                    std::cout << "[DEBUG COUNT_CANDIDATE] worker=" << tid
                              << " direction=IN"
                              << " last_y=" << state.last_ground_y
                              << " current_y=" << ground.y
                              << " line_y=" << active_roi.ground_crossing_line
                              << "\n";
                }
                // Đi vào (từ trên xuống line)
                if (!state.counted_in) {
                    ++factory_in_count;
                    state.counted_in = true;
                    publish_count_state(zenoh, opts);
                    std::cout << "[COUNT] IN: " << factory_in_count << "\n";
                }
            }
            else if (state.last_ground_y >= active_roi.ground_crossing_line
                     && ground.y < active_roi.ground_crossing_line
                     && state.last_ground_y > 0.f) {
                if (DEBUG_GROUND_LINE) {
                    std::cout << "[DEBUG COUNT_CANDIDATE] worker=" << tid
                              << " direction=OUT"
                              << " last_y=" << state.last_ground_y
                              << " current_y=" << ground.y
                              << " line_y=" << active_roi.ground_crossing_line
                              << "\n";
                }
                // Đi ra (từ dưới lên line)
                if (!state.counted_out) {
                    ++factory_out_count;
                    state.counted_out = true;
                    publish_count_state(zenoh, opts);
                    std::cout << "[COUNT] OUT: " << factory_out_count << "\n";
                }
            }
            state.last_ground_y = ground.y;

            // ---- PPE Violation check -----------------------------------
            int frame_violation_mask = 0;
            // Mở rộng vùng đầu lên trên bbox để nhận diện mũ bảo hiểm
            float head_y1 = std::max(0.f,
                track[1] - (track[3] - track[1]) * HEAD_REGION_RATIO);

            for (const auto& ppe : ppe_detections) {
                float ioa = calculate_ioa(
                    ppe.x1, ppe.y1, ppe.x2, ppe.y2,
                    track[0], head_y1, track[2], track[3]);
                if (ioa > IOA_THRESHOLD
                    && (ppe.class_id == HEAD_VIOLATION
                        || ppe.class_id == CLOTHES_VIOLATION)) {
                    if (ppe.class_id == HEAD_VIOLATION)
                        frame_violation_mask |= VIOL_NO_HELMET;
                    else
                        frame_violation_mask |= VIOL_SELF_CLOTHES;
                }
            }

            // Sliding-window vote
            state.violation_history.push_back(frame_violation_mask);
            if ((int)state.violation_history.size() > VIOLATION_HISTORY_LEN)
                state.violation_history.pop_front();

            int head_votes = 0;
            int clothes_votes = 0;
            for (int mask : state.violation_history) {
                if (mask & VIOL_NO_HELMET)    ++head_votes;
                if (mask & VIOL_SELF_CLOTHES) ++clothes_votes;
            }

            int voted_violation_mask = 0;
            if (head_votes >= VIOLATION_VOTE_THRESH)
                voted_violation_mask |= VIOL_NO_HELMET;
            if (clothes_votes >= VIOLATION_VOTE_THRESH)
                voted_violation_mask |= VIOL_SELF_CLOTHES;

            const bool violation_candidate = (voted_violation_mask != 0);

            if (violation_candidate) {
                state.is_violating = true;
                state.last_violation_mask = voted_violation_mask;
                state.safe_frames = 0;
            }
            else if (state.is_violating) {
                ++state.safe_frames;
                if (state.safe_frames >= VIOLATION_CLEAR_FRAMES) {
                    state.is_violating = false;
                    state.event_sent = false;
                    state.last_violation_mask = 0;
                    state.safe_frames = 0;
                }
            }
            else {
                state.safe_frames = 0;
            }

            // ---- Publish violation event (once per violation episode/set) --
            if (state.is_violating) {
                ++current_violations;

                const bool violation_set_changed =
                    state.last_event_mask != state.last_violation_mask;
                if (!state.event_sent || violation_set_changed) {
                    const auto cooldown = std::chrono::seconds(VIOLATION_COOLDOWN_SEC);
                    const bool has_last_event =
                        state.last_event_time.time_since_epoch().count() != 0;
                    const bool same_violation_set =
                        state.last_event_mask == state.last_violation_mask;
                    const bool cooldown_active =
                        has_last_event && same_violation_set
                        && (now - state.last_event_time < cooldown);

                    if (!cooldown_active) {
                    // [FIX-2] Metadata JSON — không nhúng ảnh vào đây
                        if (zenoh.enabled()) {
                            const int event_id = ++violation_event_count;
                            const auto event_ts = std::chrono::duration_cast<
                                std::chrono::milliseconds>(
                                now.time_since_epoch()).count();
                            std::string ev_json =
                                "{\"event_id\":\"" + opts.cam_id + "-" +
                                std::to_string(event_id) + "\""
                                ",\"id\":\"WK" + std::to_string(tid) +
                                "\",\"worker_id\":" + std::to_string(tid) +
                                ",\"type\":\"" + join_violation_types(state.last_violation_mask, "+") + "\""
                                ",\"types\":" + violation_types_json(state.last_violation_mask) +
                                ",\"location\":\"" + opts.cam_id + "\""
                                ",\"cam_id\":\"" + opts.cam_id + "\""
                                ",\"timestamp\":" + std::to_string(event_ts) +
                                ",\"snapshot_topic\":\"factory/" + opts.cam_id +
                                "/events/image\""
                                "}";
                            zenoh.publish_event(ev_json);
                    }

                    // [FIX-2] Ảnh crop worker → topic riêng → raw JPEG
                    cv::Mat raw(frame.height, frame.width,
                                CV_8UC3, frame.data, frame.stride);
                    cv::Rect roi(
                        static_cast<int>(track[0] * frame.width),
                        static_cast<int>(track[1] * frame.height),
                        static_cast<int>((track[2] - track[0]) * frame.width),
                        static_cast<int>((track[3] - track[1]) * frame.height));
                    roi &= cv::Rect(0, 0, frame.width, frame.height);

                    auto jpeg = encode_jpeg(raw(roi), 75);
                    if (!jpeg.empty() && zenoh.enabled()) {
                        // Thêm worker ID vào topic để backend biết ảnh của ai
                        // factory/cam1/events/image  (backend dùng metadata từ
                        // /events để ghép, hoặc dùng Zenoh attachment nếu cần)
                        zenoh.publish_event_image(jpeg.data(), jpeg.size());
                        std::cout << "[ZENOH] Event image sent: "
                                  << jpeg.size() << " bytes (WK"
                                  << tid << ")\n";
                    }

                    state.last_event_time = now;
                    state.last_event_mask = state.last_violation_mask;
                    }

                    state.event_sent = true;
                }
            }
            // ---- Mã hóa violation vào class_id để BBoxRenderer vẽ màu đúng ----
            // class_id âm  → vi phạm (hộp đỏ + nhãn VIOLATION)
            // class_id dương → an toàn (hộp xanh + nhãn SAFE)
            yolo::Detection det_p;
            det_p.x1 = track[0]; det_p.y1 = track[1];
            det_p.x2 = track[2]; det_p.y2 = track[3];
            det_p.confidence = track[6];
            det_p.class_id = state.is_violating ? -(tid + 1000) : (tid + 1000);
            final_detections.push_back(det_p);
        }

        // ---- Build render result before dashboard snapshot publish ----
        {
            int write_idx = 0;
            for (auto& d : final_detections)
                if (write_idx < MAX_DETECTIONS) result.detections[write_idx++] = d;
            for (auto& p : ppe_detections)
                if (write_idx < MAX_DETECTIONS) result.detections[write_idx++] = p;
            result.count = write_idx;
        }

        // ---- Periodic stats publish ------------------------------------
        if (now - last_stats_time >= stats_interval) {
            const int current_workers = static_cast<int>(tracks.size());
            float compliance = current_workers == 0
                ? 100.f
                : (1.f - (float)current_violations
                         / (float)current_workers) * 100.f;
            if (zenoh.enabled()){
                    const auto stats_ts = std::chrono::duration_cast<
                        std::chrono::milliseconds>(
                        now.time_since_epoch()).count();
                    std::string s_json =
                        "{\"cam_id\":\"" + opts.cam_id + "\""
                        ",\"status\":\"online\""
                        ",\"timestamp\":"  + std::to_string(stats_ts) +
                        ",\"total\":"      + std::to_string(current_workers) +
                        ",\"violations\":" + std::to_string(current_violations) +
                        ",\"compliance\":" + std::to_string(compliance) +
                        ",\"in\":"         + std::to_string(factory_in_count) +
                        ",\"out\":"        + std::to_string(factory_out_count) +
                        ",\"fps\":"        + std::to_string(rolling_fps) +
                        ",\"inference_ms\":" + std::to_string(rolling_inference_ms) +
                        ",\"event_count\":" + std::to_string(violation_event_count) +
                        "}";    
                    zenoh.publish_stats(s_json);
                    std::cout << "[ZENOH] Stats: " << s_json << "\n";
            }
            last_stats_time = now;
        }

        // ---- Periodic snapshot publish ---------------------------------
        //  [FIX-1] Gửi raw JPEG binary, không wrap base64-in-JSON
        if (now - last_image_time >= image_interval) {
            cv::Mat raw(frame.height, frame.width,
                        CV_8UC3, frame.data, frame.stride);
            cv::Mat annotated = raw.clone();
            draw_dashboard_snapshot_overlay(annotated, result,
                                            frame.width, frame.height,
                                            rolling_fps, rolling_inference_ms);

            cv::Mat thumb;
            cv::resize(annotated, thumb, cv::Size(SNAPSHOT_WIDTH, SNAPSHOT_HEIGHT));
            auto jpeg = encode_jpeg(thumb, SNAPSHOT_QUALITY);
            if (!jpeg.empty() && zenoh.enabled()) {
                zenoh.publish_snapshot(jpeg.data(), jpeg.size());
                std::cout << "[ZENOH] Snapshot sent: "
                          << jpeg.size() << " bytes\n";
            }
            last_image_time = now;
        }

        // ---- Evict stale workers ---------------------------------------
        //  [FIX-5] consecutive_misses đã được increment ở đầu loop,
        //           nên chỉ cần check > threshold ở đây
        for (auto it = active_workers.begin(); it != active_workers.end(); ) {
            if (it->second.consecutive_misses > 0 && it->second.consecutive_misses < WORKER_MAX_MISS) {
                // In ra cảnh báo khi worker bắt đầu bị miss (không in liên tục nếu miss quá nhiều để tránh rác)
                if (it->second.consecutive_misses == 1 || it->second.consecutive_misses % 15 == 0) {
                    std::cout << "[DEBUG MISS] Worker " << it->second.track_id 
                              << " is missing for " << it->second.consecutive_misses << " frames\n";
                }
            }

            if (it->second.consecutive_misses > WORKER_MAX_MISS) {
                std::cout << "[TRACK] Removed stale worker: "
                          << it->second.track_id << " (Missed " << it->second.consecutive_misses << " frames)\n";
                it = active_workers.erase(it);
            } else {
                ++it;
            }
        }

        // ---- Ghi final_detections (class_id mã hóa safe/violation) + ppe vào result ----
        {
            int write_idx = 0;
            for (auto& d : final_detections)
                if (write_idx < MAX_DETECTIONS) result.detections[write_idx++] = d;
            for (auto& p : ppe_detections)
                if (write_idx < MAX_DETECTIONS) result.detections[write_idx++] = p;
            result.count = write_idx;
        }

        timing.total_time_us = now_us() - total_start;
        
        timing.detection_count = result.count;
        
        // Update rolling stats for FPS overlay
        float current_fps = 1000000.0f / timing.total_time_us;
        float current_inference_ms = timing.inference_time_us / 1000.0f;
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
                
                double ui_scale = std::min(frame.width / 640.0, frame.height / 480.0);
                
                int margin_x = static_cast<int>(20 * ui_scale);
                int margin_y = static_cast<int>(60 * ui_scale);
                double font_scale = 1.0 * ui_scale;
                int thickness = std::max(1, static_cast<int>(3 * ui_scale));
                
                std::string stats = "IN: " + std::to_string(factory_in_count) +
                                    " | OUT: " + std::to_string(factory_out_count);
                
                cv::putText(
                    bgr_frame,
                    stats,
                    cv::Point(margin_x, margin_y),
                    cv::FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    cv::Scalar(0, 255, 255),
                    thickness
                );  
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
        
        
        benchmark.record_frame(timing);
        
        return !benchmark.is_complete() && g_running.load();
    };

    // ------------------------------------------------------------------
    // 10. Run
    // ------------------------------------------------------------------
    pipeline.start(process_frame);

    // ------------------------------------------------------------------
    // 11. Cleanup
    // ------------------------------------------------------------------
    std::cout << "[SHUTDOWN] Cleaning up...\n";
    zenoh.shutdown();
    neon::cleanup_preprocess_buffers();
    std::cout << "[SHUTDOWN] Done.\n";
    
    // Stop video writer (flushes remaining frames)
    if (video_writer) {
        std::cout << "Flushing video writer...\n";
        video_writer->stop();
        std::cout << "Video saved: " << opts.output_video << "\n";
        std::cout << "  Frames written: " << video_writer->frames_written() << "\n";
        std::cout << "  Frames dropped: " << video_writer->frames_dropped() << "\n";
    }
    // ------------------------------------------------------------------
    // 12. Print Benchmark Result
    // ------------------------------------------------------------------
        // Print results
    benchmark.print_summary();
    
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
    return 0;
}

// ============================================================================
// Entry Point
// ============================================================================

int main(int argc, char* argv[]) {
    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    Options opts;
    if (!parse_options(argc, argv, opts)) return 1;

    if (opts.param_path.empty() || opts.bin_path.empty()) {
        std::cerr << "[ERROR] --param and --bin are required.\n";
        print_usage(argv[0]);
        return 1;
    }

    return run_inference_pipeline(opts);
}
