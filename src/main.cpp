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

#include <zenoh-pico.h>

using namespace yolo;

// ============================================================================
// Zenoh session & Publishers
// ============================================================================
z_owned_session_t   z_session;
z_owned_publisher_t pub_stats;
z_owned_publisher_t pub_events;
z_owned_publisher_t pub_events_image;   // [FIX-2] Topic riêng cho ảnh violation
z_owned_publisher_t pub_image;
z_owned_publisher_t pub_count;

// ============================================================================
// Logic Constants
// ============================================================================
constexpr int PERSON_ID        = 0;
constexpr int HELMET_SAFE      = 1;
constexpr int CLOTHES_VIOLATION = 2;
constexpr int VEST_SAFE        = 3;
constexpr int HEAD_VIOLATION   = 4;

// Cấu hình violation voting
constexpr int VIOLATION_HISTORY_LEN  = 5;   // số frame lưu lịch sử
constexpr int VIOLATION_VOTE_THRESH  = 3;   // bao nhiêu frame viol/5 thì coi là vi phạm
constexpr int WORKER_MAX_MISS        = 30;  // frame miss liên tiếp trước khi xóa worker
constexpr float IOA_THRESHOLD        = 0.6f;
constexpr float HEAD_REGION_RATIO    = 0.15f; // mở rộng lên trên bbox để check đầu

// Cấu hình JPEG snapshot
constexpr int SNAPSHOT_WIDTH   = 320;
constexpr int SNAPSHOT_HEIGHT  = 240;
constexpr int SNAPSHOT_QUALITY = 60;  // 0-100, thấp hơn = nhỏ hơn

// Interval publish
constexpr int STATS_INTERVAL_SEC  = 2;
constexpr int IMAGE_INTERVAL_SEC  = 5;

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
};

GateROI                 active_roi;
int                     factory_in_count  = 0;
int                     factory_out_count = 0;
std::map<int, WorkerState> active_workers;

// ============================================================================
// Global State & Signal Handling
// ============================================================================
std::atomic<bool> g_running{true};

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
    int         gpu_device         = 0;
    std::string cam_id             = "cam1";
    std::string router_ip          = "192.168.1.9";
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
        << "  --router  IP     Zenoh router IP (default: 192.168.1.9)\n"
        << "  --display        Enable DRM/KMS display\n"
        << "  --fb             Enable framebuffer display\n"
        << "  --vulkan         Use Vulkan GPU backend\n"
        << "  --int8           Use INT8 quantization\n"
        << "  --verbose        Verbose logging\n";
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
        {"router",       required_argument, 0, 'r'},
        {"help",         no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "c:v:p:m:n:w:O:DBGIg:Vk:r:h",
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
            case 'r': opts.router_ip    = optarg;                   break;
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

/**
 * Publish raw bytes qua một Zenoh publisher.
 * Dùng z_bytes_copy_from_buf (không phải _from_str) để xử lý binary đúng.
 */
static void zenoh_publish_bytes(z_owned_publisher_t& publisher,
                                 const void* data, size_t size) {
    z_owned_bytes_t z_payload;
    z_bytes_copy_from_buf(&z_payload,
                          reinterpret_cast<const uint8_t*>(data),
                          size);
    z_publisher_put(z_publisher_loan(&publisher),
                    z_bytes_move(&z_payload), NULL);
}

/**
 * Publish string (JSON metadata nhỏ) — wrapper tiện lợi.
 */
static void zenoh_publish_str(z_owned_publisher_t& publisher,
                               const std::string& str) {
    zenoh_publish_bytes(publisher, str.data(), str.size());
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

// ============================================================================
// Zenoh Publisher Setup
// ============================================================================

/**
 * Khai báo một publisher.
 * [FIX-3] Nhận const std::string& để tránh dangling pointer từ temporary.
 */
static bool declare_publisher(z_owned_publisher_t& pub,
                               const std::string& topic) {
    z_view_keyexpr_t ke;
    z_view_keyexpr_from_str(&ke, topic.c_str());
    if (z_declare_publisher(z_session_loan(&z_session),
                            &pub,
                            z_view_keyexpr_loan(&ke), NULL) < 0) {
        std::cerr << "[ERROR] Failed to declare publisher: " << topic << "\n";
        return false;
    }
    std::cout << "[ZENOH] Publisher ready: " << topic << "\n";
    return true;
}

// ============================================================================
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
        cfg.source      = (opts.mode == "camera")
                          ? InputSource::CAMERA_V4L2
                          : InputSource::VIDEO_FILE;
        cfg.device_path = opts.device;
        cfg.width       = INPUT_WIDTH;
        cfg.height      = INPUT_HEIGHT;
        if (pipeline.initialize(cfg) != ErrorCode::SUCCESS) {
            std::cerr << "[FATAL] Failed to initialize input pipeline!\n";
            return 1;
        }
    }
    std::cout << "[INIT] Input pipeline ready.\n";

    // ------------------------------------------------------------------
    // 4. Zenoh session
    // ------------------------------------------------------------------
    std::cout << "[INIT] Connecting Zenoh to " << opts.router_ip << "...\n";
    {
        z_owned_config_t z_cfg;
        z_config_default(&z_cfg);
        std::string ep = "tcp/" + opts.router_ip + ":7447";
        zp_config_insert(z_config_loan_mut(&z_cfg), Z_CONFIG_CONNECT_KEY, ep.c_str());
        zp_config_insert(z_config_loan_mut(&z_cfg), Z_CONFIG_MODE_KEY,    "client");
        if (z_open(&z_session, z_config_move(&z_cfg), NULL) < 0) {
            std::cerr << "[FATAL] Zenoh connection failed to " << ep << "\n";
            return 1;
        }
    }
    zp_start_read_task(z_session_loan_mut(&z_session), NULL);
    zp_start_lease_task(z_session_loan_mut(&z_session), NULL);
    std::cout << "[INIT] Zenoh session open.\n";

    // ------------------------------------------------------------------
    // 5. Declare publishers
    //    [FIX-3] Lưu string vào biến cục bộ, không dùng temporary .c_str()
    // ------------------------------------------------------------------
    const std::string base        = "factory/" + opts.cam_id;
    const std::string t_stats     = base + "/stats";
    const std::string t_events    = base + "/events";
    const std::string t_ev_image  = base + "/events/image";  // [FIX-2] topic riêng
    const std::string t_image     = base + "/image";
    const std::string t_count     = base + "/count";

    if (!declare_publisher(pub_stats,        t_stats))    return 1;
    if (!declare_publisher(pub_events,       t_events))   return 1;
    if (!declare_publisher(pub_events_image, t_ev_image)) return 1;  // [FIX-2]
    if (!declare_publisher(pub_image,        t_image))    return 1;
    if (!declare_publisher(pub_count,        t_count))    return 1;

    // ------------------------------------------------------------------
    // 6. Tracker & ROI
    // ------------------------------------------------------------------
    ocsort::OCSort tracker(0.25f, 30, 1, 0.4f, 3, "giou", 0.2f, true);

    // Homography matrix (camera-specific, cấu hình theo thực tế)
    active_roi.H = (cv::Mat_<double>(3, 3)
                    << 1.25,  0.0,   -200.0,
                       0.0,   2.857, -571.4,
                       0.0,   0.0,      1.0);
    active_roi.ground_crossing_line = 400.0f;

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

    // ------------------------------------------------------------------
    // 9. Per-frame callback
    // ------------------------------------------------------------------
    auto process_frame = [&](const FrameBuffer& frame) -> bool {

        if (opts.verbose && frame.frame_index % 30 == 0)
            std::cout << "[FRAME] #" << frame.frame_index << "\n";

        auto now = std::chrono::steady_clock::now();

        // ---- Preprocess ------------------------------------------------
        float scale; int pad_x, pad_y;
        neon::preprocess_bgr_direct(
            frame.data, model_input.get(),
            frame.width, frame.height, frame.stride,
            &scale, &pad_x, &pad_y);
        engine.set_letterbox_params(scale, pad_x, pad_y);

        // ---- Inference -------------------------------------------------
        DetectionResult result;
        engine.infer_fp32(model_input.get(), result);

        if (opts.verbose && frame.frame_index % 30 == 0)
            std::cout << "[INFER] Detections: " << result.count << "\n";

        // ---- Split detections: person vs PPE ---------------------------
        std::vector<std::vector<float>> tracker_input;
        std::vector<yolo::Detection>    ppe_detections;
        tracker_input.reserve(result.count);
        ppe_detections.reserve(result.count);

        for (int i = 0; i < result.count; ++i) {
            const auto& d = result.detections[i];
            if (d.class_id == PERSON_ID)
                tracker_input.push_back({d.x1, d.y1, d.x2, d.y2,
                                          d.confidence, 0.f});
            else
                ppe_detections.push_back(d);
        }

        // ---- Track persons ---------------------------------------------
        auto tracks = tracker.update(to_eigen_matrix(tracker_input));
        int  current_violations = 0;

        // ---- Update worker states --------------------------------------
        // Mark all existing workers as potentially missing this frame
        for (auto& [id, ws] : active_workers)
            ws.consecutive_misses++;

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

            if (state.last_ground_y > 0.f
                && ground.y >= active_roi.ground_crossing_line
                && state.last_ground_y < active_roi.ground_crossing_line) {
                // Đi vào (từ trên xuống line)
                if (!state.counted_in) {
                    ++factory_in_count;
                    state.counted_in = true;
                    std::string val = std::to_string(factory_in_count);
                    zenoh_publish_str(pub_count, val);
                    std::cout << "[COUNT] IN: " << factory_in_count << "\n";
                }
            }
            else if (state.last_ground_y >= active_roi.ground_crossing_line
                     && ground.y < active_roi.ground_crossing_line
                     && state.last_ground_y > 0.f) {
                // Đi ra (từ dưới lên line)
                if (!state.counted_out) {
                    ++factory_out_count;
                    state.counted_out = true;
                    std::cout << "[COUNT] OUT: " << factory_out_count << "\n";
                }
            }
            state.last_ground_y = ground.y;

            // ---- PPE Violation check -----------------------------------
            int frame_viol = 0;
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
                    frame_viol = 1;
                    break;
                }
            }

            // Sliding-window vote
            state.violation_history.push_back(frame_viol);
            if ((int)state.violation_history.size() > VIOLATION_HISTORY_LEN)
                state.violation_history.pop_front();

            int votes = 0;
            for (int v : state.violation_history) votes += v;
            state.is_violating = (votes >= VIOLATION_VOTE_THRESH);

            // ---- Publish violation event (once per violation episode) --
            if (state.is_violating) {
                ++current_violations;

                if (!state.event_sent) {
                    // [FIX-2] Metadata JSON — không nhúng ảnh vào đây
                    std::string ev_json =
                        "{\"id\":\"WK" + std::to_string(tid) +
                        "\",\"type\":\"NO_PPE\""
                        ",\"location\":\"" + opts.cam_id + "\""
                        ",\"timestamp\":" +
                        std::to_string(std::chrono::duration_cast<
                            std::chrono::milliseconds>(
                            now.time_since_epoch()).count()) +
                        "}";
                    zenoh_publish_str(pub_events, ev_json);

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
                    if (!jpeg.empty()) {
                        // Thêm worker ID vào topic để backend biết ảnh của ai
                        // factory/cam1/events/image  (backend dùng metadata từ
                        // /events để ghép, hoặc dùng Zenoh attachment nếu cần)
                        zenoh_publish_bytes(pub_events_image,
                                            jpeg.data(), jpeg.size());
                        std::cout << "[ZENOH] Event image sent: "
                                  << jpeg.size() << " bytes (WK"
                                  << tid << ")\n";
                    }

                    state.event_sent = true;
                }
            }
            else {
                // Reset để có thể gửi event lần vi phạm tiếp theo
                if (state.event_sent) state.event_sent = false;
            }
        }

        // ---- Periodic stats publish ------------------------------------
        if (now - last_stats_time >= stats_interval) {
            float compliance = active_workers.empty()
                ? 100.f
                : (1.f - (float)current_violations
                         / (float)active_workers.size()) * 100.f;

            std::string s_json =
                "{\"cam_id\":\"" + opts.cam_id + "\""
                ",\"total\":"      + std::to_string(active_workers.size()) +
                ",\"violations\":" + std::to_string(current_violations) +
                ",\"compliance\":" + std::to_string(compliance) +
                ",\"in\":"         + std::to_string(factory_in_count) +
                ",\"out\":"        + std::to_string(factory_out_count) +
                "}";
            zenoh_publish_str(pub_stats, s_json);
            std::cout << "[ZENOH] Stats: " << s_json << "\n";
            last_stats_time = now;
        }

        // ---- Periodic snapshot publish ---------------------------------
        //  [FIX-1] Gửi raw JPEG binary, không wrap base64-in-JSON
        if (now - last_image_time >= image_interval) {
            cv::Mat raw(frame.height, frame.width,
                        CV_8UC3, frame.data, frame.stride);
            cv::Mat thumb;
            cv::resize(raw, thumb, cv::Size(SNAPSHOT_WIDTH, SNAPSHOT_HEIGHT));

            auto jpeg = encode_jpeg(thumb, SNAPSHOT_QUALITY);
            if (!jpeg.empty()) {
                zenoh_publish_bytes(pub_image, jpeg.data(), jpeg.size());
                std::cout << "[ZENOH] Snapshot sent: "
                          << jpeg.size() << " bytes\n";
            }
            last_image_time = now;
        }

        // ---- Evict stale workers ---------------------------------------
        //  [FIX-5] consecutive_misses đã được increment ở đầu loop,
        //           nên chỉ cần check > threshold ở đây
        for (auto it = active_workers.begin(); it != active_workers.end(); ) {
            if (it->second.consecutive_misses > WORKER_MAX_MISS) {
                std::cout << "[TRACK] Removed stale worker: "
                          << it->second.track_id << "\n";
                it = active_workers.erase(it);
            } else {
                ++it;
            }
        }

        return g_running.load();
    };

    // ------------------------------------------------------------------
    // 10. Run
    // ------------------------------------------------------------------
    pipeline.start(process_frame);

    // ------------------------------------------------------------------
    // 11. Cleanup
    // ------------------------------------------------------------------
    std::cout << "[SHUTDOWN] Cleaning up...\n";
    z_publisher_drop(z_publisher_move(&pub_stats));
    z_publisher_drop(z_publisher_move(&pub_events));
    z_publisher_drop(z_publisher_move(&pub_events_image));
    z_publisher_drop(z_publisher_move(&pub_image));
    z_publisher_drop(z_publisher_move(&pub_count));
    z_session_drop(z_session_move(&z_session));
    neon::cleanup_preprocess_buffers();
    std::cout << "[SHUTDOWN] Done.\n";
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
