/**
 * @file input_pipeline.cpp
 * @brief Unified input pipeline for V4L2 camera and video files
 * 
 * Camera & Input Systems Agent Implementation:
 * - V4L2 mmap zero-copy capture
 * - Video file decoding via FFmpeg/OpenCV
 * - Both paths produce identical FrameBuffer output
 */

#include "input_pipeline.h"

namespace yolo {

// ============================================================================
// V4L2 Helper Functions
// ============================================================================

static int xioctl(int fd, unsigned long request, void* arg) {
    int r;
    do {
        r = ioctl(fd, request, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

bool v4l2_query_format(int fd, V4L2Format* format) {
    struct v4l2_capability cap;
    if (xioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
        return false;
    }
    
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        return false;
    }
    
    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        return false;
    }
    
    // Query current format
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    
    if (xioctl(fd, VIDIOC_G_FMT, &fmt) == -1) {
        return false;
    }
    
    format->pixel_format = fmt.fmt.pix.pixelformat;
    format->width = fmt.fmt.pix.width;
    format->height = fmt.fmt.pix.height;
    format->bytesperline = fmt.fmt.pix.bytesperline;
    format->sizeimage = fmt.fmt.pix.sizeimage;
    
    return true;
}

bool v4l2_set_format(int fd, int width, int height, int fps) {
    // Set pixel format
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    
    if (xioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
        return false;
    }
    
    // Verify format was set
    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) {
        return false;
    }
    
    // Set frame rate
    struct v4l2_streamparm parm;
    memset(&parm, 0, sizeof(parm));
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = fps;
    
    xioctl(fd, VIDIOC_S_PARM, &parm);  // Ignore error, some cameras don't support
    
    return true;
}

// ============================================================================
// InputPipeline Implementation
// ============================================================================

InputPipeline::InputPipeline() = default;

InputPipeline::~InputPipeline() {
    stop();
    
    // Cleanup V4L2
    if (v4l2_buffers_) {
        for (int i = 0; i < v4l2_buffer_count_; i++) {
            if (v4l2_buffers_[i].start != MAP_FAILED) {
                munmap(v4l2_buffers_[i].start, v4l2_buffers_[i].length);
            }
        }
        delete[] v4l2_buffers_;
    }
    
    if (v4l2_fd_ >= 0) {
        close(v4l2_fd_);
    }
}

ErrorCode InputPipeline::initialize(const Config& config) {
    config_ = config;
    
    if (config.source == InputSource::CAMERA_V4L2) {
        return init_v4l2();
    } else if (config.source == InputSource::RTSP_STREAM) {
        return init_rtsp(); // <--- Call RTSP init
    } else {
        return init_video_file();
    }
}

ErrorCode InputPipeline::init_rtsp() {
    // Basic check if network address is provided
    if (config_.device_path.empty()) {
        return ErrorCode::CAMERA_OPEN_FAILED;
    }
    
    // Allocate buffer for BGR frame (Final output to Main)
    // 64-byte aligned for NEON access
    video_frame_buffer_ = make_aligned_buffer<uint8_t>(config_.width * config_.height * 3);
    
    return ErrorCode::SUCCESS;
}

ErrorCode InputPipeline::init_v4l2() {
    // Open device
    v4l2_fd_ = open(config_.device_path.c_str(), O_RDWR | O_NONBLOCK);
    if (v4l2_fd_ < 0) {
        return ErrorCode::CAMERA_OPEN_FAILED;
    }
    
    // Set format
    if (!v4l2_set_format(v4l2_fd_, config_.width, config_.height, config_.fps)) {
        close(v4l2_fd_);
        v4l2_fd_ = -1;
        return ErrorCode::CAMERA_FORMAT_FAILED;
    }
    
    // Request buffers
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = config_.buffer_count;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    
    if (xioctl(v4l2_fd_, VIDIOC_REQBUFS, &req) == -1) {
        close(v4l2_fd_);
        v4l2_fd_ = -1;
        return ErrorCode::CAMERA_MMAP_FAILED;
    }
    
    v4l2_buffer_count_ = req.count;
    v4l2_buffers_ = new V4L2Buffer[v4l2_buffer_count_];
    
    // Map buffers
    for (int i = 0; i < v4l2_buffer_count_; i++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        
        if (xioctl(v4l2_fd_, VIDIOC_QUERYBUF, &buf) == -1) {
            return ErrorCode::CAMERA_MMAP_FAILED;
        }
        
        v4l2_buffers_[i].length = buf.length;
        v4l2_buffers_[i].start = mmap(
            nullptr, buf.length,
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            v4l2_fd_, buf.m.offset
        );
        
        if (v4l2_buffers_[i].start == MAP_FAILED) {
            return ErrorCode::CAMERA_MMAP_FAILED;
        }
    }
    
    return ErrorCode::SUCCESS;
}

ErrorCode InputPipeline::init_video_file() {
    // Allocate frame buffer for video decoding
    video_frame_buffer_ = make_aligned_buffer<uint8_t>(YUYV_BUFFER_SIZE);
    if (!video_frame_buffer_) {
        return ErrorCode::MEMORY_ALLOCATION_FAILED;
    }
    
    // OpenCV will be initialized in capture loop
    return ErrorCode::SUCCESS;
}

ErrorCode InputPipeline::start(FrameCallback callback) {
    if (running_.load()) {
        return ErrorCode::SUCCESS;
    }
    
    running_.store(true);
    
    if (config_.source == InputSource::CAMERA_V4L2) {
        // Queue all buffers
        for (int i = 0; i < v4l2_buffer_count_; i++) {
            struct v4l2_buffer buf;
            memset(&buf, 0, sizeof(buf));
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            
            if (xioctl(v4l2_fd_, VIDIOC_QBUF, &buf) == -1) {
                running_.store(false);
                return ErrorCode::CAMERA_MMAP_FAILED;
            }
        }
        
        // Start streaming
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(v4l2_fd_, VIDIOC_STREAMON, &type) == -1) {
            running_.store(false);
            return ErrorCode::CAMERA_MMAP_FAILED;
        }
        
        // Run capture loop in current thread
        capture_loop_v4l2(callback);
    } else if (config_.source == InputSource::RTSP_STREAM) {
        capture_loop_rtsp(callback);
    } else {
        capture_loop_video(callback);
    }
    
    return ErrorCode::SUCCESS;
}

void InputPipeline::stop() {
    running_.store(false);
    
    if (config_.source == InputSource::CAMERA_V4L2 && v4l2_fd_ >= 0) {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(v4l2_fd_, VIDIOC_STREAMOFF, &type);
    }
}

void InputPipeline::capture_loop_v4l2(FrameCallback callback) {
    // Set thread affinity to CPU 0
    set_thread_affinity(INPUT_THREAD_CPU);
    
    fd_set fds;
    struct timeval tv;
    
    while (running_.load(std::memory_order_relaxed)) {
        FD_ZERO(&fds);
        FD_SET(v4l2_fd_, &fds);
        
        tv.tv_sec = 0;
        tv.tv_usec = 100000;  // 100ms timeout
        
        int r = select(v4l2_fd_ + 1, &fds, nullptr, nullptr, &tv);
        
        if (r == -1) {
            if (errno == EINTR) continue;
            break;
        }
        
        if (r == 0) {
            // Timeout
            continue;
        }
        
        // Dequeue buffer
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        
        if (xioctl(v4l2_fd_, VIDIOC_DQBUF, &buf) == -1) {
            if (errno == EAGAIN) continue;
            break;
        }
        
        // Create frame buffer
        FrameBuffer frame;
        frame.data = static_cast<uint8_t*>(v4l2_buffers_[buf.index].start);
        frame.size = buf.bytesused;
        frame.stride = config_.width * 2;
        frame.width = config_.width;
        frame.height = config_.height;
        frame.timestamp_ns = buf.timestamp.tv_sec * 1000000000LL + buf.timestamp.tv_usec * 1000LL;
        frame.frame_index = frame_count_.load();
        frame.valid = true;
        frame.format = PixelFormat::YUYV;  // Camera uses YUYV
        
        // Invoke callback
        bool continue_capture = callback(frame);
        frame_count_.fetch_add(1);
        
        // Re-queue buffer
        if (xioctl(v4l2_fd_, VIDIOC_QBUF, &buf) == -1) {
            break;
        }
        
        if (!continue_capture) {
            break;
        }
    }
    
    running_.store(false);
}

void InputPipeline::capture_loop_video(FrameCallback callback) {
    // Set thread affinity to CPU 0
    set_thread_affinity(INPUT_THREAD_CPU);
    
    cv::VideoCapture cap(config_.device_path);
    
    if (!cap.isOpened()) {
        running_.store(false);
        return;
    }
    
    // Get video properties
    // Note: video_width and video_height are available but not currently used
    // int video_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    // int video_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double video_fps = cap.get(cv::CAP_PROP_FPS);
    
    // Store FPS for external access
    if (video_fps > 0 && video_fps < 1000) {
        const_cast<InputPipeline*>(this)->video_fps_ = video_fps;
    }
    
    cv::Mat bgr_frame;

    while (running_.load(std::memory_order_relaxed)) {
        if (!cap.read(bgr_frame)) {
            if (config_.loop_video) {
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                continue;
            }
            break;
        }
        
        // Resize if necessary (but keep BGR format!)
        if (bgr_frame.cols != config_.width || bgr_frame.rows != config_.height) {
            cv::resize(bgr_frame, bgr_frame, cv::Size(config_.width, config_.height));
        }
        
        // OPTIMIZATION: Pass BGR directly - NO YUYV conversion!
        // Create frame buffer pointing to BGR data
        FrameBuffer frame;
        frame.data = bgr_frame.data;
        frame.size = bgr_frame.total() * bgr_frame.elemSize();
        frame.stride = bgr_frame.step[0];
        frame.width = config_.width;
        frame.height = config_.height;
        frame.timestamp_ns = get_timestamp_ns();
        frame.frame_index = frame_count_.load();
        frame.valid = true;
        frame.format = PixelFormat::BGR;  // Mark as BGR format
        
        // Invoke callback
        bool continue_capture = callback(frame);
        frame_count_.fetch_add(1);
        
        if (!continue_capture) {
            break;
        }
    }
    
    running_.store(false);
}

/*
void InputPipeline::capture_loop_rtsp(FrameCallback callback) {
    // 1. Thread Affinity: Pin to Core 0 for Cache Locality
    set_thread_affinity(INPUT_THREAD_CPU);

    // 2. FFMPEG Initialization
    AVFormatContext* fmt_ctx = avformat_alloc_context();
    AVDictionary* opts = nullptr;
    
    // 3. LOW LATENCY SETTINGS (Critical for RTSP)
    av_dict_set(&opts, "rtsp_transport", "tcp", 0); // Use TCP to prevent packet loss artifacts
    av_dict_set(&opts, "fflags", "nobuffer", 0);    // Reduce internal buffering
    av_dict_set(&opts, "flags", "low_delay", 0);    // Tell codec to output frames ASAP
    av_dict_set(&opts, "strict", "experimental", 0);

    // Open Stream
    if (avformat_open_input(&fmt_ctx, config_.device_path.c_str(), nullptr, &opts) < 0) {
        std::cerr << "Error: Could not open RTSP stream\n";
        running_.store(false);
        return;
    }

    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        running_.store(false);
        return;
    }

    // Find Video Stream
    int video_stream_idx = -1;
    for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx = i;
            break;
        }
    }

    if (video_stream_idx == -1) {
        running_.store(false);
        return;
    }

    // Setup Decoder
    AVCodecParameters* codec_par = fmt_ctx->streams[video_stream_idx]->codecpar;
    const AVCodec* codec = nullptr;

    if (codec_par->codec_id == AV_CODEC_ID_HEVC) {
        // If Video uses H.265, let FFmpeg find Hardware Decoder of Raspberry Pi 5
        codec = avcodec_find_decoder_by_name("hevc_v4l2m2m");
        if (codec) {
            std::cout << "[OPTIMIZATION] Pi 5 Hardware HEVC Decoder (hevc_v4l2m2m) ACTIVATED!\n";
        } else {
            std::cout << "[WARNING] HW Decoder not found. Falling back to Software HEVC.\n";
            codec = avcodec_find_decoder(codec_par->codec_id);
        }
    } else {
        // If using H.264 or another, Software Decoder will be used
        codec = avcodec_find_decoder(codec_par->codec_id);
    }

    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_ctx, codec_par);
    
    // If using not HW Decoder, Enable Multi-threading decoding for RPi5
    if (codec_par->codec_id != AV_CODEC_ID_HEVC || !codec || strcmp(codec->name, "hevc_v4l2m2m") != 0) {
        codec_ctx->thread_count = 4;
        codec_ctx->thread_type = FF_THREAD_FRAME;
    }
    
    // ==========================================================
    // HARDWARE DEVICE ASSIGNMENT & AUTOMATIC FALLBACK
    // ==========================================================
    AVDictionary* hw_opts = nullptr;
    
    // Explicitly force FFmpeg to use the specific device node for the hardware decoder
    if (codec && strcmp(codec->name, "hevc_v4l2m2m") == 0) {
        av_dict_set(&hw_opts, "device", "/dev/video19", 0);
    }
    
    // Pass hw_opts to avcodec_open2
    int open_ret = avcodec_open2(codec_ctx, codec, &hw_opts);
    
    // Clean up dictionary memory to prevent leaks
    if (hw_opts) {
        av_dict_free(&hw_opts);
    }

    if (open_ret < 0) {
        std::cerr << "[ERROR] Could not open codec: " << (codec ? codec->name : "Unknown") << "\n";
        
        if (codec && strcmp(codec->name, "hevc_v4l2m2m") == 0) {
            std::cout << "[FALLBACK] Hardware Decoder failed. Retrying with Software HEVC (CPU)...\n";
            
            avcodec_free_context(&codec_ctx);
            
            // find Software HEVC Decoder
            codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
            codec_ctx = avcodec_alloc_context3(codec);
            avcodec_parameters_to_context(codec_ctx, codec_par);
            codec_ctx->thread_count = 4;
            codec_ctx->thread_type = FF_THREAD_FRAME;
            
            if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
                std::cerr << "[FATAL] Could not open Software HEVC codec either!\n";
                running_.store(false);
                return;
            }
            std::cout << "[SUCCESS] Software HEVC fallback successful.\n";
        } else {
            running_.store(false);
            return;
        }
    }
    // ==========================================================
    
    // Alloc resources
    AVFrame* frame = av_frame_alloc();
    AVPacket* pkt = av_packet_alloc();
    
    // Software Scaler (SwsContext) as a fallback for NEON
    // We convert YUV420P to BGR24 (which is what Main Pipeline expects)
    struct SwsContext* sws_ctx = sws_getContext(
        codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt,
        config_.width, config_.height, AV_PIX_FMT_BGR24, // Output: BGR
        SWS_BILINEAR, nullptr, nullptr, nullptr
    );

    std::cout << "RTSP Stream Started: " << config_.width << "x" << config_.height << "\n";

    // 4. Capture Loop
    while (running_.load(std::memory_order_relaxed)) {
        if (av_read_frame(fmt_ctx, pkt) >= 0) {
            if (pkt->stream_index == video_stream_idx) {
                // Send packet to decoder
                if (avcodec_send_packet(codec_ctx, pkt) == 0) {
                    // Receive decoded frame(s)
                    while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                        
                        // 5. Convert & Zero-ish Copy
                        // Instead of full copy, we write directly to our pre-allocated BGR buffer
                        // Note: If you implement yuv420p_to_bgr_neon, call it here instead of sws_scale
                        
                        uint8_t* dest_slices[1] = { video_frame_buffer_.get() };
                        int dest_linesize[1] = { config_.width * 3 };
                        
                        // Perform scaling/conversion (YUV -> BGR)
                        sws_scale(sws_ctx, 
                            frame->data, frame->linesize, 0, codec_ctx->height,
                            dest_slices, dest_linesize);

                        //yolo::neon::yuv420p_to_bgr_neon(
                            //frame->data[0], frame->data[1], frame->data[2],
                            //video_frame_buffer_.get(),
                            //config_.width, config_.height,
                            //frame->linesize[0], frame->linesize[1],
                            //config_.width * 3
                        //);
                        // 6. Create FrameBuffer Wrapper
                        FrameBuffer fb;
                        fb.data = video_frame_buffer_.get();
                        fb.size = config_.width * config_.height * 3;
                        fb.width = config_.width;
                        fb.height = config_.height;
                        fb.stride = config_.width * 3;
                        fb.format = PixelFormat::BGR; // Mark as BGR so Main knows
                        fb.frame_index = frame_count_.load();
                        fb.timestamp_ns = get_timestamp_ns();
                        fb.valid = true;

                        // 7. Invoke Callback to Main
                        if (!callback(fb)) {
                            running_.store(false);
                        }
                        frame_count_.fetch_add(1);
                    }
                }
            }
            av_packet_unref(pkt);
        } else {
            // Error or End of stream
            // Optional: Implement reconnection logic here
            break; 
        }
    }

    // Cleanup
    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);
    sws_freeContext(sws_ctx);
    running_.store(false);
}
*/


/**
 * New RTSP callback to use Hardware Decoder with FFmpeg
**/


static enum AVPixelFormat get_hw_format(
    AVCodecContext* ctx,
    const enum AVPixelFormat* pix_fmts)
{
    std::cout << "[HW] Available pixel formats:\n";

    while (*pix_fmts != AV_PIX_FMT_NONE) {

        std::cout << "  -> " << av_get_pix_fmt_name(*pix_fmts) << "\n";

        if (*pix_fmts == AV_PIX_FMT_DRM_PRIME) {

            std::cout << "[HW] Selected DRM PRIME format\n";

            return *pix_fmts;
        }

        pix_fmts++;
    }

    std::cerr << "[HW] DRM PRIME format not offered by decoder\n";

    return AV_PIX_FMT_NONE;
}

void InputPipeline::capture_loop_rtsp(FrameCallback callback)
{
    set_thread_affinity(INPUT_THREAD_CPU);

    std::cout << "\n========================================\n";
    std::cout << " RTSP HEVC INPUT PIPELINE STARTED\n";
    std::cout << "========================================\n";

    AVFormatContext* fmt_ctx = nullptr;

    AVDictionary* opts = nullptr;

    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    av_dict_set(&opts, "fflags", "nobuffer", 0);
    av_dict_set(&opts, "flags", "low_delay", 0);

    std::cout << "[RTSP] Opening stream:\n";
    std::cout << "       " << config_.device_path << "\n";

    if (avformat_open_input(
            &fmt_ctx,
            config_.device_path.c_str(),
            nullptr,
            &opts) < 0)
    {
        std::cerr << "[FATAL] Could not open RTSP stream\n";
        running_.store(false);
        return;
    }

    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {

        std::cerr << "[FATAL] Could not find stream info\n";

        running_.store(false);
        return;
    }

    int video_stream_idx = -1;

    for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {

        if (fmt_ctx->streams[i]->codecpar->codec_type ==
            AVMEDIA_TYPE_VIDEO)
        {
            video_stream_idx = i;
            break;
        }
    }

    if (video_stream_idx < 0) {

        std::cerr << "[FATAL] No video stream found\n";

        running_.store(false);
        return;
    }

    AVCodecParameters* codec_par =
        fmt_ctx->streams[video_stream_idx]->codecpar;

    std::cout << "\n[STREAM INFO]\n";
    std::cout << "Codec: " << avcodec_get_name(codec_par->codec_id) << "\n";
    std::cout << "Resolution: "
              << codec_par->width
              << "x"
              << codec_par->height
              << "\n";

    // =========================================================
    // USE NATIVE DECODER + DRM HWACCEL
    // =========================================================

    const AVCodec* codec =
        avcodec_find_decoder(codec_par->codec_id);

    if (!codec) {

        std::cerr << "[FATAL] Decoder not found\n";

        running_.store(false);
        return;
    }

    std::cout << "\n[DECODER]\n";
    std::cout << "Using decoder: " << codec->name << "\n";

    AVCodecContext* codec_ctx =
        avcodec_alloc_context3(codec);

    avcodec_parameters_to_context(codec_ctx, codec_par);

    // =========================================================
    // DRM HW DEVICE
    // =========================================================

    AVBufferRef* hw_device_ctx = nullptr;

    std::cout << "\n[HWACCEL]\n";
    std::cout << "Creating DRM device context...\n";

    int hw_ret = av_hwdevice_ctx_create(
        &hw_device_ctx,
        AV_HWDEVICE_TYPE_DRM,
        "/dev/dri/card0",
        nullptr,
        0);

    bool hw_enabled = false;

    if (hw_ret >= 0) {

        std::cout << "[SUCCESS] DRM HW Device Created\n";

        codec_ctx->hw_device_ctx =
            av_buffer_ref(hw_device_ctx);

        codec_ctx->get_format = get_hw_format;

        hw_enabled = true;

    } else {

        std::cout << "[WARNING] DRM HW Device Creation Failed\n";
        std::cout << "[FALLBACK] Software Decode Mode\n";

        codec_ctx->thread_count = 4;
        codec_ctx->thread_type = FF_THREAD_FRAME;
    }

    // =========================================================
    // OPEN CODEC
    // =========================================================

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {

        std::cerr << "[FATAL] Could not open codec\n";

        running_.store(false);
        return;
    }

    std::cout << "\n[CODEC OPENED SUCCESSFULLY]\n";

    if (hw_enabled) {

        std::cout << "[MODE] HARDWARE HEVC DECODE ACTIVE\n";

    } else {

        std::cout << "[MODE] SOFTWARE DECODE ACTIVE\n";
    }

    // =========================================================
    // ALLOC FRAMES
    // =========================================================

    AVFrame* frame = av_frame_alloc();

    AVFrame* sw_frame = av_frame_alloc();

    AVPacket* pkt = av_packet_alloc();

    struct SwsContext* sws_ctx = sws_getContext(
        codec_par->width,
        codec_par->height,
        AV_PIX_FMT_YUV420P,
        config_.width,
        config_.height,
        AV_PIX_FMT_BGR24,
        SWS_BILINEAR,
        nullptr,
        nullptr,
        nullptr);

    std::cout << "\n[PIPELINE READY]\n";
    std::cout << "========================================\n";

    // =========================================================
    // MAIN LOOP
    // =========================================================

    while (running_.load(std::memory_order_relaxed)) {

        if (av_read_frame(fmt_ctx, pkt) < 0) {

            std::cout << "[STREAM] End or Error\n";

            break;
        }

        if (pkt->stream_index != video_stream_idx) {

            av_packet_unref(pkt);

            continue;
        }

        int send_ret = avcodec_send_packet(codec_ctx, pkt);

        if (send_ret < 0) {

            std::cerr << "[ERROR] Failed to send packet\n";

            av_packet_unref(pkt);

            continue;
        }

        while (true) {

            int recv_ret =
                avcodec_receive_frame(codec_ctx, frame);

            if (recv_ret == AVERROR(EAGAIN) ||
                recv_ret == AVERROR_EOF)
            {
                break;
            }

            if (recv_ret < 0) {

                std::cerr << "[ERROR] Failed to receive frame\n";

                break;
            }

            // =================================================
            // HARDWARE FRAME
            // =================================================

            AVFrame* usable_frame = frame;

            if (frame->format == AV_PIX_FMT_DRM_PRIME) {

                std::cout << "[HW FRAME] DRM PRIME received\n";

                if (av_hwframe_transfer_data(
                        sw_frame,
                        frame,
                        0) < 0)
                {
                    std::cerr
                        << "[ERROR] hwframe transfer failed\n";

                    continue;
                }

                usable_frame = sw_frame;

            } else {

                std::cout << "[SW FRAME] CPU frame received\n";
            }

            // =================================================
            // YUV -> BGR
            // =================================================

            uint8_t* dest_slices[1] = {
                video_frame_buffer_.get()
            };

            int dest_linesize[1] = {
                config_.width * 3
            };

            sws_scale(
                sws_ctx,
                usable_frame->data,
                usable_frame->linesize,
                0,
                codec_ctx->height,
                dest_slices,
                dest_linesize);

            // =================================================
            // CALLBACK
            // =================================================

            FrameBuffer fb;

            fb.data = video_frame_buffer_.get();
            fb.size = config_.width * config_.height * 3;
            fb.width = config_.width;
            fb.height = config_.height;
            fb.stride = config_.width * 3;
            fb.format = PixelFormat::BGR;
            fb.frame_index = frame_count_.load();
            fb.timestamp_ns = get_timestamp_ns();
            fb.valid = true;

            if (!callback(fb)) {

                running_.store(false);
            }

            frame_count_.fetch_add(1);
        }

        av_packet_unref(pkt);
    }

    // =========================================================
    // CLEANUP
    // =========================================================

    std::cout << "\n[CLEANUP]\n";

    av_packet_free(&pkt);

    av_frame_free(&frame);

    av_frame_free(&sw_frame);

    avcodec_free_context(&codec_ctx);

    avformat_close_input(&fmt_ctx);

    sws_freeContext(sws_ctx);

    if (hw_device_ctx) {

        av_buffer_unref(&hw_device_ctx);
    }

    running_.store(false);

    std::cout << "[EXIT] Pipeline stopped\n";
}



// Callback function to handle new frames from GStreamer
/**
 * @brief GStreamer static callback (The Producer).
 * Grabs raw hardware-decoded buffers, clones them, and pushes them into a thread-safe queue.
 */
//static GstFlowReturn on_new_sample(GstElement* sink, gpointer user_data) {
//    // Access the InputPipeline instance through user_data
//    auto* pipeline_ptr = static_cast<yolo::InputPipeline*>(user_data);
//    
//    GstSample* sample = nullptr;
//    // Emit signal to pull the latest sample from appsink
//    g_signal_emit_by_name(sink, "pull-sample", &sample);
//
//    if (sample) {
//        GstBuffer* buffer = gst_sample_get_buffer(sample);
//        GstCaps* caps = gst_sample_get_caps(sample);
//        GstStructure* s = gst_caps_get_structure(caps, 0);
//        
//        int width, height;
//        gst_structure_get_int(s, "width", &width);
//        gst_structure_get_int(s, "height", &height);
//
//        GstMapInfo map;
//        // Map GStreamer buffer memory for CPU access
//        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
//            
//            // Calculate stride (byte jump per row) for hardware-aligned memory
//            size_t stride = map.size / height;
//
//            // Wrap hardware memory into a cv::Mat (Zero-copy wrapper)
//            cv::Mat hw_frame(height, width, CV_8UC3, (void*)map.data, stride);
//
//            // Thread-safe queue operation
//            {
//                std::lock_guard<std::mutex> lock(pipeline_ptr->get_queue_mutex());
//                
//                // If the queue is full, drop the oldest frame to maintain low latency (Real-time)
//                if (pipeline_ptr->get_frame_queue().size() >= pipeline_ptr->get_max_queue_size()) {
//                    pipeline_ptr->get_frame_queue().pop();
//                }
//
//                // Deep copy (.clone()) is MANDATORY here because GStreamer will 
//                // recycle the 'map.data' memory immediately after this function returns.
//                pipeline_ptr->get_frame_queue().push(hw_frame.clone());
//                pipeline_ptr->get_frame_cv().notify_one();
//            }
//
//            // Unmap memory to release hardware resources
//            gst_buffer_unmap(buffer, &map);
//        }
//        
//        // Cleanup GStreamer sample reference
//        gst_sample_unref(sample);
//        return GST_FLOW_OK;
//    }
//    
//    return GST_FLOW_ERROR;
//}

/**
 * @brief Consumer loop with Condition Variable to prevent Terminal Freezing.
 */
//void yolo::InputPipeline::capture_loop_rtsp(FrameCallback callback) {
//    set_thread_affinity(INPUT_THREAD_CPU);
//    gst_init(NULL, NULL);
//
//    // Optimized pipeline with internal queue to decouple hardware decoder from app logic
//    std::string launch_str = 
//        "rtspsrc location=" + config_.device_path + " latency=0 protocols=tcp ! "
//        "rtph265depay ! h265parse ! v4l2slh265dec ! "
//        "videoconvert ! video/x-raw, format=BGR ! "
//        "appsink name=mysink emit-signals=true sync=false drop=true async=false";
//
//    GError* error = nullptr;
//    GstElement* pipeline = gst_parse_launch(launch_str.c_str(), &error);
//    if (error) { /* handle error */ return; }
//
//    GstElement* appsink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
//    g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_sample), this);
//    gst_object_unref(appsink);
//
//    gst_element_set_state(pipeline, GST_STATE_PLAYING);
//    std::cout << "[SUCCESS] Hardware Pipeline Live. Processing frames..." << std::endl;
//
//    while (running_.load(std::memory_order_relaxed)) {
//        cv::Mat frame_to_process;
//        
//        {
//            // Wait until a frame is available or timeout (to check running_ flag)
//            std::unique_lock<std::mutex> lock(queue_mutex_);
//            frame_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] {
//                return !frame_queue_.empty() || !running_.load();
//            });
//
//            if (!running_.load()) break;
//            if (frame_queue_.empty()) continue;
//
//            frame_to_process = frame_queue_.front();
//            frame_queue_.pop();
//        }
//
//        // --- INFERENCE START ---
//        yolo::FrameBuffer fb;
//        fb.data = frame_to_process.data;
//        fb.width = frame_to_process.cols;
//        fb.height = frame_to_process.rows;
//        fb.stride = frame_to_process.step;
//        fb.format = yolo::PixelFormat::BGR;
//        fb.timestamp_ns = get_timestamp_ns();
//        fb.valid = true;
//
//        if (!callback(fb)) break; 
//        // --- INFERENCE END ---
//    }
//    
//    
//    // Cleanup: Properly transition the pipeline to NULL state to release HW resources
//    std::cout << "[INFO] Shutting down capture pipeline..." << std::endl;
//    gst_element_set_state(pipeline, GST_STATE_NULL);
//    gst_object_unref(pipeline);
//}

/**
 * @brief Main capture loop for RTSP streams (The Consumer).
 * Orchestrates GStreamer pipeline and manages the delivery of frames to the AI engine.

void InputPipeline::capture_loop_rtsp(FrameCallback callback) {
    // Set CPU affinity to optimize performance on Raspberry Pi's A76 cores
    set_thread_affinity(INPUT_THREAD_CPU);
    
    // Initialize GStreamer core
    gst_init(NULL, NULL);

    // Optimized pipeline for Pi 5 Hardware Decoding (Stateless)
    // - protocols=tcp: Ensures no packet loss for reliable POC (Picture Order Count)
    // - v4l2slh265dec: Utilizes the dedicated HEVC hardware block
    // - appsink sync=false: Prevents pipeline blocking due to clock synchronization
    std::string launch_str = 
        "rtspsrc location=" + config_.device_path + " latency=0 protocols=tcp ! "
        "rtph265depay ! h265parse ! v4l2slh265dec ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink name=mysink emit-signals=true sync=false drop=true";

    GError* error = nullptr;
    GstElement* pipeline = gst_parse_launch(launch_str.c_str(), &error);

    if (error) {
        std::cerr << "[FATAL] GStreamer Pipeline Launch Error: " << error->message << std::endl;
        g_error_free(error);
        return;
    }

    // Connect the 'new-sample' signal to our producer callback
    GstElement* appsink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
    g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_sample), this);
    gst_object_unref(appsink);

    // Set the pipeline to the PLAYING state
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    std::cout << "[SUCCESS] Hardware Pipeline Live. Consumer loop starting..." << std::endl;

    // Main consumption loop: Pulls frames from the queue and triggers AI inference
    while (running_.load(std::memory_order_relaxed)) {
        cv::Mat frame_to_process;
        bool has_frame = false;

        // Scope-locked queue check
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (!frame_queue_.empty()) {
                frame_to_process = frame_queue_.front();
                frame_queue_.pop();
                has_frame = true;
            }
        }

        if (has_frame) {
            // Package the image for the Inference Engine (YOLO/NCNN)
            yolo::FrameBuffer fb;
            fb.data = frame_to_process.data;
            fb.width = static_cast<uint32_t>(frame_to_process.cols);
            fb.height = static_cast<uint32_t>(frame_to_process.rows);
            fb.stride = static_cast<uint32_t>(frame_to_process.step);
            fb.format = yolo::PixelFormat::BGR;
            fb.timestamp_ns = get_timestamp_ns();
            fb.valid = true;

            // Trigger the AI processing callback
            if (!callback(fb)) {
                break; // Stop if the AI pipeline requests termination
            }
        } else {
            // Small sleep to prevent CPU spinning when the queue is empty
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    }

    // Cleanup: Properly transition the pipeline to NULL state to release HW resources
    std::cout << "[INFO] Shutting down capture pipeline..." << std::endl;
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
}
*/

}  // namespace yolo
