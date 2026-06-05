#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <zenoh-pico.h>

struct ZenohConfig {
    bool enabled = false;
    std::string cam_id = "cam1";
    std::string router_ip = "192.168.1.9";
    std::string p2p_listen = "tcp/0.0.0.0:7448";
    std::string asn_peer;
};

class ZenohPublisher {
public:
    ZenohPublisher() = default;
    ZenohPublisher(const ZenohPublisher&) = delete;
    ZenohPublisher& operator=(const ZenohPublisher&) = delete;
    ~ZenohPublisher();

    bool initialize(const ZenohConfig& config);
    void shutdown();

    bool enabled() const { return cloud_open_ || p2p_open_; }
    bool cloud_open() const { return cloud_open_; }
    bool p2p_open() const { return p2p_open_; }

    void publish_count(const std::string& node_id, int in_count, int out_count);
    void publish_stats(const std::string& json);
    void publish_event(const std::string& json);
    void publish_event_image(const void* data, size_t size);
    void publish_snapshot(const void* data, size_t size);

private:
    bool declare_cloud_publisher(z_owned_publisher_t& pub, const std::string& topic);
    bool declare_p2p_publisher(z_owned_publisher_t& pub, const std::string& topic);
    bool declare_publisher(z_owned_session_t& session,
                           z_owned_publisher_t& pub,
                           const std::string& topic);

    void publish_bytes(z_owned_publisher_t& publisher, const void* data, size_t size);
    void publish_str(z_owned_publisher_t& publisher, const std::string& value);

    ZenohConfig config_;
    z_owned_session_t cloud_session_{};
    z_owned_session_t p2p_session_{};
    z_owned_publisher_t pub_stats_{};
    z_owned_publisher_t pub_events_{};
    z_owned_publisher_t pub_events_image_{};
    z_owned_publisher_t pub_image_{};
    z_owned_publisher_t pub_count_cloud_{};
    z_owned_publisher_t pub_count_p2p_{};
    bool cloud_open_ = false;
    bool p2p_open_ = false;
    bool publishers_declared_ = false;
};
