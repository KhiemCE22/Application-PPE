#include "zenoh_transport.h"

#include <algorithm>
#include <chrono>
#include <iostream>

ZenohPublisher::~ZenohPublisher() {
    shutdown();
}

bool ZenohPublisher::initialize(const ZenohConfig& config) {
    config_ = config;
    if (!config_.enabled) return true;

    std::cout << "[INIT] Opening Zenoh cloud session to router "
              << config_.router_ip << "...\n";
    z_owned_config_t cloud_cfg;
    z_config_default(&cloud_cfg);
//    const std::string router_ep = "tcp/" + config_.router_ip + ":7447";
    const std::string router_ep = "tcp/" + config_.router_ip + ":7449";
    zp_config_insert(z_config_loan_mut(&cloud_cfg),
                     Z_CONFIG_CONNECT_KEY,
                     router_ep.c_str());
    zp_config_insert(z_config_loan_mut(&cloud_cfg),
                     Z_CONFIG_MODE_KEY,
                     "client");

    if (z_open(&cloud_session_, z_config_move(&cloud_cfg), NULL) < 0) {
        std::cerr << "[FATAL] Zenoh cloud session failed to "
                  << router_ep << "\n";
        return false;
    }
    zp_start_read_task(z_session_loan_mut(&cloud_session_), NULL);
    zp_start_lease_task(z_session_loan_mut(&cloud_session_), NULL);
    cloud_open_ = true;
    std::cout << "[INIT] Zenoh cloud session open.\n";

    std::cout << "[INIT] Opening Zenoh P2P session for ASN"
              << " listen=" << config_.p2p_listen;
    if (!config_.asn_peer.empty()) {
        std::cout << " peer=" << config_.asn_peer;
    }
    std::cout << "...\n";

    z_owned_config_t p2p_cfg;
    z_config_default(&p2p_cfg);
    zp_config_insert(z_config_loan_mut(&p2p_cfg), Z_CONFIG_MODE_KEY, "peer");
    if (!config_.p2p_listen.empty()) {
        zp_config_insert(z_config_loan_mut(&p2p_cfg),
                         Z_CONFIG_LISTEN_KEY,
                         config_.p2p_listen.c_str());
    }
    if (!config_.asn_peer.empty()) {
        zp_config_insert(z_config_loan_mut(&p2p_cfg),
                         Z_CONFIG_CONNECT_KEY,
                         config_.asn_peer.c_str());
    }

    if (z_open(&p2p_session_, z_config_move(&p2p_cfg), NULL) < 0) {
        std::cerr << "[FATAL] Zenoh P2P session failed";
        if (!config_.asn_peer.empty()) {
            std::cerr << " for ASN peer " << config_.asn_peer;
        }
        std::cerr << "\n";
        return false;
    }
    zp_start_read_task(z_session_loan_mut(&p2p_session_), NULL);
    zp_start_lease_task(z_session_loan_mut(&p2p_session_), NULL);
    p2p_open_ = true;
    std::cout << "[INIT] Zenoh P2P session open.\n";

    const std::string base = "factory/" + config_.cam_id;
    if (!declare_cloud_publisher(pub_stats_, base + "/stats")) return false;
    if (!declare_cloud_publisher(pub_events_, base + "/events")) return false;
    if (!declare_cloud_publisher(pub_events_image_, base + "/events/image")) return false;
    if (!declare_cloud_publisher(pub_image_, base + "/image")) return false;
    if (!declare_cloud_publisher(pub_count_cloud_, "factory/count")) return false;
    if (!declare_p2p_publisher(pub_count_p2p_, "factory/count")) return false;

    publishers_declared_ = true;
    return true;
}

void ZenohPublisher::shutdown() {
    if (publishers_declared_) {
        z_publisher_drop(z_publisher_move(&pub_stats_));
        z_publisher_drop(z_publisher_move(&pub_events_));
        z_publisher_drop(z_publisher_move(&pub_events_image_));
        z_publisher_drop(z_publisher_move(&pub_image_));
        z_publisher_drop(z_publisher_move(&pub_count_cloud_));
        z_publisher_drop(z_publisher_move(&pub_count_p2p_));
        publishers_declared_ = false;
    }
    if (cloud_open_) {
        z_session_drop(z_session_move(&cloud_session_));
        cloud_open_ = false;
    }
    if (p2p_open_) {
        z_session_drop(z_session_move(&p2p_session_));
        p2p_open_ = false;
    }
}

void ZenohPublisher::publish_count(const std::string& node_id,
                                   int in_count,
                                   int out_count) {
    if (!enabled()) return;

    const int inside = std::max(0, in_count - out_count);
    const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    std::string count_json =
        "{\"node_id\":\"" + node_id + "\""
        ",\"in\":" + std::to_string(in_count) +
        ",\"out\":" + std::to_string(out_count) +
        ",\"inside\":" + std::to_string(inside) +
        ",\"timestamp\":" + std::to_string(ts) +
        "}";

//    if (cloud_open_) {
//        publish_str(pub_count_cloud_, count_json);
//    }
    if (p2p_open_) {
        publish_str(pub_count_p2p_, count_json);
    }
    std::cout << "[ZENOH] Count cloud=" 
              << " p2p=" << (p2p_open_ ? "yes" : "no")
              << ": " << count_json << "\n";
}

void ZenohPublisher::publish_stats(const std::string& json) {
    if (cloud_open_) publish_str(pub_stats_, json);
}

void ZenohPublisher::publish_event(const std::string& json) {
    if (cloud_open_) publish_str(pub_events_, json);
}

void ZenohPublisher::publish_event_image(const void* data, size_t size) {
    if (cloud_open_) publish_bytes(pub_events_image_, data, size);
}

void ZenohPublisher::publish_snapshot(const void* data, size_t size) {
    if (cloud_open_) publish_bytes(pub_image_, data, size);
}

bool ZenohPublisher::declare_cloud_publisher(z_owned_publisher_t& pub,
                                             const std::string& topic) {
    return declare_publisher(cloud_session_, pub, topic);
}

bool ZenohPublisher::declare_p2p_publisher(z_owned_publisher_t& pub,
                                           const std::string& topic) {
    return declare_publisher(p2p_session_, pub, topic);
}

bool ZenohPublisher::declare_publisher(z_owned_session_t& session,
                                       z_owned_publisher_t& pub,
                                       const std::string& topic) {
    z_view_keyexpr_t ke;
    z_view_keyexpr_from_str(&ke, topic.c_str());
    if (z_declare_publisher(z_session_loan(&session),
                            &pub,
                            z_view_keyexpr_loan(&ke), NULL) < 0) {
        std::cerr << "[ERROR] Failed to declare publisher: " << topic << "\n";
        return false;
    }
    std::cout << "[ZENOH] Publisher ready: " << topic << "\n";
    return true;
}

void ZenohPublisher::publish_bytes(z_owned_publisher_t& publisher,
                                   const void* data,
                                   size_t size) {
    z_owned_bytes_t z_payload;
    z_bytes_copy_from_buf(&z_payload,
                          reinterpret_cast<const uint8_t*>(data),
                          size);
    z_publisher_put(z_publisher_loan(&publisher),
                    z_bytes_move(&z_payload), NULL);
}

void ZenohPublisher::publish_str(z_owned_publisher_t& publisher,
                                 const std::string& value) {
    publish_bytes(publisher, value.data(), value.size());
}
