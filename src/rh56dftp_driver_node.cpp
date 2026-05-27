#include <rclcpp/rclcpp.hpp>

#include "hand_msgs/msg/rh56_dftp_angle_command.hpp"
#include "hand_msgs/msg/rh56_dftp_feedback.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

using namespace std::chrono_literals;

namespace
{
constexpr std::size_t kActuatorCount = 6;

constexpr uint16_t kAngleWriteRegister = 1486;
constexpr uint16_t kResetErrorRegister = 1004;
constexpr uint16_t kResetErrorValue = 1;

constexpr uint16_t kPosActRegister = 1534;
constexpr uint16_t kAngleActRegister = 1546;
constexpr uint16_t kForceActRegister = 1582;
constexpr uint16_t kCurrentRegister = 1594;
constexpr uint16_t kErrRegister = 1606;
constexpr uint16_t kStatusRegister = 1612;
constexpr uint16_t kTemperatureRegister = 1618;

const std::array<std::string, kActuatorCount> kFingerNames{
    "pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotate"};

std::string toLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string parseHandId(const std::string &hand_id)
{
    const auto lower = toLower(hand_id);

    if (lower.find("left") != std::string::npos || lower == "l")
        return "left";
    if (lower.find("right") != std::string::npos || lower == "r")
        return "right";

    return "";
}

std::string displayHandId(const std::string &hand)
{
    return hand == "left" ? "Left" : "Right";
}

std::vector<std::string> activeHandsFromParameter(const std::string &controlled_hand)
{
    const auto hand = toLower(controlled_hand);

    if (hand == "left")
        return {"left"};
    if (hand == "right")
        return {"right"};
    if (hand == "both")
        return {"left", "right"};

    throw std::invalid_argument("controlled_hand must be 'left', 'right', or 'both'");
}

template <typename Destination, std::size_t N>
void copyVectorToArray(const std::vector<Destination> &source, std::array<Destination, N> &dest)
{
    dest.fill(Destination{});
    const auto count = std::min<std::size_t>(source.size(), N);
    for (std::size_t i = 0; i < count; ++i)
        dest[i] = source[i];
}

int16_t clampAngle(double value, int min_angle, int max_angle)
{
    value = std::clamp(value, static_cast<double>(min_angle), static_cast<double>(max_angle));
    return static_cast<int16_t>(std::lround(value));
}

std::vector<uint8_t> buildReadHoldingRegistersRequest(
    uint16_t transaction_id,
    uint8_t unit_id,
    uint16_t start_register,
    uint16_t register_count)
{
    return {
        static_cast<uint8_t>((transaction_id >> 8) & 0xFF),
        static_cast<uint8_t>(transaction_id & 0xFF),
        0x00,
        0x00,
        0x00,
        0x06,
        unit_id,
        0x03,
        static_cast<uint8_t>((start_register >> 8) & 0xFF),
        static_cast<uint8_t>(start_register & 0xFF),
        static_cast<uint8_t>((register_count >> 8) & 0xFF),
        static_cast<uint8_t>(register_count & 0xFF)};
}

std::vector<uint8_t> buildWriteSingleRegisterRequest(
    uint16_t transaction_id,
    uint8_t unit_id,
    uint16_t register_address,
    uint16_t register_value)
{
    return {
        static_cast<uint8_t>((transaction_id >> 8) & 0xFF),
        static_cast<uint8_t>(transaction_id & 0xFF),
        0x00,
        0x00,
        0x00,
        0x06,
        unit_id,
        0x06,
        static_cast<uint8_t>((register_address >> 8) & 0xFF),
        static_cast<uint8_t>(register_address & 0xFF),
        static_cast<uint8_t>((register_value >> 8) & 0xFF),
        static_cast<uint8_t>(register_value & 0xFF)};
}

std::vector<uint8_t> buildWriteMultipleRegistersRequest(
    uint16_t transaction_id,
    uint8_t unit_id,
    uint16_t start_register,
    const std::array<int16_t, kActuatorCount> &values)
{
    const uint16_t register_count = static_cast<uint16_t>(values.size());
    const uint8_t byte_count = static_cast<uint8_t>(register_count * 2);
    const uint16_t mbap_length = static_cast<uint16_t>(7 + byte_count);

    std::vector<uint8_t> request{
        static_cast<uint8_t>((transaction_id >> 8) & 0xFF),
        static_cast<uint8_t>(transaction_id & 0xFF),
        0x00,
        0x00,
        static_cast<uint8_t>((mbap_length >> 8) & 0xFF),
        static_cast<uint8_t>(mbap_length & 0xFF),
        unit_id,
        0x10,
        static_cast<uint8_t>((start_register >> 8) & 0xFF),
        static_cast<uint8_t>(start_register & 0xFF),
        static_cast<uint8_t>((register_count >> 8) & 0xFF),
        static_cast<uint8_t>(register_count & 0xFF),
        byte_count};

    request.reserve(request.size() + byte_count);
    for (const auto value : values)
    {
        const auto raw = static_cast<uint16_t>(value);
        request.push_back(static_cast<uint8_t>((raw >> 8) & 0xFF));
        request.push_back(static_cast<uint8_t>(raw & 0xFF));
    }

    return request;
}

struct TouchRegisterBlock
{
    uint16_t start_register;
    uint16_t register_count;
};

const TouchRegisterBlock kPinkyTipTouch{3000, 9};
const TouchRegisterBlock kPinkyTopTouch{3018, 96};
const TouchRegisterBlock kPinkyPalmTouch{3210, 80};
const TouchRegisterBlock kRingTipTouch{3370, 9};
const TouchRegisterBlock kRingTopTouch{3388, 96};
const TouchRegisterBlock kRingPalmTouch{3580, 80};
const TouchRegisterBlock kMiddleTipTouch{3740, 9};
const TouchRegisterBlock kMiddleTopTouch{3758, 96};
const TouchRegisterBlock kMiddlePalmTouch{3950, 80};
const TouchRegisterBlock kIndexTipTouch{4110, 9};
const TouchRegisterBlock kIndexTopTouch{4128, 96};
const TouchRegisterBlock kIndexPalmTouch{4320, 80};
const TouchRegisterBlock kThumbTipTouch{4480, 9};
const TouchRegisterBlock kThumbTopTouch{4498, 96};
const TouchRegisterBlock kThumbMiddleTouch{4690, 9};
const TouchRegisterBlock kThumbPalmTouch{4708, 96};
const TouchRegisterBlock kPalmTouch{4900, 112};
}  // namespace

class RH56Node : public rclcpp::Node
{
public:
    RH56Node() : Node("rh56dftp_cpp_node")
    {
        declare_parameter("controlled_hand", "left");
        declare_parameter("command_topic", "/rh56dftp/angle_command");
        declare_parameter("feedback_topic", "/rh56dftp/feedback");
        declare_parameter("network_interface", "");
        declare_parameter("left_hand_ip", "192.168.123.210");
        declare_parameter("right_hand_ip", "192.168.123.211");
        declare_parameter("modbus_port", 6000);
        declare_parameter("left_modbus_unit_id", 1);
        declare_parameter("right_modbus_unit_id", 1);
        declare_parameter("modbus_timeout_ms", 1000);
        declare_parameter("polling_hz", 200.0);
        declare_parameter("feedback_rate_hz", 100.0);
        declare_parameter("command_rate_hz", 200.0);
        declare_parameter("touch_poll_divider", 50);
        declare_parameter("command_resend_period_s", 0.1);
        declare_parameter("reset_errors_on_startup", true);
        declare_parameter("open_hand_on_startup", true);
        declare_parameter("diagnostics_period_s", 5.0);
        declare_parameter("current_limit_ma", 1500);
        declare_parameter("contact_current_ma", 650);
        declare_parameter("soft_grasp_enabled", true);
        declare_parameter("unlock_current_ma", 500);
        declare_parameter("unlock_time_s", 1.0);
        declare_parameter("unlock_angle_margin", 10);
        declare_parameter("min_angle", 0);
        declare_parameter("max_angle", 1000);
        declare_parameter("higher_angle_opens", true);

        const auto active_hands =
            activeHandsFromParameter(get_parameter("controlled_hand").as_string());

        command_topic_ = get_parameter("command_topic").as_string();
        feedback_topic_ = get_parameter("feedback_topic").as_string();
        network_interface_ = get_parameter("network_interface").as_string();

        left_hand_ip_ = get_parameter("left_hand_ip").as_string();
        right_hand_ip_ = get_parameter("right_hand_ip").as_string();
        modbus_port_ = static_cast<uint16_t>(get_parameter("modbus_port").as_int());
        left_modbus_unit_id_ =
            static_cast<uint8_t>(get_parameter("left_modbus_unit_id").as_int());
        right_modbus_unit_id_ =
            static_cast<uint8_t>(get_parameter("right_modbus_unit_id").as_int());
        modbus_timeout_ms_ = static_cast<int>(get_parameter("modbus_timeout_ms").as_int());

        polling_hz_ = std::max(1.0, get_parameter("polling_hz").as_double());
        feedback_rate_hz_ = std::max(1.0, get_parameter("feedback_rate_hz").as_double());
        command_rate_hz_ = std::max(1.0, get_parameter("command_rate_hz").as_double());
        touch_poll_divider_ =
            static_cast<int>(get_parameter("touch_poll_divider").as_int());
        command_resend_period_s_ =
            std::max(0.0, get_parameter("command_resend_period_s").as_double());
        diagnostics_period_s_ =
            std::max(0.0, get_parameter("diagnostics_period_s").as_double());
        reset_errors_on_startup_ = get_parameter("reset_errors_on_startup").as_bool();
        open_hand_on_startup_ = get_parameter("open_hand_on_startup").as_bool();

        current_limit_ = get_parameter("current_limit_ma").as_int();
        contact_current_ = get_parameter("contact_current_ma").as_int();
        soft_grasp_ = get_parameter("soft_grasp_enabled").as_bool();
        unlock_current_ = get_parameter("unlock_current_ma").as_int();
        unlock_time_ = get_parameter("unlock_time_s").as_double();
        unlock_angle_margin_ = get_parameter("unlock_angle_margin").as_int();
        min_angle_ = get_parameter("min_angle").as_int();
        max_angle_ = get_parameter("max_angle").as_int();
        higher_angle_opens_ = get_parameter("higher_angle_opens").as_bool();

        if (!network_interface_.empty())
        {
            RCLCPP_INFO(
                get_logger(),
                "network_interface='%s' ignored in direct Modbus mode",
                network_interface_.c_str());
        }

        cmd_sub_ = create_subscription<hand_msgs::msg::RH56DFTPAngleCommand>(
            command_topic_,
            10,
            std::bind(&RH56Node::cmdCallback, this, std::placeholders::_1));

        feedback_pub_ =
            create_publisher<hand_msgs::msg::RH56DFTPFeedback>(feedback_topic_, 10);

        feedback_timer_ = create_wall_timer(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::duration<double>(1.0 / feedback_rate_hz_)),
            std::bind(&RH56Node::publishFeedback, this));

        if (diagnostics_period_s_ > 0.0)
        {
            diagnostics_timer_ = create_wall_timer(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::duration<double>(diagnostics_period_s_)),
                std::bind(&RH56Node::reportDiagnostics, this));
        }

        for (const auto &hand : active_hands)
        {
            auto context = createHandContext(hand);
            hand_by_id_[hand] = context;
            hands_.push_back(context);
        }

        running_ = true;
        for (auto &hand : hands_)
        {
            if (reset_errors_on_startup_)
                resetErrors(*hand);
            if (open_hand_on_startup_)
                openHand(*hand);

            hand->poll_thread = std::thread(&RH56Node::pollLoop, this, hand.get());
            hand->control_thread = std::thread(&RH56Node::controlLoop, this, hand.get());
        }

        RCLCPP_INFO(
            get_logger(),
            "RH56 direct Modbus driver started: command_topic=%s feedback_topic=%s polling_hz=%.1f command_rate_hz=%.1f touch_poll_divider=%d",
            command_topic_.c_str(),
            feedback_topic_.c_str(),
            polling_hz_,
            command_rate_hz_,
            touch_poll_divider_);
    }

    ~RH56Node() override
    {
        running_ = false;
        for (auto &hand : hands_)
        {
            {
                std::lock_guard<std::mutex> socket_lock(hand->socket_mutex);
                closeSocketLocked(*hand);
            }

            if (hand->poll_thread.joinable())
                hand->poll_thread.join();
            if (hand->control_thread.joinable())
                hand->control_thread.join();
        }
    }

private:
    struct HandContext
    {
        explicit HandContext(
            const std::string &hand_name,
            const std::string &ip_address,
            uint8_t modbus_unit_id)
            : hand(hand_name),
              hand_id(displayHandId(hand_name)),
              ip(ip_address),
              unit_id(modbus_unit_id)
        {
            desired_angles.fill(1000);
            safe_angles.fill(1000);
            last_sent_angles.fill(-1);
            hard_locked.fill(false);
            contact_locked.fill(false);
            unlock_start_time.fill(0.0);
            latest_feedback.hand_id = hand_id;
        }

        std::string hand;
        std::string hand_id;
        std::string ip;
        uint8_t unit_id{1};
        int socket_fd{-1};
        std::mutex socket_mutex;
        std::mutex data_mutex;

        std::array<int16_t, kActuatorCount> desired_angles{};
        std::array<int16_t, kActuatorCount> safe_angles{};
        std::array<int16_t, kActuatorCount> last_sent_angles{};
        std::array<bool, kActuatorCount> hard_locked{};
        std::array<bool, kActuatorCount> contact_locked{};
        std::array<double, kActuatorCount> unlock_start_time{};

        hand_msgs::msg::RH56DFTPFeedback latest_feedback;
        bool has_feedback{false};
        bool has_sent_command{false};
        uint64_t poll_iteration{0};

        double last_command_time_s{0.0};
        double last_poll_time_s{0.0};
        double last_write_time_s{0.0};
        uint64_t command_count{0};
        uint64_t poll_success_count{0};
        uint64_t poll_failure_count{0};
        uint64_t write_success_count{0};
        uint64_t write_failure_count{0};

        std::thread poll_thread;
        std::thread control_thread;
    };

    std::string command_topic_;
    std::string feedback_topic_;
    std::string network_interface_;
    std::string left_hand_ip_;
    std::string right_hand_ip_;
    uint16_t modbus_port_{6000};
    uint8_t left_modbus_unit_id_{1};
    uint8_t right_modbus_unit_id_{1};
    int modbus_timeout_ms_{1000};
    double polling_hz_{200.0};
    double feedback_rate_hz_{50.0};
    double command_rate_hz_{50.0};
        int touch_poll_divider_{50};
    double command_resend_period_s_{0.1};
    double diagnostics_period_s_{5.0};
    bool reset_errors_on_startup_{true};
    bool open_hand_on_startup_{true};

    int current_limit_{1500};
    int contact_current_{650};
    bool soft_grasp_{true};
    int unlock_current_{500};
    double unlock_time_{1.0};
    int unlock_angle_margin_{10};
    int min_angle_{0};
    int max_angle_{1000};
    bool higher_angle_opens_{true};

    std::vector<std::shared_ptr<HandContext>> hands_;
    std::unordered_map<std::string, std::shared_ptr<HandContext>> hand_by_id_;
    std::atomic_bool running_{false};
    std::atomic_uint16_t transaction_counter_{1};

    rclcpp::Subscription<hand_msgs::msg::RH56DFTPAngleCommand>::SharedPtr cmd_sub_;
    rclcpp::Publisher<hand_msgs::msg::RH56DFTPFeedback>::SharedPtr feedback_pub_;
    rclcpp::TimerBase::SharedPtr feedback_timer_;
    rclcpp::TimerBase::SharedPtr diagnostics_timer_;

    std::shared_ptr<HandContext> createHandContext(const std::string &hand)
    {
        const auto ip = hand == "left" ? left_hand_ip_ : right_hand_ip_;
        const auto unit_id =
            hand == "left" ? left_modbus_unit_id_ : right_modbus_unit_id_;

        auto context = std::make_shared<HandContext>(hand, ip, unit_id);
        RCLCPP_INFO(
            get_logger(),
            "%s hand direct Modbus target: %s:%u unit_id=%u",
            context->hand_id.c_str(),
            context->ip.c_str(),
            modbus_port_,
            static_cast<unsigned>(context->unit_id));
        return context;
    }

    std::vector<std::shared_ptr<HandContext>> commandTargets(const std::string &hand_id)
    {
        const auto requested_hand = parseHandId(hand_id);

        if (requested_hand.empty())
            return hands_;

        auto it = hand_by_id_.find(requested_hand);
        if (it == hand_by_id_.end())
            return {};

        return {it->second};
    }

    bool connectSocketLocked(HandContext &hand)
    {
        if (hand.socket_fd >= 0)
            return true;

        hand.socket_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (hand.socket_fd < 0)
            return false;

        timeval timeout{};
        timeout.tv_sec = modbus_timeout_ms_ / 1000;
        timeout.tv_usec = (modbus_timeout_ms_ % 1000) * 1000;
        ::setsockopt(hand.socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        ::setsockopt(hand.socket_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(modbus_port_);
        if (::inet_pton(AF_INET, hand.ip.c_str(), &address.sin_addr) != 1)
        {
            closeSocketLocked(hand);
            return false;
        }

        if (::connect(
                hand.socket_fd,
                reinterpret_cast<const sockaddr *>(&address),
                sizeof(address)) != 0)
        {
            closeSocketLocked(hand);
            return false;
        }

        return true;
    }

    void closeSocketLocked(HandContext &hand)
    {
        if (hand.socket_fd >= 0)
        {
            ::close(hand.socket_fd);
            hand.socket_fd = -1;
        }
    }

    bool readExactLocked(HandContext &hand, uint8_t *buffer, std::size_t length)
    {
        std::size_t total = 0;
        while (total < length)
        {
            const auto received = ::recv(
                hand.socket_fd,
                buffer + total,
                length - total,
                0);
            if (received <= 0)
                return false;

            total += static_cast<std::size_t>(received);
        }

        return true;
    }

    bool transactLocked(
        HandContext &hand,
        const std::vector<uint8_t> &request,
        std::vector<uint8_t> &response)
    {
        if (!connectSocketLocked(hand))
            return false;

        const auto sent = ::send(hand.socket_fd, request.data(), request.size(), 0);
        if (sent != static_cast<ssize_t>(request.size()))
        {
            closeSocketLocked(hand);
            return false;
        }

        std::array<uint8_t, 6> mbap_header{};
        if (!readExactLocked(hand, mbap_header.data(), mbap_header.size()))
        {
            closeSocketLocked(hand);
            return false;
        }

        const uint16_t remaining_length =
            static_cast<uint16_t>((mbap_header[4] << 8) | mbap_header[5]);

        response.resize(mbap_header.size() + remaining_length);
        std::copy(mbap_header.begin(), mbap_header.end(), response.begin());

        if (!readExactLocked(hand, response.data() + mbap_header.size(), remaining_length))
        {
            closeSocketLocked(hand);
            return false;
        }

        return true;
    }

    bool readHoldingRegistersSignedLocked(
        HandContext &hand,
        uint16_t start_register,
        uint16_t register_count,
        std::vector<int16_t> &values)
    {
        const auto transaction_id = transaction_counter_.fetch_add(1);
        const auto request = buildReadHoldingRegistersRequest(
            transaction_id,
            hand.unit_id,
            start_register,
            register_count);

        std::vector<uint8_t> response;
        if (!transactLocked(hand, request, response))
            return false;

        if (response.size() < 9 || response[7] != 0x03)
            return false;

        const auto byte_count = static_cast<std::size_t>(response[8]);
        const auto expected_byte_count = static_cast<std::size_t>(register_count) * 2;
        if (byte_count != expected_byte_count || response.size() < 9 + byte_count)
            return false;

        values.resize(register_count);
        for (std::size_t i = 0; i < register_count; ++i)
        {
            const std::size_t offset = 9 + i * 2;
            const uint16_t raw =
                static_cast<uint16_t>((response[offset] << 8) | response[offset + 1]);
            values[i] = static_cast<int16_t>(raw);
        }

        return true;
    }

    bool readHoldingRegistersBytesLocked(
        HandContext &hand,
        uint16_t start_register,
        uint16_t register_count,
        std::vector<uint8_t> &values)
    {
        const auto transaction_id = transaction_counter_.fetch_add(1);
        const auto request = buildReadHoldingRegistersRequest(
            transaction_id,
            hand.unit_id,
            start_register,
            register_count);

        std::vector<uint8_t> response;
        if (!transactLocked(hand, request, response))
            return false;

        if (response.size() < 9 || response[7] != 0x03)
            return false;

        const auto byte_count = static_cast<std::size_t>(response[8]);
        const auto expected_byte_count = static_cast<std::size_t>(register_count) * 2;
        if (byte_count != expected_byte_count || response.size() < 9 + byte_count)
            return false;

        values.assign(response.begin() + 9, response.begin() + 9 + byte_count);
        return true;
    }

    bool writeSingleRegisterLocked(HandContext &hand, uint16_t address, uint16_t value)
    {
        const auto transaction_id = transaction_counter_.fetch_add(1);
        const auto request = buildWriteSingleRegisterRequest(
            transaction_id,
            hand.unit_id,
            address,
            value);

        std::vector<uint8_t> response;
        if (!transactLocked(hand, request, response))
            return false;

        return response.size() >= 12 &&
               response[7] == 0x06 &&
               response[8] == request[8] &&
               response[9] == request[9] &&
               response[10] == request[10] &&
               response[11] == request[11];
    }

    bool writeAnglesLocked(
        HandContext &hand,
        const std::array<int16_t, kActuatorCount> &angles)
    {
        const auto transaction_id = transaction_counter_.fetch_add(1);
        const auto request = buildWriteMultipleRegistersRequest(
            transaction_id,
            hand.unit_id,
            kAngleWriteRegister,
            angles);

        std::vector<uint8_t> response;
        if (!transactLocked(hand, request, response))
            return false;

        return response.size() >= 12 &&
               response[7] == 0x10 &&
               response[8] == request[8] &&
               response[9] == request[9];
    }

    void resetErrors(HandContext &hand)
    {
        std::lock_guard<std::mutex> lock(hand.socket_mutex);
        if (writeSingleRegisterLocked(hand, kResetErrorRegister, kResetErrorValue))
        {
            RCLCPP_INFO(
                get_logger(),
                "Startup error reset sent to %s hand via Modbus TCP (%s:%u, unit_id=%u)",
                hand.hand_id.c_str(),
                hand.ip.c_str(),
                modbus_port_,
                static_cast<unsigned>(hand.unit_id));
        }
        else
        {
            RCLCPP_WARN(
                get_logger(),
                "Startup error reset failed for %s hand via Modbus TCP (%s:%u, unit_id=%u)",
                hand.hand_id.c_str(),
                hand.ip.c_str(),
                modbus_port_,
                static_cast<unsigned>(hand.unit_id));
        }
    }

    void openHand(HandContext &hand)
    {
        std::array<int16_t, kActuatorCount> open_angles{1000, 1000, 1000, 1000, 1000, 0};
        {
            std::lock_guard<std::mutex> lock(hand.socket_mutex);
            if (!writeAnglesLocked(hand, open_angles))
            {
                RCLCPP_WARN(
                    get_logger(),
                    "Failed to open %s hand at startup",
                    hand.hand_id.c_str());
                return;
            }
        }

        {
            std::lock_guard<std::mutex> lock(hand.data_mutex);
            hand.desired_angles = open_angles;
            hand.safe_angles = open_angles;
            hand.last_sent_angles = open_angles;
            hand.has_sent_command = true;
            hand.last_write_time_s = now().seconds();
        }

        std::this_thread::sleep_for(1s);
        RCLCPP_INFO(get_logger(), "%s hand opened at startup", hand.hand_id.c_str());
    }

    bool readTouchBlockLocked(
        HandContext &hand,
        const TouchRegisterBlock &block,
        std::vector<int16_t> &values)
    {
        return readHoldingRegistersSignedLocked(
            hand,
            block.start_register,
            block.register_count,
            values);
    }

    bool pollHand(HandContext &hand)
    {
        std::vector<int16_t> position;
        std::vector<int16_t> angle;
        std::vector<int16_t> force;
        std::vector<int16_t> current;
        std::vector<uint8_t> error;
        std::vector<uint8_t> status;
        std::vector<uint8_t> temperature;
        std::vector<int16_t> pinky_tip;
        std::vector<int16_t> pinky_top;
        std::vector<int16_t> pinky_palm;
        std::vector<int16_t> ring_tip;
        std::vector<int16_t> ring_top;
        std::vector<int16_t> ring_palm;
        std::vector<int16_t> middle_tip;
        std::vector<int16_t> middle_top;
        std::vector<int16_t> middle_palm;
        std::vector<int16_t> index_tip;
        std::vector<int16_t> index_top;
        std::vector<int16_t> index_palm;
        std::vector<int16_t> thumb_tip;
        std::vector<int16_t> thumb_top;
        std::vector<int16_t> thumb_middle;
        std::vector<int16_t> thumb_palm;
        std::vector<int16_t> palm;
        bool read_touch_this_cycle = false;

        {
            std::lock_guard<std::mutex> socket_lock(hand.socket_mutex);
            read_touch_this_cycle =
                touch_poll_divider_ > 0 &&
                (hand.poll_iteration % static_cast<uint64_t>(touch_poll_divider_)) == 0;

            if (!readHoldingRegistersSignedLocked(hand, kPosActRegister, 6, position) ||
                !readHoldingRegistersSignedLocked(hand, kAngleActRegister, 6, angle) ||
                !readHoldingRegistersSignedLocked(hand, kForceActRegister, 6, force) ||
                !readHoldingRegistersSignedLocked(hand, kCurrentRegister, 6, current) ||
                !readHoldingRegistersBytesLocked(hand, kErrRegister, 3, error) ||
                !readHoldingRegistersBytesLocked(hand, kStatusRegister, 3, status) ||
                !readHoldingRegistersBytesLocked(hand, kTemperatureRegister, 3, temperature))
            {
                closeSocketLocked(hand);
                return false;
            }

            if (read_touch_this_cycle &&
                (!readTouchBlockLocked(hand, kPinkyTipTouch, pinky_tip) ||
                 !readTouchBlockLocked(hand, kPinkyTopTouch, pinky_top) ||
                 !readTouchBlockLocked(hand, kPinkyPalmTouch, pinky_palm) ||
                 !readTouchBlockLocked(hand, kRingTipTouch, ring_tip) ||
                 !readTouchBlockLocked(hand, kRingTopTouch, ring_top) ||
                 !readTouchBlockLocked(hand, kRingPalmTouch, ring_palm) ||
                 !readTouchBlockLocked(hand, kMiddleTipTouch, middle_tip) ||
                 !readTouchBlockLocked(hand, kMiddleTopTouch, middle_top) ||
                 !readTouchBlockLocked(hand, kMiddlePalmTouch, middle_palm) ||
                 !readTouchBlockLocked(hand, kIndexTipTouch, index_tip) ||
                 !readTouchBlockLocked(hand, kIndexTopTouch, index_top) ||
                 !readTouchBlockLocked(hand, kIndexPalmTouch, index_palm) ||
                 !readTouchBlockLocked(hand, kThumbTipTouch, thumb_tip) ||
                 !readTouchBlockLocked(hand, kThumbTopTouch, thumb_top) ||
                 !readTouchBlockLocked(hand, kThumbMiddleTouch, thumb_middle) ||
                 !readTouchBlockLocked(hand, kThumbPalmTouch, thumb_palm) ||
                 !readTouchBlockLocked(hand, kPalmTouch, palm)))
            {
                closeSocketLocked(hand);
                return false;
            }
        }

        {
            std::lock_guard<std::mutex> data_lock(hand.data_mutex);
            copyVectorToArray(position, hand.latest_feedback.position);
            copyVectorToArray(angle, hand.latest_feedback.angle);
            copyVectorToArray(force, hand.latest_feedback.force);
            copyVectorToArray(current, hand.latest_feedback.current);
            copyVectorToArray(error, hand.latest_feedback.error);
            copyVectorToArray(status, hand.latest_feedback.status);
            copyVectorToArray(temperature, hand.latest_feedback.temperature);
            if (read_touch_this_cycle)
            {
                copyVectorToArray(pinky_tip, hand.latest_feedback.pinky_tip_touch);
                copyVectorToArray(pinky_top, hand.latest_feedback.pinky_top_touch);
                copyVectorToArray(pinky_palm, hand.latest_feedback.pinky_palm_touch);
                copyVectorToArray(ring_tip, hand.latest_feedback.ring_tip_touch);
                copyVectorToArray(ring_top, hand.latest_feedback.ring_top_touch);
                copyVectorToArray(ring_palm, hand.latest_feedback.ring_palm_touch);
                copyVectorToArray(middle_tip, hand.latest_feedback.middle_tip_touch);
                copyVectorToArray(middle_top, hand.latest_feedback.middle_top_touch);
                copyVectorToArray(middle_palm, hand.latest_feedback.middle_palm_touch);
                copyVectorToArray(index_tip, hand.latest_feedback.index_tip_touch);
                copyVectorToArray(index_top, hand.latest_feedback.index_top_touch);
                copyVectorToArray(index_palm, hand.latest_feedback.index_palm_touch);
                copyVectorToArray(thumb_tip, hand.latest_feedback.thumb_tip_touch);
                copyVectorToArray(thumb_top, hand.latest_feedback.thumb_top_touch);
                copyVectorToArray(thumb_middle, hand.latest_feedback.thumb_middle_touch);
                copyVectorToArray(thumb_palm, hand.latest_feedback.thumb_palm_touch);
                copyVectorToArray(palm, hand.latest_feedback.palm_touch);
            }
            hand.latest_feedback.hand_id = hand.hand_id;
            hand.has_feedback = true;
            hand.last_poll_time_s = now().seconds();
            ++hand.poll_iteration;
            ++hand.poll_success_count;
            checkCurrentLocked(hand);
        }

        return true;
    }

    void cmdCallback(hand_msgs::msg::RH56DFTPAngleCommand::SharedPtr msg)
    {
        const auto targets = commandTargets(msg->hand_id);
        if (targets.empty())
            return;

        for (auto &hand : targets)
        {
            std::lock_guard<std::mutex> lock(hand->data_mutex);
            hand->last_command_time_s = now().seconds();
            ++hand->command_count;

            for (std::size_t i = 0; i < kActuatorCount; ++i)
            {
                const auto requested = clampAngle(msg->angles[i], min_angle_, max_angle_);
                if (commandOpensFromSafeAngle(*hand, i, requested))
                {
                    hand->contact_locked[i] = false;
                    hand->hard_locked[i] = false;
                    hand->unlock_start_time[i] = 0.0;
                }
                hand->desired_angles[i] = requested;
            }
        }
    }

    bool commandOpensFromSafeAngle(
        const HandContext &hand,
        std::size_t index,
        int16_t requested) const
    {
        if (higher_angle_opens_)
            return requested > hand.safe_angles[index] + unlock_angle_margin_;

        return requested < hand.safe_angles[index] - unlock_angle_margin_;
    }

    bool commandIsClosing(
        const HandContext &hand,
        std::size_t index,
        int16_t actual_angle) const
    {
        if (higher_angle_opens_)
            return hand.desired_angles[index] < actual_angle - unlock_angle_margin_;

        return hand.desired_angles[index] > actual_angle + unlock_angle_margin_;
    }

    void checkCurrentLocked(HandContext &hand)
    {
        for (std::size_t i = 0; i < kActuatorCount; ++i)
        {
            const int current_abs = std::abs(static_cast<int>(hand.latest_feedback.current[i]));
            const auto actual_angle = hand.latest_feedback.angle[i];
            const double now_s = now().seconds();

            if (current_abs > current_limit_ && !hand.hard_locked[i])
            {
                hand.hard_locked[i] = true;
                hand.contact_locked[i] = false;
                hand.safe_angles[i] = actual_angle;
                hand.unlock_start_time[i] = 0.0;

                RCLCPP_WARN(
                    get_logger(),
                    "%s hand overcurrent on %s: %d mA",
                    hand.hand_id.c_str(),
                    kFingerNames[i].c_str(),
                    current_abs);
            }

            if (soft_grasp_ &&
                current_abs > contact_current_ &&
                !hand.contact_locked[i] &&
                !hand.hard_locked[i] &&
                commandIsClosing(hand, i, actual_angle))
            {
                hand.contact_locked[i] = true;
                hand.safe_angles[i] = actual_angle;

                RCLCPP_INFO(
                    get_logger(),
                    "%s hand contact on %s: %d mA",
                    hand.hand_id.c_str(),
                    kFingerNames[i].c_str(),
                    current_abs);
            }

            if (!hand.hard_locked[i])
                continue;

            if (current_abs < unlock_current_)
            {
                if (hand.unlock_start_time[i] == 0.0)
                    hand.unlock_start_time[i] = now_s;

                if (now_s - hand.unlock_start_time[i] > unlock_time_)
                {
                    hand.hard_locked[i] = false;
                    hand.unlock_start_time[i] = 0.0;

                    RCLCPP_INFO(
                        get_logger(),
                        "%s hand unlocked %s",
                        hand.hand_id.c_str(),
                        kFingerNames[i].c_str());
                }
            }
            else
            {
                hand.unlock_start_time[i] = 0.0;
            }
        }
    }

    void pollLoop(HandContext *hand)
    {
        rclcpp::Rate rate(polling_hz_);

        while (running_ && rclcpp::ok())
        {
            if (!pollHand(*hand))
            {
                std::lock_guard<std::mutex> lock(hand->data_mutex);
                ++hand->poll_failure_count;
            }

            rate.sleep();
        }
    }

    void controlLoop(HandContext *hand)
    {
        rclcpp::Rate rate(command_rate_hz_);

        while (running_ && rclcpp::ok())
        {
            std::array<int16_t, kActuatorCount> command_angles{};
            bool should_write = false;
            double now_s = 0.0;

            {
                std::lock_guard<std::mutex> lock(hand->data_mutex);
                now_s = now().seconds();

                for (std::size_t i = 0; i < kActuatorCount; ++i)
                {
                    if (!hand->hard_locked[i] && !hand->contact_locked[i])
                        hand->safe_angles[i] = hand->desired_angles[i];

                    command_angles[i] = hand->safe_angles[i];
                    if (!hand->has_sent_command ||
                        hand->last_sent_angles[i] != command_angles[i])
                    {
                        should_write = true;
                    }
                }

                if (!should_write &&
                    (now_s - hand->last_write_time_s) >= command_resend_period_s_)
                {
                    should_write = true;
                }
            }

            if (should_write)
            {
                bool write_ok = false;
                {
                    std::lock_guard<std::mutex> socket_lock(hand->socket_mutex);
                    write_ok = writeAnglesLocked(*hand, command_angles);
                    if (!write_ok)
                        closeSocketLocked(*hand);
                }

                std::lock_guard<std::mutex> data_lock(hand->data_mutex);
                hand->last_write_time_s = now_s;
                hand->has_sent_command = true;
                hand->last_sent_angles = command_angles;

                if (write_ok)
                {
                    ++hand->write_success_count;
                }
                else
                {
                    ++hand->write_failure_count;
                    RCLCPP_WARN_THROTTLE(
                        get_logger(),
                        *get_clock(),
                        5000,
                        "Modbus write failed for %s hand (%s:%u unit_id=%u)",
                        hand->hand_id.c_str(),
                        hand->ip.c_str(),
                        modbus_port_,
                        static_cast<unsigned>(hand->unit_id));
                }
            }

            rate.sleep();
        }
    }

    void publishFeedback()
    {
        for (auto &hand : hands_)
        {
            hand_msgs::msg::RH56DFTPFeedback feedback;
            bool should_publish = false;

            {
                std::lock_guard<std::mutex> lock(hand->data_mutex);
                if (!hand->has_feedback)
                    continue;

                feedback = hand->latest_feedback;
                should_publish = true;
            }

            if (should_publish)
            {
                feedback.header.stamp = now();
                feedback_pub_->publish(feedback);
            }
        }
    }

    void reportDiagnostics()
    {
        const auto now_s = now().seconds();

        for (const auto &hand : hands_)
        {
            std::lock_guard<std::mutex> lock(hand->data_mutex);
            const auto last_command_age =
                hand->last_command_time_s > 0.0 ? now_s - hand->last_command_time_s : -1.0;
            const auto last_poll_age =
                hand->last_poll_time_s > 0.0 ? now_s - hand->last_poll_time_s : -1.0;
            const auto last_write_age =
                hand->last_write_time_s > 0.0 ? now_s - hand->last_write_time_s : -1.0;

            RCLCPP_INFO(
                get_logger(),
                "%s diag: commands=%lu last_cmd=%.2fs polls_ok=%lu polls_fail=%lu last_poll=%.2fs writes_ok=%lu writes_fail=%lu last_write=%.2fs",
                hand->hand_id.c_str(),
                static_cast<unsigned long>(hand->command_count),
                last_command_age,
                static_cast<unsigned long>(hand->poll_success_count),
                static_cast<unsigned long>(hand->poll_failure_count),
                last_poll_age,
                static_cast<unsigned long>(hand->write_success_count),
                static_cast<unsigned long>(hand->write_failure_count),
                last_write_age);
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RH56Node>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
