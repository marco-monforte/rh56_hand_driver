#include <rclcpp/rclcpp.hpp>

#include "hand_msgs/msg/hand_landmarks.hpp"
#include "hand_msgs/msg/rh56_dftp_angle_command.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace
{
constexpr std::size_t kLandmarkCount = 21;
constexpr std::size_t kFingerFeatureCount = 5;
constexpr std::size_t kActuatorCount = 6;

struct Vec3
{
    double x{0.0};
    double y{0.0};
    double z{0.0};
};

Vec3 operator-(const Vec3 &lhs, const Vec3 &rhs)
{
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

double dot(const Vec3 &lhs, const Vec3 &rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

double norm(const Vec3 &value)
{
    return std::sqrt(dot(value, value));
}

double clamp01(double value)
{
    return std::clamp(value, 0.0, 1.0);
}

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

std::string trim(const std::string &value)
{
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return "";

    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string expandUserPath(const std::string &path)
{
    if (path.rfind("~/", 0) != 0)
        return path;

    const char *home = std::getenv("HOME");
    if (home == nullptr)
        return path;

    return std::string(home) + path.substr(1);
}

bool fileExists(const std::string &path)
{
    std::ifstream input(path);
    return input.good();
}

std::vector<double> parseInlineList(std::string value)
{
    value = trim(value);
    if (!value.empty() && value.front() == '[')
        value.erase(value.begin());
    if (!value.empty() && value.back() == ']')
        value.pop_back();

    std::replace(value.begin(), value.end(), ',', ' ');

    std::vector<double> result;
    std::istringstream stream(value);
    double number = 0.0;
    while (stream >> number)
        result.push_back(number);

    return result;
}

std::string arrayToString(const std::array<double, kFingerFeatureCount> &values, int precision = 3)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << "[";
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (i > 0)
            stream << ", ";
        stream << values[i];
    }
    stream << "]";
    return stream.str();
}

std::string arrayToString(const std::array<double, kActuatorCount> &values, int precision = 2)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << "[";
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (i > 0)
            stream << ", ";
        stream << values[i];
    }
    stream << "]";
    return stream.str();
}
}  // namespace

class HandMapperNode : public rclcpp::Node
{
public:
    HandMapperNode() : Node("hand_mapper_node")
    {
        declare_parameter("calibration", false);
        declare_parameter("controlled_hand", "left");
        declare_parameter("input_topic", "/hand_landmarks");
        declare_parameter("command_topic", "/rh56dftp/angle_command");
        declare_parameter("calibration_file", "~/.rh56_hand_calibration.yaml");
        declare_parameter("alpha", 0.25);
        declare_parameter("deadband", 15.0);
        declare_parameter("min_angle", 20.0);
        declare_parameter("max_angle", 980.0);
        declare_parameter("debug_every_n", 10);

        calibration_enabled_ = get_parameter("calibration").as_bool();
        controlled_hand_ = toLower(get_parameter("controlled_hand").as_string());
        input_topic_ = get_parameter("input_topic").as_string();
        command_topic_ = get_parameter("command_topic").as_string();
        calibration_file_ = expandUserPath(get_parameter("calibration_file").as_string());
        alpha_ = std::clamp(get_parameter("alpha").as_double(), 0.0, 1.0);
        deadband_ = std::max(0.0, get_parameter("deadband").as_double());
        min_angle_ = get_parameter("min_angle").as_double();
        max_angle_ = get_parameter("max_angle").as_double();
        debug_every_n_ = std::max(1, static_cast<int>(get_parameter("debug_every_n").as_int()));

        if (controlled_hand_ != "left" && controlled_hand_ != "right" && controlled_hand_ != "both")
            throw std::invalid_argument("controlled_hand must be 'left', 'right', or 'both'");
        if (min_angle_ > max_angle_)
            throw std::invalid_argument("min_angle must be <= max_angle");

        filtered_angles_["left"].fill(0.0);
        filtered_angles_["right"].fill(0.0);

        angle_pub_ = create_publisher<hand_msgs::msg::RH56DFTPAngleCommand>(command_topic_, 10);
        landmarks_sub_ = create_subscription<hand_msgs::msg::HandLandmarks>(
            input_topic_,
            rclcpp::SensorDataQoS(),
            std::bind(&HandMapperNode::landmarksCallback, this, std::placeholders::_1));

        RCLCPP_INFO(
            get_logger(),
            "Hand mapper configured: controlled_hand=%s, input_topic=%s, command_topic=%s",
            controlled_hand_.c_str(),
            input_topic_.c_str(),
            command_topic_.c_str());
    }

    void initializeCalibration()
    {
        if (calibration_enabled_ || !fileExists(calibration_file_))
        {
            runCalibration();
            return;
        }

        if (!loadCalibration())
        {
            RCLCPP_WARN(
                get_logger(),
                "Could not load calibration from %s; starting calibration",
                calibration_file_.c_str());
            runCalibration();
        }
        else
        {
            RCLCPP_INFO(get_logger(), "Calibration loaded from file");
        }

        RCLCPP_INFO(get_logger(), "=== RH56 Hand Mapper READY ===");
    }

private:
    using FeatureArray = std::array<double, kFingerFeatureCount>;
    using AngleArray = std::array<double, kActuatorCount>;

    bool calibration_enabled_{false};
    std::string controlled_hand_{"left"};
    std::string input_topic_;
    std::string command_topic_;
    std::string calibration_file_;

    double alpha_{0.25};
    double deadband_{15.0};
    double min_angle_{20.0};
    double max_angle_{980.0};
    int debug_every_n_{10};
    int debug_counter_{0};

    std::vector<FeatureArray> open_close_data_;
    std::vector<double> opposition_data_;

    FeatureArray open_ref_{};
    FeatureArray close_ref_{};
    double opposition_min_{0.0};
    double opposition_max_{1.0};
    bool calibration_loaded_{false};

    std::map<std::string, AngleArray> filtered_angles_;

    rclcpp::Publisher<hand_msgs::msg::RH56DFTPAngleCommand>::SharedPtr angle_pub_;
    rclcpp::Subscription<hand_msgs::msg::HandLandmarks>::SharedPtr landmarks_sub_;

    void waitForEnter(const std::string &message)
    {
        RCLCPP_WARN(get_logger(), "%s", message.c_str());
        std::string unused;
        std::getline(std::cin, unused);
    }

    void progressBar(double elapsed, double total)
    {
        const double percent = std::min(elapsed / total, 1.0);
        const int bars = static_cast<int>(percent * 20.0);

        std::cout << "\r["
                  << std::string(bars, '#')
                  << std::string(20 - bars, '-')
                  << "] "
                  << std::setw(5)
                  << std::fixed
                  << std::setprecision(1)
                  << (percent * 100.0)
                  << "% "
                  << std::flush;
    }

    void collectCalibrationSamples(double seconds)
    {
        rclcpp::executors::SingleThreadedExecutor executor;
        executor.add_node(get_node_base_interface());

        const auto start = std::chrono::steady_clock::now();
        while (rclcpp::ok())
        {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - start).count();
            if (elapsed >= seconds)
                break;

            progressBar(elapsed, seconds);
            executor.spin_some();
            std::this_thread::sleep_for(10ms);
        }

        executor.spin_some();
        executor.remove_node(get_node_base_interface());
        progressBar(seconds, seconds);
        std::cout << std::endl;
    }

    FeatureArray meanFeatures() const
    {
        if (open_close_data_.empty())
            throw std::runtime_error("No hand landmark samples collected");

        FeatureArray result{};
        result.fill(0.0);

        for (const auto &sample : open_close_data_)
        {
            for (std::size_t i = 0; i < result.size(); ++i)
                result[i] += sample[i];
        }

        for (auto &value : result)
            value /= static_cast<double>(open_close_data_.size());

        return result;
    }

    double meanOpposition() const
    {
        if (opposition_data_.empty())
            throw std::runtime_error("No thumb opposition samples collected");

        const double sum = std::accumulate(opposition_data_.begin(), opposition_data_.end(), 0.0);
        return sum / static_cast<double>(opposition_data_.size());
    }

    void runCalibration()
    {
        RCLCPP_WARN(get_logger(), "=== CALIBRATION MODE ===");
        calibration_enabled_ = true;

        try
        {
            waitForEnter("Keep hand OPEN, then press ENTER to start OPEN acquisition (5s)...");
            open_close_data_.clear();
            opposition_data_.clear();
            collectCalibrationSamples(5.0);
            open_ref_ = meanFeatures();
            opposition_min_ = meanOpposition();
            RCLCPP_INFO(get_logger(), "OPEN reference: %s", arrayToString(open_ref_).c_str());
            RCLCPP_INFO(get_logger(), "Opposition min: %.4f", opposition_min_);

            waitForEnter("Keep hand CLOSED, then press ENTER to start CLOSED acquisition (5s)...");
            open_close_data_.clear();
            opposition_data_.clear();
            collectCalibrationSamples(5.0);
            close_ref_ = meanFeatures();
            RCLCPP_INFO(get_logger(), "CLOSE reference: %s", arrayToString(close_ref_).c_str());

            waitForEnter("Move thumb in MAXIMUM OPPOSITION, then press ENTER to start acquisition (5s)...");
            opposition_data_.clear();
            collectCalibrationSamples(5.0);
            opposition_max_ = meanOpposition();
            RCLCPP_INFO(get_logger(), "Opposition max: %.4f", opposition_max_);

            saveCalibration();
            calibration_loaded_ = true;
            RCLCPP_INFO(get_logger(), "=== CALIBRATION COMPLETE ===");
        }
        catch (const std::exception &error)
        {
            calibration_enabled_ = false;
            throw;
        }

        calibration_enabled_ = false;
        RCLCPP_INFO(get_logger(), "=== RH56 Hand Mapper READY ===");
    }

    void saveCalibration()
    {
        std::ofstream output(calibration_file_);
        if (!output)
            throw std::runtime_error("Could not write calibration file: " + calibration_file_);

        output << std::fixed << std::setprecision(10);

        output << "open_ref:\n";
        for (const auto value : open_ref_)
            output << "- " << value << "\n";

        output << "close_ref:\n";
        for (const auto value : close_ref_)
            output << "- " << value << "\n";

        output << "opposition_min: " << opposition_min_ << "\n";
        output << "opposition_max: " << opposition_max_ << "\n";

        RCLCPP_INFO(get_logger(), "Calibration saved to %s", calibration_file_.c_str());
    }

    bool loadCalibration()
    {
        std::ifstream input(calibration_file_);
        if (!input)
            return false;

        std::map<std::string, std::vector<double>> lists;
        std::map<std::string, double> scalars;

        std::string current_key;
        std::string line;
        while (std::getline(input, line))
        {
            const auto comment = line.find('#');
            if (comment != std::string::npos)
                line = line.substr(0, comment);

            const auto value = trim(line);
            if (value.empty())
                continue;

            if (value.rfind("-", 0) == 0)
            {
                if (!current_key.empty())
                    lists[current_key].push_back(std::stod(trim(value.substr(1))));
                continue;
            }

            const auto separator = value.find(':');
            if (separator == std::string::npos)
                continue;

            current_key = trim(value.substr(0, separator));
            const auto rest = trim(value.substr(separator + 1));

            if (rest.empty())
                continue;
            if (!rest.empty() && rest.front() == '[')
                lists[current_key] = parseInlineList(rest);
            else
                scalars[current_key] = std::stod(rest);
        }

        if (lists["open_ref"].size() != kFingerFeatureCount ||
            lists["close_ref"].size() != kFingerFeatureCount ||
            scalars.find("opposition_min") == scalars.end() ||
            scalars.find("opposition_max") == scalars.end())
        {
            return false;
        }

        for (std::size_t i = 0; i < kFingerFeatureCount; ++i)
        {
            open_ref_[i] = lists["open_ref"][i];
            close_ref_[i] = lists["close_ref"][i];
        }

        opposition_min_ = scalars["opposition_min"];
        opposition_max_ = scalars["opposition_max"];
        calibration_loaded_ = true;
        return true;
    }

    bool shouldProcessHand(const std::string &hand) const
    {
        if (hand.empty())
            return false;
        if (controlled_hand_ == "both")
            return hand == "left" || hand == "right";

        return hand == controlled_hand_;
    }

    Vec3 getPoint(const hand_msgs::msg::HandLandmarks &msg, std::size_t index) const
    {
        const auto &pose = msg.landmarks[index];
        return {pose.position.x, pose.position.y, pose.position.z};
    }

    FeatureArray normalizeFeatures(const FeatureArray &raw) const
    {
        FeatureArray result{};

        for (std::size_t i = 0; i < raw.size(); ++i)
        {
            double denominator = open_ref_[i] - close_ref_[i];
            if (std::abs(denominator) < 1e-6)
                denominator = 1e-6;

            result[i] = clamp01((raw[i] - close_ref_[i]) / denominator);
        }

        return result;
    }

    AngleArray applyDeadband(const std::string &hand, const AngleArray &target) const
    {
        AngleArray result = filtered_angles_.at(hand);

        for (std::size_t i = 0; i < target.size(); ++i)
        {
            if (std::abs(target[i] - result[i]) > deadband_)
                result[i] = target[i];
        }

        return result;
    }

    AngleArray lowPassFilter(const std::string &hand, const AngleArray &target)
    {
        auto &filtered = filtered_angles_[hand];

        for (std::size_t i = 0; i < target.size(); ++i)
            filtered[i] = alpha_ * target[i] + (1.0 - alpha_) * filtered[i];

        return filtered;
    }

    void landmarksCallback(hand_msgs::msg::HandLandmarks::SharedPtr msg)
    {
        if (msg->landmarks.size() != kLandmarkCount)
            return;

        const auto hand = parseHandId(msg->hand_id);
        if (!shouldProcessHand(hand))
            return;

        const Vec3 wrist = getPoint(*msg, 0);
        const double palm_size = norm(getPoint(*msg, 9) - wrist);
        if (palm_size < 1e-6)
            return;

        const double index = norm(getPoint(*msg, 8) - wrist) / palm_size;
        const double middle = norm(getPoint(*msg, 12) - wrist) / palm_size;
        const double ring = norm(getPoint(*msg, 16) - wrist) / palm_size;
        const double pinky = norm(getPoint(*msg, 20) - wrist) / palm_size;

        const Vec3 v1 = getPoint(*msg, 3) - getPoint(*msg, 2);
        const Vec3 v2 = getPoint(*msg, 4) - getPoint(*msg, 3);
        const double thumb_angle = std::acos(std::clamp(
            dot(v1, v2) / (norm(v1) * norm(v2) + 1e-6),
            -1.0,
            1.0));

        const double opposition_dist = norm(getPoint(*msg, 4) - getPoint(*msg, 9));

        if (calibration_enabled_)
        {
            open_close_data_.push_back({index, middle, ring, pinky, thumb_angle});
            opposition_data_.push_back(opposition_dist);
            return;
        }

        if (!calibration_loaded_)
            return;

        double opposition_denominator = opposition_max_ - opposition_min_;
        if (std::abs(opposition_denominator) < 1e-6)
            opposition_denominator = 1e-6;

        const double opposition = clamp01(
            (opposition_dist - opposition_min_) / opposition_denominator);

        const FeatureArray raw_features{index, middle, ring, pinky, thumb_angle};
        const auto fingers_norm = normalizeFeatures(raw_features);

        const double thumb_flex = fingers_norm[4];
        const double thumb_opp = 1.0 - opposition;

        AngleArray target_angles{
            fingers_norm[3] * 1000.0,
            fingers_norm[2] * 1000.0,
            fingers_norm[1] * 1000.0,
            fingers_norm[0] * 1000.0,
            thumb_flex * 1000.0,
            thumb_opp * 1000.0};

        target_angles = lowPassFilter(hand, applyDeadband(hand, target_angles));
        for (auto &angle : target_angles)
            angle = std::clamp(angle, min_angle_, max_angle_);

        auto command = hand_msgs::msg::RH56DFTPAngleCommand();
        command.header.stamp = now();
        command.hand_id = displayHandId(hand);

        for (std::size_t i = 0; i < target_angles.size(); ++i)
            command.angles[i] = static_cast<float>(target_angles[i]);

        angle_pub_->publish(command);

        ++debug_counter_;
        if (debug_counter_ % debug_every_n_ == 0)
        {
            RCLCPP_INFO(get_logger(), "----- DEBUG -----");
            RCLCPP_INFO(get_logger(), "Hand: %s", command.hand_id.c_str());
            RCLCPP_INFO(get_logger(), "Fingers norm: %s", arrayToString(fingers_norm, 2).c_str());
            RCLCPP_INFO(get_logger(), "Thumb flex: %.2f | Thumb opp: %.2f", thumb_flex, thumb_opp);
            RCLCPP_INFO(get_logger(), "Robot angles: %s", arrayToString(target_angles, 2).c_str());
            RCLCPP_INFO(get_logger(), "-----------------");
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<HandMapperNode>();

    try
    {
        node->initializeCalibration();
        rclcpp::spin(node);
    }
    catch (const std::exception &error)
    {
        RCLCPP_ERROR(node->get_logger(), "%s", error.what());
    }

    rclcpp::shutdown();
    return 0;
}
