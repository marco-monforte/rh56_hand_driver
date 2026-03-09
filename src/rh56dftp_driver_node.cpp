#include <rclcpp/rclcpp.hpp>

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "inspire_hand_ctrl.hpp"
#include "inspire_hand_state.hpp"
#include "inspire_hand_touch.hpp"

#include "hand_msgs/msg/rh56_dftp_angle_command.hpp"
#include "hand_msgs/msg/rh56_dftp_feedback.hpp"

#include <thread>
#include <mutex>
#include <vector>
#include <chrono>
#include <string>

using namespace std::chrono_literals;

class RH56Node : public rclcpp::Node
{
public:
    RH56Node() : Node("rh56dftp_cpp_node")
    {
        // Parameters
        declare_parameter("current_limit_ma", 1500);
        declare_parameter("contact_current_ma", 650);
        declare_parameter("soft_grasp_enabled", true);

        current_limit = get_parameter("current_limit_ma").as_int();
        contact_current = get_parameter("contact_current_ma").as_int();
        soft_grasp = get_parameter("soft_grasp_enabled").as_bool();

        unlock_current = 500;
        unlock_time = 1.0;

        finger_names = {"pinky","ring","middle","index","thumb_bend","thumb_rotate"};

        desired_angles.resize(6,1000);
        safe_angles.resize(6,1000);

        finger_locked.resize(6,false);
        contact_locked.resize(6,false);

        unlock_timer.resize(6,0.0);

        // ROS interfaces
        cmd_sub = create_subscription<hand_msgs::msg::RH56DFTPAngleCommand>(
            "/rh56dftp/angle_command", 10,
            std::bind(&RH56Node::cmd_callback,this,std::placeholders::_1)
        );

        feedback_pub = create_publisher<hand_msgs::msg::RH56DFTPFeedback>(
            "/rh56dftp/feedback",10
        );

        timer = create_wall_timer(
            20ms,
            std::bind(&RH56Node::publish_feedback,this)
        );

        // DDS init
        unitree::robot::ChannelFactory::Instance()->Init(0,"");

        handcmd = std::make_shared<
            unitree::robot::ChannelPublisher<inspire::inspire_hand_ctrl>
        >("rt/inspire_hand/ctrl/r");

        handstate = std::make_shared<
            unitree::robot::ChannelSubscriber<inspire::inspire_hand_state>
        >("rt/inspire_hand/state/r");

        handtouch = std::make_shared<
            unitree::robot::ChannelSubscriber<inspire::inspire_hand_touch>
        >("rt/inspire_hand/touch/r");

        handcmd->InitChannel();

        handstate->InitChannel([this](const void* message)
        {
            std::lock_guard<std::mutex> lock(mtx);
            state = *(inspire::inspire_hand_state*)message;
            check_current();
        });

        handtouch->InitChannel([this](const void* message)
        {
            std::lock_guard<std::mutex> lock(mtx);
            touch = *(inspire::inspire_hand_touch*)message;
        });

        running = true;
        control_thread = std::thread(&RH56Node::control_loop,this);

        RCLCPP_INFO(get_logger(),"RH56 C++ node started");
    }

    ~RH56Node()
    {
        running = false;
        if(control_thread.joinable())
            control_thread.join();
    }

private:
    // Parameters
    int current_limit;
    int contact_current;
    bool soft_grasp;
    int unlock_current;
    double unlock_time;

    // State
    std::vector<int> desired_angles;
    std::vector<int> safe_angles;
    std::vector<bool> finger_locked;
    std::vector<bool> contact_locked;
    std::vector<double> unlock_timer;
    std::vector<std::string> finger_names;

    inspire::inspire_hand_state state;
    inspire::inspire_hand_touch touch;
    std::mutex mtx;

    // DDS
    unitree::robot::ChannelPublisherPtr<inspire::inspire_hand_ctrl> handcmd;
    unitree::robot::ChannelSubscriberPtr<inspire::inspire_hand_state> handstate;
    unitree::robot::ChannelSubscriberPtr<inspire::inspire_hand_touch> handtouch;

    // ROS
    rclcpp::Subscription<hand_msgs::msg::RH56DFTPAngleCommand>::SharedPtr cmd_sub;
    rclcpp::Publisher<hand_msgs::msg::RH56DFTPFeedback>::SharedPtr feedback_pub;
    rclcpp::TimerBase::SharedPtr timer;

    // Threads
    std::thread control_thread;
    bool running;

    // ------------------------
    void cmd_callback(hand_msgs::msg::RH56DFTPAngleCommand::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        for(int i=0;i<6;i++)
        {
            if(!finger_locked[i])
                desired_angles[i] = msg->angles[i];

            if(msg->angles[i] > safe_angles[i])
                contact_locked[i] = false;
        }
    }

    void check_current()
    {
        auto currents = state.current();
        auto angles = state.angle_act();
        double now = this->now().seconds();

        for(int i=0;i<6;i++)
        {
            int current = currents[i];

            // Overcurrent
            if(current > current_limit && !finger_locked[i])
            {
                finger_locked[i]=true;
                safe_angles[i]=angles[i];
                RCLCPP_WARN(get_logger(),"OVERCURRENT %s %d mA",
                            finger_names[i].c_str(), current);
            }

            // Contact
            if(soft_grasp &&
               current > contact_current &&
               !contact_locked[i] &&
               !finger_locked[i])
            {
                contact_locked[i]=true;
                safe_angles[i]=angles[i];
                RCLCPP_INFO(get_logger(),"CONTACT %s %d mA",
                            finger_names[i].c_str(), current);
            }

            // Unlock
            if(finger_locked[i])
            {
                if(current < unlock_current)
                {
                    if(unlock_timer[i]==0)
                        unlock_timer[i]=now;

                    if(now-unlock_timer[i] > unlock_time)
                    {
                        finger_locked[i]=false;
                        unlock_timer[i]=0;
                        RCLCPP_INFO(get_logger(),"UNLOCK %s",
                                    finger_names[i].c_str());
                    }
                }
                else
                    unlock_timer[i]=0;
            }
        }
    }

    void control_loop()
    {
        rclcpp::Rate rate(200);
        while(running)
        {
            inspire::inspire_hand_ctrl cmd;
            cmd.angle_set().resize(6);
            cmd.mode(0b0001); // angle control mode

            {
                std::lock_guard<std::mutex> lock(mtx);
                for(int i=0;i<6;i++)
                {
                    if(!finger_locked[i] && !contact_locked[i])
                        safe_angles[i] = desired_angles[i];

                    cmd.angle_set()[i] = safe_angles[i];
                }
            }

            handcmd->Write(cmd);
            rate.sleep();
        }
    }

    void publish_feedback()
    {
        auto msg = hand_msgs::msg::RH56DFTPFeedback();
        std::lock_guard<std::mutex> lock(mtx);

        auto current = state.current();
        auto angle = state.angle_act();
        auto force = state.force_act();

        for(int i=0;i<6;i++)
        {
            msg.current[i] = current[i];
            msg.angle[i] = angle[i];
            msg.force[i] = force[i];
        }

        feedback_pub->publish(msg);
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