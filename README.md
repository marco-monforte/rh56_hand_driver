Quick bringup

```bash
ros2 launch rh56_hand_driver rh56_hand_driver.launch.py \
  controlled_hand:=left
```

Notes

- The current driver uses direct Modbus TCP for command writes and feedback reads.
- `network_interface` is kept only for backward compatibility and is ignored by the driver.
- `controlled_hand` selects which configured Modbus endpoint is used: `left`, `right`, or `both`.
- Startup behavior matches the working Python reference more closely:
  - optional error reset
  - optional full open command
  - continuous direct Modbus polling for feedback
- Touch matrices are heavier than joint/status registers, so tactile data is read every `touch_poll_divider` polling cycles by default. Set `touch_poll_divider:=0` to disable touch polling and prioritize control responsiveness.

Useful launch overrides

```bash
ros2 launch rh56_hand_driver rh56_hand_driver.launch.py \
  controlled_hand:=left \
  left_hand_ip:=192.168.123.210 \
  right_hand_ip:=192.168.123.211 \
  reset_errors_on_startup:=true \
  open_hand_on_startup:=true \
  polling_hz:=200.0 \
  command_rate_hz:=200.0 \
  feedback_rate_hz:=100.0 \
  touch_poll_divider:=50
```
