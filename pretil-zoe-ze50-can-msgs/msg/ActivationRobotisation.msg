# Robotisation activation flags
std_msgs/Header header      # header
uint8 mode    # Desired auto mode [0...3] (0 "manual_mode" 1 "robotized_mode" 2 "cooperative_mode" 3 "driver_selector_mode")
uint8 steering     # Activation of the EPS (Electronic Power Steering) command through the robotized CAN [0,1]
uint8 acceleration # Activation of the acceleration command through the robotized CAN [0,1]
uint8 brake        # Activation of the brake command through the robotized CAN [0,1]
uint8 prnd         # Activation of the PRND (Park, Reverse, Neutral, Drive) command through the robotized CAN [0,1]
uint8 blinker      # Activation of the blinker command through the robotized CAN [0,1]
uint8 checksum_id  # Checksum ID [0...15 CRC]
uint8 counter_id   # Counter ID [0...15]

