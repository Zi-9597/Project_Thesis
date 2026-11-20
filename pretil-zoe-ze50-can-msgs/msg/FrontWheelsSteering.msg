# Options 2 state flags
std_msgs/Header header          # header
float64 vit_roue_avg              # Vitesse de la roue [0 -- 32767.5 tr/min]
float64 vit_roue_avd              # Vitesse de la roue [0 -- 32767.5 tr/min]
float64 pedale_accel_enfoncement  # Position mesurée de la pedale [0 -- 6553.5 ]
uint8 pedale_frein_enfoncement  # Position mesurée du frein [0 -- 255]
uint8 counter_id                # Counter ID [0...15]
