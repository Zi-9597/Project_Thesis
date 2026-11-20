# Options 1 state flags
std_msgs/Header header          # header
float64 vit_roue_arg              # Vitesse de la roue [0 -- 32767.5 tr/min]
float64 vit_roue_ard              # Vitesse de la roue [0 -- 32767.5 tr/min]
float64 angle_volant              # Position mesurée du volant [-3276.8 -- 3276.8 ]
uint8 sens_roue_ard             # Sens de roulage du vehicule  [0=="arret" , 1=="marche avant" , 2=="marche arriere"]
uint8 sens_roue_arg             # Sens de roulage du vehicule [0=="arret" , 1=="marche avant" , 2=="marche arriere"]
uint8 counter_id                # Counter ID [0...15]
