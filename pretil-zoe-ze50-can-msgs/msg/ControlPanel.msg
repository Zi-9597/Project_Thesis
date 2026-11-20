# Control Panel state flags
std_msgs/Header header      # header
uint8 selector_mode # Pupitre : etat du rotacteur [0=="mode_manuel" , 1=="mode_robotise" , 2=="mode_cooperatif" , 3=="mode_Selecteur_conducteur"]
uint8 switch_dae    # Pupitre : etat de l interrupteur DAE [0=="off" , 1=="on"]
uint8 switch_accel  # Pupitre : etat de l interrupteur Accel [0=="off" , 1=="on"]
uint8 switch_brake  # Pupitre : etat de l interrupteur Brake [0=="off" , 1=="on"]
uint8 switch_rndb   # Pupitre : etat de l interrupteur RNDB [0=="off" , 1=="on"]
uint8 led_accel     # Pupitre : etat de LED Accel [0=="off" , 1=="on"]
uint8 led_brake     # Pupitre : etat de LED Brake [0=="off" , 1=="on"]
uint8 led_dae       # Pupitre : etat de LED DAE [0=="off" , 1=="on"]
uint8 led_rndb      # Pupitre : etat de LED RNDB [0=="off" , 1=="on"]
uint8 led_default   # Pupitre : etat de LED Default [0=="off" , 1=="on"]
uint8 checksum_id   # Checksum ID [0...15 CRC]
uint8 counter_id    # Counter ID [0...15]
