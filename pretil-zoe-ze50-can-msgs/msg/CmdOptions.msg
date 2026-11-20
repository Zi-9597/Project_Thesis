# Options Command Control
std_msgs/Header header      # header
uint8 blinker_l         # consigne d activation du clignotant gauche [0=="off" , 1=="on"]
uint8 blinker_r         # consigne d activation du clignotant droit [0=="off" , 1=="on"]
uint8 warning           # consigne d activation des warning (a confirmer si realisable sur ZE50 !!!!) [0=="off" , 1=="on"]
uint8 honk              # consigne d activation du klaxon [0=="off" , 1=="on"]
uint8 checksum_id       # Checksum ID [0...15 CRC]
uint8 counter_id        # Counter ID [0...15]
