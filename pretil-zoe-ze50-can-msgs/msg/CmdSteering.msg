# Steering Command Control
std_msgs/Header header      # header
float32 angle # consigne de position de la direction [-720 deg -- 720 deg]
float32 rotation_speed # consigne de vitesse de rotation de la direction deg/s [ ??? -- ??? ]
uint8 switch_accel   #Reactivation de la DAE rapide en cas de desactivation [0==pas de demande de reactivation DAE en cooperatif , 1==demande de reactivation DAE en cooperatif]
bool can_publish # true or false
