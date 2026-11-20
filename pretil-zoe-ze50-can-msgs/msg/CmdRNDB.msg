# RNDB Command Control
std_msgs/Header header      # header
uint8 rndb           # consigne du rapport a engager [1==R , 2==N , 3==D , 4==B]
uint8 rndb_retry     # RNDB permet de redemander le changement invalide precedent [0=="pas de redemande Changement invalide precedent" , 1=="redemande Changement invalide precedent"]
bool can_publish # true or false
