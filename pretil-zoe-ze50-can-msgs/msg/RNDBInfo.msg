# RNDB control info (410)
std_msgs/Header header                    # header
uint8 rndb_engage_mesure                  # Rapport engage mesuré par FH [1=="R" , 2=="N" , 3=="D" , 4=="B"]
uint8 rndb_engage_genere                  # Pseudo rapport engage generé par FH [1=="R" , 2=="N" , 3=="D" , 4=="B"]
uint8 rndb_changement_valide              # RNDB dernier changement valide [0=="no error" , 1=="error"]
uint8 rndb_changement_invalide_pied_frein # RNDB dernier changement non valide [0=="no error" , 1=="error"]
uint8 rndb_changement_invalide_renault    # RNDB dernier changement non valide [0=="no error" , 1=="error"]
uint8 mode_rndb_en_cours                  # mode en cours [0 "manual_mode" 1 "robotized_mode" 2 "cooperative_mode" 3 "driver_selector_mode"]
