# Steering control info (110)
std_msgs/Header header                # header
float64 position_dae                    # Position mesurée du volant [-720 deg -- 720 deg]
float64 ecart_pos_dae_cons_pos         # ecart de position du volant entre la consigne et sa position [-720 deg -- 720 deg]
float64 couple_volant_mesure_dae        # Couple mesuré sur le volant exercé par le conducteur [-200 % -- 200 %]
float64 couple_volant_genere_dae        # Pseudo couple sur volant generé par FH pour robotiser la DAE [-200 % -- 200 %]
uint8 mode_dae_en_cours               # mode en cours [0 "manual_mode" 1 "robotized_mode" 2 "cooperative_mode" 3 "driver_selector_mode"]
uint8 dae_robotisable_plage_ok        # La consigne CAN d'angle volant est proche de la mesure de l angle volant [0=="no error" , 1=="error"]
