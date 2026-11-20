# Accel control info (310)
std_msgs/Header header                 # header
uint16 position_frein_piste_1_mesure    # Position frein (Ped1) mesurée par FH [0 SENT -- 4095 SENT]
uint16 position_frein_piste_2_mesure    # Position frein (Ped2) mesurée par FH [ 0 micros -- 4095 micros]
uint16 position_frein_piste_1_genere    # Pseudo position frein (Ped1) generée par FH [0 SENT -- 4095 SENT]
float64 pression_frein_mesure            # [0 bar -- 204.7 bar]
uint8 mode_frein_en_cours              # mode en cours [0 "manual_mode" 1 "robotized_mode" 2 "cooperative_mode" 3 "driver_selector_mode"]
