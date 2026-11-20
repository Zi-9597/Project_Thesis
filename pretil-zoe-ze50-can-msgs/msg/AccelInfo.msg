# Accel control info (210)
std_msgs/Header header                # header
float64 position_accel_piste_1_mesure   # Position accel (Ped1) mesurée par FH [0 % -- 100 %]
float64 position_accel_piste_2_mesure   # Position accel (Ped2) mesurée par FH [0 % -- 100 %]
float64 position_accel_piste_1_genere   # Pseudo position accel (Ped1) generée par FH [0 % -- 100 %]
uint8 mode_accel_en_cours             # mode en cours [0 "manual_mode" 1 "robotized_mode" 2 "cooperative_mode" 3 "driver_selector_mode"]
