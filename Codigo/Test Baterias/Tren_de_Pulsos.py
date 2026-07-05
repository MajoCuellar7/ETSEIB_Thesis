capacity = 27.6
current = 10
current_2 = 3
current_3 = 1

Porcentaje_5_Ah = capacity * 0.05
time_5p = 60*(Porcentaje_5_Ah / current)

time_3A = 60*(Porcentaje_5_Ah / current_2)

Porcentaje = (15/60) * (current_2) * (100/capacity)



print(f"tiempo para 5%: {time_5p} minutos")
print(f"tiempo durante 3 A: {time_3A} minutos")
print(f"Porcentaje de descargado en 3 A por 15 min: {Porcentaje}%")

# 18 pulsos de 10 A por 8 minutos
# 1 pulso de 3 A 