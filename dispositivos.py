import sounddevice as sd

# O comando retorna uma lista de dicionários com os detalhes de cada dispositivo.
dispositivos = sd.query_devices()

print(dispositivos)

print("--- Dispositivos de Áudio Disponíveis ---")
for i, dispositivo in enumerate(dispositivos):
    print(f"[{i}] Nome: {dispositivo['name']}")
    print(f"    Entrada: {dispositivo['max_input_channels']} canais")
    print(f"    Saída: {dispositivo['max_output_channels']} canais")
    print("-" * 20)

# Depois de ver a lista, você pode selecionar um dispositivo pelo seu índice:
# sd.default.device = 4 
# (assumindo que 4 é o índice do microfone USB desejado)