#Ejercicio 1

# Función para convertir Celsius a Fahrenheit
def to_fahrenheit(temp):
    return (temp * 9/5) + 32

celsius = [0, 10, 20, 30, 40]

fahrenheit = list(map(to_fahrenheit, celsius))
print(fahrenheit)

"""
Ejercicio crear un administrador de configuración de usuarios hecho por mi.
"""
def add_setting(diccionario, tupla):
    tupla_mi = tuple([item.lower() for item in tupla])
    if tupla_mi[0] in diccionario:
        return f"Setting '{tupla_mi[0]}' already exists! Cannot add a new setting with this name."
    else:
        diccionario[tupla_mi[0]] = tupla_mi[1]
        return f"Setting '{tupla_mi[0]}' added with value '{tupla_mi[1]}' successfully!"


def update_setting(diccionario, tupla):
    tupla_mi = tuple([item.lower() for item in tupla])
    if tupla_mi[0] in diccionario:
        diccionario.update({tupla_mi})
        return f"Setting '{tupla_mi[0]}' updated to '{tupla_mi[1]}' successfully!"
    else:
        return f"Setting '{tupla_mi[0]}' does not exist! Cannot update a non-existing setting."


def delete_setting(diccionario, clave):
    clave_mi = clave.lower()
    if clave_mi in diccionario:
        diccionario.pop(clave_mi)
        return f"Setting '{clave_mi}' deleted successfully!"
    else:
        return "Setting not found!"


def view_settings(diccionario):
    if diccionario == {}:
        return "No settings available."
    else:
        inicio = "Current User Settings:\n"
        llave = diccionario.keys()
        valor = diccionario.values()
        for l, v in zip(llave, valor):
            inicio += l.capitalize() + ": " + v + "\n"

        return inicio


test_settings = {"TEMA": "NATURAL", "IDIOMA": "ESPAÑOL", "NOTIFICACIONES": "VACIAS"}


"""
Ejercicio crear un administrador de configuración de usuarios hecho por gemini.
"""

def add_setting(settings, pair):
    key, value = pair[0].lower(), pair[1].lower()
    if key in settings:
        return f"Setting '{key}' already exists! Cannot add a new setting with this name."
    settings[key] = value
    return f"Setting '{key}' added with value '{value}' successfully!"


def update_setting(settings, pair):
    key, value = pair[0].lower(), pair[1].lower()
    if key not in settings:
        return f"Setting '{key}' does not exist! Cannot update a non-existing setting."
    settings[key] = value
    return f"Setting '{key}' updated to '{value}' successfully!"


def delete_setting(settings, key):
    key = key.lower()
    if key in settings:
        settings.pop(key)
        return f"Setting '{key}' deleted successfully!"
    return "Setting not found!"


def view_settings(settings):
    if not settings:
        return "No settings available."

    result = "Current User Settings:"
    for key, value in settings.items():
        # Agregamos el salto de línea y formateamos la clave
        result += f"\n{key.capitalize()}: {value}"
    return result


# Para probarlo:
test_settings = {"TEMA": "NATURAL", "IDIOMA": "ESPAÑOL", "NOTIFICACIONES": "VACIAS"}

