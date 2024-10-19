# Tecnicatura Universitaria en Inteligencia Artificial {align=center}
## FCEIA - UNR {align=center}
## Procesamiento de imágenes 1 {align=center}

**Integrantes:**
- Alsop, Agustín. Legajo: A-4651/7
- Asad, Gonzalo. Legajo: A-4595/1
- Castells, Sergio. Legajo: C-7334/2
- Hachen, Rocío. Legajo: H-1184/3

---

## Preparación del Entorno

### Linux
Para inicializar el cliente, debe tener dentro de una misma carpeta en su PC los tres archivos que componen al cliente:
main.py
client.py
requirements.txt
Luego, debe abrir una ventana de comandos (CMD) e ingresar los siguientes comandos:
cd Ruta de la carpeta que contiene los archivos del cliente
python -m venv ./.venv
..venv\Scripts\activate
pip install -r requirements.txt
python main.py
Al paso 4 solo será necesario realizarlo únicamente la primera vez. Para poder realizar cualquier acción sobre la base de datos (Navegar la Aplicación), necesitará en primer lugar establecer la IP donde está alojado el servidor API. Siga los pasos en pantalla para configurar la IP del servidor. Habiendo ingresado de forma correcta la IP, se le solicitará que ingrese sus credenciales. Si el inicio de sesión es exitoso, será llevado al menú principal donde verá las siguientes opciones:
Navegar la Aplicación
Salir
Si ingresa el número 1 con el teclado y presiona “enter”, será llevado a un nuevo menú que le permitirá interactuar con la base de datos de los 100 mejores libros de la historia. Podrá encontrar las opciones:
Ver Catálogo de Libros
Buscar un Libro
Descatalogar un Libro
Agregar Nuevo Libro
Modificar Datos de un Libro
Reestablecer el Catálogo
Volver al Menú Anterior
Para seleccionar la forma en que desea interactuar con la base de datos, debe ingresar con el teclado el número de la opción seguido de la tecla “enter”. Le será indicado por pantalla cómo continuar.
¡Disfrute!

## Ejecución

### Linux
```bash
source .venv/bin/activate

python3 main.py
```

### Windows
```bash
.\.venv\Scripts\activate

python main.py
```