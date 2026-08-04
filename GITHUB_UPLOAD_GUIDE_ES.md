# Guía para publicar el proyecto en GitHub

Esta guía parte de que la cuenta de GitHub ya está creada.

## 1. Revisar el contenido antes de publicarlo

La carpeta debe contener, como mínimo:

```text
certainty_ratio.py
reproduce_paper_experiments.py
example_basic.py
README.md
requirements.txt
pyproject.toml
.gitignore
tests/
data/
```

Antes de continuar:

1. Sustituye cualquier marcador de posición del README.
2. Decide la licencia del código. No se ha añadido una automáticamente porque los archivos originales no especifican ninguna.
3. No copies los datasets a GitHub si su licencia o condiciones de distribución no lo permiten.
4. Comprueba que `alldata/` y `results/` aparecen en `.gitignore`.

## 2. Probar el proyecto localmente

Abre Terminal y entra en la carpeta del repositorio:

```bash
cd /ruta/a/certainty_ratio_github
```

Crea un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

En Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Instala el proyecto y las dependencias de prueba:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Para los experimentos completos:

```bash
python -m pip install -e ".[experiments,test]"
```

Ejecuta el ejemplo:

```bash
python example_basic.py
```

Ejecuta las pruebas:

```bash
python -m pytest
```

No continúes hasta que las pruebas terminen correctamente.

## 3. Crear el repositorio vacío en GitHub

En la web de GitHub:

1. Pulsa el botón para crear un repositorio nuevo.
2. Asigna un nombre, por ejemplo `probability-mass-confusion-matrix`.
3. Añade una descripción breve.
4. Elige si será público o privado.
5. No inicialices el repositorio remoto con README, `.gitignore` o licencia, porque esos archivos ya existen localmente.
6. Crea el repositorio.

GitHub mostrará la dirección remota. Tendrá una forma similar a:

```text
https://github.com/USUARIO/NOMBRE-REPOSITORIO.git
```

## 4. Inicializar Git en la carpeta local

Desde la carpeta del proyecto:

```bash
git init
git branch -M main
```

Comprueba qué archivos se van a incluir:

```bash
git status
```

Los datasets, los resultados y el entorno `.venv` no deben aparecer como archivos preparados para publicar.

## 5. Crear el primer commit

Añade los archivos:

```bash
git add .
```

Revisa de nuevo:

```bash
git status
```

Crea el commit inicial:

```bash
git commit -m "Initial public implementation"
```

## 6. Conectar con el repositorio remoto

Sustituye la dirección del ejemplo por la que GitHub te haya proporcionado:

```bash
git remote add origin https://github.com/USUARIO/NOMBRE-REPOSITORIO.git
```

Comprueba la conexión configurada:

```bash
git remote -v
```

## 7. Subir el proyecto

```bash
git push -u origin main
```

GitHub puede solicitar autenticación mediante el navegador, una clave SSH o un token personal. La contraseña normal de la cuenta no debe insertarse como contraseña Git cuando el sistema solicita un token.

Después de completar la autenticación, recarga la página del repositorio y comprueba que aparecen todos los archivos.

## 8. Comprobaciones después de la subida

Verifica en GitHub:

1. El README se renderiza correctamente.
2. `certainty_ratio.py` contiene únicamente la implementación esencial.
3. El ejemplo es comprensible sin leer el código experimental.
4. No se han subido datasets, credenciales, rutas personales ni resultados pesados.
5. La URL de reproducibilidad que se incluya en el artículo apunta a este repositorio o a una versión archivada.

## 9. Actualizaciones posteriores

Después de modificar archivos:

```bash
git status
git add archivo_modificado.py README.md
git commit -m "Describe the change"
git push
```

Usa mensajes de commit concretos, por ejemplo:

```text
Clarify tie handling in Q decomposition
Add paper experiment reproduction script
Document isotonic calibration protocol
```

## 10. Crear una versión estable

Cuando el código esté validado:

1. Actualiza el número de versión en `pyproject.toml`.
2. Ejecuta de nuevo el ejemplo y las pruebas.
3. Crea un commit final.
4. En GitHub, crea una release con una etiqueta, por ejemplo `v1.0.0`.
5. No modifiques esa release después de citarla en el artículo.

Para máxima reproducibilidad, puede archivarse la release en un servicio que proporcione un DOI permanente. El DOI o la URL permanente debe sustituir la dirección provisional de la sección de reproducibilidad del artículo.

## 11. Clonar y verificar como lo haría un lector

Realiza una comprobación final en otra carpeta:

```bash
cd /tmp
git clone https://github.com/USUARIO/NOMBRE-REPOSITORIO.git
cd NOMBRE-REPOSITORIO
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python example_basic.py
python -m pytest
```

Esta prueba detecta archivos locales olvidados, dependencias no declaradas y rutas que solo funcionan en el equipo del autor.
