# Guía de Uso - Quimática Orgánica

## Introducción

Esta guía te ayudará a aprovechar al máximo todas las funcionalidades de Quimática - Orgánica, desde la visualización básica de moléculas hasta la combinación avanzada usando inteligencia artificial.

## Primeros Pasos

### 1. Acceso a la Aplicación

1. Abre tu navegador web (Chrome, Firefox, Safari, Edge)
2. Navega a `http://localhost:5000` (si ejecutas localmente)
3. La aplicación se cargará automáticamente

### 2. Interfaz Principal

La aplicación está dividida en cuatro secciones principales:

- **Banco de Moléculas**: Moléculas predefinidas disponibles
- **Zona de Combinación**: Área para combinar moléculas con IA
- **Visor 2D**: Visualización bidimensional con SVG
- **Visor 3D**: Visualización tridimensional interactiva

---

## Funcionalidades Básicas

### Visualización de Moléculas

#### Paso 1: Seleccionar una Molécula
1. En el **Banco de Moléculas**, verás tarjetas con íconos animados
2. Haz clic en cualquier tarjeta (Estradiol o Fulvestrant)
3. La molécula se cargará automáticamente en ambos visores

#### Paso 2: Explorar la Vista 2D
1. **Pan (Desplazamiento)**: Mantén clic y arrastra para mover la vista
2. **Zoom**: Usa la rueda del mouse para acercar/alejar
3. **Controles de Índices**:
   - ☑️ **Índices de Átomos**: Muestra números en cada átomo
   - ☑️ **Índices de Enlaces**: Muestra números en cada enlace
4. **Reiniciar Vista**: Usa los controles del visor para centrar

#### Paso 3: Explorar la Vista 3D
1. **Rotación**: Mantén clic y arrastra para rotar la molécula
2. **Zoom**: Rueda del mouse o pinch en dispositivos táctiles
3. **Estilos de Visualización**:
   - 🔘 **Palos**: Representación clásica con palos
   - 🔘 **Alambre**: Solo las conexiones entre átomos
   - 🔘 **Esfera**: Esferas de van der Waals
   - 🔘 **Bolas y Palos**: Combinación de esferas y palos

4. **Opciones Adicionales**:
   - ☑️ **Mostrar Hidrógenos**: Incluir átomos de hidrógeno
   - ☑️ **Índices de Átomos**: Etiquetas numéricas en 3D
   - ☑️ **Índices de Enlaces**: Numeración de enlaces

5. **Reiniciar Vista**: Botón para volver a la posición inicial

### Ejemplo Práctico: Explorando Estradiol

```
1. Clic en la tarjeta "Estradiol"
2. Observa la estructura 2D: nota los anillos aromáticos y grupos OH
3. En 3D, cambia a estilo "Esfera" para ver el volumen molecular
4. Activa "Índices de Átomos" para identificar posiciones específicas
5. Rota la molécula para ver la conformación tridimensional
```

---

## Funcionalidades Avanzadas

### Combinación de Moléculas con IA

#### Paso 1: Preparar las Moléculas
1. En la **Zona de Combinación**, verás dos slots vacíos
2. Arrastra una molécula del banco al **Slot 1**
3. Arrastra otra molécula al **Slot 2**
4. Los slots mostrarán las moléculas seleccionadas

#### Paso 2: Configurar el Modelo de IA
1. En el selector desplegable, elige un modelo:
   - **Gemini 1.5 Flash**: Rápido, ideal para pruebas
   - **Gemini 1.5 Pro**: Más potente, mejores resultados
   - **Gemini 2.5 Flash Lite**: Ultrarrápido, para sugerencias
   - **Gemini 2.5 Pro**: Última generación, máxima calidad

#### Paso 3: Ejecutar la Combinación
1. El botón cambiará a **"Combinar Moléculas"**
2. Haz clic para iniciar el proceso
3. La IA analizará las estructuras y generará una nueva molécula
4. El resultado aparecerá en una nueva sección

#### Paso 4: Analizar el Resultado
1. Se mostrará la nueva estructura en 2D y 3D
2. Podrás ver el SMILES generado
3. Compara con las moléculas originales

### Ejemplo Práctico: Combinando Estradiol y Fulvestrant

```
1. Arrastra "Estradiol" al Slot 1
2. Arrastra "Fulvestrant" al Slot 2
3. Selecciona "Gemini 1.5 Pro" para mejor calidad
4. Clic en "Combinar Moléculas"
5. Espera mientras la IA procesa (30-60 segundos)
6. Analiza el resultado: ¿qué características conservó de cada molécula?
```

---

### Sugerencias Inteligentes con Menú Contextual

#### Activación del Menú
1. En el **Visor 2D**, haz clic en cualquier átomo o enlace
2. Aparecerá un menú contextual inteligente
3. La IA generará sugerencias basadas en el contexto

#### Tipos de Sugerencias
- **Modificaciones de Grupos Funcionales**: Cambiar OH por NH2, etc.
- **Adición de Grupos**: Añadir metilos, fenilos, etc.
- **Transformaciones Químicas**: Oxidaciones, reducciones, etc.
- **Cambios Estructurales**: Ciclizaciones, aperturas de anillo, etc.

#### Aplicar Sugerencias
1. Revisa las opciones generadas automáticamente
2. Haz clic en cualquier sugerencia para aplicarla
3. O usa el campo de texto para instrucciones personalizadas
4. La nueva estructura se renderizará automáticamente

### Ejemplo Práctico: Modificando Estradiol

```
1. Carga "Estradiol" en los visores
2. En el visor 2D, haz clic en el átomo de oxígeno del grupo OH
3. Revisa las sugerencias: "Convertir a NH2", "Añadir grupo metilo", etc.
4. Selecciona "Convertir OH en NH2"
5. Observa cómo cambia la estructura molecular
```

---

### Guardado y Gestión de Compuestos

#### Descripción General
La aplicación permite guardar compuestos generados mediante combinaciones de IA para uso posterior. Los compuestos se almacenan en Firebase Firestore y pueden ser recuperados en cualquier momento.

#### Guardando un Compuesto

##### Paso 1: Generar un Compuesto
1. Realiza una **Combinación de Moléculas** exitosa (ver sección anterior)
2. Espera a que aparezca el análisis completo del compuesto
3. Verifica que el resultado sea satisfactorio

##### Paso 2: Guardar el Compuesto
1. Al final del análisis, aparecerá un botón **"Guardar Compuesto"**
2. Haz clic en el botón
3. El sistema mostrará "Guardando..." mientras procesa
4. Una vez guardado, aparecerá la confirmación "Compuesto guardado con éxito!"
5. El botón cambiará a "Guardado" para indicar el estado

##### ¿Qué se Guarda?
- **Nombre**: Generado automáticamente por la IA
- **SMILES**: Notación química de la molécula
- **Análisis completo**: Propiedades farmacológicas y características
- **Fecha de creación**: Timestamp automático

#### Cargando Compuestos Guardados

##### Visualizar Compuestos Disponibles
1. En la sección **"Compuestos Guardados"** (parte inferior de la página)
2. Verás tarjetas verdes con los nombres de tus compuestos
3. Cada tarjeta tiene un botón **"Cargar"** y un ícono de **papelera** (visible al pasar el mouse)
4. Si no tienes compuestos guardados, aparecerá un estado vacío con ícono y mensaje informativo

##### Cargar un Compuesto Específico
1. Busca el compuesto deseado en las tarjetas verdes
2. Haz clic en el botón **"Cargar"** de la tarjeta
3. El compuesto se cargará automáticamente en ambos visores (2D y 3D)
4. Aparecerá el análisis guardado en la sección de resultados

#### Eliminando Compuestos Guardados

##### Proceso de Eliminación Segura
1. **Identificar el compuesto**: Localiza la tarjeta del compuesto que deseas eliminar
2. **Activar eliminación**: Pasa el mouse sobre la tarjeta para ver el ícono de papelera 🗑️
3. **Iniciar eliminación**: Haz clic en el ícono de papelera (esquina superior derecha)
4. **Confirmar acción**: Se abrirá un popup de confirmación con el nombre del compuesto
5. **Ejecutar eliminación**: Haz clic en **"Eliminar"** para confirmar, o **"Cancelar"** para abortar

##### Características de Seguridad
- **Confirmación requerida**: Siempre aparece un popup antes de eliminar
- **Nombre visible**: El popup muestra exactamente qué compuesto se eliminará
- **Advertencia clara**: "Esta acción no se puede deshacer"
- **Estados de loading**: El botón muestra "Eliminando..." durante el proceso
- **Manejo de errores**: Si falla, se muestra un mensaje de error y se puede reintentar

##### Popup de Confirmación
```
┌─────────────────────────────────────┐
│  ⚠️   Confirmar Eliminación         │
├─────────────────────────────────────┤
│ ¿Estás seguro de que deseas         │
│ eliminar este compuesto?            │
│                                     │
│ "Nombre del Compuesto Específico"  │
│                                     │
│ Esta acción no se puede deshacer.   │
├─────────────────────────────────────┤
│    [Cancelar]    [🗑️ Eliminar]     │
└─────────────────────────────────────┘
```

### Ejemplo Práctico: Flujo Completo de Guardado

```
1. Combina "Estradiol" + "Fulvestrant" usando Gemini 1.5 Pro
2. Espera el análisis completo (2-3 minutos)
3. Revisa el nombre generado, ej: "17β-Estradiol-7α-pentafluorobutilsulfinildecanoato"
4. Haz clic en "Guardar Compuesto"
5. Confirma que aparezca "Guardado"
6. Desplázate hacia abajo para ver tu compuesto en "Compuestos Guardados"
7. Prueba cargar el compuesto haciendo clic en "Cargar"
```

### Ejemplo Práctico: Flujo Completo de Eliminación

```
1. Ve a la sección "Compuestos Guardados"
2. Localiza el compuesto que deseas eliminar
3. Pasa el mouse sobre la tarjeta verde
4. Observa cómo aparece el ícono de papelera 🗑️ en la esquina superior derecha
5. Haz clic en el ícono de papelera
6. Lee el popup de confirmación que aparece
7. Verifica que el nombre del compuesto sea correcto
8. Haz clic en "Eliminar" para confirmar (o "Cancelar" para abortar)
9. Observa el estado "Eliminando..." en el botón
10. Confirma que el compuesto desaparece de la lista
11. Si no hay más compuestos, verás el estado vacío con ícono informativo
```

---

### Exportación de Modelos 3D

#### Formatos Disponibles
- **STL**: Para impresión 3D
- **OBJ**: Para software de modelado 3D
- **FBX**: Para animación y juegos
- **3MF**: Formato Microsoft 3D
- **AMF**: Formato de manufactura aditiva

#### Proceso de Descarga
1. Asegúrate de tener una molécula cargada en 3D
2. Selecciona el formato deseado en el menú desplegable
3. Haz clic en **"Descargar Modelo 3D"**
4. El archivo se descargará automáticamente

### Ejemplo Práctico: Preparar para Impresión 3D

```
1. Carga la molécula que deseas imprimir
2. Ajusta el estilo 3D a "Bolas y Palos" para mejor definición
3. Selecciona formato "STL"
4. Descarga el archivo
5. Importa en tu software de impresión 3D (Cura, PrusaSlicer, etc.)
```

---

## Trabajando con SMILES Personalizados

### Renderizado Manual

#### Usando la API REST
```bash
# Ejemplo: Renderizar cafeína
curl -X POST http://localhost:5000/api/render_smiles \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "show_atoms": true,
    "show_bonds": true
  }'
```

#### Integración en JavaScript
```javascript
// Función para renderizar SMILES personalizado
async function renderCustomMolecule(smiles) {
    const response = await fetch('/api/render_smiles', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            smiles: smiles,
            show_atoms: true,
            show_bonds: true
        })
    });
    
    const data = await response.json();
    if (data.error) {
        console.error('Error:', data.error);
        return;
    }
    
    // Actualizar visores con nueva molécula
    updateViewers(data);
}

// Uso
renderCustomMolecule('CCO'); // Etanol
```

### Moléculas de Ejemplo

#### Moléculas Simples
```
Metano: C
Etanol: CCO
Acetona: CC(=O)C
Benceno: c1ccccc1
Tolueno: Cc1ccccc1
```

#### Moléculas Complejas
```
Cafeína: CN1C=NC2=C1C(=O)N(C(=O)N2C)C
Aspirina: CC(=O)OC1=CC=CC=C1C(=O)O
Penicilina G: CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C
```

---

## Casos de Uso Educativos

### 1. Enseñanza de Química Orgánica

#### Comparación de Isómeros
```
1. Renderiza diferentes isómeros del mismo compuesto
2. Compara sus estructuras 2D y 3D
3. Analiza diferencias en conformación
4. Usa sugerencias IA para interconversiones
```

#### Ejemplo: Isómeros de Butano
```
n-Butano: CCCC
Isobutano: CC(C)C

1. Renderiza ambos SMILES
2. Compara en vista 3D
3. Observa diferencias en ramificación
```

### 2. Análisis de Fármacos

#### Estudio de Relación Estructura-Actividad
```
1. Carga un fármaco conocido (ej: Estradiol)
2. Usa sugerencias IA para modificaciones
3. Analiza cómo los cambios afectan la estructura
4. Exporta modelos para análisis detallado
```

### 3. Diseño de Moléculas

#### Prototipado Rápido
```
1. Comienza con una estructura base
2. Usa combinaciones IA para generar variantes
3. Aplica modificaciones específicas con el menú contextual
4. Exporta candidatos prometedores
```

---

## Solución de Problemas Comunes

### Problema: "Molécula no encontrada"
**Causa**: Nombre incorrecto en la URL o base de datos
**Solución**:
```
1. Verifica que el nombre sea exacto: 'estradiol' o 'fulvestrant'
2. Usa la API: GET /api/molecule/estradiol
3. Para moléculas personalizadas, usa POST /api/render_smiles
```

### Problema: "SMILES no válido"
**Causa**: Notación SMILES incorrecta o demasiado compleja
**Solución**:
```
1. Verifica la sintaxis SMILES en herramientas como ChemSketch
2. Asegúrate de que la longitud sea menor a 500 caracteres
3. Verifica que no haya espacios en el SMILES
4. Ejemplos válidos:
   - Metano: C
   - Etanol: CCO
   - Benceno: c1ccccc1
5. Evita caracteres especiales no químicos
6. Verifica que la molécula tenga menos de 150 átomos
```

### Problema: Visor 3D no carga
**Causa**: Problemas con librerías JavaScript
**Solución**:
```
1. Refresca la página (F5)
2. Verifica conexión a internet
3. Comprueba la consola del navegador (F12)
4. Prueba en modo incógnito
```

### Problema: IA no responde
**Causa**: Límites de API o conectividad
**Solución**:
```
1. Verifica conexión a internet
2. Intenta con un modelo más rápido (Gemini 2.5 Flash Lite)
3. Reduce la complejidad de la consulta
4. Espera unos minutos antes de reintentar
```

### Problema: Descarga 3D falla
**Causa**: Modelo no inicializado o formato no soportado
**Solución**:
```
1. Asegúrate de que la molécula esté cargada en 3D
2. Prueba con formato STL primero
3. Verifica que el navegador permita descargas
```

### Problema: Error al guardar compuesto
**Causa**: Problemas de conectividad o límites de Firebase
**Solución**:
```
1. Verifica tu conexión a internet
2. Refresca la página e intenta nuevamente
3. Verifica que el análisis del compuesto esté completo
4. Si persiste, usa modo incógnito del navegador
```

### Problema: Compuestos guardados no aparecen
**Causa**: Problemas de carga desde Firebase o cache del navegador
**Solución**:
```
1. Refresca la página (F5)
2. Verifica conexión a internet
3. Borra cache del navegador para el sitio
4. Intenta en modo incógnito
```

### Problema: SMILES demasiado complejo
**Causa**: Molécula generada excede límites del sistema
**Solución**:
```
1. Usa moléculas más simples para la combinación
2. Cambia a un modelo de IA diferente (ej: Gemini 2.5 Flash Lite)
3. Intenta la combinación nuevamente
4. Verifica que el SMILES no contenga espacios o caracteres inválidos
```

### Problema: Error al eliminar compuesto
**Causa**: Problemas de conectividad o permisos de Firebase
**Solución**:
```
1. Verifica tu conexión a internet
2. Refresca la página e intenta nuevamente
3. Asegúrate de que el compuesto existe (no fue eliminado por otra sesión)
4. Si persiste, cierra y abre el navegador
5. En caso extremo, usa modo incógnito
```

### Problema: Botón de eliminar no aparece
**Causa**: Problemas de CSS o JavaScript
**Solución**:
```
1. Asegúrate de pasar el mouse sobre la tarjeta completamente
2. Espera un momento para que la animación CSS se active
3. Verifica que no haya errores de JavaScript en la consola (F12)
4. Refresca la página si los íconos no cargan
```

### Problema: Popup de confirmación no se cierra
**Causa**: Problema con event listeners o JavaScript
**Solución**:
```
1. Haz clic en "Cancelar" o "Eliminar" dentro del popup
2. Intenta hacer clic fuera del popup para cerrarlo
3. Presiona la tecla Escape
4. Si no responde, refresca la página (F5)
```

### Problema: Compuesto eliminado sigue apareciendo
**Causa**: Problemas de sincronización con Firebase o cache
**Solución**:
```
1. Refresca la página manualmente
2. Espera unos segundos para que se sincronice con Firebase
3. Verifica tu conexión a internet
4. Borra el cache del navegador para este sitio
```

---

## Consejos y Mejores Prácticas

### Optimización del Rendimiento

#### Para Moléculas Grandes
```
1. Usa "Gemini 2.5 Flash Lite" para sugerencias rápidas
2. Desactiva hidrógenos en vista 3D si no son necesarios
3. Usa estilo "Line" para moléculas muy complejas
4. Cierra pestañas innecesarias del navegador
```

#### Para Uso Intensivo
```
1. Mantén la aplicación en una pestaña dedicada
2. Evita cambiar estilos 3D frecuentemente
3. Usa cache del navegador (no borres datos)
4. Reinicia la aplicación si se vuelve lenta
5. Guarda compuestos importantes regularmente
6. Organiza tus compuestos guardados con nombres descriptivos
```

### Flujo de Trabajo Recomendado

#### Para Exploración General
```
1. Comienza con moléculas predefinidas
2. Explora diferentes estilos de visualización
3. Experimenta con sugerencias IA
4. Documenta hallazgos interesantes
```

#### Para Investigación Específica
```
1. Define objetivos claros
2. Prepara lista de SMILES a analizar
3. Usa modelos IA apropiados (Pro para calidad, Flash para velocidad)
4. Exporta resultados importantes
5. Mantén registro de modificaciones exitosas
6. Guarda compuestos prometedores para análisis posterior
7. Documenta hallazgos en los análisis guardados
```

#### Para Gestión de Compuestos Guardados
```
1. Usa nombres descriptivos para las combinaciones
2. Guarda solo resultados químicamente viables
3. Revisa periódicamente tus compuestos guardados
4. Usa compuestos guardados como base para nuevas modificaciones
5. Exporta análisis importantes antes de experimentar
6. Ten cuidado al eliminar: la acción es permanente
7. Verifica siempre el nombre antes de confirmar eliminación
8. Mantén un número razonable de compuestos para mejor organización
```

#### Para Eliminación Segura de Compuestos
```
1. Lee siempre el popup de confirmación completamente
2. Verifica que el nombre del compuesto sea correcto
3. Considera exportar el análisis antes de eliminar
4. No elimines compuestos que podrías necesitar más tarde
5. Ten en cuenta que la eliminación es inmediata y permanente
6. Si dudas, cancela y reconsidéalo más tarde
```

### Personalización Avanzada

#### Modificar Base de Datos de Moléculas
```python
# En app.py, añadir nuevas moléculas
molecules_db = {
    "estradiol": "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CC[C@@H]2O",
    "fulvestrant": "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CC[C@H](C(F)(F)C(F)(F)S(=O)CCCCCCCCC)C2",
    "cafeina": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Nueva molécula
    "aspirina": "CC(=O)OC1=CC=CC=C1C(=O)O"      # Nueva molécula
}
```

#### Personalizar Prompts de IA
```javascript
// En combination.js, modificar prompts
const customPrompt = `
Eres un químico medicinal experto. Combina estas moléculas considerando:
- Actividad biológica
- Propiedades ADMET
- Síntesis viable
- Selectividad

Molécula 1: ${smiles1}
Molécula 2: ${smiles2}

Genera un híbrido molecular optimizado.
`;
```

---

## Integración con Otras Herramientas

### Exportación de Datos

#### Para Análisis en Python
```python
import requests
import json

# Obtener datos de molécula
response = requests.get('http://localhost:5000/api/molecule/estradiol')
data = response.json()

# Procesar con RDKit
from rdkit import Chem
mol = Chem.MolFromSmiles(data['smiles'])
print(f"Peso molecular: {Chem.rdMolDescriptors.CalcExactMolWt(mol)}")
```

#### Para ChemDraw/ChemSketch
```
1. Copia el SMILES de la respuesta API
2. Pega en ChemDraw: Structure → Convert Name to Structure
3. O importa el archivo MOL descargado
```

#### Para PyMOL
```python
# Guardar MOL block como archivo
mol_data = "..."  # Datos del endpoint /api/molecule/
with open('molecule.mol', 'w') as f:
    f.write(mol_data)

# En PyMOL
# load molecule.mol
# show sticks
```

### Automatización

#### Script para Análisis Masivo
```python
import requests
import time

molecules = ['estradiol', 'fulvestrant']
results = {}

for mol in molecules:
    response = requests.get(f'http://localhost:5000/api/molecule/{mol}')
    if response.status_code == 200:
        results[mol] = response.json()
        print(f"Procesado: {mol}")
    time.sleep(1)  # Respetar límites

# Análisis de resultados
for name, data in results.items():
    print(f"{name}: {data['smiles']}")
```

### Integración con Firebase

#### Acceso Directo a la Base de Datos
```javascript
// Para desarrolladores avanzados
import { getFirestore, collection, getDocs } from 'firebase/firestore';

async function getAllCompounds() {
    const db = getFirestore();
    const querySnapshot = await getDocs(collection(db, "compounds"));
    
    querySnapshot.forEach((doc) => {
        console.log(doc.id, " => ", doc.data());
    });
}
```

#### Backup de Compuestos Guardados
```javascript
// Exportar todos los compuestos a JSON
async function exportCompounds() {
    const compounds = await loadCompounds();
    const dataStr = JSON.stringify(compounds, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'compuestos_guardados.json';
    link.click();
}
```

---

## Recursos Adicionales

### Documentación Técnica
- [README.md](../README.md) - Información general del proyecto
- [API_REFERENCE.md](API_REFERENCE.md) - Documentación completa de APIs
- [COMPONENTS.md](COMPONENTS.md) - Guía de componentes de interfaz

### Herramientas Complementarias
- **RDKit**: Librería de quimioinformática
- **ChemDraw**: Software profesional de dibujo químico
- **PyMOL**: Visualizador molecular avanzado
- **Avogadro**: Editor molecular gratuito

### Recursos de Aprendizaje
- **SMILES Tutorial**: [daylight.com/smiles](https://www.daylight.com/smiles/)
- **Química Orgánica**: Libros de texto especializados
- **3D Molecular Visualization**: Tutoriales de 3Dmol.js

### Comunidad y Soporte
- **Issues**: Reportar problemas en el repositorio
- **Discussions**: Compartir casos de uso y mejoras
- **Wiki**: Documentación colaborativa

---

## Glosario de Términos

**SMILES**: Simplified Molecular Input Line Entry System - Notación para representar estructuras moleculares

**MOL Block**: Formato de archivo para estructuras moleculares 3D

**RDKit**: Librería open-source para quimioinformática

**3Dmol.js**: Librería JavaScript para visualización molecular 3D

**CPK Colors**: Esquema de colores estándar para elementos químicos

**MMFF94**: Campo de fuerza para optimización de geometría molecular

**ETKDG**: Algoritmo para generación de conformaciones 3D

**API REST**: Interfaz de programación de aplicaciones basada en HTTP

**SVG**: Scalable Vector Graphics - Formato de imagen vectorial

**Drag & Drop**: Funcionalidad de arrastrar y soltar

**Firebase Firestore**: Base de datos NoSQL en la nube para almacenar compuestos

**CompoundData**: Estructura de datos que incluye nombre, SMILES, análisis y fecha de creación

**Validación SMILES**: Proceso de verificación de formato y complejidad de notaciones químicas

**Popup de Confirmación**: Ventana modal que requiere confirmación del usuario antes de ejecutar acciones destructivas

**Estado Vacío**: Interfaz que se muestra cuando no hay elementos para mostrar, con íconos y mensajes informativos

**Eliminación Permanente**: Proceso de borrado definitivo de datos que no se puede deshacer

**Event Delegation**: Patrón de JavaScript que maneja eventos en el contenedor padre en lugar de elementos individuales

**Flaticon (fi)**: Biblioteca de íconos utilizada en la interfaz para elementos visuales como papelera, advertencias, etc.

---

*Guía de Uso v1.0 - Quimática Orgánica*

¡Disfruta explorando el fascinante mundo de la química molecular!