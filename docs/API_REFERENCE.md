# API Reference - Quimática Orgánica

## Descripción General

Esta documentación describe todas las APIs públicas disponibles en la aplicación Quimática - Orgánica, incluyendo endpoints REST, funciones JavaScript y componentes de interfaz.

## APIs Backend (Flask)

### Base URL
```
http://localhost:5000
```

### Autenticación
No se requiere autenticación para las APIs públicas.

### Content-Type
- Para requests con body: `application/json`
- Para responses: `application/json`

---

## Endpoints REST

### 1. Página Principal

#### `GET /`

**Descripción**: Sirve la página principal de la aplicación web.

**Headers**:
- Ninguno requerido

**Parámetros**:
- Ninguno

**Respuesta**:
- **Status**: 200 OK
- **Content-Type**: `text/html`
- **Body**: HTML de la página principal

**Ejemplo**:
```bash
curl -X GET http://localhost:5000/
```

---

### 2. Obtener Molécula por Nombre

#### `GET /api/molecule/<molecule_name>`

**Descripción**: Obtiene la estructura 2D y 3D de una molécula específica del banco de datos.

**Parámetros de Ruta**:
- `molecule_name` (string, requerido): Nombre de la molécula
  - Valores válidos: `"estradiol"`, `"fulvestrant"`

**Parámetros de Consulta**:
- `show_atoms` (boolean, opcional): Mostrar índices de átomos en la imagen SVG
  - Default: `true`
  - Valores: `"true"`, `"false"`
- `show_bonds` (boolean, opcional): Mostrar índices de enlaces en la imagen SVG
  - Default: `true`
  - Valores: `"true"`, `"false"`

**Respuesta Exitosa (200)**:
```json
{
  "mol": "string",      // Bloque MOL 3D en formato texto
  "smiles": "string",   // Notación SMILES de la molécula
  "svg": "string",      // Imagen SVG 2D como string
  "bonds": [            // Array de información de enlaces
    {
      "bond_index": 0,        // Índice del enlace
      "atom1_index": 1,       // Índice del primer átomo
      "atom2_index": 2        // Índice del segundo átomo
    }
  ]
}
```

**Respuesta de Error (404)**:
```json
{
  "error": "Molécula no encontrada"
}
```

**Respuesta de Error (500)**:
```json
{
  "error": "No se pudo generar la estructura 3D o 2D"
}
```

**Ejemplos**:

```bash
# Obtener estradiol con todos los índices
curl "http://localhost:5000/api/molecule/estradiol"

# Obtener estradiol sin índices de átomos
curl "http://localhost:5000/api/molecule/estradiol?show_atoms=false"

# Obtener fulvestrant solo con índices de enlaces
curl "http://localhost:5000/api/molecule/fulvestrant?show_atoms=false&show_bonds=true"
```

---

### 3. Renderizar SMILES Personalizado

#### `POST /api/render_smiles`

**Descripción**: Genera estructuras 2D y 3D a partir de una notación SMILES proporcionada por el usuario.

**Headers**:
- `Content-Type: application/json`

**Cuerpo de la Petición**:
```json
{
  "smiles": "string",           // Notación SMILES (requerido)
  "show_atoms": boolean,        // Mostrar índices de átomos (opcional, default: true)
  "show_bonds": boolean         // Mostrar índices de enlaces (opcional, default: true)
}
```

**Respuesta Exitosa (200)**:
```json
{
  "mol": "string",      // Bloque MOL 3D optimizado
  "smiles": "string",   // SMILES de entrada (eco)
  "svg": "string",      // Imagen SVG 2D generada
  "bonds": [            // Información de enlaces extraída
    {
      "bond_index": 0,
      "atom1_index": 1,
      "atom2_index": 2
    }
  ]
}
```

**Respuesta de Error (400)**:
```json
{
  "error": "No se proporcionó SMILES"
}
```

**Respuesta de Error (422)**:
```json
{
  "error": "El SMILES proporcionado no es válido o no se pudo generar la estructura 3D."
}
```

**Ejemplos**:

```bash
# Renderizar etanol (CCO)
curl -X POST http://localhost:5000/api/render_smiles \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": "CCO",
    "show_atoms": true,
    "show_bonds": true
  }'

# Renderizar cafeína sin índices de enlaces
curl -X POST http://localhost:5000/api/render_smiles \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "show_atoms": true,
    "show_bonds": false
  }'

# Renderizar benceno con configuración mínima
curl -X POST http://localhost:5000/api/render_smiles \
  -H "Content-Type: application/json" \
  -d '{"smiles": "c1ccccc1"}'
```

---

## Funciones Backend (Python)

### 1. smiles_to_mol_block()

```python
def smiles_to_mol_block(smiles: str) -> str | None
```

**Descripción**: Convierte una cadena SMILES a un bloque MOL 3D optimizado usando RDKit.

**Parámetros**:
- `smiles` (str): Notación SMILES válida de la molécula

**Retorna**:
- `str`: Bloque MOL 3D en formato texto, o `None` si hay error

**Proceso**:
1. Parsea SMILES con RDKit
2. Añade hidrógenos explícitos
3. Genera conformación 3D con ETKDG
4. Optimiza con campo de fuerza MMFF94
5. Convierte a formato MOL

**Manejo de Errores**:
- Captura excepciones y retorna `None`
- Imprime errores en consola para debugging

**Ejemplo**:
```python
# Uso básico
mol_block = smiles_to_mol_block("CCO")
if mol_block:
    print("Estructura 3D generada exitosamente")
else:
    print("Error al procesar SMILES")

# Ejemplo con molécula compleja
complex_smiles = "CC(C)(C)c1ccc(O)cc1"  # BHT
mol_block = smiles_to_mol_block(complex_smiles)
```

### 2. smiles_to_svg()

```python
def smiles_to_svg(smiles: str, show_atom_indices: bool = True, show_bond_indices: bool = True) -> tuple[str | None, list | None]
```

**Descripción**: Convierte SMILES a imagen SVG 2D y extrae información de enlaces.

**Parámetros**:
- `smiles` (str): Notación SMILES de la molécula
- `show_atom_indices` (bool): Mostrar números de átomos en la imagen
- `show_bond_indices` (bool): Mostrar números de enlaces en la imagen

**Retorna**:
- `tuple`: (svg_string, bonds_data) donde:
  - `svg_string` (str): Código SVG de la imagen 2D
  - `bonds_data` (list): Lista de diccionarios con información de enlaces

**Estructura de bonds_data**:
```python
[
    {
        'bond_index': int,      # Índice único del enlace
        'atom1_index': int,     # Índice del primer átomo
        'atom2_index': int      # Índice del segundo átomo
    }
]
```

**Configuración de Imagen**:
- Tamaño: 400x350 píxeles
- Fondo: Transparente
- Coordenadas 2D calculadas automáticamente

**Ejemplo**:
```python
# Generar SVG con todos los índices
svg, bonds = smiles_to_svg("CCO", True, True)
print(f"SVG generado: {len(svg)} caracteres")
print(f"Enlaces: {len(bonds)}")

# Generar solo estructura sin etiquetas
svg, bonds = smiles_to_svg("c1ccccc1", False, False)

# Procesar información de enlaces
for bond in bonds:
    print(f"Enlace {bond['bond_index']}: átomo {bond['atom1_index']} - átomo {bond['atom2_index']}")
```

---

## APIs Frontend (JavaScript)

### Módulo: api.js

#### 1. getGenerativeModel()

```javascript
function getGenerativeModel(modelName: string): GenerativeModel
```

**Descripción**: Obtiene una instancia de modelo generativo de Google AI.

**Parámetros**:
- `modelName` (string): Nombre del modelo de IA
  - `"gemini-1.5-flash"`: Rápido, balanceado
  - `"gemini-1.5-pro"`: Más potente, mejor calidad
  - `"gemini-2.5-flash-lite"`: Ultrarrápido, sugerencias
  - `"gemini-2.5-pro"`: Última generación, máximo rendimiento

**Retorna**:
- `GenerativeModel`: Instancia configurada del modelo

**Ejemplo**:
```javascript
import { getGenerativeModel } from './api.js';

// Obtener modelo rápido para uso general
const flashModel = getGenerativeModel('gemini-1.5-flash');

// Usar modelo para generar contenido
const result = await flashModel.generateContent('Describe el benceno');
console.log(result.response.text());
```

#### 2. suggestionModel

```javascript
const suggestionModel: GenerativeModel
```

**Descripción**: Instancia preconfigurada del modelo Gemini 2.5 Flash Lite optimizada para generar sugerencias rápidas de modificaciones moleculares.

**Uso**:
```javascript
import { suggestionModel } from './api.js';

// Generar sugerencias para una molécula
const prompt = `Sugiere 5 modificaciones para esta molécula: ${smiles}`;
const result = await suggestionModel.generateContent(prompt);
```

---

### Módulo: viewer.js

#### 1. initViewer()

```javascript
function initViewer(): void
```

**Descripción**: Inicializa el visor 3D usando 3Dmol.js en el elemento DOM correspondiente.

**Configuración**:
- Elemento: `#viewer-3d`
- Estilo inicial: `stick`
- Colores: Esquema CPK estándar
- Controles: Rotación, zoom, pan habilitados

**Ejemplo**:
```javascript
import { initViewer } from './viewer.js';

// Inicializar al cargar la página
document.addEventListener('DOMContentLoaded', () => {
    initViewer();
});
```

#### 2. loadMolecule()

```javascript
function loadMolecule(moleculeName: string): Promise<void>
```

**Descripción**: Carga y visualiza una molécula en ambos visores (2D y 3D).

**Parámetros**:
- `moleculeName` (string): Nombre de la molécula a cargar

**Proceso**:
1. Realiza petición a `/api/molecule/${moleculeName}`
2. Actualiza visor 2D con SVG
3. Carga estructura 3D en 3Dmol.js
4. Actualiza etiquetas de interfaz

**Ejemplo**:
```javascript
import { loadMolecule } from './viewer.js';

// Cargar estradiol
await loadMolecule('estradiol');

// Cargar con manejo de errores
try {
    await loadMolecule('fulvestrant');
    console.log('Molécula cargada exitosamente');
} catch (error) {
    console.error('Error al cargar molécula:', error);
}
```

#### 3. update3DStyle()

```javascript
function update3DStyle(): void
```

**Descripción**: Actualiza el estilo de visualización del visor 3D basado en los controles de la interfaz.

**Estilos Soportados**:
- `stick`: Representación de palos (default)
- `line`: Estructura de alambre
- `sphere`: Esferas de van der Waals
- `ballandstick`: Combinación bolas y palos

**Configuraciones Adicionales**:
- Mostrar/ocultar hidrógenos
- Colores por elemento químico
- Transparencia configurable

**Ejemplo**:
```javascript
import { update3DStyle } from './viewer.js';

// Cambiar a estilo esfera
document.querySelector('input[value="sphere"]').checked = true;
update3DStyle();

// Ocultar hidrógenos
document.getElementById('show-hydrogens').checked = false;
update3DStyle();
```

#### 4. toggle3DLabels()

```javascript
function toggle3DLabels(): void
```

**Descripción**: Alterna la visualización de etiquetas (índices de átomos y enlaces) en el visor 3D.

**Controles**:
- `#show-atom-indices-3d`: Checkbox para índices de átomos
- `#show-bond-indices-3d`: Checkbox para índices de enlaces

**Ejemplo**:
```javascript
import { toggle3DLabels } from './viewer.js';

// Mostrar índices de átomos
document.getElementById('show-atom-indices-3d').checked = true;
toggle3DLabels();
```

#### 5. getViewer()

```javascript
function getViewer(): 3Dmol.GLViewer | null
```

**Descripción**: Obtiene la instancia actual del visor 3D.

**Retorna**:
- `3Dmol.GLViewer`: Instancia del visor, o `null` si no está inicializado

**Ejemplo**:
```javascript
import { getViewer } from './viewer.js';

// Obtener visor y resetear vista
const viewer = getViewer();
if (viewer) {
    viewer.zoomTo();
    viewer.render();
}
```

#### 6. rerenderCurrentMolecule()

```javascript
function rerenderCurrentMolecule(): void
```

**Descripción**: Vuelve a renderizar la molécula actual en el visor 2D con la configuración actualizada de índices.

**Uso**: Se llama automáticamente cuando cambian los checkboxes de visualización 2D.

---

### Módulo: ui.js

#### 1. populateMoleculeCards()

```javascript
function populateMoleculeCards(): void
```

**Descripción**: Puebla las tarjetas de moléculas en el banco molecular con íconos animados.

**Proceso**:
1. Obtiene todas las tarjetas con clase `.molecule-card`
2. Carga íconos de Lord Icon
3. Configura eventos de drag & drop
4. Añade tooltips con nombres de moléculas

#### 2. updateCardSelection()

```javascript
function updateCardSelection(moleculeName: string): void
```

**Descripción**: Actualiza la selección visual de las tarjetas de moléculas.

**Parámetros**:
- `moleculeName` (string): Nombre de la molécula seleccionada

**Efectos Visuales**:
- Añade clase `selected` a la tarjeta activa
- Remueve selección de otras tarjetas
- Actualiza colores y bordes

#### 3. updateButtonState()

```javascript
function updateButtonState(text: string, loading: boolean, disabled: boolean): void
```

**Descripción**: Actualiza el estado visual de botones con indicadores de carga.

**Parámetros**:
- `text` (string): Texto a mostrar en el botón
- `loading` (boolean): Si mostrar indicador de carga
- `disabled` (boolean): Si deshabilitar el botón

**Características**:
- Animación de carga con Lord Icon
- Texto animado tipo carrusel
- Estados visuales diferenciados

#### 4. initContextMenu()

```javascript
function initContextMenu(): void
```

**Descripción**: Inicializa el menú contextual para sugerencias de IA.

**Funcionalidades**:
- Detección de clics en átomos/enlaces
- Generación automática de sugerencias
- Interfaz arrastrable
- Input personalizado para modificaciones

#### 5. extractJson()

```javascript
function extractJson(rawText: string): object | null
```

**Descripción**: Extrae objetos JSON de respuestas de IA de forma robusta.

**Parámetros**:
- `rawText` (string): Respuesta cruda de la IA

**Retorna**:
- `object`: Objeto JSON parseado, o `null` si no se encuentra

**Métodos de Extracción**:
1. Bloques de código markdown (```json ... ```)
2. Arrays JSON embebidos en texto
3. Fallback con regex patterns

**Ejemplo**:
```javascript
import { extractJson } from './ui.js';

const aiResponse = `
Aquí tienes las sugerencias:
\`\`\`json
[
  {"name": "Añadir grupo metilo", "description": "..."},
  {"name": "Oxidar alcohol", "description": "..."}
]
\`\`\`
`;

const suggestions = extractJson(aiResponse);
if (suggestions) {
    suggestions.forEach(suggestion => {
        console.log(suggestion.name);
    });
}
```

---

### Módulo: combination.js

#### 1. setupDragAndDrop()

```javascript
function setupDragAndDrop(): void
```

**Descripción**: Configura la funcionalidad completa de arrastrar y soltar para moléculas.

**Características**:
- Drop zones para dos moléculas
- Efectos visuales durante el arrastre
- Validación de drops válidos
- Actualización automática de UI

**Eventos Manejados**:
- `dragover`: Permite el drop
- `drop`: Procesa la molécula soltada
- `dragleave`: Remueve efectos visuales

#### 2. combineMolecules()

```javascript
async function combineMolecules(): Promise<void>
```

**Descripción**: Combina dos moléculas usando IA generativa para crear nuevas estructuras.

**Proceso**:
1. Valida que hay dos moléculas seleccionadas
2. Obtiene modelo de IA seleccionado
3. Genera prompt de combinación
4. Procesa respuesta de IA
5. Renderiza resultado

**Prompt Template**:
```
Eres un químico experto. Combina estas dos moléculas de manera realista:
Molécula 1: [SMILES]
Molécula 2: [SMILES]

Genera un producto de reacción químicamente viable en formato SMILES.
```

#### 3. clearCombination()

```javascript
function clearCombination(selectedMolecule: string): void
```

**Descripción**: Limpia la zona de combinación y restaura la vista individual.

**Parámetros**:
- `selectedMolecule` (string): Molécula a mostrar tras la limpieza

**Acciones**:
- Limpia slots de combinación
- Oculta sección de resultados
- Restaura vista individual
- Actualiza estado de botones

#### 4. handleSuggestionClick()

```javascript
async function handleSuggestionClick(suggestion: object): Promise<void>
```

**Descripción**: Maneja clics en sugerencias de modificación molecular del menú contextual.

**Parámetros**:
- `suggestion` (object): Objeto con datos de la sugerencia
  - `name` (string): Nombre de la modificación
  - `description` (string): Descripción detallada
  - `smiles` (string, opcional): SMILES modificado

**Proceso**:
1. Valida estructura de la sugerencia
2. Si tiene SMILES, lo renderiza directamente
3. Si no, usa IA para generar modificación
4. Actualiza visualización con resultado

---

## Tipos de Datos

### MoleculeData

```typescript
interface MoleculeData {
  mol: string;          // Bloque MOL 3D
  smiles: string;       // Notación SMILES
  svg: string;          // Imagen SVG 2D
  bonds: BondInfo[];    // Información de enlaces
}
```

### BondInfo

```typescript
interface BondInfo {
  bond_index: number;     // Índice único del enlace
  atom1_index: number;    // Índice del primer átomo
  atom2_index: number;    // Índice del segundo átomo
}
```

### Suggestion

```typescript
interface Suggestion {
  name: string;           // Nombre de la modificación
  description: string;    // Descripción detallada
  smiles?: string;        // SMILES modificado (opcional)
}
```

---

## Códigos de Estado HTTP

| Código | Descripción | Uso |
|--------|-------------|-----|
| 200 | OK | Respuesta exitosa |
| 400 | Bad Request | Parámetros faltantes o inválidos |
| 404 | Not Found | Molécula no encontrada |
| 422 | Unprocessable Entity | SMILES inválido |
| 500 | Internal Server Error | Error del servidor |

---

## Límites y Restricciones

### Backend
- **Longitud SMILES**: Máximo 1000 caracteres (recomendado)
- **Timeout**: 30 segundos para procesamiento 3D
- **Memoria**: Limitado por disponibilidad del sistema

### Frontend
- **Modelos IA**: Sujeto a cuotas de Google AI
- **Tamaño SVG**: Máximo 400x350 píxeles
- **Navegadores**: Requiere soporte para ES6 modules

### Rate Limiting
- No implementado actualmente
- Recomendado para producción: 100 requests/minuto por IP

---

## Ejemplos de Integración

### Cliente Python

```python
import requests
import json

class QuimaticaClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def get_molecule(self, name, show_atoms=True, show_bonds=True):
        """Obtiene datos de una molécula por nombre."""
        url = f"{self.base_url}/api/molecule/{name}"
        params = {
            'show_atoms': str(show_atoms).lower(),
            'show_bonds': str(show_bonds).lower()
        }
        response = requests.get(url, params=params)
        return response.json()
    
    def render_smiles(self, smiles, show_atoms=True, show_bonds=True):
        """Renderiza una molécula desde SMILES."""
        url = f"{self.base_url}/api/render_smiles"
        data = {
            'smiles': smiles,
            'show_atoms': show_atoms,
            'show_bonds': show_bonds
        }
        response = requests.post(url, json=data)
        return response.json()

# Uso
client = QuimaticaClient()
molecule = client.get_molecule('estradiol')
print(f"SMILES: {molecule['smiles']}")
```

### Cliente JavaScript (Node.js)

```javascript
import fetch from 'node-fetch';

class QuimaticaClient {
    constructor(baseUrl = 'http://localhost:5000') {
        this.baseUrl = baseUrl;
    }
    
    async getMolecule(name, showAtoms = true, showBonds = true) {
        const params = new URLSearchParams({
            show_atoms: showAtoms.toString(),
            show_bonds: showBonds.toString()
        });
        
        const response = await fetch(
            `${this.baseUrl}/api/molecule/${name}?${params}`
        );
        
        return await response.json();
    }
    
    async renderSmiles(smiles, showAtoms = true, showBonds = true) {
        const response = await fetch(
            `${this.baseUrl}/api/render_smiles`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    smiles,
                    show_atoms: showAtoms,
                    show_bonds: showBonds
                })
            }
        );
        
        return await response.json();
    }
}

// Uso
const client = new QuimaticaClient();
const molecule = await client.getMolecule('estradiol');
console.log(`SMILES: ${molecule.smiles}`);
```

---

*Documentación API v1.0 - Quimática Orgánica*