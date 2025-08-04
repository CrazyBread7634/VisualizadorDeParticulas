# 🧬 Quimática Orgánica

**Visualización y Manipulación Molecular Impulsada por IA**

Una aplicación web interactiva moderna para la visualización, combinación y análisis de estructuras moleculares usando inteligencia artificial generativa y herramientas de química computacional.

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![Flask](https://img.shields.io/badge/flask-2.0+-red.svg)
![JavaScript](https://img.shields.io/badge/javascript-ES6+-yellow.svg)
![Firebase](https://img.shields.io/badge/firebase-v9+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

---

## ✨ Características Principales

### 🔬 **Visualización Molecular Avanzada**
- **Vista 2D**: Renderizado SVG interactivo con pan & zoom
- **Vista 3D**: Visualización tridimensional usando 3Dmol.js
- **Múltiples estilos**: Palos, alambre, esferas, bolas y palos
- **Índices configurables**: Átomos y enlaces numerados
- **Exportación 3D**: Múltiples formatos (STL, OBJ, FBX, 3MF, AMF)

### 🤖 **Inteligencia Artificial Generativa**
- **Combinación molecular**: Fusión inteligente de compuestos usando Gemini AI
- **Sugerencias contextuales**: Modificaciones químicas basadas en IA
- **Múltiples modelos**: Gemini 1.5 Flash, Pro, 2.5 Flash Lite, 2.5 Pro
- **Análisis farmacológico**: Propiedades y aplicaciones generadas automáticamente

### 💾 **Persistencia y Gestión**
- **Firebase Firestore**: Almacenamiento en la nube de compuestos generados
- **CRUD completo**: Crear, leer y eliminar compuestos guardados
- **Confirmación segura**: Popups de confirmación para acciones destructivas
- **Estados informativos**: Interfaz vacía con iconografía guía

### 🎨 **Interfaz de Usuario Moderna**
- **Drag & Drop**: Arrastrar moléculas para combinar
- **Diseño responsivo**: Adaptable a diferentes dispositivos
- **Iconografía profesional**: Integración Flaticon consistente
- **Animaciones suaves**: Transiciones CSS elegantes
- **Estados de loading**: Feedback visual durante operaciones

### 🔧 **Backend Robusto**
- **RDKit**: Procesamiento químico profesional
- **Validaciones estrictas**: SMILES < 500 chars, < 150 átomos
- **API REST**: Endpoints documentados completamente
- **Manejo de errores**: Respuestas detalladas con sugerencias

---

## 🚀 Instalación Rápida

### Prerrequisitos
```bash
- Python 3.8+
- Node.js (para desarrollo frontend)
- Navegador moderno con soporte ES6
```

### 1. Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/quimatica-organica.git
cd quimatica-organica
```

### 2. Configurar Backend (Python)
```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Configurar Firebase (Opcional pero Recomendado)
```javascript
// Editar static/js/firebase/config.js con tu configuración
const firebaseConfig = {
  apiKey: "tu-api-key",
  authDomain: "tu-proyecto.firebaseapp.com",
  projectId: "tu-proyecto-id",
  // ... más configuración
};
```

### 4. Configurar Google AI
```javascript
// Editar static/js/api.js con tu API key
const API_KEY = "tu-google-ai-api-key";
```

### 5. Ejecutar la Aplicación
```bash
python app.py
```

Visita `http://localhost:5000` para usar la aplicación.

---

## 🎯 Uso Rápido

### Visualización Básica
1. **Seleccionar molécula**: Haz clic en Estradiol o Fulvestrant
2. **Explorar vistas**: Observa en 2D y 3D simultáneamente
3. **Cambiar estilos**: Prueba diferentes representaciones 3D
4. **Exportar**: Descarga modelos 3D para impresión

### Combinación con IA
1. **Arrastrar moléculas**: Del banco a los slots de combinación
2. **Elegir modelo IA**: Selecciona Gemini según velocidad/calidad
3. **Combinar**: Haz clic en "Combinar Moléculas"
4. **Analizar**: Revisa el compuesto generado y su análisis
5. **Guardar**: Persiste resultados prometedores

### Gestión de Compuestos
1. **Ver guardados**: Sección inferior con tarjetas verdes
2. **Cargar compuesto**: Botón "Cargar" en cada tarjeta
3. **Eliminar compuesto**: Ícono papelera con confirmación segura
4. **Estados vacíos**: Guías visuales cuando no hay compuestos

---

## 🏗️ Arquitectura del Sistema

```
📁 Quimática Orgánica/
├── 🐍 Backend (Flask + RDKit)
│   ├── app.py                 # Servidor principal
│   ├── requirements.txt       # Dependencias Python
│   └── templates/            
│       └── index.html         # SPA principal
│
├── 🌐 Frontend (Vanilla JS + ES6 Modules)
│   ├── static/js/
│   │   ├── main.js           # Controlador principal
│   │   ├── api.js            # Cliente Google AI
│   │   ├── viewer.js         # Visualización molecular
│   │   ├── combination.js    # Lógica de combinación
│   │   ├── ui.js             # Interfaz de usuario
│   │   └── firebase/         # Integración Firebase
│   │       ├── config.js     # Configuración
│   │       └── db.js         # Operaciones CRUD
│   │
│   ├── static/css/
│   │   └── style.css         # Estilos responsivos
│   │
│   └── static/media/         # Recursos estáticos
│
├── 🔥 Cloud Services
│   ├── Firebase Firestore    # Base de datos NoSQL
│   ├── Google AI (Gemini)    # Modelos generativos
│   └── 3Dmol.js CDN          # Visualización 3D
│
└── 📚 Documentation/
    ├── README.md             # Este archivo
    ├── docs/INDEX.md         # Índice de documentación
    ├── docs/API_REFERENCE.md # Referencia técnica
    ├── docs/COMPONENTS.md    # Guía de componentes
    └── docs/USAGE_GUIDE.md   # Manual de usuario
```

---

## 🔬 Tecnologías Utilizadas

### Backend
- **[Flask](https://flask.palletsprojects.com/)**: Framework web minimalista
- **[RDKit](https://rdkit.readthedocs.io/)**: Química computacional y validación
- **Python 3.8+**: Lenguaje de programación principal

### Frontend
- **Vanilla JavaScript ES6+**: Sin frameworks, máximo rendimiento
- **[3Dmol.js](https://3dmol.csb.pitt.edu/)**: Visualización molecular 3D
- **[SVG-Pan-Zoom](https://github.com/ariutta/svg-pan-zoom)**: Interactividad 2D
- **CSS3**: Animaciones y diseño responsivo

### Cloud & APIs
- **[Firebase Firestore](https://firebase.google.com/docs/firestore)**: Base de datos en tiempo real
- **[Google AI Gemini](https://ai.google.dev/)**: Modelos de lenguaje generativo
- **[Flaticon](https://flaticon.com/)**: Iconografía profesional

---

## 📖 Documentación Completa

| Documento | Descripción | Ideal para |
|-----------|-------------|------------|
| **[📋 Índice](docs/INDEX.md)** | Navegación completa de documentación | Todos los usuarios |
| **[🔧 API Reference](docs/API_REFERENCE.md)** | Referencia técnica completa | Desarrolladores |
| **[🎨 Componentes](docs/COMPONENTS.md)** | Guía de componentes UI | Frontend developers |
| **[📖 Guía de Uso](docs/USAGE_GUIDE.md)** | Manual paso a paso | Usuarios finales |

### Guía Rápida por Rol

#### 👨‍💻 Desarrolladores
1. **Instalación**: Este README
2. **APIs**: [API_REFERENCE.md](docs/API_REFERENCE.md)
3. **Componentes**: [COMPONENTS.md](docs/COMPONENTS.md)

#### 👨‍🔬 Usuarios Finales
1. **Tutorial**: [USAGE_GUIDE.md](docs/USAGE_GUIDE.md)
2. **Problemas**: [Solución de problemas](docs/USAGE_GUIDE.md#solución-de-problemas-comunes)

#### 🎓 Educadores
1. **Casos educativos**: [Casos de uso](docs/USAGE_GUIDE.md#casos-de-uso-educativos)
2. **Ejemplos**: [Ejemplos prácticos](docs/USAGE_GUIDE.md#ejemplos-prácticos)

---

## 🧪 Ejemplos de Uso

### API REST
```bash
# Obtener molécula predefinida
curl "http://localhost:5000/api/molecule/estradiol"

# Renderizar SMILES personalizado
curl -X POST http://localhost:5000/api/render_smiles \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO", "show_atoms": true}'
```

### JavaScript Frontend
```javascript
import { loadMolecule } from './static/js/viewer.js';
import { saveCompound } from './static/js/firebase/db.js';

// Cargar molécula en visores
await loadMolecule('estradiol');

// Guardar compuesto generado
await saveCompound({
    name: "Mi Compuesto",
    smiles: "CC1=CC=CC=C1",
    analysis: "<p>Análisis...</p>",
    createdAt: new Date()
});
```

### Python Backend
```python
from app import smiles_to_mol_block, smiles_to_svg

# Generar estructura 3D
mol_block = smiles_to_mol_block("CCO")

# Generar imagen 2D
svg_image, bonds = smiles_to_svg("CCO", show_atoms=True)
```

---

## 🚦 Estado del Proyecto

### ✅ Características Completadas
- ✅ Visualización 2D/3D completa
- ✅ Integración Google AI (4 modelos)
- ✅ Sistema Firebase completo (CRUD)
- ✅ Popup de confirmación segura
- ✅ Exportación 3D múltiples formatos
- ✅ Drag & Drop interactivo
- ✅ Validaciones SMILES robustas
- ✅ Documentación completa (100%)

### 🔄 En Desarrollo
- 🔄 Autenticación de usuarios
- 🔄 Historial de modificaciones
- 🔄 Más formatos de exportación
- 🔄 API rate limiting

### 💡 Funcionalidades Futuras
- 💡 Colaboración en tiempo real
- 💡 Búsqueda de similitud molecular
- 💡 Integración con bases de datos químicas
- 💡 Análisis QSAR automatizado

---

## 🤝 Contribución

¡Las contribuciones son bienvenidas! Por favor:

1. **Fork** el repositorio
2. **Crear** una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. **Push** a la rama (`git push origin feature/AmazingFeature`)
5. **Abrir** un Pull Request

### Estándares de Código
- **Python**: PEP 8, type hints cuando sea posible
- **JavaScript**: ES6+, JSDoc para funciones públicas
- **CSS**: BEM methodology, variables CSS
- **Documentación**: Markdown con ejemplos funcionales

---

## 📝 Notas de la Versión

### v1.0.0 (Actual)
- 🎉 Lanzamiento inicial
- ✨ Sistema Firebase completo
- 🗑️ Eliminación segura de compuestos
- ⚠️ Popups de confirmación
- 🎨 Estados vacíos informativos
- 📚 Documentación completa

---

## ⚠️ Consideraciones Importantes

### Seguridad
- **🔓 API Keys expuestas**: Para fines educativos. En producción, usar variables de entorno
- **🔥 Firebase Rules**: Configurar reglas de seguridad apropiadas
- **🌐 CORS**: Configurar dominios permitidos para producción

### Límites Técnicos
- **SMILES**: Máximo 500 caracteres
- **Átomos**: Máximo 150 por molécula
- **Firebase**: Cuotas gratuitas aplicables
- **Google AI**: Límites de API según plan

### Compatibilidad
- **Navegadores**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **Dispositivos**: Desktop recomendado, responsive para tablet/móvil
- **Resolución**: Mínimo 1024x768 para experiencia óptima

---

## 📞 Soporte y Contacto

### Canales de Soporte
- **🐛 Issues**: [GitHub Issues](https://github.com/tu-usuario/quimatica-organica/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/tu-usuario/quimatica-organica/discussions)
- **📖 Wiki**: [Documentación Colaborativa](https://github.com/tu-usuario/quimatica-organica/wiki)

### Tiempo de Respuesta
- **Issues críticos**: 24-48 horas
- **Preguntas generales**: 3-5 días
- **Feature requests**: Según disponibilidad

---

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

### MIT License Summary
- ✅ Uso comercial permitido
- ✅ Modificación permitida  
- ✅ Distribución permitida
- ✅ Uso privado permitido
- ❌ Sin garantía
- ❌ Sin responsabilidad del autor

---

## 🙏 Agradecimientos

- **[RDKit](https://rdkit.org/)** - Herramientas de química computacional
- **[3Dmol.js](https://3dmol.csb.pitt.edu/)** - Visualización molecular 3D
- **[Google AI](https://ai.google.dev/)** - Modelos generativos Gemini
- **[Firebase](https://firebase.google.com/)** - Infraestructura cloud
- **[Flaticon](https://flaticon.com/)** - Iconografía profesional

---

## 🌟 Dale una Estrella

Si este proyecto te resultó útil, por favor ⭐ **deja una estrella** en GitHub. ¡Ayuda mucho!

---

**Desarrollado con ❤️ para la comunidad científica y educativa**

*Quimática Orgánica v1.0 - Visualización Molecular Inteligente*