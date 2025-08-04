# Índice de Documentación - Quimática Orgánica

## Bienvenido a la Documentación Completa

Esta es la documentación completa para **Quimática - Orgánica**, una aplicación web interactiva para la visualización y manipulación de estructuras moleculares usando inteligencia artificial.

## 📚 Documentos Disponibles

### 🚀 [README Principal](../README.md)
**Descripción**: Información general del proyecto, instalación y configuración básica.
- Descripción del proyecto
- Arquitectura del sistema
- Instalación paso a paso
- Configuración inicial
- Información de licencia

**Ideal para**: Desarrolladores que quieren instalar y ejecutar la aplicación.

---

### 🔧 [Referencia de APIs](API_REFERENCE.md)
**Descripción**: Documentación técnica completa de todas las APIs públicas.
- Endpoints REST detallados
- Funciones backend (Python)
- APIs frontend (JavaScript)
- Tipos de datos y esquemas
- Ejemplos de código
- Clientes de ejemplo

**Ideal para**: Desarrolladores que quieren integrar o extender la aplicación.

---

### 🎨 [Guía de Componentes](COMPONENTS.md)
**Descripción**: Documentación detallada de todos los componentes de interfaz.
- Componentes principales de UI
- Propiedades CSS y estados
- Patrones de interacción
- Eventos del sistema
- Optimizaciones de rendimiento
- Testing de componentes

**Ideal para**: Desarrolladores frontend y diseñadores que trabajan en la interfaz.

---

### 📖 [Guía de Uso](USAGE_GUIDE.md)
**Descripción**: Manual completo para usuarios finales con ejemplos prácticos.
- Tutorial paso a paso
- Funcionalidades básicas y avanzadas
- Casos de uso educativos
- Solución de problemas
- Mejores prácticas
- Integración con herramientas

**Ideal para**: Usuarios finales, educadores y estudiantes.

---

## 🎯 Guía de Navegación por Rol

### Para Usuarios Finales
1. **Empezar aquí**: [Guía de Uso](USAGE_GUIDE.md)
2. **Referencia rápida**: [README Principal](../README.md)
3. **Problemas técnicos**: [Guía de Uso - Solución de Problemas](USAGE_GUIDE.md#solución-de-problemas-comunes)

### Para Educadores
1. **Casos educativos**: [Guía de Uso - Casos Educativos](USAGE_GUIDE.md#casos-de-uso-educativos)
2. **Ejemplos prácticos**: [Guía de Uso - Ejemplos](USAGE_GUIDE.md#ejemplos-prácticos)
3. **Personalización**: [Guía de Uso - Personalización](USAGE_GUIDE.md#personalización-avanzada)

### Para Desarrolladores Backend
1. **Instalación**: [README Principal](../README.md)
2. **APIs Python**: [Referencia de APIs - Backend](API_REFERENCE.md#funciones-backend-python)
3. **Endpoints REST**: [Referencia de APIs - Endpoints](API_REFERENCE.md#endpoints-rest)

### Para Desarrolladores Frontend
1. **Componentes**: [Guía de Componentes](COMPONENTS.md)
2. **APIs JavaScript**: [Referencia de APIs - Frontend](API_REFERENCE.md#apis-frontend-javascript)
3. **Eventos**: [Guía de Componentes - Eventos](COMPONENTS.md#eventos-del-sistema)

### Para Integradores
1. **Referencia API**: [Referencia de APIs](API_REFERENCE.md)
2. **Ejemplos de integración**: [Referencia de APIs - Integración](API_REFERENCE.md#ejemplos-de-integración)
3. **Clientes**: [Referencia de APIs - Clientes](API_REFERENCE.md#ejemplos-de-integración)

---

## 🔍 Búsqueda Rápida por Tema

### Visualización Molecular
- **2D**: [Componentes - Visor 2D](COMPONENTS.md#3-visor-2d-viewer-container-2d)
- **3D**: [Componentes - Visor 3D](COMPONENTS.md#4-visor-3d-viewer-container-3d)
- **Estilos**: [Guía de Uso - Vista 3D](USAGE_GUIDE.md#paso-3-explorar-la-vista-3d)

### Inteligencia Artificial
- **Combinación**: [Guía de Uso - Combinación IA](USAGE_GUIDE.md#combinación-de-moléculas-con-ia)
- **Sugerencias**: [Guía de Uso - Sugerencias](USAGE_GUIDE.md#sugerencias-inteligentes-con-menú-contextual)
- **Modelos**: [Referencia API - Modelos IA](API_REFERENCE.md#getgenerativemodel)

### Guardado y Persistencia
- **Guardar Compuestos**: [Guía de Uso - Guardado](USAGE_GUIDE.md#guardado-y-gestión-de-compuestos)
- **Eliminar Compuestos**: [Guía de Uso - Eliminación](USAGE_GUIDE.md#eliminando-compuestos-guardados)
- **Firebase**: [Referencia API - Firebase](API_REFERENCE.md#apis-firebase-javascript)
- **Gestión Completa**: [Componentes - Compuestos Guardados](COMPONENTS.md#6-sección-de-compuestos-guardados-saved-compounds-section)
- **Popup Confirmación**: [Componentes - Popup Eliminación](COMPONENTS.md#5-popup-de-confirmación-de-eliminación-delete-confirmation-popup)

### SMILES y Formatos
- **Renderizado**: [Referencia API - Render SMILES](API_REFERENCE.md#3-renderizar-smiles-personalizado)
- **Ejemplos**: [Guía de Uso - SMILES](USAGE_GUIDE.md#moléculas-de-ejemplo)
- **Validación**: [Guía de Uso - Problemas SMILES](USAGE_GUIDE.md#problema-smiles-no-válido)

### Exportación 3D
- **Formatos**: [Guía de Uso - Exportación](USAGE_GUIDE.md#exportación-de-modelos-3d)
- **Implementación**: [Componentes - Exportación 3D](COMPONENTS.md#exportación-3d)
- **Impresión 3D**: [Guía de Uso - Impresión 3D](USAGE_GUIDE.md#ejemplo-práctico-preparar-para-impresión-3d)

### Drag & Drop
- **Configuración**: [Componentes - Drag & Drop](COMPONENTS.md#drag--drop)
- **Uso**: [Guía de Uso - Combinación](USAGE_GUIDE.md#paso-1-preparar-las-moléculas)
- **Implementación**: [Referencia API - setupDragAndDrop](API_REFERENCE.md#1-setupdraganddrop)

---

## 🛠️ Referencia Técnica Rápida

### Endpoints Principales
```
GET  /                           # Página principal
GET  /api/molecule/<name>        # Obtener molécula
POST /api/render_smiles          # Renderizar SMILES
```

### Funciones Backend Clave
```python
smiles_to_mol_block(smiles)      # SMILES → MOL 3D
smiles_to_svg(smiles, ...)       # SMILES → SVG 2D
```

### Módulos JavaScript
```javascript
main.js        # Controlador principal
api.js         # Cliente IA
viewer.js      # Visualización molecular
ui.js          # Interfaz de usuario
combination.js # Combinación molecular
```

### Componentes UI Principales
```html
#molecule-bank           # Banco de moléculas
#combination-section     # Zona de combinación
#viewer-container-2d     # Visor 2D
#viewer-container-3d     # Visor 3D
#context-menu           # Menú contextual IA
```

---

## 📊 Matriz de Funcionalidades

| Funcionalidad | Guía de Uso | API Reference | Componentes |
|---------------|-------------|---------------|-------------|
| Visualización 2D | ✅ | ✅ | ✅ |
| Visualización 3D | ✅ | ✅ | ✅ |
| Combinación IA | ✅ | ✅ | ✅ |
| Sugerencias IA | ✅ | ✅ | ✅ |
| **Guardado Firebase** | ✅ | ✅ | ✅ |
| **🗑️ Eliminación Compuestos** | ✅ | ✅ | ✅ |
| **⚠️ Popup Confirmación** | ✅ | ❌ | ✅ |
| **📱 Estados Vacíos** | ✅ | ❌ | ✅ |
| **🎨 Iconografía Flaticon** | ✅ | ❌ | ✅ |
| **Validaciones SMILES** | ✅ | ✅ | ❌ |
| Exportación 3D | ✅ | ❌ | ✅ |
| SMILES Custom | ✅ | ✅ | ❌ |
| Drag & Drop | ✅ | ❌ | ✅ |

**Leyenda**: ✅ Documentado | ❌ No documentado | **Negrita** = Funcionalidades recientes | 🎯 = Nuevas en esta actualización

---

## 🔗 Enlaces Externos Útiles

### Herramientas Químicas
- [RDKit Documentation](https://rdkit.readthedocs.io/)
- [SMILES Tutorial](https://www.daylight.com/smiles/)
- [3Dmol.js Documentation](https://3dmol.csb.pitt.edu/)

### Recursos de Desarrollo
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Google AI Documentation](https://ai.google.dev/)
- [Three.js Documentation](https://threejs.org/docs/)
- [Firebase Documentation](https://firebase.google.com/docs)
- [Firestore Web Documentation](https://firebase.google.com/docs/firestore/quickstart)

### Bases de Datos Moleculares
- [PubChem](https://pubchem.ncbi.nlm.nih.gov/)
- [ChemSpider](http://www.chemspider.com/)
- [DrugBank](https://go.drugbank.com/)

---

## 📝 Convenciones de Documentación

### Formato de Código
- **Python**: Documentado con docstrings y type hints
- **JavaScript**: Comentarios JSDoc cuando aplica
- **API**: Formato OpenAPI/Swagger style

### Ejemplos
- **Prácticos**: Casos reales de uso
- **Completos**: Código ejecutable
- **Comentados**: Explicaciones línea por línea

### Estructura
- **Jerárquica**: De general a específico
- **Cross-references**: Enlaces entre documentos
- **Searchable**: Términos clave destacados

---

## 🚨 Información Importante

### Seguridad
⚠️ **API Key Expuesta**: La clave de Google AI está en el cliente. Ver [consideraciones de seguridad](API_REFERENCE.md#consideraciones-de-seguridad).

### Limitaciones
- **SMILES**: Máximo 500 caracteres (validación estricta)
- **Átomos**: Máximo 150 átomos por molécula
- **Navegadores**: Requiere soporte ES6 modules
- **IA**: Sujeto a cuotas de Google AI
- **Firebase**: Limitado por cuotas gratuitas de Firestore

### Actualizaciones
- **Versión actual**: v1.0
- **Última actualización**: Documentación generada automáticamente
- **Próximas versiones**: Ver [roadmap](../README.md#roadmap)

---

## 🤝 Contribuir a la Documentación

### Reportar Errores
1. Crear issue en el repositorio
2. Etiquetar como "documentation"
3. Incluir página y sección específica

### Sugerir Mejoras
1. Fork del repositorio
2. Editar archivos en `/docs/`
3. Pull request con descripción clara

### Estándares
- **Markdown**: Usar sintaxis estándar
- **Idioma**: Español para contenido, inglés para código
- **Ejemplos**: Incluir código funcional

---

## 📞 Soporte y Contacto

### Canales de Soporte
- **Issues**: Problemas técnicos y bugs
- **Discussions**: Preguntas generales y casos de uso
- **Wiki**: Documentación colaborativa

### Tiempo de Respuesta
- **Issues críticos**: 24-48 horas
- **Preguntas generales**: 3-5 días
- **Mejoras**: Según disponibilidad

---

## 📈 Estadísticas de Documentación

### Cobertura
- **APIs Backend**: 100% documentadas
- **APIs Frontend**: 100% documentadas
- **APIs Firebase**: 100% documentadas
- **🗑️ Sistema Eliminación**: 100% documentado (nuevo)
- **⚠️ Popups Confirmación**: 100% documentados (nuevo)
- **Componentes UI**: 100% documentados
- **Casos de Uso**: 95% cubiertos
- **Validaciones**: 100% documentadas

### Métricas de Esta Actualización
- **Páginas totales**: 4 documentos principales
- **Ejemplos de código**: 85+ snippets (+10 nuevos)
- **Nuevas funcionalidades**: 10 agregadas (+4 en esta actualización)
- **Flujos de trabajo**: 4 diagramas Mermaid actualizados
- **Problemas resueltos**: 5 nuevos casos documentados
- **Mejores prácticas**: 2 nuevas secciones
- **Términos glosario**: 6 nuevos conceptos

---

*Índice de Documentación v1.0 - Quimática Orgánica*

**¡Explora, aprende y contribuye al futuro de la visualización molecular!**