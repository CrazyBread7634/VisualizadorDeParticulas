import { suggestionModel } from './api.js';
import { getCurrentSmiles } from './viewer.js';
import { handleSuggestionClick } from './combination.js';

const contextMenu = document.getElementById('context-menu');
const contextMenuContent = document.getElementById('context-menu-content');
const contextMenuTitle = document.getElementById('context-menu-title');
const contextMenuLoader = document.getElementById('context-menu-loader');
const customPromptBtn = document.getElementById('custom-prompt-btn');
const customPromptInput = document.getElementById('custom-prompt-input');
const dragHandle = document.getElementById('drag-handle');
const clearCombinationBtn = document.getElementById('clear-combination-btn');
const combinationResultSection = document.getElementById('combination-result-section');
const moleculeCards = document.querySelectorAll('.molecule-card');

let isFetchingSuggestions = false; // Variable de bloqueo
let currentContext = {}; // Para guardar el contexto del átomo/enlace

/**
 * Extrae de forma robusta un objeto JSON de una cadena de texto sin procesar.
 * Intenta encontrar bloques de código markdown o arrays JSON directos.
 * @param {string} rawText - La respuesta de la IA.
 * @returns {object|null} El objeto JSON parseado o null si no se encuentra.
 */
function extractJson(rawText) {
    // Método 1: Buscar un bloque de código markdown (```json ... ``` o ``` ... ```)
    const markdownMatch = rawText.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
    if (markdownMatch && markdownMatch[1]) {
        try {
            return JSON.parse(markdownMatch[1]);
        } catch (e) {
            console.warn("No se pudo parsear el JSON del bloque markdown, intentando otro método.", e);
        }
    }

    // Método 2: Buscar un array JSON crudo incrustado en el texto
    const arrayMatch = rawText.match(/(\[\s*{[\s\S]*?}\s*\])/);
     if (arrayMatch && arrayMatch[0]) {
        try {
            return JSON.parse(arrayMatch[0]);
        } catch (e) {
             console.warn("La cadena que coincide con el regex de array no era un JSON válido.", e);
        }
    }

    // Si ambos métodos fallan
    return null;
}

/**
 * Habilita o deshabilita la interacción con las tarjetas de moléculas.
 * @param {boolean} disabled - True para deshabilitar, false para habilitar.
 */
export function setMoleculeCardsDisabled(disabled) {
    moleculeCards.forEach(card => {
        if (disabled) {
            card.classList.add('disabled');
        } else {
            card.classList.remove('disabled');
        }
    });
}

/**
 * Limpia una cadena de texto para extraer únicamente la parte que parece ser un SMILES.
 * Se asume que el SMILES es la última "palabra" en la cadena.
 * @param {string} rawSmiles - La cadena SMILES potencialmente sucia de la IA.
 * @returns {string} La cadena SMILES limpia.
 */
function cleanSmilesFromAiResponse(rawSmiles) {
    if (!rawSmiles) return "";
    const cleaned = String(rawSmiles).trim();
    // Divide por espacios y toma la última parte.
    const parts = cleaned.split(/\s+/);
    let potentialSmiles = parts[parts.length - 1];
    
    // Eliminar puntuación común al final de la cadena.
    potentialSmiles = potentialSmiles.replace(/[.,;:]+$/, '');
    
    return potentialSmiles;
}

export function handleAtomClick(atom, event = null) {
    if (isFetchingSuggestions) return; // Bloquear si ya hay una petición
    currentContext = { type: 'atom', id: atom.serial };
    const atomElement = atom.elem;

    const title = `Sugerencias para Átomo ${atomElement}-${atom.serial}`;
    if (event) {
        event.preventDefault();
        showContextMenu(event, title);
    } else {
        const mockEvent = { pageX: 400, pageY: 300 };
        showContextMenu(mockEvent, title);
    }
}

export function handleBondClick(event, bondData) {
    if (isFetchingSuggestions) return; // Bloquear si ya hay una petición
    currentContext = { type: 'bond', id: bondData.bond_index };
    const title = `Sugerencias para Enlace E-${bondData.bond_index}`;
    showContextMenu(event, title);
}

function hideContextMenu() {
    if (contextMenu) {
        contextMenu.style.display = 'none';
        contextMenuContent.innerHTML = '';
        contextMenuTitle.textContent = '';
        isFetchingSuggestions = false; // Desbloquear al cerrar
    }
}

async function showContextMenu(event, title) {
    contextMenuTitle.textContent = title;
    contextMenu.style.left = `${event.pageX + 15}px`;
    contextMenu.style.top = `${event.pageY}px`;
    contextMenu.style.display = 'flex';
    contextMenuContent.style.display = 'none';
    contextMenuLoader.style.display = 'block';
    isFetchingSuggestions = true;

    try {
        const currentSmiles = getCurrentSmiles();
        const contextType = currentContext.type === 'atom' ? `el átomo con índice ${currentContext.id}` : `el enlace con índice ${currentContext.id}`;
        
        const prompt = `
Eres un asistente de química que responde únicamente en formato JSON. No añadas texto introductorio, explicaciones ni formato markdown.

Para la molécula con SMILES: "${currentSmiles}"

Sugiere exactamente 3 modificaciones químicas realistas para ${contextType}.

Tu respuesta DEBE ser un array de objetos JSON válido. Cada objeto debe contener dos claves:
1. "description": Una descripción muy breve de la modificación (máximo 5 palabras).
2. "new_smiles": La cadena SMILES válida de la molécula resultante.

Ejemplo de formato de respuesta:
[
  {"description": "Añadir grupo metilo", "new_smiles": "CC..."},
  {"description": "Crear un doble enlace", "new_smiles": "C=C..."},
  {"description": "Reemplazar por nitrógeno", "new_smiles": "CN..."}
]

Responde SÓLO con el array JSON.
`;

        const result = await suggestionModel.generateContent(prompt);
        const rawResponse = (await result.response).text();

        // Usar la nueva función de extracción robusta
        const suggestions = extractJson(rawResponse);

        if (!suggestions) {
            console.error("No se pudo extraer el JSON de la respuesta de la IA:", rawResponse);
            throw new Error("La respuesta de la IA no contenía un JSON válido.");
        }

        contextMenuContent.innerHTML = '';
        if (suggestions && suggestions.length > 0) {
            suggestions.forEach(sugg => {
                const btn = document.createElement('button');
                btn.innerHTML = `<i class="fi fi-rr-sparkles"></i><span>${sugg.description}</span>`;
                btn.onclick = () => {
                    // Limpieza robusta del SMILES de la sugerencia.
                    const cleanedSmiles = cleanSmilesFromAiResponse(sugg.new_smiles);
                    handleSuggestionClick(cleanedSmiles);
                    hideContextMenu();
                };
                contextMenuContent.appendChild(btn);
            });
        } else {
            contextMenuContent.textContent = 'No se encontraron sugerencias.';
        }
        contextMenuContent.style.display = 'flex';

    } catch (error) {
        console.error('Error fetching suggestions:', error);
        let errorMessage = 'Error al obtener sugerencias.';
        if (error.message && error.message.includes('503')) {
            errorMessage = 'El modelo de IA está sobrecargado. Por favor, inténtalo de nuevo en unos momentos.';
        }
        contextMenuContent.innerHTML = `<p>${errorMessage}</p>`;
        contextMenuContent.style.display = 'flex';
    } finally {
        contextMenuLoader.style.display = 'none';
        isFetchingSuggestions = false; // Desbloquear al finalizar
    }
}

async function handleCustomPrompt() {
    if (isFetchingSuggestions || !customPromptInput.value) return;
    
    hideContextMenu();
    const userPrompt = customPromptInput.value;
    const prompt = `Dada la molécula con SMILES '${getCurrentSmiles()}', y la instrucción del usuario: "${userPrompt}", genera el nuevo SMILES resultante. Devuelve solo la cadena SMILES.`;
    
    // Muestra un loader global o en el botón principal
    updateButtonState('Procesando IA...', true);

    try {
        const result = await suggestionModel.generateContent(prompt); // Usar el modelo rápido aquí también
        const rawResponse = (await result.response).text();
        const newSmiles = rawResponse.replace(/```(smiles|json|html)?/g, '').replace(/```/g, '').trim().split('\n').pop().trim();
        
        if (!newSmiles || newSmiles.includes(' ')) {
            throw new Error("La IA no devolvió un SMILES válido.");
        }

        await handleSuggestionClick(newSmiles);
        updateButtonState('Modificación Aplicada', false);

    } catch (error) {
        console.error("Error con el prompt personalizado:", error);
        let alertMessage = `Error: ${error.message}`;
        if (error.message && error.message.includes('503')) {
            alertMessage = 'El modelo de IA está sobrecargado. Por favor, inténtalo de nuevo en unos momentos.';
        }
        alert(alertMessage);
        updateButtonState('Error de IA', false, false);
    } finally {
        customPromptInput.value = '';
    }
}


// --- Lógica de la Interfaz de Usuario Principal ---

const feedbackPhrases = [
    "Mezclando reactivos...",
    "Calentando el matraz...",
    "Ajustando el pH...",
    "Consultando orbitales...",
    "Calculando entalpía...",
    "Destilando el producto...",
    "Evitando reacciones secundarias...",
    "Verificando la quiralidad...",
    "Purificando la muestra...",
    "Dibujando diagramas de energía...",
    "Combinando moléculas...",
    "Buscando sitios reactivos...",
    "Formando enlaces covalentes...",
    "Optimizando propiedades farmacológicas...",
    "Generando moléculas sintéticas...",
    "Analizando moléculas...",
    "Evaluando estabilidad química...",
    "Calculando propiedades fisicoquímicas...",
    "Evaluando propiedades farmacológicas...",
    "Evaluando propiedades farmacocinéticas...",
    "Evaluando propiedades farmacodinámicas...",
    "Evaluando propiedades farmacotoxicologicas...",
];
let feedbackInterval = null;

function startFeedbackCarousel(btnText) {
    if (feedbackInterval) return; // Evitar iniciar múltiples carruseles

    let phraseIndex = 0;
    const showNextPhrase = () => {
        btnText.classList.add('fading');
        setTimeout(() => {
            phraseIndex = (phraseIndex + 1) % feedbackPhrases.length;
            btnText.textContent = feedbackPhrases[phraseIndex];
            btnText.classList.remove('fading');
        }, 400);
    };
    
    btnText.textContent = feedbackPhrases[0];
    feedbackInterval = setInterval(showNextPhrase, 2000);
}

function stopFeedbackCarousel() {
    clearInterval(feedbackInterval);
    feedbackInterval = null;
}

const initialMolecules = {
    "estradiol": "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CC[C@@H]2O",
    "fulvestrant": "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CC[C@H](C(F)(F)C(F)(F)S(=O)CCCCCCCCC)C2"
};

export function populateMoleculeCards() {
    moleculeCards.forEach(card => {
        const moleculeName = card.dataset.moleculeName;
        if (initialMolecules[moleculeName]) {
            card.dataset.smiles = initialMolecules[moleculeName];
            
            card.innerHTML = ''; // Limpiar loader
            
            const img = document.createElement('img');
            img.src = `/static/media/${moleculeName}.png`; // Cargar desde la carpeta static/media
            img.alt = moleculeName;
            
            const nameLabel = document.createElement('span');
            nameLabel.textContent = moleculeName;

            card.appendChild(img);
            card.appendChild(nameLabel);
        } else {
            card.innerHTML = `<p>Error</p>`;
            console.error(`No se encontró la molécula inicial: ${moleculeName}`);
        }
    });
}

export function updateCardSelection(selectedName) {
    moleculeCards.forEach(card => {
        if (card.dataset.moleculeName === selectedName) {
            card.classList.add('selected');
        } else {
            card.classList.remove('selected');
        }
    });
}

export function updateButtonState(text, isLoading, disabled = false) {
    const btnText = clearCombinationBtn.querySelector('.carrusel-text');
    clearCombinationBtn.classList.toggle('loading', isLoading);
    clearCombinationBtn.disabled = isLoading || disabled;


    if (isLoading) {
        startFeedbackCarousel(btnText);
    } else {
        stopFeedbackCarousel();
        btnText.textContent = text;
    }
}



export function scrollToResults() {
    combinationResultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

export function showCombinationResult(htmlContent) {
    combinationResultSection.innerHTML = `<h3>Análisis de la Nueva Molécula</h3>${htmlContent}`;
    combinationResultSection.classList.add('visible');
}

export function showCombinationLoading() {
    combinationResultSection.innerHTML = `<h3>Análisis de la Nueva Molécula</h3>
        <lord-icon 
            src="https://cdn.lordicon.com/tewlfgbl.json" 
            trigger="loop" 
            state="loop-dispersion"
            colors="primary:#616161"
            style="width:80px;height:80px;display:block;margin:auto;"></lord-icon>
        <p style="text-align:center;">Generando análisis con IA...</p>`;
    combinationResultSection.classList.add('visible');
}

export function hideCombinationResult() {
    combinationResultSection.classList.remove('visible');
    setTimeout(() => {
        combinationResultSection.innerHTML = '';
    }, 500);
} 

function makeDraggable() {
    let isDragging = false;
    let initialX, initialY, offsetX, offsetY;

    dragHandle.addEventListener('mousedown', (e) => {
        isDragging = true;
        
        initialX = e.clientX;
        initialY = e.clientY;

        offsetX = contextMenu.offsetLeft;
        offsetY = contextMenu.offsetTop;

        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);

        // Evitar la selección de texto mientras se arrastra
        document.body.style.userSelect = 'none';
    });

    function onMouseMove(e) {
        if (!isDragging) return;
        const dx = e.clientX - initialX;
        const dy = e.clientY - initialY;
        contextMenu.style.left = `${offsetX + dx}px`;
        contextMenu.style.top = `${offsetY + dy}px`;
    }

    function onMouseUp() {
        isDragging = false;
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
        // Restaurar la selección de texto
        document.body.style.userSelect = '';
    }
}

// (Se asume que esto se llamará desde main.js)
export function initContextMenu() {
    customPromptBtn.addEventListener('click', handleCustomPrompt);
    document.getElementById('context-menu-close').addEventListener('click', hideContextMenu);
    makeDraggable(); // Activar la lógica de arrastre
} 