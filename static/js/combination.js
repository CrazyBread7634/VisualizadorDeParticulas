import { getGenerativeModel } from './api.js';
import { loadMolecule, loadMoleculeFromData, showViewerLoaders } from './viewer.js';
import { 
    updateButtonState, 
    scrollToResults, 
    showCombinationResult, 
    showCombinationLoading, 
    hideCombinationResult,
    setMoleculeCardsDisabled
} from './ui.js';

let molSlot1 = null;
let molSlot2 = null;

const molSlot1Element = document.getElementById('mol-slot-1');
const molSlot2Element = document.getElementById('mol-slot-2');
const combinationDropZone = document.getElementById('combination-drop-zone');

export function setupDragAndDrop() {
    combinationDropZone.addEventListener('dragover', (event) => {
        event.preventDefault(); 
        event.dataTransfer.dropEffect = 'copy';
        const targetSlot = getSlotFromElement(event.target);
        if (targetSlot && !targetSlot.classList.contains('occupied')) {
            targetSlot.classList.add('drag-over');
        }
    });

    combinationDropZone.addEventListener('dragleave', (event) => {
        const targetSlot = getSlotFromElement(event.target);
        if (targetSlot) {
            targetSlot.classList.remove('drag-over');
        }
    });

    combinationDropZone.addEventListener('drop', handleMoleculeDrop);
}

function getSlotFromElement(element) {
    if (element.classList.contains('mol-slot')) {
        return element;
    }
    return element.closest('.mol-slot');
}

async function handleMoleculeDrop(event) {
    event.preventDefault();
    const droppedMoleculeName = event.dataTransfer.getData('text/plain');
    const targetSlotElement = getSlotFromElement(event.target);
    if (!targetSlotElement) return;
    targetSlotElement.classList.remove('drag-over');
    const isSlot1 = targetSlotElement.id === 'mol-slot-1';
    
    if ((isSlot1 && molSlot2?.name === droppedMoleculeName) || 
        (!isSlot1 && molSlot1?.name === droppedMoleculeName)) {
        alert(`La molécula ${droppedMoleculeName} ya está en el otro slot.`);
        return;
    }

    try {
        const card = document.querySelector(`.molecule-card[data-molecule-name="${droppedMoleculeName}"]`);
        const moleculeData = {
            name: droppedMoleculeName,
            smiles: card.dataset.smiles
        };
        
        if (isSlot1) {
            molSlot1 = moleculeData;
        } else {
            molSlot2 = moleculeData;
        }
        updateSlotUI(targetSlotElement, moleculeData);
        
        if (molSlot1 && molSlot2) {
            updateButtonState('Combinar Moléculas', false);
        } else {
            // Si solo hay una molécula, el botón sigue deshabilitado pero podría decir "Limpiar"
            updateButtonState('Limpiar Combinación', false, false); 
        }

    } catch (error) {
        console.error('Error al procesar datos de la molécula:', error);
        alert('No se pudieron obtener los datos de la molécula desde la tarjeta.');
    }
}

function updateSlotUI(slotElement, moleculeData) {
    slotElement.innerHTML = '';
    slotElement.classList.add('occupied');
    const img = document.createElement('img');
    img.src = `/static/media/${moleculeData.name}.png`; // Corregido: Usar la imagen PNG local
    img.alt = moleculeData.name;
    const nameLabel = document.createElement('p');
    nameLabel.textContent = moleculeData.name;
    nameLabel.style.fontWeight = 'bold';
    slotElement.appendChild(img);
    slotElement.appendChild(nameLabel);
}

export function clearCombination(currentSelectedMolecule) {
    molSlot1 = null;
    molSlot2 = null;
    [molSlot1Element, molSlot2Element].forEach(slot => {
        slot.innerHTML = `<p>Molécula ${slot.id.slice(-1)}</p>`;
        slot.classList.remove('occupied');
    });
    updateButtonState('Limpiar Combinación', false, true);
    document.querySelector('h1').textContent = 'Quimica Orgánica';
    hideCombinationResult();
    loadMolecule(currentSelectedMolecule);
}

export async function combineMolecules() {
    if (!molSlot1 || !molSlot2) return;
    
    updateButtonState('', true); // Inicia carrusel y loader
    setMoleculeCardsDisabled(true); // Deshabilitar tarjetas
    showViewerLoaders("Combinando moléculas..."); // Mostrar loaders en visores

    try {
        const selectedModelName = document.getElementById('model-select').value;
        const combinationModel = getGenerativeModel(selectedModelName);

        const prompt = `Eres un experto en química orgánica y diseño racional de fármacos. Tu tarea es combinar dos moléculas para generar una nueva con propiedades farmacológicas mejoradas.

Molécula 1: ${molSlot1.smiles}
Molécula 2: ${molSlot2.smiles}

Instrucciones:
1. Analiza los sitios reactivos de ambas moléculas.
2. Forma un enlace covalente químicamente viable y estable entre ellas (p. ej., éster, éter, amida).
3. La nueva molécula debe preservar o mejorar la afinidad por los receptores de estrógeno y tener propiedades farmacocinéticas razonables.
4. El resultado final debe ser una molécula estructuralmente plausible y sintéticamente accesible.
5. Por ninguna razon debes añadir texto adicional a la respuesta. Ni comenzar con texto, ni finalizar con texto.
6. No añadas comentarios adicionales a la respuesta.
7. Genera unicamente la cadena SMILES de la molécula combinada.

Genera una respuesta en formato JSON con la siguiente estructura:
{
  "combinedSmiles": "la_cadena_smiles_resultante"
}

Asegúrate de que el valor de "combinedSmiles" sea únicamente la cadena SMILES de la molécula combinada, sin texto adicional.`;
        
        const result = await combinationModel.generateContent(prompt);
        const rawResponse = (await result.response).text();
        
        // La respuesta de la API puede venir con "```json" y "```", los quitamos para un parseo seguro.
        const jsonString = rawResponse.replace(/```(json)?/g, '').trim();
        const responseObject = JSON.parse(jsonString);
        const combinedSmiles = responseObject.combinedSmiles;

        // Validaciones múltiples del SMILES generado
        if (!combinedSmiles || typeof combinedSmiles !== 'string') {
            console.error("La IA no generó un SMILES válido en el JSON:", rawResponse);
            throw new Error('La IA no generó un SMILES válido.');
        }
        
        if (combinedSmiles.includes(' ') || combinedSmiles.length > 300) {
            console.error("SMILES inválido o demasiado largo:", combinedSmiles);
            throw new Error('El SMILES generado es inválido o demasiado complejo.');
        }
        
        // Verificar que no sea una simple concatenación de las moléculas originales
        if (combinedSmiles === molSlot1.smiles + molSlot2.smiles || 
            combinedSmiles === molSlot2.smiles + molSlot1.smiles) {
            console.error("La IA concatenó las moléculas en lugar de combinarlas:", combinedSmiles);
            throw new Error('La IA no realizó una combinación química válida.');
        }
        
        updateButtonState('Renderizando...', true);
        await handleSuggestionClick(combinedSmiles);
        document.querySelector('h1').textContent = 'Resultado de la Combinación';
        await analyzeCombinedMolecule(combinedSmiles);
        scrollToResults();

    } catch (error) {
        console.error('Error durante la combinación de moléculas:', error);
        
        // Limpiar completamente la UI
        hideCombinationResult();
        showViewerLoaders(""); // Limpiar loaders de visualización
        
        let errorMessage = 'Error de Combinación';
        if (error.message && error.message.includes('503')) {
            errorMessage = 'IA Sobrecargada';
        } else if (error.message && error.message.includes('no es válido')) {
            errorMessage = 'SMILES Inválido';
        }
        
        updateButtonState(errorMessage, false, false);
        setMoleculeCardsDisabled(false); // Rehabilitar tarjetas en caso de error
        
        // Mostrar mensaje de error más detallado
        showCombinationResult(`
            <div style="color: #d32f2f; background-color: #ffebee; padding: 20px; border-radius: 8px; text-align: center;">
                <h4>❌ Error en la Combinación</h4>
                <p><strong>La IA generó una molécula demasiado compleja o químicamente imposible.</strong></p>
                <p>Esto puede ocurrir porque:</p>
                <ul style="text-align: left; margin: 15px 0;">
                    <li>El modelo de IA concatenó las moléculas en lugar de combinarlas químicamente</li>
                    <li>La estructura resultante es demasiado grande para procesar</li>
                    <li>La molécula generada no es químicamente estable</li>
                </ul>
                <p><em>Consejo: Intenta usar un modelo de IA diferente.</em></p>
            </div>
        `);
        
        setTimeout(() => {
            updateButtonState('Limpiar Combinación', false, false);
        }, 5000); // Dar más tiempo para leer el error detallado
    }
}

async function analyzeCombinedMolecule(smiles) {
    showCombinationLoading();
    
    try {
        const selectedModelName = document.getElementById('model-select').value;
        const combinationModel = getGenerativeModel(selectedModelName);

        const prompt = `Eres un químico medicinal y farmacólogo molecular experto. Analiza la molécula con SMILES "${smiles}" y proporciona un informe detallado.

La molécula es un derivado combinado de Estradiol y Fulvestrant. Tu análisis debe ser profesional, técnico y conciso.

Genera una respuesta en formato JSON con la siguiente estructura exacta:
{
  "suggestedName": "Un nombre químico apropiado y científicamente razonable.",
  "keyChemicalFeatures": "Un resumen técnico de las características estructurales: grupos funcionales, puntos de interacción, sitios reactivos.",
  "potentialPharmacologicalProperties": "Una explicación de las posibles propiedades biológicas y farmacológicas, considerando su perfil agonista/antagonista sobre receptores de estrógeno.",
  "potentialUses": "Al menos un posible uso farmacológico o área de investigación relevante para esta molécula."
}

No incluyas explicaciones adicionales fuera del formato JSON. Redacta cada campo como si fuera parte de un informe de investigación.`; 
        
        const result = await combinationModel.generateContent(prompt);
        const rawResponse = (await result.response).text();

        // La respuesta de la API puede venir con "```json" y "```", los quitamos.
        const jsonString = rawResponse.replace(/```(json)?/g, '').trim();
        const analysisObject = JSON.parse(jsonString);

        const resultSection = document.getElementById('combination-result-section');
        resultSection.dataset.name = analysisObject.suggestedName;
        resultSection.dataset.smiles = smiles;

        const analysisHtml = `
            <h3>${analysisObject.suggestedName}</h3>
            <div id="ai-analysis-content">
                <h4>Características Químicas Clave</h4>
                <p>${analysisObject.keyChemicalFeatures}</p>
                <h4>Propiedades Farmacológicas Potenciales</h4>
                <p>${analysisObject.potentialPharmacologicalProperties}</p>
                <h4>Posibles Usos o Campos de Investigación</h4>
                <p>${analysisObject.potentialUses}</p>
            </div>
            <button id="save-compound-btn" class="control-btn">Guardar Compuesto</button>
        `;

        showCombinationResult(analysisHtml);
        updateButtonState('Combinación Exitosa', false, false);
        setMoleculeCardsDisabled(false); // Rehabilitar tarjetas al finalizar con éxito

    } catch (error) {
        console.error('Error al analizar la molécula combinada:', error);
        showCombinationResult('<p>Ocurrió un error al generar el análisis.</p>');
        let errorMessage = 'Error en Análisis';
        if (error.message && error.message.includes('503')) {
            errorMessage = 'IA Sobrecargada';
        }
        updateButtonState(errorMessage, false, false);
        setMoleculeCardsDisabled(false); // Rehabilitar tarjetas en caso de error
    }
}

export async function handleSuggestionClick(newSmiles) {
    try {
        console.log('Intentando procesar SMILES:', newSmiles);
        
        // Validación básica del SMILES antes de enviarlo al servidor
        if (!newSmiles || newSmiles.length > 500 || newSmiles.includes(' ')) {
            throw new Error('SMILES inválido o demasiado complejo');
        }
        
        const response = await fetch('/api/render_smiles', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ smiles: newSmiles, show_atoms: true, show_bonds: true })
        });
        if (!response.ok) {
            const errorData = await response.json();
            console.error('Error del servidor al procesar SMILES:', newSmiles, 'Error:', errorData);
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }
        const data = await response.json();
        loadMoleculeFromData(data);
    } catch (error) {
        console.error('Error applying suggestion:', error);
        console.error('SMILES problemático:', newSmiles);
        
        // Limpiar loaders
        showViewerLoaders("");
        updateButtonState('Error SMILES', false, false);
        setMoleculeCardsDisabled(false);
        
        // Mostrar un mensaje de error más amigable al usuario
        const resultSection = document.getElementById('combination-result-section');
        resultSection.innerHTML = `
            <h3>❌ Error al Procesar Molécula</h3>
            <div style="color: #d32f2f; background-color: #ffebee; padding: 20px; border-radius: 8px;">
                <p><strong>No se pudo procesar la molécula generada por la IA.</strong></p>
                <details style="margin-top: 15px;">
                    <summary style="cursor: pointer; font-weight: bold;">Ver detalles técnicos</summary>
                    <div style="background-color: #f5f5f5; padding: 10px; margin-top: 10px; border-radius: 4px;">
                        <p><strong>Error:</strong> ${error.message}</p>
                        <p><strong>SMILES problemático:</strong> <code style="word-break: break-all;">${newSmiles}</code></p>
                        <p><strong>Longitud:</strong> ${newSmiles ? newSmiles.length : 0} caracteres</p>
                    </div>
                </details>
                <div style="margin-top: 15px; padding: 10px; background-color: #e3f2fd; border-left: 4px solid #2196f3;">
                    <strong>💡 Sugerencias:</strong>
                    <ul style="margin: 5px 0; padding-left: 20px;">
                        <li>Prueba con un modelo de IA diferente (ej. GPT-4 en lugar de Gemini)</li>
                        <li>Combina moléculas más simples</li>
                        <li>Intenta la combinación nuevamente</li>
                    </ul>
                </div>
            </div>
        `;
        resultSection.style.display = 'block';
        resultSection.classList.add('visible');
        
        setTimeout(() => {
            updateButtonState('Limpiar Combinación', false, false);
        }, 3000);
        
        throw error; // Re-lanzar para que el error se propague correctamente
    }
}
