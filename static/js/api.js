import { GoogleGenerativeAI } from "https://esm.run/@google/generative-ai";

let genAI = null;
let currentApiKey = null;

// Función para inicializar la API con la clave del usuario
function initializeAPI(apiKey) {
    if (!apiKey || apiKey === currentApiKey) {
        return;
    }
    
    try {
        genAI = new GoogleGenerativeAI(apiKey);
        currentApiKey = apiKey;
        console.log('API de Gemini inicializada con nueva clave');
    } catch (error) {
        console.error('Error al inicializar la API de Gemini:', error);
        throw new Error('Clave API inválida');
    }
}

// Función para obtener el modelo generativo
export function getGenerativeModel(modelName) {
    if (!genAI) {
        throw new Error('API de Gemini no inicializada. Por favor, configura tu clave API en la configuración.');
    }
    return genAI.getGenerativeModel({ model: modelName });
}

// Función para obtener el modelo de sugerencias
export function getSuggestionModel() {
    if (!genAI) {
        throw new Error('API de Gemini no inicializada. Por favor, configura tu clave API en la configuración.');
    }
    return genAI.getGenerativeModel({ model: "gemini-2.5-flash-lite" });
}

// Función para verificar si la API está disponible
export function isAPIAvailable() {
    return genAI !== null;
}

// Función para obtener el estado de la API
export function getAPIStatus() {
    return {
        isAvailable: isAPIAvailable(),
        hasApiKey: currentApiKey !== null
    };
}

// Escuchar cambios en la configuración
if (typeof window !== 'undefined') {
    window.addEventListener('configUpdated', (event) => {
        const { geminiApiKey } = event.detail;
        if (geminiApiKey && geminiApiKey.trim() !== '') {
            initializeAPI(geminiApiKey);
        } else {
            genAI = null;
            currentApiKey = null;
        }
    });

    // Inicializar con la configuración existente si está disponible
    document.addEventListener('DOMContentLoaded', () => {
        if (window.configManager) {
            const apiKey = window.configManager.getApiKey();
            if (apiKey) {
                initializeAPI(apiKey);
            }
        }
    });
} 