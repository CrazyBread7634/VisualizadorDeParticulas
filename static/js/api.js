import { GoogleGenerativeAI } from "https://esm.run/@google/generative-ai";

const API_KEY = "AIzaSyAox4uJYUeLZet5YrR7R7BT7q0vmiluI4w";
const genAI = new GoogleGenerativeAI(API_KEY);

/**
 * Obtiene una instancia de un modelo generativo de Google AI por su nombre.
 * @param {string} modelName - El nombre del modelo a obtener (ej. 'gemini-1.5-flash').
 * @returns {import("@google/generative-ai").GenerativeModel}
 */
export function getGenerativeModel(modelName) {
    return genAI.getGenerativeModel({ model: modelName });
}

// Modelo más rápido y ligero, ideal para las sugerencias rápidas
export const suggestionModel = genAI.getGenerativeModel({ model: "gemini-2.5-flash-lite" }); 