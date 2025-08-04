import { GoogleGenerativeAI } from "https://esm.run/@google/generative-ai";

const API_KEY = "AIzaSyAox4uJYUeLZet5YrR7R7BT7q0vmiluI4w"; //Se deja aqui para fines practicos
const genAI = new GoogleGenerativeAI(API_KEY);

export function getGenerativeModel(modelName) {
    return genAI.getGenerativeModel({ model: modelName });
}

export const suggestionModel = genAI.getGenerativeModel({ model: "gemini-2.5-flash-lite" }); 