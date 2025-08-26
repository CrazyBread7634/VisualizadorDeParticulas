// Configuración del sistema
class ConfigManager {
    constructor() {
        this.config = {
            geminiApiKey: '',
            defaultModel: 'gemini-2.5-flash-lite'
        };
        this.init();
    }

    init() {
        this.loadConfig();
        this.setupEventListeners();
        this.updateUI();
    }

    setupEventListeners() {
        // Botón de configuración
        const configBtn = document.getElementById('config-btn');
        if (configBtn) {
            configBtn.addEventListener('click', () => this.openConfigModal());
        }

        // Botón de cerrar configuración
        const configCloseBtn = document.getElementById('config-close-btn');
        if (configCloseBtn) {
            configCloseBtn.addEventListener('click', () => this.closeConfigModal());
        }

        // Botón de guardar configuración
        const configSaveBtn = document.getElementById('config-save-btn');
        if (configSaveBtn) {
            configSaveBtn.addEventListener('click', () => this.saveConfig());
        }

        // Botón de restablecer configuración
        const configResetBtn = document.getElementById('config-reset-btn');
        if (configResetBtn) {
            configResetBtn.addEventListener('click', () => this.resetConfig());
        }

        // Toggle para mostrar/ocultar clave API
        const toggleApiKeyBtn = document.getElementById('toggle-api-key');
        if (toggleApiKeyBtn) {
            toggleApiKeyBtn.addEventListener('click', () => this.toggleApiKeyVisibility());
        }

        // Cerrar modal al hacer clic fuera
        const configModal = document.getElementById('config-modal');
        if (configModal) {
            configModal.addEventListener('click', (e) => {
                if (e.target === configModal) {
                    this.closeConfigModal();
                }
            });
        }

        // Cerrar modal con Escape
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeConfigModal();
            }
        });
    }

    openConfigModal() {
        const modal = document.getElementById('config-modal');
        if (modal) {
            modal.style.display = 'flex';
            this.updateUI();
        }
    }

    closeConfigModal() {
        const modal = document.getElementById('config-modal');
        if (modal) {
            modal.style.display = 'none';
        }
    }

    updateUI() {
        // Actualizar campos con la configuración actual
        const apiKeyInput = document.getElementById('gemini-api-key');
        const defaultModelSelect = document.getElementById('default-model');

        if (apiKeyInput) {
            apiKeyInput.value = this.config.geminiApiKey;
        }

        if (defaultModelSelect) {
            defaultModelSelect.value = this.config.defaultModel;
        }
    }

    async saveConfig() {
        const apiKeyInput = document.getElementById('gemini-api-key');
        const defaultModelSelect = document.getElementById('default-model');

        if (apiKeyInput && defaultModelSelect) {
            const newConfig = {
                geminiApiKey: apiKeyInput.value.trim(),
                defaultModel: defaultModelSelect.value
            };

            // Validar la clave API
            if (newConfig.geminiApiKey && !this.isValidApiKey(newConfig.geminiApiKey)) {
                this.showNotification('La clave API no parece ser válida. Verifica que tenga el formato correcto.', 'error');
                return;
            }

            try {
                // Guardar en localStorage
                this.config = newConfig;
                localStorage.setItem('quimaticaConfig', JSON.stringify(this.config));

                // Actualizar el modelo por defecto en el selector principal
                this.updateMainModelSelector();

                this.showNotification('Configuración guardada exitosamente.', 'success');
                this.closeConfigModal();

                // Disparar evento para que otros módulos se actualicen
                window.dispatchEvent(new CustomEvent('configUpdated', { detail: this.config }));
                
                // Actualizar el estado del botón de combinación
                if (window.updateButtonStateForAPI) {
                    window.updateButtonStateForAPI();
                }

            } catch (error) {
                console.error('Error al guardar la configuración:', error);
                this.showNotification('Error al guardar la configuración.', 'error');
            }
        }
    }

    resetConfig() {
        if (confirm('¿Estás seguro de que deseas restablecer la configuración? Esto eliminará tu clave API guardada.')) {
            this.config = {
                geminiApiKey: '',
                defaultModel: 'gemini-2.5-flash-lite'
            };
            
            localStorage.removeItem('quimaticaConfig');
            this.updateUI();
            this.updateMainModelSelector();
            
            this.showNotification('Configuración restablecida.', 'info');
            
            // Disparar evento para que otros módulos se actualicen
            window.dispatchEvent(new CustomEvent('configUpdated', { detail: this.config }));
            
            // Actualizar el estado del botón de combinación
            if (window.updateButtonStateForAPI) {
                window.updateButtonStateForAPI();
            }
        }
    }

    toggleApiKeyVisibility() {
        const apiKeyInput = document.getElementById('gemini-api-key');
        const toggleBtn = document.getElementById('toggle-api-key');
        const icon = toggleBtn.querySelector('i');

        if (apiKeyInput.type === 'password') {
            apiKeyInput.type = 'text';
            icon.className = 'fi fi-sr-eye-crossed';
            toggleBtn.title = 'Ocultar';
        } else {
            apiKeyInput.type = 'password';
            icon.className = 'fi fi-sr-eye';
            toggleBtn.title = 'Mostrar';
        }
    }

    loadConfig() {
        try {
            const savedConfig = localStorage.getItem('quimaticaConfig');
            if (savedConfig) {
                this.config = { ...this.config, ...JSON.parse(savedConfig) };
            }
        } catch (error) {
            console.error('Error al cargar la configuración:', error);
        }
    }

    getConfig() {
        return { ...this.config };
    }

    getApiKey() {
        return this.config.geminiApiKey;
    }

    getDefaultModel() {
        return this.config.defaultModel;
    }

    isValidApiKey(apiKey) {
        // Validación básica: debe tener al menos 20 caracteres y empezar con "AIza"
        return apiKey.length >= 20 && apiKey.startsWith('AIza');
    }

    updateMainModelSelector() {
        const mainModelSelect = document.getElementById('model-select');
        if (mainModelSelect) {
            mainModelSelect.value = this.config.defaultModel;
        }
    }

    showNotification(message, type = 'info') {
        // Crear notificación temporal
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fi fi-br-${type === 'success' ? 'check' : type === 'error' ? 'cross-small' : 'info'}"></i>
                <span>${message}</span>
            </div>
        `;

        // Agregar estilos
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? '#28a745' : type === 'error' ? '#dc3545' : '#17a2b8'};
            color: white;
            padding: 15px 20px;
            border-radius: var(--border-radius);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            z-index: 10000;
            animation: slideInRight 0.3s ease-out;
            max-width: 400px;
            font-family: var(--font-family-sans);
        `;

        // Agregar animación CSS
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideInRight {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
        `;
        document.head.appendChild(style);

        document.body.appendChild(notification);

        // Remover después de 4 segundos
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-in';
            notification.style.transform = 'translateX(100%)';
            notification.style.opacity = '0';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 4000);
    }

    // Método para verificar si la configuración está completa
    isConfigComplete() {
        return this.config.geminiApiKey && this.config.geminiApiKey.trim() !== '';
    }

    // Método para obtener la configuración actualizada
    getCurrentConfig() {
        return this.config;
    }
}

// Inicializar el gestor de configuración cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', () => {
    window.configManager = new ConfigManager();
});

// Exportar para uso en otros módulos
export { ConfigManager };
