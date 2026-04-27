<template>
    <div class="app-container">
        <div class="chat-app">
            <header class="app-header">
                <h1>AI Assessment Center</h1>
                <span class="phase-badge" :class="currentPhase">
                    {{ phaseLabels[currentPhase] }}
                </span>
            </header>

            <template v-if="currentPhase !== 'result'">
                <ChatWindow 
                    :messages="chatHistory" 
                    :loading="loading" 
                />
                <ChatInput 
                    :disabled="loading"
                    @send="handleUserMessage"
                />
            </template>

            <ResultCard 
                v-else 
                :result="finalResult"
                @restart="restart"
            />
        </div>
    </div>
</template>

<script setup>
import ChatWindow from './components/ChatWindow.vue'
import ChatInput from './components/ChatInput.vue'
import ResultCard from './components/ResultCard.vue'
import { useAssessment } from './composables/useAssessment.js'

// State-Variablen und Funktionen aus dem Composable holen
const { 
    chatHistory, 
    loading, 
    currentPhase, 
    finalResult, 
    handleUserMessage, 
    restart 
} = useAssessment()

const phaseLabels = {
    host: '👔 Interview',
    angry_customer: '😠 Kundenservice',
    scorer: '⏳ Auswertung...',
    result: '✅ Abgeschlossen'
}
</script>

<style scoped>
.app-container {
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 20px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
}

.chat-app {
    width: 100%;
    max-width: 500px;
    height: 700px;
    background: white;
    border-radius: 16px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.app-header {
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    text-align: center;
}

.app-header h1 {
    font-size: 1.3rem;
    margin: 0 0 10px 0;
}

.phase-badge {
    display: inline-block;
    padding: 6px 14px;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
}

.phase-badge.angry {
    background: rgba(255, 100, 100, 0.3);
}

.phase-badge.scorer {
    background: rgba(255, 200, 100, 0.3);
}

.phase-badge.result {
    background: rgba(100, 255, 150, 0.3);
}

.debug-bar {
    padding: 8px;
    background: #333;
    color: #0f0;
    font-family: monospace;
    font-size: 0.75rem;
    text-align: center;
}
</style>
