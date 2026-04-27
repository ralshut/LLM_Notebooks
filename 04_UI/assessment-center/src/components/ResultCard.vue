<template>
    <div class="result-card">
        <div class="result-header">
            <span class="icon">📊</span>
            <h2>Dein Assessment Ergebnis</h2>
        </div>
        
        <div class="result-content">
            <pre>{{ result }}</pre>
        </div>
        
        <button @click="$emit('restart')" class="restart-btn">
            🔄 Neues Assessment starten
        </button>
    </div>
</template>

<script setup>
/**
 * ResultCard - Zeigt das Ergebnis des Scorers an
 * 
 * Props:
 * - result: string - Das Feedback vom Scorer
 * 
 * Emits:
 * - restart: Wird ausgelöst wenn User neu starten will
 */
defineProps({
    result: {
        type: String,
        required: true
    }
})

defineEmits(['restart'])
</script>

<style scoped>
.result-card {
    padding: 30px;
    display: flex;
    flex-direction: column;
    gap: 20px;
    min-height: 400px;
}

.result-header {
    text-align: center;
}

.result-header .icon {
    font-size: 3rem;
    display: block;
    margin-bottom: 10px;
}

.result-header h2 {
    color: #333;
    font-size: 1.3rem;
    margin: 0;
}

.result-content {
    flex: 1;
    background: #f8f9fa;
    padding: 20px;
    border-radius: 12px;
    overflow-y: auto;
}

.result-content pre {
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: inherit;
    margin: 0;
    line-height: 1.6;
    color: #333;
}

.restart-btn {
    padding: 14px 28px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 25px;
    cursor: pointer;
    font-weight: 600;
    font-size: 1rem;
    transition: transform 0.2s;
}

.restart-btn:hover {
    transform: scale(1.02);
}
</style>
